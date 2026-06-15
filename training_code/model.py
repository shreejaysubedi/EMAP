"""EMAP arousal model — two-stream (raw MLP + topomap CNN) + BiLSTM.

Subject distribution shift is the main hazard: train/val use disjoint
participants, so absolute EEG levels are not directly comparable across
subjects. The forward pass therefore subtracts each trial's running mean for
both the EEG and peripheral inputs before computing features. This makes the
model see deviations from that subject/trial baseline.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_idw_kernel(pos2d: np.ndarray, grid_size: int = 24, power: float = 4.0):
    xs = np.linspace(-1, 1, grid_size)
    ys = np.linspace(-1, 1, grid_size)
    gx, gy = np.meshgrid(xs, ys, indexing='xy')
    grid = np.stack([gx, gy], axis=-1).reshape(-1, 2)
    d = np.linalg.norm(grid[:, None, :] - pos2d[None, :, :], axis=-1)
    w = 1.0 / (d ** power + 1e-9)
    w = w / w.sum(axis=1, keepdims=True)
    mask = (gx ** 2 + gy ** 2 <= 1.05).astype(np.float32).reshape(grid_size, grid_size)
    return w.astype(np.float32), mask


class SpatialCNN(nn.Module):
    """Compact CNN over (B*, 4, H, W) topomap. No BN/GN; uses 1x1 LayerNorm style affine.

    GroupNorm was empirically problematic (slow learning); we instead rely on
    inputs being already z-scored and use a small CNN with strong weight decay.
    """

    def __init__(self, in_bands: int = 4, base: int = 16, d_out: int = 64):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.net = nn.Sequential(
            nn.Conv2d(in_bands, c1, 3, padding=1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c2, c2, 3, padding=1), nn.GELU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, d_out), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class EmapModel(nn.Module):
    def __init__(self, idw_kernel: np.ndarray, head_mask: np.ndarray,
                 grid_size: int = 24, d_raw: int = 160, d_spatial: int = 64,
                 d_hidden: int = 192, lstm_layers: int = 2, dropout: float = 0.35,
                 n_bands: int = 4, use_cnn: bool = True, demean: bool = True,
                 noise_std: float = 0.0, ch_drop: float = 0.0):
        super().__init__()
        self.grid = grid_size
        self.n_bands = n_bands
        self.use_cnn = use_cnn
        self.demean = demean
        self.noise_std = noise_std
        self.ch_drop = ch_drop
        self.register_buffer('idw', torch.from_numpy(idw_kernel).float())
        self.register_buffer('mask_img',
                             torch.from_numpy(head_mask).float().view(1, 1, grid_size, grid_size))
        # Raw input dim: 256 EEG (z) + 256 EEG (de-meaned) + 16 per-band stats
        #              + 4 peri (z) + 4 peri (de-meaned) + 8 peri context (d1+d3)
        #              + 2 position features (bin frac, log length)
        raw_in = n_bands * 64 * 2 + n_bands * 4 + 4 * 2 + 4 * 2 + 2
        self.raw_mlp = nn.Sequential(
            nn.Linear(raw_in, d_raw), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(d_raw, d_raw), nn.GELU(),
        )
        if use_cnn:
            self.cnn = SpatialCNN(in_bands=n_bands, base=16, d_out=d_spatial)
            self.cnn_proj = nn.Linear(d_spatial, d_raw)
        self.fuse = nn.Sequential(
            nn.Linear(d_raw, d_hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(d_hidden, d_hidden, num_layers=lstm_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_hidden), nn.Dropout(dropout),
            nn.Linear(2 * d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )
        with torch.no_grad():
            self.head[-1].bias.fill_(0.486)
        # Aux head for trial-mean arousal regression (multi-task regularization)
        self.aux_trial_head = nn.Linear(2 * d_hidden, 1)
        with torch.no_grad():
            self.aux_trial_head.bias.fill_(0.486)

    def topomap(self, eeg_bt_chb: torch.Tensor) -> torch.Tensor:
        x = eeg_bt_chb.transpose(-1, -2)
        flat = x @ self.idw.t()
        H = self.grid
        img = flat.view(*flat.shape[:-1], H, H) * self.mask_img
        return img

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor):
        """Mean over the time dim weighted by mask. x: (B, T, ...); mask: (B, T)."""
        m = mask.unsqueeze(-1) if x.dim() == 3 else mask.unsqueeze(-1).unsqueeze(-1)
        msum = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        if x.dim() == 4:
            msum = msum.unsqueeze(-1)
            return (x * m).sum(dim=1) / msum
        return (x * m).sum(dim=1) / msum

    def forward(self, eeg: torch.Tensor, peri: torch.Tensor, mask: torch.Tensor,
                bin_idx: torch.Tensor = None, trial_len: torch.Tensor = None,
                train_aug: bool = False):
        """
        eeg: (B, T, 64, 4)  peri: (B, T, 4)  mask: (B, T)
        bin_idx: (B, T) int — bin position within trial (0-based)
        trial_len: (B,) — total bins per trial (or any positive number)
        train_aug: enable per-batch augmentations (channel dropout, noise)
        """
        B, T = eeg.shape[:2]
        # ----- Augmentations -----
        if train_aug and self.training:
            if self.ch_drop > 0:
                # Drop a random subset of channels for the whole batch
                ch_mask = (torch.rand(64, device=eeg.device) > self.ch_drop).float()
                eeg = eeg * ch_mask.view(1, 1, 64, 1)
            if self.noise_std > 0:
                eeg = eeg + torch.randn_like(eeg) * self.noise_std
                peri = peri + torch.randn_like(peri) * self.noise_std

        # ----- Trial-level de-meaning -----
        if self.demean:
            eeg_mu = self._masked_mean(eeg, mask)        # (B, 64, 4)
            eeg_d = eeg - eeg_mu.unsqueeze(1)
            peri_mu = self._masked_mean(peri, mask)      # (B, 4)
            peri_d = peri - peri_mu.unsqueeze(1)
        else:
            eeg_d = eeg
            peri_d = peri

        # ----- Feature engineering -----
        eeg_flat = eeg.flatten(2)
        eeg_d_flat = eeg_d.flatten(2)
        g_mean = eeg_d.mean(dim=2)
        g_std  = eeg_d.std(dim=2)
        g_max  = eeg_d.amax(dim=2)
        g_min  = eeg_d.amin(dim=2)
        peri_d1 = torch.diff(peri, dim=1, prepend=peri[:, :1])
        peri_lag3 = F.pad(peri[:, :-3], (0, 0, 3, 0))
        peri_d3 = peri - peri_lag3

        # Position features
        if bin_idx is None or trial_len is None:
            bin_frac = torch.zeros(B, T, 1, device=eeg.device)
            log_len = torch.zeros(B, T, 1, device=eeg.device)
        else:
            bin_frac = (bin_idx.float() / trial_len.unsqueeze(1).clamp_min(1).float()).unsqueeze(-1)
            log_len = torch.log1p(trial_len.float()).unsqueeze(1).unsqueeze(-1).expand(B, T, 1)

        raw_in = torch.cat([eeg_flat, eeg_d_flat,
                            g_mean, g_std, g_max, g_min,
                            peri, peri_d,
                            peri_d1, peri_d3,
                            bin_frac, log_len], dim=-1)
        raw = self.raw_mlp(raw_in)
        if self.use_cnn:
            topo = self.topomap(eeg_d)        # use de-meaned EEG for spatial signal
            spatial = self.cnn(topo.flatten(0, 1)).view(B, T, -1)
            raw = raw + self.cnn_proj(spatial)
        fused = self.fuse(raw) * mask.unsqueeze(-1)
        h, _ = self.lstm(fused)
        out_bin = self.head(h).squeeze(-1)                            # (B, T)
        # Trial-mean prediction from masked-mean pooled hidden
        h_pool = self._masked_mean(h, mask)                           # (B, 2*d_hidden)
        out_trial = self.aux_trial_head(h_pool).squeeze(-1)           # (B,)
        return out_bin, out_trial


# Keep backwards-compatible alias for prediction.py
EmapTopoLSTM = EmapModel
