"""Enhanced EmapModel with optional spatial and feature flags.

Flags:
  - peri_in_topo: broadcast the 4 peripherals as additional channels in the
                  topomap input (4 bands + 4 peripherals = 8 channels)
  - use_quartile: append loop quartile (0..3) as both an ordinal scalar and a
                  4-way one-hot embedding into the raw-feature MLP
  - use_trial_id: learn an embedding for stimulus / trial id and add it to the
                  per-timestep feature stack
  - csd_maps:     compute a discrete spatial Laplacian on each topomap and
                  concatenate it as extra channels (the "arrow maps" idea, best
                  guess = current source density / surface Laplacian)
  - grid_size:    bumped up (24 -> 48 etc) for the "larger topomaps" idea

Everything else (CNN + raw MLP + BiLSTM + head) keeps the same shape as the
original `EmapModel` so this is a clean ablation.
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


def laplacian_kernel():
    """3x3 discrete Laplacian — the CSD approximation."""
    return torch.tensor([[0.0, -1.0, 0.0],
                         [-1.0, 4.0, -1.0],
                         [0.0, -1.0, 0.0]]).view(1, 1, 3, 3)


class SpatialCNNv2(nn.Module):
    """Compact CNN that adapts to whatever channel count the topomap stack has."""

    def __init__(self, in_channels: int, base: int = 16, d_out: int = 64):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1), nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c2, c2, 3, padding=1), nn.GELU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, d_out), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class EmapModelV2(nn.Module):
    def __init__(self, idw_kernel: np.ndarray, head_mask: np.ndarray,
                 grid_size: int = 24, d_raw: int = 192, d_spatial: int = 64,
                 d_hidden: int = 256, lstm_layers: int = 2, dropout: float = 0.3,
                 n_bands: int = 4, n_peri: int = 4,
                 # --- optional feature flags ---
                 peri_in_topo: bool = False,
                 use_quartile: bool = False,
                 use_trial_id: bool = False,
                 csd_maps: bool = False,
                 n_trials: int = 25,        # trial-id vocab size (24 stimuli + 1 pad)
                 trial_emb_dim: int = 8,
                 train_mean: float = 0.486):
        super().__init__()
        self.grid = grid_size
        self.n_bands = n_bands
        self.n_peri = n_peri
        self.peri_in_topo = peri_in_topo
        self.use_quartile = use_quartile
        self.use_trial_id = use_trial_id
        self.csd_maps = csd_maps
        self.train_mean = float(train_mean)

        self.register_buffer('idw', torch.from_numpy(idw_kernel).float())
        self.register_buffer('mask_img',
                             torch.from_numpy(head_mask).float()
                                  .view(1, 1, grid_size, grid_size))
        if csd_maps:
            self.register_buffer('lap_kernel', laplacian_kernel())

        # ---- Spatial encoder ----
        topo_channels = n_bands + (n_peri if peri_in_topo else 0)
        if csd_maps:
            topo_channels *= 2     # original + Laplacian (CSD)
        self.cnn = SpatialCNNv2(in_channels=topo_channels, base=16, d_out=d_spatial)

        # ---- Raw feature MLP ----
        # 256 EEG + 16 per-band stats + 12 peri (raw + d1 + d3) + 2 position
        raw_in = n_bands * 64 + n_bands * 4 + n_peri * 3 + 2
        if use_quartile:
            raw_in += 1 + 4        # ordinal + one-hot
        if use_trial_id:
            self.trial_emb = nn.Embedding(n_trials, trial_emb_dim, padding_idx=0)
            raw_in += trial_emb_dim
        self.raw_mlp = nn.Sequential(
            nn.Linear(raw_in, d_raw), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(d_raw, d_raw), nn.GELU(),
        )
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
            self.head[-1].bias.fill_(self.train_mean)

    # ------------------------------------------------------------------
    def topomap(self, eeg_bt_chb: torch.Tensor) -> torch.Tensor:
        """(N, T, 64, 4) -> (N, T, 4, H, W)."""
        x = eeg_bt_chb.transpose(-1, -2)
        flat = x @ self.idw.t()
        H = self.grid
        return flat.view(*flat.shape[:-1], H, H) * self.mask_img

    def peri_topomap(self, peri: torch.Tensor) -> torch.Tensor:
        """(N, T, 4) -> (N, T, 4, H, W) by broadcasting peripheral values
        uniformly over the scalp grid."""
        N, T, P = peri.shape
        H = self.grid
        img = peri.view(N, T, P, 1, 1).expand(N, T, P, H, H)
        return img * self.mask_img

    def csd(self, topo: torch.Tensor) -> torch.Tensor:
        """Apply a 3x3 discrete Laplacian (CSD-style) per channel."""
        N, T, C, H, W = topo.shape
        x = topo.view(N * T * C, 1, H, W)
        x = F.conv2d(x, self.lap_kernel, padding=1)
        return x.view(N, T, C, H, W) * self.mask_img

    # ------------------------------------------------------------------
    def forward(self, eeg, peri, mask, bin_idx=None, trial_len=None,
                trial_id=None, train_aug=False, noise_std=0.0, ch_drop=0.0):
        """eeg: (N, T, 64, 4)  peri: (N, T, 4)  mask: (N, T)
        bin_idx: (N, T) int  trial_len: (N,) int
        trial_id: (N,) int (optional, only used if use_trial_id)
        """
        N, T = eeg.shape[:2]

        # ---- Augmentation ----
        if train_aug and self.training:
            if ch_drop > 0:
                cm = (torch.rand(64, device=eeg.device) > ch_drop).float()
                eeg = eeg * cm.view(1, 1, 64, 1)
            if noise_std > 0:
                eeg = eeg + torch.randn_like(eeg) * noise_std
                peri = peri + torch.randn_like(peri) * noise_std

        # ---- Spatial path: topomap + optional peri + optional CSD ----
        topo = self.topomap(eeg)                          # (N, T, 4, H, W)
        if self.peri_in_topo:
            peri_img = self.peri_topomap(peri)            # (N, T, 4, H, W)
            topo = torch.cat([topo, peri_img], dim=2)     # (N, T, 8, H, W)
        if self.csd_maps:
            lap = self.csd(topo)                          # (N, T, C, H, W)
            topo = torch.cat([topo, lap], dim=2)          # (N, T, 2C, H, W)
        spatial = self.cnn(topo.flatten(0, 1)).view(N, T, -1)

        # ---- Raw feature path ----
        eeg_flat = eeg.flatten(2)
        g_mean = eeg.mean(dim=2); g_std = eeg.std(dim=2)
        g_max = eeg.amax(dim=2); g_min = eeg.amin(dim=2)
        peri_d1 = torch.diff(peri, dim=1, prepend=peri[:, :1])
        peri_lag3 = F.pad(peri[:, :-3], (0, 0, 3, 0))
        peri_d3 = peri - peri_lag3

        if bin_idx is None or trial_len is None:
            bin_frac = torch.zeros(N, T, 1, device=eeg.device)
            log_len = torch.zeros(N, T, 1, device=eeg.device)
        else:
            bin_frac = (bin_idx.float() /
                        trial_len.unsqueeze(1).clamp_min(1).float()).unsqueeze(-1)
            log_len = torch.log1p(trial_len.float()).view(N, 1, 1).expand(N, T, 1)

        raw_parts = [eeg_flat, g_mean, g_std, g_max, g_min,
                     peri, peri_d1, peri_d3, bin_frac, log_len]

        # ---- (b) Quartile loop label ----
        if self.use_quartile and bin_idx is not None and trial_len is not None:
            quart = (bin_idx.float() * 4.0 /
                     trial_len.unsqueeze(1).clamp_min(1).float()).long().clamp(0, 3)
            quart_ord = quart.float().unsqueeze(-1) / 3.0            # ordinal in [0,1]
            quart_oh = F.one_hot(quart, num_classes=4).float()       # (N, T, 4)
            raw_parts.extend([quart_ord, quart_oh])

        # ---- (c) Trial id embedding ----
        if self.use_trial_id and trial_id is not None:
            te = self.trial_emb(trial_id)                            # (N, D)
            te = te.unsqueeze(1).expand(N, T, te.shape[-1])          # broadcast over time
            raw_parts.append(te)

        raw_in = torch.cat(raw_parts, dim=-1)
        raw = self.raw_mlp(raw_in)
        raw = raw + self.cnn_proj(spatial)

        fused = self.fuse(raw) * mask.unsqueeze(-1)
        h, _ = self.lstm(fused)
        return self.head(h).squeeze(-1)
