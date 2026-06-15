"""Dataset utilities for the EMAP cached arrays."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


CACHE_PATH = "/Users/shreejaysubedi/Coding/EMAP3/work/cache.npz"
MONT_PATH = "/Users/shreejaysubedi/Coding/EMAP3/montage_2d.npz"


def load_cache():
    d = np.load(CACHE_PATH, allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_montage_pos(channel_order):
    """Return pos2d (n_ch, 2) in channel_order."""
    m = np.load(MONT_PATH, allow_pickle=True)
    name_to_pos = {str(c): p for c, p in zip(m['channels'], m['pos2d'])}
    return np.stack([name_to_pos[c] for c in channel_order], axis=0).astype(np.float32)


def compute_norm_stats(X, train_mask, n_ch=64, n_bands=4):
    """Compute per-feature stats on log1p(EEG) and on peripherals.

    Returns dict with arrays for normalisation.
    """
    eeg = X[train_mask, :n_ch * n_bands]
    per = X[train_mask, n_ch * n_bands:]
    # EEG: log1p + clip extreme values + z-score per-feature
    log_eeg = np.log1p(np.maximum(eeg, 0.0))
    # robust clip per-feature to suppress artefact spikes
    p_lo, p_hi = np.percentile(log_eeg, [0.5, 99.5], axis=0)
    log_eeg_c = np.clip(log_eeg, p_lo, p_hi)
    mu = log_eeg_c.mean(axis=0)
    sd = log_eeg_c.std(axis=0) + 1e-6
    # Peripheral: robust z-score using median/MAD scaled
    med = np.median(per, axis=0)
    mad = np.median(np.abs(per - med), axis=0) * 1.4826 + 1e-6
    return dict(eeg_lo=p_lo.astype(np.float32),
                eeg_hi=p_hi.astype(np.float32),
                eeg_mu=mu.astype(np.float32),
                eeg_sd=sd.astype(np.float32),
                per_med=med.astype(np.float32),
                per_mad=mad.astype(np.float32))


def apply_norm(X, stats, n_ch=64, n_bands=4):
    eeg = X[:, :n_ch * n_bands].astype(np.float32)
    per = X[:, n_ch * n_bands:].astype(np.float32)
    log_eeg = np.log1p(np.maximum(eeg, 0.0))
    log_eeg = np.clip(log_eeg, stats['eeg_lo'], stats['eeg_hi'])
    log_eeg = (log_eeg - stats['eeg_mu']) / stats['eeg_sd']
    per_z = (per - stats['per_med']) / stats['per_mad']
    per_z = np.clip(per_z, -5.0, 5.0)
    return log_eeg.astype(np.float32), per_z.astype(np.float32)


class TrialDataset(Dataset):
    """Each item is one trial; we will pad in the collate."""

    def __init__(self, X_eeg, X_per, y, offsets, trial_indices, n_ch=64, n_bands=4):
        self.eeg = X_eeg  # (N, 256)
        self.per = X_per  # (N, 4)
        self.y = y
        self.offsets = offsets
        self.indices = trial_indices
        self.n_ch = n_ch
        self.n_bands = n_bands

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ti = self.indices[idx]
        a, b = self.offsets[ti], self.offsets[ti + 1]
        T = b - a
        eeg = self.eeg[a:b].reshape(T, self.n_ch, self.n_bands)  # (T, 64, 4)
        per = self.per[a:b]                                       # (T, 4)
        y = self.y[a:b]                                           # (T,)
        bin_idx = np.arange(T, dtype=np.int64)
        return (torch.from_numpy(eeg), torch.from_numpy(per), torch.from_numpy(y),
                torch.from_numpy(bin_idx), int(T), int(ti))


def pad_collate(batch):
    Ts = [b[0].shape[0] for b in batch]
    Tmax = max(Ts)
    B = len(batch)
    eeg = torch.zeros(B, Tmax, batch[0][0].shape[1], batch[0][0].shape[2])
    per = torch.zeros(B, Tmax, batch[0][1].shape[1])
    y = torch.zeros(B, Tmax)
    bin_idx = torch.zeros(B, Tmax, dtype=torch.long)
    trial_len = torch.zeros(B, dtype=torch.long)
    mask = torch.zeros(B, Tmax)
    tids = []
    for i, (e, p, yy, bi, tl, t) in enumerate(batch):
        T = e.shape[0]
        eeg[i, :T] = e
        per[i, :T] = p
        y[i, :T] = yy
        bin_idx[i, :T] = bi
        trial_len[i] = tl
        mask[i, :T] = 1.0
        tids.append(t)
    return eeg, per, y, mask, bin_idx, trial_len, tids
