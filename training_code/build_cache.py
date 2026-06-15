"""Build cached numpy dataset from raw CSVs.

Each trial CSV becomes one row block in concatenated arrays:
- X:        (N, 260) float32  (256 EEG + 4 peripheral)
- y:        (N,)     float32  arousal label
- group:    (N,)     int32    participant id
- trial:    (N,)     int32    trial id
- bin_idx:  (N,)     int32    bin position within the trial
- offsets:  (T+1,)   int64    cumulative row count per trial (slice X[offsets[i]:offsets[i+1]])
- split:    (T,)     int8     0=train, 1=val
"""
import os, glob, re, time, sys
import numpy as np
import pandas as pd

ROOT = "/Users/shreejaysubedi/Coding/EMAP3"
OUT  = f"{ROOT}/work/cache.npz"

CHANNELS = ['AF3','AF4','AF7','AF8','C1','C2','C3','C4','C5','C6','CP1','CP2','CP3','CP4','CP5','CP6','CPz','Cz','F1','F2','F3','F4','F5','F6','F7','F8','FC1','FC2','FC3','FC4','FC5','FC6','Fp1','Fp2','FT10','FT7','FT8','FT9','Fz','O1','O2','Oz','P1','P2','P3','P4','P5','P6','P7','P8','PO3','PO4','PO7','PO8','POz','Pz','T7','T8','TP10','TP7','TP8','TP9','AFz','FCz']
BANDS = ['Theta','Alpha','Beta','Gamma']
EEG_COLS = [f'EEG_{c}_{b}' for c in CHANNELS for b in BANDS]
PERI_COLS = ['heartrate_mean','GSR_mean','IRPleth_mean','Respir_mean']
LABEL = 'LABEL_SR_Arousal'
FEAT_COLS = EEG_COLS + PERI_COLS

FNAME_RE = re.compile(r'Features_P(\d+)-T(\d+)\.csv$')


def collect_files():
    train_dirs = [f"{ROOT}/train/P1-40", f"{ROOT}/train/P41-80", f"{ROOT}/train/P81-90"]
    val_dir = f"{ROOT}/val"
    items = []
    for d in train_dirs:
        for fp in sorted(glob.glob(f"{d}/Features_P*-T*.csv")):
            m = FNAME_RE.search(fp)
            if not m: continue
            items.append((fp, int(m.group(1)), int(m.group(2)), 0))
    for fp in sorted(glob.glob(f"{val_dir}/Features_P*-T*.csv")):
        m = FNAME_RE.search(fp)
        if not m: continue
        items.append((fp, int(m.group(1)), int(m.group(2)), 1))
    return items


def main():
    items = collect_files()
    print(f"trials found: {len(items)}", flush=True)

    parts_x, parts_y, parts_g, parts_t, parts_b = [], [], [], [], []
    offsets = [0]
    splits = []
    t0 = time.time()
    bad = 0
    for i, (fp, pid, tid, sp) in enumerate(items):
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            bad += 1; print(f"skip {fp}: {e}"); continue
        # Make sure all expected columns are present
        miss = [c for c in FEAT_COLS + [LABEL] if c not in df.columns]
        if miss:
            bad += 1; print(f"skip {fp}: missing {miss[:3]}..."); continue
        x = df[FEAT_COLS].to_numpy(dtype=np.float32, copy=False)
        y = df[LABEL].to_numpy(dtype=np.float32, copy=False)
        # Drop rows with NaN label or features
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
        x, y = x[ok], y[ok]
        if x.shape[0] == 0:
            bad += 1; continue
        n = x.shape[0]
        parts_x.append(x); parts_y.append(y)
        parts_g.append(np.full(n, pid, dtype=np.int32))
        parts_t.append(np.full(n, tid, dtype=np.int32))
        parts_b.append(np.arange(n, dtype=np.int32))
        offsets.append(offsets[-1] + n)
        splits.append(sp)
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(items)} ({time.time()-t0:.1f}s)", flush=True)
    X = np.concatenate(parts_x, axis=0)
    y = np.concatenate(parts_y, axis=0)
    g = np.concatenate(parts_g, axis=0)
    t = np.concatenate(parts_t, axis=0)
    b = np.concatenate(parts_b, axis=0)
    offsets = np.array(offsets, dtype=np.int64)
    splits = np.array(splits, dtype=np.int8)
    print(f"X: {X.shape}, y: {y.shape}, trials: {len(splits)} (val={int((splits==1).sum())})")
    print(f"y range: {y.min():.3f}..{y.max():.3f} mean={y.mean():.3f} std={y.std():.3f}")
    print(f"feature stats sample: HR {X[:,256].mean():.2f}±{X[:,256].std():.2f}; GSR {X[:,257].mean():.4f}±{X[:,257].std():.4f}")
    print(f"bad files: {bad}")
    np.savez_compressed(OUT,
        X=X, y=y, group=g, trial=t, bin_idx=b,
        offsets=offsets, split=splits,
        feat_cols=np.array(FEAT_COLS),
        channels=np.array(CHANNELS), bands=np.array(BANDS))
    print(f"Wrote {OUT}, size = {os.path.getsize(OUT)/1e6:.1f} MB", flush=True)

if __name__ == '__main__':
    main()
