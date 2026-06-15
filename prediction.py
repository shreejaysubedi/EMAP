"""EMAP Dataset Challenge — End-to-end prediction script (final ensemble).

The final prediction blends three neural snapshots, each carrying its own
normalisation statistics and IDW topomap kernel:

  - model.pt          EmapModel,   24x24 topomap, predicts arousal     (weight 0.366)
  - model_quartile.pt EmapModelV2, 24x24 topomap, predicts arousal,    (weight 0.318)
                                   uses the loop-quartile feature
  - model_q48dev.pt   EmapModelV2, 48x48 topomap, predicts y-trial_mean(weight 0.317)
                                   (larger topomap, within-trial specialist)

Pipeline per trial CSV:
  1. log1p + percentile-clip + z-score the 256 EEG band-power features; robust
     median/MAD normalisation of the 4 peripheral signals. (Per-model stats.)
  2. Reshape EEG to (T, 64, 4); interpolate each timestep onto the model's
     topomap grid using that model's precomputed IDW kernel.
  3. Run each model. Re-centre the deviation model by adding the training global
     mean to its per-trial centred output.
  4. Convex-blend (weights in blend_weights.json).
  5. Gaussian smoothing (sigma=0.5) within each trial, then per-trial variance
     calibration (k=1.318). Clip to [0, 1].

Usage:
    python prediction.py --input <CSV_OR_FOLDER> --output predictions.csv

Selected features: all 256 EEG + 4 peripheral features are kept (see
selected_features.csv). The two EmapModelV2 snapshots additionally derive a
loop-quartile feature (which of the 4 video repetitions each bin belongs to).
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
import numpy as np
import pandas as pd
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = THIS_DIR  # model.py / model_v2.py are co-located
sys.path.insert(0, WORK_DIR)
from model import EmapModel          # noqa: E402
from model_v2 import EmapModelV2     # noqa: E402

CHANNELS = ['AF3','AF4','AF7','AF8','C1','C2','C3','C4','C5','C6','CP1','CP2','CP3','CP4','CP5','CP6','CPz','Cz','F1','F2','F3','F4','F5','F6','F7','F8','FC1','FC2','FC3','FC4','FC5','FC6','Fp1','Fp2','FT10','FT7','FT8','FT9','Fz','O1','O2','Oz','P1','P2','P3','P4','P5','P6','P7','P8','PO3','PO4','PO7','PO8','POz','Pz','T7','T8','TP10','TP7','TP8','TP9','AFz','FCz']
BANDS = ['Theta','Alpha','Beta','Gamma']
EEG_COLS = [f'EEG_{c}_{b}' for c in CHANNELS for b in BANDS]
PERI_COLS = ['heartrate_mean','GSR_mean','IRPleth_mean','Respir_mean']
DERIVED_COLS = ['bin_index', 'bin_fraction', 'trial_length', 'loop_quartile']
ALL_LGBM_COLS = EEG_COLS + PERI_COLS + DERIVED_COLS
FNAME_RE = re.compile(r'Features_P(\d+)-T(\d+)\.csv$', re.IGNORECASE)

# Optional LightGBM regressors for the heart-rate and GSR targets (feature-
# selected). Loaded only if present in the assets folder.
try:
    import lightgbm as lgb
    _HAVE_LGBM = True
except Exception:
    _HAVE_LGBM = False


def build_lgbm_frame(X, eeg_z):
    """Per-bin feature frame for the HR/GSR LightGBM models: log-z EEG (256) +
    raw peripherals (4) + bin-position features (4). Column names match training."""
    T = X.shape[0]
    peri = X[:, 256:260]
    bi = np.arange(T, dtype=np.float32)
    bin_frac = bi / max(T, 1)
    trial_len = np.full(T, T, dtype=np.float32)
    loop_q = np.minimum((bi * 4) // max(T, 1), 3).astype(np.float32)
    data = np.concatenate([eeg_z, peri,
                           bi[:, None], bin_frac[:, None], trial_len[:, None], loop_q[:, None]],
                          axis=1).astype(np.float32)
    return pd.DataFrame(data, columns=ALL_LGBM_COLS)


def pick_device(pref='auto'):
    if pref != 'auto':
        return torch.device(pref)
    if torch.cuda.is_available(): return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def normalise(X, stats):
    eeg = X[:, :256].astype(np.float32); per = X[:, 256:260].astype(np.float32)
    log_eeg = np.log1p(np.maximum(eeg, 0.0))
    log_eeg = np.clip(log_eeg, stats['eeg_lo'], stats['eeg_hi'])
    log_eeg = (log_eeg - stats['eeg_mu']) / stats['eeg_sd']
    per_z = np.clip((per - stats['per_med']) / stats['per_mad'], -5.0, 5.0)
    return log_eeg.astype(np.float32), per_z.astype(np.float32)


def build_model(name, meta, assets, device):
    base = name[:-3]  # strip '.pt'
    ck = torch.load(os.path.join(assets, name), map_location='cpu', weights_only=False)
    a = ck['args']
    idw_npz = np.load(os.path.join(assets, f'{base}.idw.npz'))
    idw, mask = idw_npz['idw'], idw_npz['mask']
    if meta['type'] == 'EmapModelV2':
        m = EmapModelV2(idw, mask, grid_size=a.get('grid', 24),
                        d_raw=a.get('d_raw', 192), d_spatial=a.get('d_spatial', 64),
                        d_hidden=a.get('d_hidden', 224), lstm_layers=a.get('lstm_layers', 2),
                        dropout=a.get('dropout', 0.30),
                        peri_in_topo=a.get('peri_in_topo', False),
                        use_quartile=a.get('use_quartile', True),
                        use_trial_id=a.get('use_trial_id', False),
                        csd_maps=a.get('csd_maps', False),
                        n_trials=26, train_mean=a.get('train_mean', 0.486))
    else:
        m = EmapModel(idw, mask, grid_size=a.get('grid', 24),
                      d_raw=a.get('d_raw', 160), d_spatial=a.get('d_spatial', 64),
                      d_hidden=a.get('d_hidden', 192), lstm_layers=a.get('lstm_layers', 2),
                      dropout=a.get('dropout', 0.3), use_cnn=a.get('use_cnn', True),
                      demean=a.get('demean', True), noise_std=0.0, ch_drop=0.0)
    m.load_state_dict(ck['state_dict']); m.to(device).eval()
    stats = dict(np.load(os.path.join(assets, f'{base}.norm.npz')))
    return {'model': m, 'stats': stats, 'meta': meta}


def run_one(entry, X, device, trial_id=1):
    stats = entry['stats']; meta = entry['meta']; m = entry['model']
    eeg_z, per_z = normalise(X, stats)
    T = eeg_z.shape[0]
    e = torch.from_numpy(eeg_z.reshape(1, T, 64, 4)).to(device)
    p = torch.from_numpy(per_z.reshape(1, T, 4)).to(device)
    mask = torch.ones(1, T, device=device)
    bi = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)
    tl = torch.tensor([T], device=device, dtype=torch.long)
    with torch.no_grad():
        if meta['type'] == 'EmapModelV2':
            tid = torch.tensor([trial_id], device=device, dtype=torch.long)
            yp = m(e, p, mask, bi, tl, tid, train_aug=False).cpu().numpy().reshape(-1)
        else:
            out = m(e, p, mask, bi, tl, train_aug=False)
            yp = (out[0] if isinstance(out, tuple) else out).cpu().numpy().reshape(-1)
    return yp


def gauss_smooth(arr, sigma):
    if sigma <= 0: return arr
    r = int(np.ceil(3 * sigma)); xs = np.arange(-r, r + 1)
    k = np.exp(-(xs ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    return np.convolve(arr, k, mode='same')


def predict_trial(df, entries, cfg, device, trial_id=1):
    X = df[EEG_COLS + PERI_COLS].to_numpy(dtype=np.float32)
    mu = cfg['train_global_mean']
    comp, wts = [], []
    for name, meta in cfg['models'].items():
        if meta['weight'] <= 0: continue
        yp = run_one(entries[name], X, device, trial_id=trial_id)
        if meta['target'] == 'deviation':
            yp = mu + (yp - yp.mean())
        comp.append(yp); wts.append(meta['weight'])
    wts = np.array(wts); wts = wts / wts.sum()
    blend = sum(w * c for w, c in zip(wts, comp))
    blend = gauss_smooth(blend, cfg.get('smoothing_sigma', 0.0))
    k = cfg.get('calibration_k', 1.0)
    if k != 1.0:
        m = blend.mean(); blend = m + k * (blend - m)
    return np.clip(blend, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', default='predictions.csv')
    ap.add_argument('--assets', default=os.path.join(THIS_DIR, 'model_assets'))
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    device = pick_device(args.device)
    cfg = json.load(open(os.path.join(args.assets, 'blend_weights.json')))
    entries = {name: build_model(name, meta, args.assets, device)
               for name, meta in cfg['models'].items() if meta['weight'] > 0}
    print(f"loaded {len(entries)} arousal models on {device}", flush=True)

    # Optional feature-selected LightGBM regressors for HR and GSR.
    hr_model = gsr_model = None
    if _HAVE_LGBM:
        hr_path = os.path.join(args.assets, 'HR_lgbm.txt')
        gsr_path = os.path.join(args.assets, 'GSR_lgbm.txt')
        if os.path.exists(hr_path):
            hr_model = lgb.Booster(model_file=hr_path)
        if os.path.exists(gsr_path):
            gsr_model = lgb.Booster(model_file=gsr_path)
        print(f"HR model: {'yes' if hr_model else 'no'} | GSR model: {'yes' if gsr_model else 'no'}", flush=True)

    # Shared EEG normalisation stats (the HR/GSR feature frame uses log-z EEG).
    base_stats = entries[next(iter(entries))]['stats']

    files = sorted(glob.glob(os.path.join(args.input, 'Features_P*-T*.csv'))) \
        if os.path.isdir(args.input) else [args.input]
    print(f"scoring {len(files)} files", flush=True)
    rows = []
    for i, fp in enumerate(files):
        m = FNAME_RE.search(fp)
        pid = int(m.group(1)) if m else -1
        tid = int(m.group(2)) if m else 1
        df = pd.read_csv(fp)
        miss = [c for c in EEG_COLS + PERI_COLS if c not in df.columns]
        if miss:
            print(f"  skip {os.path.basename(fp)}: missing {miss[:3]}..."); continue
        arousal = predict_trial(df, entries, cfg, device, trial_id=min(max(tid, 1), 24))
        # HR / GSR from the feature-selected LightGBM models
        hr_pred = gsr_pred = None
        if hr_model is not None or gsr_model is not None:
            X = df[EEG_COLS + PERI_COLS].to_numpy(dtype=np.float32)
            eeg_z, _ = normalise(X, base_stats)
            frame = build_lgbm_frame(X, eeg_z)
            if hr_model is not None:
                hr_pred = hr_model.predict(frame[hr_model.feature_name()])
            if gsr_model is not None:
                gsr_pred = gsr_model.predict(frame[gsr_model.feature_name()])
        for bi in range(len(arousal)):
            row = {'Participant': pid, 'Trial': tid, 'Bin': bi,
                   'Pred_Arousal': float(arousal[bi])}
            if hr_pred is not None:  row['Pred_HR'] = float(hr_pred[bi])
            if gsr_pred is not None: row['Pred_GSR'] = float(gsr_pred[bi])
            rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(out)} rows, columns: {list(out.columns)})")


if __name__ == '__main__':
    main()
