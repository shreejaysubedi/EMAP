"""Final ensemble combining the new quartile-flag models (run_quartile_raw,
run_quartile_dev) with the previous best snapshots (run4, run_dev_7) and the
LightGBM baseline.

Steps:
  1. Score every available model on every trial.
  2. Re-center the dev-target models (predict y - trial_mean) by adding the
     training global mean to each per-trial centered output.
  3. Fit convex blend weights on val via Nelder-Mead.
  4. Sweep Gaussian smoothing sigma.
  5. Write final_ensemble_v2.npz.
"""
import os, sys, json
import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
from data import load_cache, load_montage_pos, apply_norm
from model import EmapModel, build_idw_kernel
from model_v2 import EmapModelV2

OUT = '/Users/shreejaysubedi/Coding/EMAP3/work/final_ensemble_v2.npz'


def predict_v1_model(run, eeg_z, per_z, offsets, splits, channels, device):
    """Predict with original EmapModel (run4, run_dev_7)."""
    ck = torch.load(f'{run}/best.pt', map_location='cpu', weights_only=False)
    a = ck['args']
    pos2d = load_montage_pos(list(channels))
    idw, mh = build_idw_kernel(pos2d, grid_size=a.get('grid', 24), power=4.0)
    model = EmapModel(idw, mh, grid_size=a.get('grid', 24),
                      d_raw=a.get('d_raw', 160), d_spatial=a.get('d_spatial', 64),
                      d_hidden=a.get('d_hidden', 192),
                      lstm_layers=a.get('lstm_layers', 2),
                      dropout=a.get('dropout', 0.3),
                      use_cnn=a.get('use_cnn', True),
                      demean=a.get('demean', True),
                      noise_std=0.0, ch_drop=0.0)
    model.load_state_dict(ck['state_dict']); model.to(device).eval()
    n = offsets[-1]; pred = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for ti in range(len(splits)):
            a0, b0 = offsets[ti], offsets[ti + 1]; T = b0 - a0
            e = torch.from_numpy(eeg_z[a0:b0].reshape(1, T, 64, 4)).to(device)
            p = torch.from_numpy(per_z[a0:b0].reshape(1, T, 4)).to(device)
            m = torch.ones(1, T, device=device)
            bi = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)
            tl = torch.tensor([T], device=device, dtype=torch.long)
            out = model(e, p, m, bi, tl, train_aug=False)
            yp = (out[0] if isinstance(out, tuple) else out).cpu().numpy().reshape(-1)
            pred[a0:b0] = yp
    return pred


def predict_v2_model(run, eeg_z, per_z, offsets, splits, channels, trial_id_arr, device):
    """Predict with EmapModelV2 (run_quartile_raw, run_quartile_dev)."""
    ck = torch.load(f'{run}/best.pt', map_location='cpu', weights_only=False)
    a = ck['args']
    pos2d = load_montage_pos(list(channels))
    idw, mh = build_idw_kernel(pos2d, grid_size=a.get('grid', 24), power=4.0)
    n_trials_vocab = int(trial_id_arr.max()) + 2
    model = EmapModelV2(idw, mh, grid_size=a.get('grid', 24),
                        d_raw=a.get('d_raw', 192),
                        d_spatial=a.get('d_spatial', 64),
                        d_hidden=a.get('d_hidden', 224),
                        lstm_layers=a.get('lstm_layers', 2),
                        dropout=a.get('dropout', 0.30),
                        peri_in_topo=a.get('peri_in_topo', False),
                        use_quartile=a.get('use_quartile', True),
                        use_trial_id=a.get('use_trial_id', False),
                        csd_maps=a.get('csd_maps', False),
                        n_trials=n_trials_vocab,
                        train_mean=a.get('train_mean', 0.486))
    model.load_state_dict(ck['state_dict']); model.to(device).eval()
    n = offsets[-1]; pred = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for ti in range(len(splits)):
            a0, b0 = offsets[ti], offsets[ti + 1]; T = b0 - a0
            e = torch.from_numpy(eeg_z[a0:b0].reshape(1, T, 64, 4)).to(device)
            p = torch.from_numpy(per_z[a0:b0].reshape(1, T, 4)).to(device)
            m = torch.ones(1, T, device=device)
            bi = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0)
            tl = torch.tensor([T], device=device, dtype=torch.long)
            tid = torch.tensor([int(trial_id_arr[a0])], device=device, dtype=torch.long)
            yp = model(e, p, m, bi, tl, tid, train_aug=False).cpu().numpy().reshape(-1)
            pred[a0:b0] = yp
    return pred


def gauss_smooth(arr, sigma):
    if sigma <= 0: return arr
    r = int(np.ceil(3 * sigma))
    xs = np.arange(-r, r + 1)
    k = np.exp(-(xs ** 2) / (2 * sigma ** 2)); k = k / k.sum()
    return np.convolve(arr, k, mode='same')


def main():
    cache = load_cache()
    y = cache['y']; offsets = cache['offsets']; splits = cache['split']
    channels = cache['channels']; X = cache['X']
    trial_id_arr = cache['trial']
    val_mask = np.zeros_like(y, dtype=bool); train_mask = np.zeros_like(y, dtype=bool)
    for ti in range(len(splits)):
        (val_mask if splits[ti] == 1 else train_mask)[offsets[ti]:offsets[ti + 1]] = True

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    global_mu = float(y[train_mask].mean())

    # Load each model and predict
    raw_preds = {}

    # Previous best run4 (raw target)
    stats = dict(np.load('/Users/shreejaysubedi/Coding/EMAP3/work/run4/norm_stats.npz'))
    eeg_z, per_z = apply_norm(X, stats)
    raw_preds['run4'] = predict_v1_model('/Users/shreejaysubedi/Coding/EMAP3/work/run4',
                                          eeg_z, per_z, offsets, splits, channels, device)

    # Previous best run_dev_7 (dev target)
    stats = dict(np.load('/Users/shreejaysubedi/Coding/EMAP3/work/run_dev_7/norm_stats.npz'))
    eeg_z, per_z = apply_norm(X, stats)
    raw_preds['run_dev_7'] = predict_v1_model('/Users/shreejaysubedi/Coding/EMAP3/work/run_dev_7',
                                               eeg_z, per_z, offsets, splits, channels, device)

    # New v2 quartile raw
    stats = dict(np.load('/Users/shreejaysubedi/Coding/EMAP3/work/run_quartile_raw/norm_stats.npz'))
    eeg_z, per_z = apply_norm(X, stats)
    raw_preds['quartile_raw'] = predict_v2_model('/Users/shreejaysubedi/Coding/EMAP3/work/run_quartile_raw',
                                                  eeg_z, per_z, offsets, splits, channels, trial_id_arr, device)

    # New v2 quartile dev
    stats = dict(np.load('/Users/shreejaysubedi/Coding/EMAP3/work/run_quartile_dev/norm_stats.npz'))
    eeg_z, per_z = apply_norm(X, stats)
    raw_preds['quartile_dev'] = predict_v2_model('/Users/shreejaysubedi/Coding/EMAP3/work/run_quartile_dev',
                                                  eeg_z, per_z, offsets, splits, channels, trial_id_arr, device)

    # Re-center the dev-target predictions
    dev_runs = {'run_dev_7', 'quartile_dev'}
    preds = {}
    print("\nPer-model val RMSE after recentring:")
    for name, p in raw_preds.items():
        if name in dev_runs:
            pp = np.zeros_like(p)
            for ti in range(len(splits)):
                a0, b0 = offsets[ti], offsets[ti + 1]
                dev_c = p[a0:b0] - p[a0:b0].mean()
                pp[a0:b0] = global_mu + dev_c
        else:
            pp = p
        pp = np.clip(pp, 0, 1)
        preds[name] = pp
        rmse = float(np.sqrt(((pp[val_mask] - y[val_mask]) ** 2).mean()))
        print(f"  {name:<14} val RMSE = {rmse:.4f}  raw_range=[{p.min():.3f},{p.max():.3f}]")

    # LightGBM
    lgbm = np.clip(np.load('/Users/shreejaysubedi/Coding/EMAP3/work/lgbm/lgbm_pred.npz')['pred'], 0, 1)
    print(f"  {'lgbm':<14} val RMSE = {np.sqrt(((lgbm[val_mask] - y[val_mask])**2).mean()):.4f}")

    names = ['run4', 'run_dev_7', 'quartile_raw', 'quartile_dev', 'lgbm']
    P = [preds['run4'], preds['run_dev_7'], preds['quartile_raw'], preds['quartile_dev'], lgbm]

    # Nelder-Mead blend
    def loss(w):
        w = np.maximum(w, 0); s = w.sum()
        if s < 1e-9: return 1.0
        w = w / s
        pred = sum(wi * p for wi, p in zip(w, P))
        pred = np.clip(pred, 0, 1)
        return float(np.sqrt(((pred[val_mask] - y[val_mask]) ** 2).mean()))

    best_loss = 1e9; best_w = None
    for x0 in [np.ones(5) / 5,
               np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
               np.array([0.0, 0.0, 0.5, 0.5, 0.0]),
               np.array([0.25, 0.25, 0.25, 0.25, 0.0])]:
        res = minimize(loss, x0, method='Nelder-Mead',
                       options={'xatol': 1e-4, 'fatol': 1e-5, 'maxiter': 3000})
        if res.fun < best_loss:
            best_loss = res.fun
            best_w = np.maximum(res.x, 0); best_w = best_w / best_w.sum()
    print(f"\nBEST blend weights: {dict(zip(names, np.round(best_w, 3)))}")
    print(f"BEST blend val RMSE (no smoothing): {best_loss:.4f}")

    blend = sum(wi * p for wi, p in zip(best_w, P))
    best_sigma = 0; best_sigma_rmse = best_loss
    for sigma in [0, 0.5, 1, 1.5, 2]:
        p = blend.copy()
        if sigma > 0:
            for ti in range(len(splits)):
                a0, b0 = offsets[ti], offsets[ti + 1]
                p[a0:b0] = gauss_smooth(p[a0:b0], sigma)
        p = np.clip(p, 0, 1)
        rmse = float(np.sqrt(((p[val_mask] - y[val_mask]) ** 2).mean()))
        print(f"  sigma={sigma}: val RMSE = {rmse:.4f}")
        if rmse < best_sigma_rmse:
            best_sigma = sigma; best_sigma_rmse = rmse

    final = blend.copy()
    if best_sigma > 0:
        for ti in range(len(splits)):
            a0, b0 = offsets[ti], offsets[ti + 1]
            final[a0:b0] = gauss_smooth(final[a0:b0], best_sigma)
    final = np.clip(final, 0, 1)
    rmse_final = float(np.sqrt(((final[val_mask] - y[val_mask]) ** 2).mean()))
    mae_final = float(np.abs(final[val_mask] - y[val_mask]).mean())
    print(f"\n>>> FINAL val RMSE: {rmse_final:.4f}  MAE: {mae_final:.4f}")

    np.savez(OUT, pred=final, weights=best_w, sigma=best_sigma,
             val_rmse=rmse_final, names=np.array(names))
    json.dump({'weights': {n: float(w) for n, w in zip(names, best_w)},
               'sigma': float(best_sigma), 'val_rmse': rmse_final, 'val_mae': mae_final,
               'prev_v1_ensemble_val_rmse': 0.3067},
              open('/Users/shreejaysubedi/Coding/EMAP3/work/final_ensemble_v2.json', 'w'),
              indent=2)


if __name__ == '__main__':
    main()
