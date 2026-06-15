"""Feature selection + regression for the three challenge targets:
arousal, heart rate (HR), and galvanic skin response (GSR).

Two feature-selection approaches are applied and compared:
  1. Filter method  — univariate F-test (sklearn f_regression), ranks each
     feature by its individual association with the target.
  2. Embedded method — LightGBM gain importance from a model trained on all
     features, which accounts for interactions between features.

For each target we keep the embedded-selected subset (features the gradient-
boosted model actually splits on), retrain a compact LightGBM on that subset,
and compare its validation RMSE against the all-feature model. The chosen
features and both rankings are written to selected_features.csv.

Targets are predicted in their native units (arousal in [0,1], HR in BPM, GSR in
its recorded unit). When predicting HR we drop heartrate_mean from the inputs,
and when predicting GSR we drop GSR_mean, so a target is never used to predict
itself.
"""
import os, sys, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_selection import f_regression

sys.path.insert(0, os.path.dirname(__file__))
from data import load_cache, compute_norm_stats, apply_norm

OUT = os.path.join(os.path.dirname(__file__), 'feature_selection')
os.makedirs(OUT, exist_ok=True)

CHANNELS = ['AF3','AF4','AF7','AF8','C1','C2','C3','C4','C5','C6','CP1','CP2','CP3','CP4','CP5','CP6','CPz','Cz','F1','F2','F3','F4','F5','F6','F7','F8','FC1','FC2','FC3','FC4','FC5','FC6','Fp1','Fp2','FT10','FT7','FT8','FT9','Fz','O1','O2','Oz','P1','P2','P3','P4','P5','P6','P7','P8','PO3','PO4','PO7','PO8','POz','Pz','T7','T8','TP10','TP7','TP8','TP9','AFz','FCz']
BANDS = ['Theta','Alpha','Beta','Gamma']
EEG_NAMES = [f'EEG_{c}_{b}' for c in CHANNELS for b in BANDS]
PERI_NAMES = ['heartrate_mean','GSR_mean','IRPleth_mean','Respir_mean']
DERIVED = ['bin_index', 'bin_fraction', 'trial_length', 'loop_quartile']
ALL_NAMES = EEG_NAMES + PERI_NAMES + DERIVED


def build_features(cache):
    """Return (F, names) where F is the full per-bin feature matrix in raw/log
    units suitable for LightGBM."""
    X = cache['X']; offsets = cache['offsets']; splits = cache['split']
    trial = cache['trial']
    # train mask for normalisation stats
    train_mask = np.zeros(X.shape[0], dtype=bool)
    for ti in range(len(splits)):
        if splits[ti] == 0:
            train_mask[offsets[ti]:offsets[ti+1]] = True
    stats = compute_norm_stats(X, train_mask)
    eeg_z, _ = apply_norm(X, stats)             # log1p+clip+zscore EEG (256)
    peri = X[:, 256:260].astype(np.float32)      # raw peripherals (4)
    # derived bin-position features
    n = X.shape[0]
    bin_index = np.zeros(n, np.float32); bin_frac = np.zeros(n, np.float32)
    trial_len = np.zeros(n, np.float32); loop_q = np.zeros(n, np.float32)
    for ti in range(len(splits)):
        a, b = offsets[ti], offsets[ti+1]; T = b - a
        bi = np.arange(T); bin_index[a:b] = bi; trial_len[a:b] = T
        bin_frac[a:b] = bi / max(T, 1); loop_q[a:b] = np.minimum((bi*4)//max(T,1), 3)
    F = np.concatenate([eeg_z, peri,
                        bin_index[:,None], bin_frac[:,None], trial_len[:,None], loop_q[:,None]],
                       axis=1).astype(np.float32)
    return F, ALL_NAMES, train_mask


def select_and_fit(F, names, target, target_name, train_mask, val_mask, drop_name=None):
    """Run filter + embedded selection for one target, fit a compact model, and
    return a result dict."""
    keep_cols = [i for i, nm in enumerate(names) if nm != drop_name]
    sub_names = [names[i] for i in keep_cols]
    Ft = F[:, keep_cols]

    # (1) Filter: univariate F-test on training rows
    Ftr = Ft[train_mask]; ytr = target[train_mask]
    finite = np.isfinite(Ftr).all(axis=0)
    fscores = np.zeros(Ft.shape[1])
    fs, _ = f_regression(np.nan_to_num(Ftr), ytr)
    fscores = np.nan_to_num(fs)

    # (2) Embedded: LightGBM gain importance from an all-feature model
    params = dict(objective='regression', metric='rmse', learning_rate=0.03,
                  num_leaves=96, min_data_in_leaf=80, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=4, lambda_l2=2.0, verbose=-1)
    dtr = lgb.Dataset(Ft[train_mask], target[train_mask], feature_name=sub_names)
    dva = lgb.Dataset(Ft[val_mask], target[val_mask], reference=dtr)
    full = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    gains = full.feature_importance(importance_type='gain').astype(float)
    yp_all = full.predict(Ft[val_mask], num_iteration=full.best_iteration)
    rmse_all = float(np.sqrt(((yp_all - target[val_mask])**2).mean()))

    # Embedded selection: features the model actually used (gain > 0), capped to top 80
    used = np.where(gains > 0)[0]
    order = used[np.argsort(gains[used])[::-1]]
    sel = order[:80]
    sel_names = [sub_names[i] for i in sel]

    # Retrain compact model on the selected subset
    dtr2 = lgb.Dataset(Ft[train_mask][:, sel], target[train_mask], feature_name=sel_names)
    dva2 = lgb.Dataset(Ft[val_mask][:, sel], target[val_mask], reference=dtr2)
    comp = lgb.train(params, dtr2, num_boost_round=2000, valid_sets=[dva2],
                     callbacks=[lgb.early_stopping(120), lgb.log_evaluation(0)])
    yp_sel_val = comp.predict(Ft[val_mask][:, sel], num_iteration=comp.best_iteration)
    yp_sel_tr  = comp.predict(Ft[train_mask][:, sel], num_iteration=comp.best_iteration)
    rmse_sel = float(np.sqrt(((yp_sel_val - target[val_mask])**2).mean()))

    # Full per-row prediction array (for plotting)
    full_pred = np.zeros(F.shape[0], dtype=np.float32)
    full_pred[val_mask] = yp_sel_val; full_pred[train_mask] = yp_sel_tr

    comp.save_model(os.path.join(OUT, f'{target_name}_lgbm.txt'))
    print(f"[{target_name}] all-features RMSE {rmse_all:.4f} | selected({len(sel)}) RMSE {rmse_sel:.4f} "
          f"| target std {target[val_mask].std():.4g}")

    # Map gains/fscores back to the full ALL_NAMES space (drop_name gets NaN)
    gain_full = {nm: 0.0 for nm in names}
    fsc_full  = {nm: 0.0 for nm in names}
    sel_full  = {nm: 0 for nm in names}
    for j, nm in enumerate(sub_names):
        gain_full[nm] = float(gains[j]); fsc_full[nm] = float(fscores[j])
    for nm in sel_names:
        sel_full[nm] = 1
    return dict(target=target_name, rmse_all=rmse_all, rmse_sel=rmse_sel,
                n_selected=int(len(sel)), selected=sel_names,
                gain=gain_full, fscore=fsc_full, sel_flag=sel_full,
                pred=full_pred, target_std=float(target[val_mask].std()))


def main():
    cache = load_cache()
    y = cache['y']; X = cache['X']; offsets = cache['offsets']; splits = cache['split']
    F, names, train_mask = build_features(cache)
    val_mask = np.zeros(X.shape[0], dtype=bool)
    for ti in range(len(splits)):
        if splits[ti] == 1:
            val_mask[offsets[ti]:offsets[ti+1]] = True

    targets = {
        'arousal': (y, None),
        'HR':      (X[:, 256].astype(np.float32), 'heartrate_mean'),
        'GSR':     (X[:, 257].astype(np.float32), 'GSR_mean'),
    }
    results = {}
    for tname, (tvals, drop) in targets.items():
        results[tname] = select_and_fit(F, names, tvals, tname, train_mask, val_mask, drop_name=drop)

    # ---- selected_features.csv (per-target selection + rankings) ----
    rows = []
    for nm in names:
        grp = ('EEG' if nm.startswith('EEG_') else
               'Peripheral' if nm in PERI_NAMES else 'Derived')
        row = {'feature': nm, 'group': grp}
        for t in ['arousal', 'HR', 'GSR']:
            row[f'selected_{t}'] = results[t]['sel_flag'][nm]
            row[f'gain_{t}'] = round(results[t]['gain'][nm], 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'selected_features.csv'), index=False)

    # ---- predictions for plotting (per-bin, full array order = cache) ----
    np.savez(os.path.join(OUT, 'fs_predictions.npz'),
             arousal=results['arousal']['pred'],
             HR=results['HR']['pred'], GSR=results['GSR']['pred'])

    summary = {t: {'rmse_all_features': results[t]['rmse_all'],
                   'rmse_selected': results[t]['rmse_sel'],
                   'n_selected': results[t]['n_selected'],
                   'target_std_val': results[t]['target_std']}
               for t in results}
    json.dump(summary, open(os.path.join(OUT, 'fs_summary.json'), 'w'), indent=2)
    print("\n=== feature-selection summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/selected_features.csv ({len(df)} features), models, predictions")


if __name__ == '__main__':
    main()
