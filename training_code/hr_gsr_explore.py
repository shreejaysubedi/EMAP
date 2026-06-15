"""Broad exploration of heart-rate and GSR regression.

For each target we try several algorithm families (ridge / random forest /
LightGBM / a small MLP) crossed with several feature views (EEG only,
peripherals + position, a physiology-aware subset, all features), plus a
temporal-context variant that adds lagged and rolling peripheral features.

The target is always excluded from its own inputs. We report validation RMSE and
R² (fraction of variance explained; 0 = no better than predicting the mean).
"""
import os, sys, json, time, warnings
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from data import load_cache, compute_norm_stats, apply_norm

OUT = os.path.join(os.path.dirname(__file__), 'hr_gsr_explore')
os.makedirs(OUT, exist_ok=True)

CHANNELS_EEG = list(range(256))
HR_I, GSR_I, IRP_I, RESP_I = 256, 257, 258, 259


def build(cache):
    X = cache['X']; offsets = cache['offsets']; splits = cache['split']
    tm = np.zeros(X.shape[0], bool); vm = np.zeros(X.shape[0], bool)
    for ti in range(len(splits)):
        (tm if splits[ti] == 0 else vm)[offsets[ti]:offsets[ti+1]] = True
    stats = compute_norm_stats(X, tm)
    eeg_z, _ = apply_norm(X, stats)
    peri = X[:, 256:260].astype(np.float32)
    n = X.shape[0]
    bin_i = np.zeros(n, np.float32); bin_f = np.zeros(n, np.float32)
    tlen = np.zeros(n, np.float32); loopq = np.zeros(n, np.float32)
    # lagged + rolling peripheral features (per trial)
    peri_lag1 = np.zeros_like(peri); peri_lag3 = np.zeros_like(peri); peri_roll = np.zeros_like(peri)
    for ti in range(len(splits)):
        a, b = offsets[ti], offsets[ti+1]; T = b - a
        bi = np.arange(T); bin_i[a:b] = bi; tlen[a:b] = T
        bin_f[a:b] = bi/max(T,1); loopq[a:b] = np.minimum((bi*4)//max(T,1),3)
        p = peri[a:b]
        peri_lag1[a:b] = np.vstack([p[:1], p[:-1]])
        peri_lag3[a:b] = np.vstack([np.repeat(p[:1],min(3,T),0)[:T][:3] if T>=3 else p[:1].repeat(T,0), p[:-3]])[:T] if T>3 else p
        # rolling mean window 5
        roll = np.copy(p)
        for j in range(T):
            roll[j] = p[max(0,j-4):j+1].mean(0)
        peri_roll[a:b] = roll
    return dict(eeg=eeg_z, peri=peri, peri_lag1=peri_lag1, peri_lag3=peri_lag3,
                peri_roll=peri_roll, bin_i=bin_i, bin_f=bin_f, tlen=tlen, loopq=loopq,
                tm=tm, vm=vm, offsets=offsets, splits=splits)


def feature_set(B, name, target_i):
    """Return feature matrix for a named view, excluding the target peripheral."""
    other_peri = [i for i in [HR_I, GSR_I, IRP_I, RESP_I] if i != target_i]
    op_idx = [i-256 for i in other_peri]
    ctx = np.stack([B['bin_i'], B['bin_f'], B['tlen'], B['loopq']], axis=1)
    eeg = B['eeg']; peri = B['peri'][:, op_idx]
    if name == 'EEG-only':
        return eeg
    if name == 'Peri+ctx':
        return np.concatenate([peri, ctx], axis=1)
    if name == 'Physio-aware':
        # cardio-respiratory channels + their lags/rolling + context
        if target_i == HR_I:      keep = [IRP_I-256, RESP_I-256]
        else:                     keep = [HR_I-256, IRP_I-256, RESP_I-256]
        return np.concatenate([B['peri'][:, keep], B['peri_lag1'][:, keep],
                               B['peri_lag3'][:, keep], B['peri_roll'][:, keep], ctx], axis=1)
    if name == 'All':
        return np.concatenate([eeg, peri, ctx], axis=1)
    if name == 'All+temporal':
        return np.concatenate([eeg, peri, B['peri_lag1'][:, op_idx],
                               B['peri_lag3'][:, op_idx], B['peri_roll'][:, op_idx], ctx], axis=1)
    raise ValueError(name)


def fit_eval(algo, Xtr, ytr, Xva, yva, sub=None):
    if sub is not None and len(Xtr) > sub:
        idx = np.random.RandomState(0).choice(len(Xtr), sub, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    if algo == 'Ridge':
        sc = StandardScaler().fit(np.nan_to_num(Xtr))
        m = Ridge(alpha=10.0).fit(sc.transform(np.nan_to_num(Xtr)), ytr)
        yp = m.predict(sc.transform(np.nan_to_num(Xva)))
    elif algo == 'RandomForest':
        m = RandomForestRegressor(n_estimators=120, max_depth=14, n_jobs=-1,
                                  min_samples_leaf=20, random_state=0)
        m.fit(np.nan_to_num(Xtr), ytr); yp = m.predict(np.nan_to_num(Xva))
    elif algo == 'LightGBM':
        params = dict(objective='regression', metric='rmse', learning_rate=0.03,
                      num_leaves=96, min_data_in_leaf=80, feature_fraction=0.7,
                      bagging_fraction=0.8, bagging_freq=4, lambda_l2=2.0, verbose=-1)
        d = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=1500,
                      valid_sets=[lgb.Dataset(Xva, yva)],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
        yp = d.predict(Xva, num_iteration=d.best_iteration)
    elif algo == 'MLP':
        sc = StandardScaler().fit(np.nan_to_num(Xtr))
        m = MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=60,
                         early_stopping=True, random_state=0)
        m.fit(sc.transform(np.nan_to_num(Xtr)), ytr)
        yp = m.predict(sc.transform(np.nan_to_num(Xva)))
    rmse = float(np.sqrt(((yp - yva)**2).mean()))
    r2 = float(1 - ((yp - yva)**2).mean()/yva.var())
    return rmse, r2


def main():
    cache = load_cache(); X = cache['X']
    B = build(cache)
    tm, vm = B['tm'], B['vm']
    results = {}
    for tname, ti in [('HR', HR_I), ('GSR', GSR_I)]:
        target = X[:, ti].astype(np.float32)
        ytr, yva = target[tm], target[vm]
        print(f"\n===== {tname}  (val std {yva.std():.4g}, mean-predictor R2=0) =====")
        print(f"{'feature view':14s} {'algo':14s} {'RMSE':>11s} {'R2':>8s}")
        rows = []
        views = ['EEG-only', 'Peri+ctx', 'Physio-aware', 'All']
        algos = ['Ridge', 'RandomForest', 'LightGBM', 'MLP']
        for view in views:
            F = feature_set(B, view, ti)
            Xtr, Xva = F[tm], F[vm]
            for algo in algos:
                sub = 80000 if algo in ('RandomForest', 'MLP') else None
                t0 = time.time()
                rmse, r2 = fit_eval(algo, Xtr, ytr, Xva, yva, sub=sub)
                print(f"{view:14s} {algo:14s} {rmse:>11.4g} {r2:>8.3f}   ({time.time()-t0:.0f}s)")
                rows.append(dict(view=view, algo=algo, rmse=rmse, r2=r2))
        # temporal-context variant with LightGBM
        F = feature_set(B, 'All+temporal', ti)
        rmse, r2 = fit_eval('LightGBM', F[tm], ytr, F[vm], yva)
        print(f"{'All+temporal':14s} {'LightGBM':14s} {rmse:>11.4g} {r2:>8.3f}")
        rows.append(dict(view='All+temporal', algo='LightGBM', rmse=rmse, r2=r2))
        best = max(rows, key=lambda r: r['r2'])
        print(f">>> best {tname}: {best['view']} / {best['algo']}  R2={best['r2']:.3f}  RMSE={best['rmse']:.4g}")
        results[tname] = dict(rows=rows, best=best, val_std=float(yva.std()))
    json.dump(results, open(os.path.join(OUT, 'explore_results.json'), 'w'), indent=2)
    print(f"\nsaved {OUT}/explore_results.json")


if __name__ == '__main__':
    main()
