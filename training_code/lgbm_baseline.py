"""Train a LightGBM arousal regressor with the log-transformed EEG + peripheral
features and bin-position context. Saves predictions on the val set so they can
be averaged with the neural-network predictions."""
import os, sys, json, numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from data import load_cache, compute_norm_stats, apply_norm

OUT = os.path.join(os.path.dirname(__file__), 'lgbm')
os.makedirs(OUT, exist_ok=True)


def main():
    cache = load_cache()
    X = cache['X']; y = cache['y']; offsets = cache['offsets']; splits = cache['split']
    trial = cache['trial']; bin_idx = cache['bin_idx']
    tm = np.zeros(X.shape[0], dtype=bool); vm = np.zeros(X.shape[0], dtype=bool)
    for ti in range(len(splits)):
        a, b = offsets[ti], offsets[ti + 1]
        (tm if splits[ti] == 0 else vm)[a:b] = True
    # log1p + clip + z-score per training stats
    stats = compute_norm_stats(X, tm)
    eeg_z, per_z = apply_norm(X, stats)
    # Trial-length & bin position
    trial_len = np.zeros(X.shape[0], dtype=np.int32)
    bin_frac = np.zeros(X.shape[0], dtype=np.float32)
    for ti in range(len(splits)):
        a, b = offsets[ti], offsets[ti + 1]
        T = b - a
        trial_len[a:b] = T
        bin_frac[a:b] = np.arange(T) / max(T, 1)
    Xfeat = np.concatenate([
        eeg_z, per_z,
        bin_idx.reshape(-1, 1).astype(np.float32),
        trial_len.reshape(-1, 1).astype(np.float32),
        bin_frac.reshape(-1, 1),
        trial.reshape(-1, 1).astype(np.float32),
    ], axis=1)
    print(f"feature count: {Xfeat.shape[1]}; train rows={tm.sum()} val rows={vm.sum()}")
    params = dict(objective='regression', metric='rmse', learning_rate=0.03,
                  num_leaves=128, min_data_in_leaf=64, feature_fraction=0.8,
                  bagging_fraction=0.85, bagging_freq=4, lambda_l2=1.0, verbose=-1)
    ds_tr = lgb.Dataset(Xfeat[tm], y[tm])
    ds_va = lgb.Dataset(Xfeat[vm], y[vm], reference=ds_tr)
    m = lgb.train(params, ds_tr, num_boost_round=3000, valid_sets=[ds_va],
                  callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])
    yp_val = np.clip(m.predict(Xfeat[vm], num_iteration=m.best_iteration), 0, 1)
    yp_train = np.clip(m.predict(Xfeat[tm], num_iteration=m.best_iteration), 0, 1)
    rmse = np.sqrt(((yp_val - y[vm]) ** 2).mean())
    print(f"LGBM val RMSE: {rmse:.4f}")
    # Save predictions for ensembling — keep row order matching the cache.
    full_pred = np.zeros_like(y, dtype=np.float32)
    full_pred[tm] = yp_train
    full_pred[vm] = yp_val
    np.savez(os.path.join(OUT, 'lgbm_pred.npz'), pred=full_pred, val_rmse=rmse)
    m.save_model(os.path.join(OUT, 'lgbm.txt'))
    json.dump({'val_rmse': float(rmse)}, open(os.path.join(OUT, 'lgbm.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
