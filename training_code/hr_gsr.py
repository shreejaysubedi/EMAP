"""Regression baselines for heart rate and GSR using LightGBM on the same cache.

For each side-task we drop the target column from the feature matrix and add a
participant-bin time index. We train one global LightGBM per target with the
existing train/val split and report RMSE.
"""
import os, sys, json
import numpy as np
import pandas as pd
import lightgbm as lgb

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from data import load_cache


def fit_target(X, y, train_mask, val_mask, name: str, out_dir: str):
    train = lgb.Dataset(X[train_mask], y[train_mask])
    val = lgb.Dataset(X[val_mask], y[val_mask], reference=train)
    params = dict(objective='regression', metric='rmse', learning_rate=0.05,
                  num_leaves=128, feature_fraction=0.8, bagging_fraction=0.85,
                  bagging_freq=4, min_data_in_leaf=64, verbose=-1)
    model = lgb.train(params, train, num_boost_round=3000, valid_sets=[val],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    yp = model.predict(X[val_mask], num_iteration=model.best_iteration)
    rmse = float(np.sqrt(((yp - y[val_mask]) ** 2).mean()))
    print(f"{name}: val RMSE = {rmse:.4f}")
    model.save_model(os.path.join(out_dir, f'{name}_lgbm.txt'))
    return rmse, model


def main():
    out = os.path.join(THIS, 'side_tasks'); os.makedirs(out, exist_ok=True)
    cache = load_cache()
    X = cache['X']; offsets = cache['offsets']; splits = cache['split']
    feat = list(cache['feat_cols'])
    # train/val row masks
    tm = np.zeros(X.shape[0], dtype=bool); vm = np.zeros(X.shape[0], dtype=bool)
    for ti, sp in enumerate(splits):
        a, b = offsets[ti], offsets[ti + 1]
        if sp == 0: tm[a:b] = True
        else: vm[a:b] = True
    # Add bin-position context feature
    bin_idx = cache['bin_idx'].astype(np.float32).reshape(-1, 1)
    targets = {'HR': 'heartrate_mean', 'GSR': 'GSR_mean'}
    summary = {}
    for tag, col in targets.items():
        ci = feat.index(col)
        keep = [i for i in range(len(feat)) if i != ci]
        Xt = np.concatenate([X[:, keep], bin_idx], axis=1).astype(np.float32)
        y = X[:, ci].astype(np.float32)
        rmse, _ = fit_target(Xt, y, tm, vm, tag, out)
        summary[tag] = rmse
    with open(os.path.join(out, 'side_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
