"""Train a model that predicts only the within-trial deviation (y - trial_mean).

Hypothesis: the existing model already captures the within-trial dynamic with
corr ≈ 0.50 on val and pred_dev_std ≈ 0.07 vs true_dev_std ≈ 0.18. If we free
the model from trying to predict absolute level, the within-trial signal should
sharpen. Trial-mean is then provided by a separate model (or fixed) at inference.

We also experiment with stronger augmentation (mixup, channel dropout, noise) and
a higher-capacity backbone, since the multi-task pressure from trial-mean is
removed.
"""
import os, sys, time, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data import (load_cache, load_montage_pos, compute_norm_stats, apply_norm,
                  TrialDataset, pad_collate)
from model import EmapModel, build_idw_kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=3e-4)
    ap.add_argument('--grid', type=int, default=24)
    ap.add_argument('--d_spatial', type=int, default=64)
    ap.add_argument('--d_raw', type=int, default=192)
    ap.add_argument('--d_hidden', type=int, default=256)
    ap.add_argument('--lstm_layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--noise_std', type=float, default=0.05)
    ap.add_argument('--ch_drop', type=float, default=0.05)
    ap.add_argument('--mixup_prob', type=float, default=0.3)
    ap.add_argument('--mixup_alpha', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--use_cnn', type=int, default=1)
    args = ap.parse_args()
    args.use_cnn = bool(args.use_cnn)

    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"device: {dev}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cache = load_cache()
    X = cache['X']; y = cache['y']; offsets = cache['offsets']; splits = cache['split']
    train_trials = np.where(splits == 0)[0]
    val_trials = np.where(splits == 1)[0]

    train_row_mask = np.zeros(len(y), dtype=bool)
    for ti in train_trials:
        train_row_mask[offsets[ti]:offsets[ti + 1]] = True
    stats = compute_norm_stats(X, train_row_mask)
    np.savez(os.path.join(args.out, 'norm_stats.npz'), **stats)
    eeg_z, per_z = apply_norm(X, stats)

    # Compute per-trial mean of y (this is what we subtract; recorded per row for fast access)
    y_trial_mean_per_row = np.zeros_like(y, dtype=np.float32)
    y_trial_mean_per_trial = np.zeros(len(splits), dtype=np.float32)
    for ti in range(len(splits)):
        a, b = offsets[ti], offsets[ti + 1]
        tm = y[a:b].mean()
        y_trial_mean_per_row[a:b] = tm
        y_trial_mean_per_trial[ti] = tm
    # Train target = deviation from trial mean
    y_dev = (y - y_trial_mean_per_row).astype(np.float32)
    print(f"true dev std (train): {y_dev[train_row_mask].std():.4f}", flush=True)

    channels = list(cache['channels'])
    pos2d = load_montage_pos(channels)
    idw, mask_head = build_idw_kernel(pos2d, grid_size=args.grid, power=4.0)
    np.savez(os.path.join(args.out, 'idw_kernel.npz'), idw=idw, mask=mask_head, pos2d=pos2d, channels=np.array(channels))

    train_ds = TrialDataset(eeg_z, per_z, y_dev, offsets, train_trials)
    val_ds   = TrialDataset(eeg_z, per_z, y_dev, offsets, val_trials)
    # For evaluation: also need the true y to compute the actual RMSE after re-adding trial_mean
    val_y_true = y.copy()
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=pad_collate, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          collate_fn=pad_collate, num_workers=0)

    model = EmapModel(idw, mask_head, grid_size=args.grid,
                      d_raw=args.d_raw, d_spatial=args.d_spatial,
                      d_hidden=args.d_hidden,
                      lstm_layers=args.lstm_layers, dropout=args.dropout,
                      use_cnn=args.use_cnn, demean=False,        # we already demeaned the labels
                      noise_std=args.noise_std, ch_drop=args.ch_drop).to(dev)
    # Override head bias to 0 (predicting deviation)
    with torch.no_grad():
        model.head[-1].bias.fill_(0.0)
        model.aux_trial_head.bias.fill_(0.0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = math.inf; best_epoch = -1; log = []

    for ep in range(args.epochs):
        model.train(); t0 = time.time(); loss_tot = 0.0; ntok = 0
        for eeg, per, yt, m, bi, tl, _ in train_dl:
            eeg, per, yt, m, bi, tl = [t.to(dev) for t in (eeg, per, yt, m, bi, tl)]
            if args.mixup_prob > 0 and torch.rand(1).item() < args.mixup_prob:
                perm = torch.randperm(eeg.shape[0], device=dev)
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                lam = max(lam, 1 - lam)
                eeg = lam * eeg + (1 - lam) * eeg[perm]
                per = lam * per + (1 - lam) * per[perm]
                yt  = lam * yt  + (1 - lam) * yt[perm]
                m   = m * m[perm]
            yp, _ = model(eeg, per, m, bi, tl, train_aug=True)
            loss = ((yp - yt) ** 2 * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_tot += float(loss.detach()) * m.sum().item()
            ntok += int(m.sum().item())
        sched.step()
        tr_loss = loss_tot / max(ntok, 1)
        model.eval()
        with torch.no_grad():
            se_pure = 0.0; se_with_mean = 0.0; n_sum = 0
            for eeg, per, yt, m, bi, tl, tids in val_dl:
                eeg, per, yt, m, bi, tl = [t.to(dev) for t in (eeg, per, yt, m, bi, tl)]
                yp, _ = model(eeg, per, m, bi, tl, train_aug=False)
                se_pure += float(((yp - yt) ** 2 * m).sum())
                # Re-add trial-mean for "true" RMSE evaluation
                tmeans = torch.tensor([y_trial_mean_per_trial[t] for t in tids], device=dev).float()
                yp_full = (yp + tmeans.unsqueeze(1)).clamp(0, 1)
                yt_full = yt + tmeans.unsqueeze(1)
                se_with_mean += float(((yp_full - yt_full) ** 2 * m).sum())
                n_sum += int(m.sum().item())
            dev_rmse = math.sqrt(se_pure / max(n_sum, 1))
            full_rmse = math.sqrt(se_with_mean / max(n_sum, 1))
        dt = time.time() - t0
        log.append(dict(epoch=ep, train_loss=tr_loss, dev_rmse=dev_rmse, full_rmse=full_rmse, secs=dt))
        print(f"ep {ep:02d}  trL {tr_loss:.4f}  dev_RMSE {dev_rmse:.4f}  full_RMSE {full_rmse:.4f}  ({dt:.1f}s)", flush=True)
        if dev_rmse < best - 1e-4:
            best = dev_rmse; best_epoch = ep
            torch.save({'state_dict': model.state_dict(), 'args': vars(args), 'dev_rmse': best,
                        'note': 'predicts y - trial_mean'},
                       os.path.join(args.out, 'best.pt'))
        elif ep - best_epoch >= args.patience:
            print(f"early stop at ep {ep} (best dev_rmse={best:.4f} @ ep {best_epoch})", flush=True)
            break
    with open(os.path.join(args.out, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    print(f"BEST dev_rmse = {best:.4f} (ep {best_epoch})", flush=True)


if __name__ == '__main__':
    main()
