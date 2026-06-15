"""Train EmapModel and report val RMSE.

The model now returns (per-bin predictions, per-trial-mean prediction). We
optimise a multi-task loss: bin-level MSE + aux trial-mean MSE. Trial-mean
acts as a regulariser anchoring the BiLSTM's pooled representation.
"""
import os, sys, time, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
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
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--grid', type=int, default=24)
    ap.add_argument('--d_spatial', type=int, default=64)
    ap.add_argument('--d_raw', type=int, default=160)
    ap.add_argument('--d_hidden', type=int, default=192)
    ap.add_argument('--lstm_layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.35)
    ap.add_argument('--noise_std', type=float, default=0.10)
    ap.add_argument('--ch_drop', type=float, default=0.10)
    ap.add_argument('--aux_weight', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--out', type=str, default='/Users/shreejaysubedi/Coding/EMAP3/work/run')
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--use_cnn', type=int, default=1)
    ap.add_argument('--demean', type=int, default=1)
    args = ap.parse_args()
    args.use_cnn = bool(args.use_cnn)
    args.demean = bool(args.demean)

    if args.device == 'auto':
        if torch.backends.mps.is_available(): dev = torch.device('mps')
        elif torch.cuda.is_available(): dev = torch.device('cuda')
        else: dev = torch.device('cpu')
    else:
        dev = torch.device(args.device)
    print(f"device: {dev}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cache = load_cache()
    X = cache['X']; y = cache['y']; offsets = cache['offsets']; splits = cache['split']
    train_trials = np.where(splits == 0)[0]
    val_trials = np.where(splits == 1)[0]
    print(f"trials: train={len(train_trials)}, val={len(val_trials)}", flush=True)

    train_row_mask = np.zeros(len(y), dtype=bool)
    for ti in train_trials:
        train_row_mask[offsets[ti]:offsets[ti + 1]] = True
    stats = compute_norm_stats(X, train_row_mask)
    np.savez(os.path.join(args.out, 'norm_stats.npz'), **stats)
    eeg_z, per_z = apply_norm(X, stats)

    channels = list(cache['channels'])
    pos2d = load_montage_pos(channels)
    idw, mask_head = build_idw_kernel(pos2d, grid_size=args.grid, power=4.0)
    np.savez(os.path.join(args.out, 'idw_kernel.npz'), idw=idw, mask=mask_head, pos2d=pos2d, channels=np.array(channels))

    train_ds = TrialDataset(eeg_z, per_z, y, offsets, train_trials)
    val_ds   = TrialDataset(eeg_z, per_z, y, offsets, val_trials)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=pad_collate, num_workers=args.num_workers)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          collate_fn=pad_collate, num_workers=args.num_workers)

    model = EmapModel(idw, mask_head, grid_size=args.grid,
                      d_raw=args.d_raw, d_spatial=args.d_spatial, d_hidden=args.d_hidden,
                      lstm_layers=args.lstm_layers, dropout=args.dropout,
                      use_cnn=args.use_cnn, demean=args.demean,
                      noise_std=args.noise_std, ch_drop=args.ch_drop).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = math.inf
    best_epoch = -1
    log = []
    for ep in range(args.epochs):
        model.train(); t0 = time.time()
        loss_tot = 0.0; ntok = 0
        for eeg, per, yt, m, bi, tl, _ in train_dl:
            eeg, per, yt, m, bi, tl = [t.to(dev) for t in (eeg, per, yt, m, bi, tl)]
            yp, yp_trial = model(eeg, per, m, bi, tl, train_aug=True)
            # Bin-level MSE
            bin_loss = ((yp - yt) ** 2 * m).sum() / m.sum().clamp_min(1.0)
            # Trial-mean aux loss
            trial_true = (yt * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
            trial_loss = ((yp_trial - trial_true) ** 2).mean()
            loss = bin_loss + args.aux_weight * trial_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_tot += float(bin_loss.detach()) * m.sum().item()
            ntok += int(m.sum().item())
        sched.step()
        tr_loss = loss_tot / max(ntok, 1)
        model.eval()
        with torch.no_grad():
            se_sum = 0.0; n_sum = 0; mae_sum = 0.0
            for eeg, per, yt, m, bi, tl, _ in val_dl:
                eeg, per, yt, m, bi, tl = [t.to(dev) for t in (eeg, per, yt, m, bi, tl)]
                yp, _ = model(eeg, per, m, bi, tl, train_aug=False)
                yp = yp.clamp(0.0, 1.0)
                se_sum += float(((yp - yt) ** 2 * m).sum())
                mae_sum += float(((yp - yt).abs() * m).sum())
                n_sum += int(m.sum().item())
            val_rmse = math.sqrt(se_sum / max(n_sum, 1))
            val_mae = mae_sum / max(n_sum, 1)
        dt = time.time() - t0
        log.append(dict(epoch=ep, train_loss=tr_loss, val_rmse=val_rmse, val_mae=val_mae, secs=dt))
        print(f"ep {ep:02d}  trL {tr_loss:.4f}  valRMSE {val_rmse:.4f}  valMAE {val_mae:.4f}  ({dt:.1f}s)", flush=True)
        if val_rmse < best - 1e-4:
            best = val_rmse; best_epoch = ep
            torch.save({'state_dict': model.state_dict(), 'args': vars(args), 'val_rmse': best},
                       os.path.join(args.out, 'best.pt'))
        elif ep - best_epoch >= args.patience:
            print(f"early stop at ep {ep} (best={best:.4f} @ ep {best_epoch})", flush=True)
            break
    with open(os.path.join(args.out, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    print(f"BEST val RMSE = {best:.4f} (epoch {best_epoch})", flush=True)


if __name__ == '__main__':
    main()
