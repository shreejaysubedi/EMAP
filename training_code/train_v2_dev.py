"""Like train_v2 but predicts (y - trial_mean) for the residual-target branch.

Used to make a dev-target counterpart of the new quartile-flag model that goes
into the final ensemble.
"""
import os, sys, time, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data import load_cache, load_montage_pos, compute_norm_stats, apply_norm
from model_v2 import EmapModelV2, build_idw_kernel
from train_v2 import TrialDatasetV2, pad_collate_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=3e-4)
    ap.add_argument('--grid', type=int, default=24)
    ap.add_argument('--d_raw', type=int, default=192)
    ap.add_argument('--d_hidden', type=int, default=256)
    ap.add_argument('--d_spatial', type=int, default=64)
    ap.add_argument('--lstm_layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.30)
    ap.add_argument('--noise_std', type=float, default=0.05)
    ap.add_argument('--ch_drop', type=float, default=0.05)
    ap.add_argument('--mixup_prob', type=float, default=0.4)
    ap.add_argument('--mixup_alpha', type=float, default=0.3)
    # optional feature flags
    ap.add_argument('--peri_in_topo', type=int, default=0)
    ap.add_argument('--use_quartile', type=int, default=1)
    ap.add_argument('--use_trial_id', type=int, default=0)
    ap.add_argument('--csd_maps', type=int, default=0)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    args.peri_in_topo = bool(args.peri_in_topo)
    args.use_quartile = bool(args.use_quartile)
    args.use_trial_id = bool(args.use_trial_id)
    args.csd_maps = bool(args.csd_maps)

    if torch.backends.mps.is_available(): dev = torch.device('mps')
    elif torch.cuda.is_available(): dev = torch.device('cuda')
    else: dev = torch.device('cpu')
    print(f"device: {dev}  TAG={args.tag}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cache = load_cache()
    X = cache['X']; y = cache['y']; offsets = cache['offsets']; splits = cache['split']
    trial_id_arr = cache['trial']
    train_trials = np.where(splits == 0)[0]; val_trials = np.where(splits == 1)[0]

    train_row_mask = np.zeros(len(y), dtype=bool)
    for ti in train_trials:
        train_row_mask[offsets[ti]:offsets[ti + 1]] = True
    stats = compute_norm_stats(X, train_row_mask)
    np.savez(os.path.join(args.out, 'norm_stats.npz'), **stats)
    eeg_z, per_z = apply_norm(X, stats)
    train_mu = float(y[train_row_mask].mean())

    # Replace labels with y - trial_mean
    y_dev = np.zeros_like(y, dtype=np.float32)
    trial_mean_per_trial = np.zeros(len(splits), dtype=np.float32)
    for ti in range(len(splits)):
        a, b = offsets[ti], offsets[ti + 1]
        tm = y[a:b].mean()
        y_dev[a:b] = y[a:b] - tm
        trial_mean_per_trial[ti] = tm
    print(f"true dev std (train): {y_dev[train_row_mask].std():.4f}", flush=True)

    channels = list(cache['channels'])
    pos2d = load_montage_pos(channels)
    idw, mh = build_idw_kernel(pos2d, grid_size=args.grid, power=4.0)
    np.savez(os.path.join(args.out, 'idw_kernel.npz'),
             idw=idw, mask=mh, pos2d=pos2d, channels=np.array(channels))

    train_ds = TrialDatasetV2(eeg_z, per_z, y_dev, offsets, train_trials, trial_id_arr)
    val_ds = TrialDatasetV2(eeg_z, per_z, y_dev, offsets, val_trials, trial_id_arr)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=pad_collate_v2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        collate_fn=pad_collate_v2)

    n_trials_vocab = int(trial_id_arr.max()) + 2
    model = EmapModelV2(idw, mh, grid_size=args.grid,
                        d_raw=args.d_raw, d_spatial=args.d_spatial,
                        d_hidden=args.d_hidden,
                        lstm_layers=args.lstm_layers, dropout=args.dropout,
                        peri_in_topo=args.peri_in_topo,
                        use_quartile=args.use_quartile,
                        use_trial_id=args.use_trial_id,
                        csd_maps=args.csd_maps,
                        n_trials=n_trials_vocab,
                        train_mean=0.0).to(dev)
    # Set bias to 0 since target is deviation
    with torch.no_grad():
        model.head[-1].bias.fill_(0.0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = math.inf; best_ep = -1; log = []

    for ep in range(args.epochs):
        model.train(); t0 = time.time(); loss_tot = 0.0; ntok = 0
        for eeg, per, yt, m, bi, tl, tid, _ in train_dl:
            eeg, per, yt, m, bi, tl, tid = [t.to(dev) for t in (eeg, per, yt, m, bi, tl, tid)]
            # Cross-subject mixup at trial level
            if args.mixup_prob > 0 and torch.rand(1).item() < args.mixup_prob:
                perm = torch.randperm(eeg.shape[0], device=dev)
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                lam = max(lam, 1 - lam)
                eeg = lam * eeg + (1 - lam) * eeg[perm]
                per = lam * per + (1 - lam) * per[perm]
                yt = lam * yt + (1 - lam) * yt[perm]
                m = m * m[perm]
            yp = model(eeg, per, m, bi, tl, tid,
                       train_aug=True, noise_std=args.noise_std, ch_drop=args.ch_drop)
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
            se = 0.0; n = 0
            for eeg, per, yt, m, bi, tl, tid, _ in val_dl:
                eeg, per, yt, m, bi, tl, tid = [t.to(dev) for t in (eeg, per, yt, m, bi, tl, tid)]
                yp = model(eeg, per, m, bi, tl, tid, train_aug=False)
                se += float(((yp - yt) ** 2 * m).sum())
                n += int(m.sum().item())
            dev_rmse = math.sqrt(se / max(n, 1))
        dt = time.time() - t0
        log.append(dict(epoch=ep, train_loss=tr_loss, dev_rmse=dev_rmse, secs=dt))
        print(f"ep {ep:02d}  trL {tr_loss:.4f}  dev_RMSE {dev_rmse:.4f}  ({dt:.1f}s)", flush=True)
        if dev_rmse < best - 1e-4:
            best = dev_rmse; best_ep = ep
            torch.save({'state_dict': model.state_dict(), 'args': vars(args),
                        'dev_rmse': best, 'train_mean': train_mu,
                        'note': 'predicts y - trial_mean (v2 with quartile)'},
                       os.path.join(args.out, 'best.pt'))
        elif ep - best_ep >= args.patience:
            print(f"early stop ep {ep} (best {best:.4f} @ ep {best_ep})", flush=True)
            break
    with open(os.path.join(args.out, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    print(f"BEST dev_rmse = {best:.4f} (ep {best_ep})  TAG={args.tag}", flush=True)


if __name__ == '__main__':
    main()
