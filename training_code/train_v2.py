"""Train EmapModelV2 with the optional feature flags exposed."""
import os, sys, time, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data import (load_cache, load_montage_pos, compute_norm_stats, apply_norm)
from model_v2 import EmapModelV2, build_idw_kernel


class TrialDatasetV2(Dataset):
    """Same as TrialDataset but also returns trial_id (stimulus id)."""

    def __init__(self, X_eeg, X_per, y, offsets, trial_indices,
                 trial_id_arr, n_ch=64, n_bands=4):
        self.eeg = X_eeg; self.per = X_per; self.y = y
        self.offsets = offsets; self.indices = trial_indices
        self.trial_id = trial_id_arr
        self.n_ch = n_ch; self.n_bands = n_bands

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        ti = self.indices[idx]
        a, b = self.offsets[ti], self.offsets[ti + 1]
        T = b - a
        eeg = self.eeg[a:b].reshape(T, self.n_ch, self.n_bands)
        per = self.per[a:b]
        y = self.y[a:b]
        bi = np.arange(T, dtype=np.int64)
        tid = int(self.trial_id[a])
        return (torch.from_numpy(eeg), torch.from_numpy(per), torch.from_numpy(y),
                torch.from_numpy(bi), int(T), int(tid), int(ti))


def pad_collate_v2(batch):
    Ts = [b[0].shape[0] for b in batch]
    Tmax = max(Ts); B = len(batch)
    eeg = torch.zeros(B, Tmax, batch[0][0].shape[1], batch[0][0].shape[2])
    per = torch.zeros(B, Tmax, batch[0][1].shape[1])
    y = torch.zeros(B, Tmax)
    bin_idx = torch.zeros(B, Tmax, dtype=torch.long)
    trial_len = torch.zeros(B, dtype=torch.long)
    trial_id = torch.zeros(B, dtype=torch.long)
    mask = torch.zeros(B, Tmax)
    tids = []
    for i, (e, p, yy, bi, tl, tid, t) in enumerate(batch):
        T = e.shape[0]
        eeg[i, :T] = e; per[i, :T] = p; y[i, :T] = yy
        bin_idx[i, :T] = bi
        trial_len[i] = tl; trial_id[i] = tid
        mask[i, :T] = 1.0
        tids.append(t)
    return eeg, per, y, mask, bin_idx, trial_len, trial_id, tids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=24)
    ap.add_argument('--lr', type=float, default=1.5e-3)
    ap.add_argument('--wd', type=float, default=3e-4)
    ap.add_argument('--grid', type=int, default=24)
    ap.add_argument('--d_raw', type=int, default=192)
    ap.add_argument('--d_hidden', type=int, default=224)
    ap.add_argument('--d_spatial', type=int, default=64)
    ap.add_argument('--lstm_layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.30)
    ap.add_argument('--noise_std', type=float, default=0.05)
    ap.add_argument('--ch_drop', type=float, default=0.05)
    # --- optional feature flags ---
    ap.add_argument('--peri_in_topo', type=int, default=0)
    ap.add_argument('--use_quartile', type=int, default=0)
    ap.add_argument('--use_trial_id', type=int, default=0)
    ap.add_argument('--csd_maps', type=int, default=0)
    # ---
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--tag', type=str, default='', help='Just for logging.')
    args = ap.parse_args()
    args.peri_in_topo = bool(args.peri_in_topo)
    args.use_quartile = bool(args.use_quartile)
    args.use_trial_id = bool(args.use_trial_id)
    args.csd_maps = bool(args.csd_maps)

    if torch.backends.mps.is_available(): dev = torch.device('mps')
    elif torch.cuda.is_available(): dev = torch.device('cuda')
    else: dev = torch.device('cpu')
    print(f"device: {dev}  TAG={args.tag}", flush=True)
    print(f"flags: peri_in_topo={args.peri_in_topo}  use_quartile={args.use_quartile}  "
          f"use_trial_id={args.use_trial_id}  csd_maps={args.csd_maps}  grid={args.grid}",
          flush=True)
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

    channels = list(cache['channels'])
    pos2d = load_montage_pos(channels)
    idw, mh = build_idw_kernel(pos2d, grid_size=args.grid, power=4.0)
    np.savez(os.path.join(args.out, 'idw_kernel.npz'),
             idw=idw, mask=mh, pos2d=pos2d, channels=np.array(channels))

    train_ds = TrialDatasetV2(eeg_z, per_z, y, offsets, train_trials, trial_id_arr)
    val_ds = TrialDatasetV2(eeg_z, per_z, y, offsets, val_trials, trial_id_arr)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=pad_collate_v2, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        collate_fn=pad_collate_v2, num_workers=0)

    n_trials_vocab = int(trial_id_arr.max()) + 2     # +1 for 0-padding safety
    model = EmapModelV2(idw, mh, grid_size=args.grid,
                        d_raw=args.d_raw, d_spatial=args.d_spatial,
                        d_hidden=args.d_hidden,
                        lstm_layers=args.lstm_layers, dropout=args.dropout,
                        peri_in_topo=args.peri_in_topo,
                        use_quartile=args.use_quartile,
                        use_trial_id=args.use_trial_id,
                        csd_maps=args.csd_maps,
                        n_trials=n_trials_vocab,
                        train_mean=train_mu).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = math.inf; best_ep = -1; log = []

    for ep in range(args.epochs):
        model.train(); t0 = time.time(); loss_tot = 0.0; ntok = 0
        for eeg, per, yt, m, bi, tl, tid, _ in train_dl:
            eeg, per, yt, m, bi, tl, tid = [t.to(dev) for t in (eeg, per, yt, m, bi, tl, tid)]
            yp = model(eeg, per, m, bi, tl, tid,
                       train_aug=True, noise_std=args.noise_std, ch_drop=args.ch_drop)
            bin_loss = ((yp - yt) ** 2 * m).sum() / m.sum().clamp_min(1.0)
            opt.zero_grad(); bin_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_tot += float(bin_loss.detach()) * m.sum().item()
            ntok += int(m.sum().item())
        sched.step()
        tr_loss = loss_tot / max(ntok, 1)

        model.eval()
        with torch.no_grad():
            se = 0.0; n = 0; mae_s = 0.0
            for eeg, per, yt, m, bi, tl, tid, _ in val_dl:
                eeg, per, yt, m, bi, tl, tid = [t.to(dev) for t in (eeg, per, yt, m, bi, tl, tid)]
                yp = model(eeg, per, m, bi, tl, tid, train_aug=False).clamp(0, 1)
                se += float(((yp - yt) ** 2 * m).sum())
                mae_s += float(((yp - yt).abs() * m).sum())
                n += int(m.sum().item())
            val_rmse = math.sqrt(se / max(n, 1))
            val_mae = mae_s / max(n, 1)
        dt = time.time() - t0
        log.append(dict(epoch=ep, train_loss=tr_loss, val_rmse=val_rmse, val_mae=val_mae, secs=dt))
        print(f"ep {ep:02d}  trL {tr_loss:.4f}  valRMSE {val_rmse:.4f}  valMAE {val_mae:.4f}  ({dt:.1f}s)", flush=True)
        if val_rmse < best - 1e-4:
            best = val_rmse; best_ep = ep
            torch.save({'state_dict': model.state_dict(), 'args': vars(args),
                        'val_rmse': best, 'train_mean': train_mu},
                       os.path.join(args.out, 'best.pt'))
        elif ep - best_ep >= args.patience:
            print(f"early stop ep {ep} (best {best:.4f} @ ep {best_ep})", flush=True)
            break
    with open(os.path.join(args.out, 'train_log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    print(f"BEST val RMSE = {best:.4f} (ep {best_ep})  TAG={args.tag}", flush=True)


if __name__ == '__main__':
    main()
