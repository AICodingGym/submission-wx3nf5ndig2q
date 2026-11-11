"""Sequence model for ventilator pressure. One run = one row in results.tsv.

Usage examples
  python train.py --name lstm_v1 --epochs 30
  python train.py --name lstm_v2 --epochs 80 --hidden 384 --layers 4
  python train.py --name full_s0 --epochs 80 --full        # train on train+val for the final submission

Outputs: checkpoints/<name>.pt (resumable), outputs/<name>_val.npy, outputs/<name>_test.npy
Runs on cuda (DataHub), mps (Apple silicon) or cpu.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from common import insp_mae, load, log_result, per_setting, snap

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--bs", type=int, default=256)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--dropout", type=float, default=0.0)
p.add_argument("--exp_weight", type=float, default=0.0, help="loss weight on expiratory (unscored) rows")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--full", action="store_true", help="fit on train+val (no honest val score)")
p.add_argument("--init", default="", help="warm-start weights from checkpoints/<init>.pt")
p.add_argument("--note", default="")
a = p.parse_args()

torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

d = load()
X, y, u_out, is_val = d["X"], d["y"], d["u_out"], d["is_val"]
tr_idx = np.arange(len(X)) if a.full else np.where(~is_val)[0]
va_idx = np.where(is_val)[0]
T = lambda v: torch.from_numpy(v)
Xtr, ytr, mtr = T(X[tr_idx]).to(dev), T(y[tr_idx]).to(dev), T(u_out[tr_idx] == 0).to(dev)
Xva, Xte = T(X[va_idx]).to(dev), T(d["Xte"]).to(dev)


Y_MU, Y_SD = float(y[~is_val].mean()), float(y[~is_val].std())


class Net(nn.Module):
    def __init__(s, f, h, layers, dropout):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(f, h), nn.SiLU())
        s.rnn = nn.LSTM(h, h, num_layers=layers, batch_first=True, bidirectional=True, dropout=dropout)
        s.head = nn.Sequential(nn.Linear(2 * h, h), nn.SiLU(), nn.Linear(h, 1))

    def forward(s, x):
        # predict in standardized units, return cmH2O (train-split pressure mean/std)
        return s.head(s.rnn(s.inp(x))[0]).squeeze(-1) * Y_SD + Y_MU


model = Net(X.shape[-1], a.hidden, a.layers, a.dropout).to(dev)
nparams = sum(q.numel() for q in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
steps = a.epochs * ((len(tr_idx) + a.bs - 1) // a.bs)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

if a.init:
    model.load_state_dict(torch.load(f"checkpoints/{a.init}.pt", map_location=dev)["model"])
ckpt = f"checkpoints/{a.name}.pt"
start, elapsed = 0, 0.0
if os.path.exists(ckpt):  # resume (DataHub containers time out)
    s = torch.load(ckpt, map_location=dev)
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
    start, elapsed = s["epoch"], s["elapsed"]
    print(f"resumed from epoch {start}")


@torch.no_grad()
def predict(Xs):
    model.eval()
    out = torch.cat([model(Xs[i:i + 256]).float().cpu() for i in range(0, len(Xs), 256)]).numpy()
    if dev == "mps":
        torch.mps.empty_cache()
    return out


print(f"{a.name}: device={dev} params={nparams/1e6:.2f}M train_breaths={len(tr_idx)} epochs={a.epochs}", flush=True)
t0 = time.time()
for ep in range(start, a.epochs):
    model.train()
    perm = torch.randperm(len(Xtr), device=dev)
    tot = 0.0
    for i in range(0, len(perm), a.bs):
        b = perm[i:i + a.bs]
        err = (model(Xtr[b]) - ytr[b]).abs()
        m = mtr[b]
        loss = err[m].mean()  # L1 on the scored rows = the competition metric
        if a.exp_weight > 0:
            loss = loss + a.exp_weight * err[~m].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        tot += err[m].mean().item() * len(b)
    msg = f"ep {ep+1:3d}/{a.epochs} train {tot/len(perm):.4f}"
    if not a.full and ((ep + 1) % 5 == 0 or ep + 1 == a.epochs):
        msg += f" val {insp_mae(predict(Xva), y[va_idx], u_out[va_idx]):.4f}"
    print(msg + f"  [{(elapsed + time.time() - t0)/60:.1f} min]", flush=True)
    if (ep + 1) % 5 == 0 or ep + 1 == a.epochs:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1, "elapsed": elapsed + time.time() - t0, "args": vars(a)}, ckpt)

rt = (elapsed + time.time() - t0) / 60
np.save(f"outputs/{a.name}_test.npy", predict(Xte))
cfg = (f"init={a.init} " if a.init else "") + f"BiLSTM h{a.hidden} L{a.layers} bs{a.bs} lr{a.lr} do{a.dropout} expw{a.exp_weight} seed{a.seed}"
if a.full:
    log_result(name=a.name, val_mae="n/a (full fit)", val_mae_snapped="", runtime_min=round(rt, 1), device=dev,
               epochs=a.epochs, params=nparams, config=cfg, note=a.note)
else:
    pv = predict(Xva)
    np.save(f"outputs/{a.name}_val.npy", pv)
    yv, uv = y[va_idx], u_out[va_idx]
    mae, mae_s = insp_mae(pv, yv, uv), insp_mae(snap(pv, d["grid"]), yv, uv)
    print(f"FINAL val MAE {mae:.4f}  snapped {mae_s:.4f}  runtime {rt:.1f} min")
    print("per (R_C):", per_setting(pv, yv, uv, d["rc"][va_idx]))
    log_result(name=a.name, val_mae=round(mae, 5), val_mae_snapped=round(mae_s, 5), runtime_min=round(rt, 1),
               device=dev, epochs=a.epochs, params=nparams, config=cfg, note=a.note)
