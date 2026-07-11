"""End-to-end smoke test: build synthetic data, train a few hundred steps,
confirm the mean loss drops and inspect whether the variance head collapses.

    /home/andi/venv/bin/python scripts/smoke_train.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfaker.data import collate, make_synthetic
from lcfaker.losses import curve_loss
from lcfaker.model import LearningCurveModel, ModelConfig


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds, vocab = make_synthetic(n_tasks=24, n_configs=80, seed=0)
    print(f"dataset: {len(ds)} points | param_types={vocab.n_param_types} "
          f"cat_values={vocab.n_cat_values} tasks={vocab.n_tasks}")

    n_val = len(ds) // 10
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(1))
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=512, shuffle=False, collate_fn=collate)

    model = LearningCurveModel(vocab, ModelConfig()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e3:.1f}k")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    step = 0
    for epoch in range(15):
        model.train()
        for batch in train_dl:
            batch = batch.to(device)
            mean, var = model(batch)
            loss, m = curve_loss(mean, var, batch.y, batch.w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        if epoch % 3 == 0 or epoch == 14:
            print(f"[ep {epoch:2d} step {step:4d}] train loss={m['loss']:.4f} "
                  f"mean_err={m['mean_err']:.4f} var_err={m['var_err']:.4f} "
                  f"var_pred={m['var_pred']:.5f}")

    # held-out calibration: predicted var vs empirical squared residual
    model.eval()
    sq_res, var_pred, mae = [], [], []
    with torch.no_grad():
        for batch in val_dl:
            batch = batch.to(device)
            mean, var = model(batch)
            sq_res.append(((mean - batch.y) ** 2).cpu())
            var_pred.append(var.cpu())
            mae.append((mean - batch.y).abs().cpu())
    sq_res = torch.cat(sq_res); var_pred = torch.cat(var_pred); mae = torch.cat(mae)
    print("\n--- held-out ---")
    print(f"mean |err| (log-loss)        : val={mae[:,0].mean():.4f} train={mae[:,1].mean():.4f}")
    print(f"empirical E[resid^2]         : {sq_res.mean():.5f}")
    print(f"predicted var (mean)         : {var_pred.mean():.5f}")
    ratio = (var_pred.mean() / sq_res.mean().clamp_min(1e-9)).item()
    print(f"var_pred / E[resid^2]        : {ratio:.3f}  "
          f"({'collapsed' if ratio < 0.3 else 'ok'})")

    # show a few predictions vs targets
    b = next(iter(val_dl)).to(device)
    with torch.no_grad():
        mean, var = model(b)
    print("\nsample (log-loss): pred_val  targ_val | pred_train targ_train | t_rel")
    for i in range(min(6, len(b))):
        print(f"  {mean[i,0]:+.3f}   {b.y[i,0]:+.3f}  | "
              f"{mean[i,1]:+.3f}    {b.y[i,1]:+.3f}   | {b.t_rel[i]:.2f}")


if __name__ == "__main__":
    main()
