"""Plot real vs predicted learning curves for held-out (validation) configs.

Reconstructs the exact config-level split train.py uses (seed 0, val_frac 0.1),
picks held-out configs spanning the loss range across ALL four datasets, and for
each plots the real val/train curve against the model's predicted mean and
residual-quantile bands (central 50% and 90%).

Targets are per-task-normalized log(loss/ref); we recover absolute loss with
loss = exp(y_norm + log ref_task) for reals, predicted mean, and quantiles alike,
so the panels read in real cross-entropy / loss units.

    /home/andi/venv/bin/python scripts/plot_val_curves.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfaker.data.base import collate           # noqa: E402
from lcfaker.model import LearningCurveModel, ModelConfig  # noqa: E402
from lcfaker.train import build_source           # noqa: E402

CKPT = Path(__file__).resolve().parents[1] / "checkpoint_joint4.pt"
OUT = Path(__file__).resolve().parents[1] / "plots" / "val_curves_joint4.png"
PER_DATASET = 4  # configs spanning the loss range per dataset -> 4x4 grid


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, vocab = build_source("joint4")

    # reconstruct the held-out config set exactly as train.config_split (seed 0)
    rng = np.random.default_rng(0)
    n_cfg = int(ds.config_idx.max()) + 1
    is_val_cfg = rng.random(n_cfg) < 0.1

    # per-config task id and mean normalized val-loss, in one pass over points
    counts = np.bincount(ds.config_idx, minlength=n_cfg)
    sum_yv = np.bincount(ds.config_idx, weights=ds.y[:, 0], minlength=n_cfg)
    cfg_mean = np.where(counts > 0, sum_yv / np.maximum(counts, 1), np.nan)
    cfg_task = np.zeros(n_cfg, np.int64)
    cfg_task[ds.config_idx] = ds.task_id            # each config -> its one task

    task_name = {v: k for k, v in vocab.task_id.items()}
    log_ref = np.log(ds.task_ref)                    # (n_tasks,)

    # group held-out configs by dataset prefix, pick low/high-loss spread from each
    picks = []
    for dset in ("lcbench", "pd1", "fcnet", "taskset"):
        members = [c for c in np.nonzero(is_val_cfg)[0]
                   if task_name[cfg_task[c]].startswith(dset) and counts[c] > 3]
        if not members:
            continue
        members.sort(key=lambda c: cfg_mean[c])
        idxs = np.linspace(0, len(members) - 1, PER_DATASET).round().astype(int)
        picks.extend(members[i] for i in idxs)

    ckpt = torch.load(CKPT, map_location=device)
    model = LearningCurveModel(vocab, ModelConfig(**ckpt["cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    taus = model.taus.cpu().numpy()

    def qidx(level):
        return int(np.argmin(np.abs(taus - level)))
    lo90, hi90 = qidx(0.05), qidx(0.95)
    lo50, hi50 = qidx(0.25), qidx(0.75)

    ncol = 4
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    for ax, c in zip(axes.flat, picks):
        rows = np.flatnonzero(ds.config_idx == c)
        rows = rows[np.argsort(ds.t_rel[rows])]
        batch = collate([ds[i] for i in rows]).to(device)
        with torch.no_grad():
            pred = model(batch)
        median = pred.median.cpu().numpy()           # (n, 2) normalized log, point est
        quant = pred.quantiles.cpu().numpy()         # (n, 2, Q) absolute quantiles
        tid = int(batch.task_id[0]); lr = log_ref[tid]
        t = ds.t_rel[rows]
        y = ds.y[rows]                               # normalized log-loss [val, train]

        def to_loss(a):                              # normalized log -> absolute loss
            return np.exp(a + lr)

        for ch, (nm, col) in enumerate([("val", "C0"), ("train", "C1")]):
            ax.plot(t, to_loss(y[:, ch]), "o", ms=3, color=col, label=f"{nm} real")
            ax.plot(t, to_loss(median[:, ch]), "-", color=col, lw=1.6, label=f"{nm} pred (median)")
            ax.fill_between(t, to_loss(quant[:, ch, lo90]),
                            to_loss(quant[:, ch, hi90]), color=col, alpha=0.12)
            ax.fill_between(t, to_loss(quant[:, ch, lo50]),
                            to_loss(quant[:, ch, hi50]), color=col, alpha=0.22)
        tn = task_name[tid]
        mae = np.abs(median - y).mean()
        ax.set_title(f"{tn}\ncfg {c} | ref={ds.task_ref[tid]:.3g} | mae(logn)={mae:.3f}", fontsize=8)
        ax.set_xlabel("t_rel (fraction of budget)")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
    for ax in axes.flat[len(picks):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, loc="best")
    fig.suptitle("joint4 held-out configs: real vs predicted (bands = residual central 50% / 90%)",
                 fontsize=13)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"saved {OUT}  ({len(picks)} panels)")


if __name__ == "__main__":
    main()
