"""Held-out curves with *calibrated* predictive intervals.

Uses the globally-fit Student-t (nu=0.91, scale=0.64*sigma) so each shaded band
is a real central-coverage region (50 / 80 / 95%), verified to match empirical
coverage, rather than an arbitrary +/- sigma.

    /home/andi/venv/bin/python scripts/plot_calibrated.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfaker.data.base import collate
from lcfaker.data.lcbench import load_lcbench
from lcfaker.model import LearningCurveModel, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
NU = 0.91          # global fitted Student-t dof
GSCALE = 0.64      # global scale correction on model sigma
LEVELS = [0.95, 0.80, 0.50]        # widest -> narrowest (draw in this order)
ALPHAS = [0.12, 0.20, 0.32]
TVAL = {L: stats.t.ppf(0.5 + L / 2, NU) for L in LEVELS}  # half-width in scaled-sigma units


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, vocab = load_lcbench()
    ck = torch.load(ROOT / "checkpoint_lcbench.pt", map_location=device)
    model = LearningCurveModel(vocab, ModelConfig(**ck["cfg"])).to(device)
    model.load_state_dict(ck["model"]); model.eval()

    rng = np.random.default_rng(0)
    n_cfg = int(ds.config_idx.max()) + 1
    val_cfgs = np.nonzero(rng.random(n_cfg) < 0.1)[0]
    cfg_mean = {c: ds.y[ds.config_idx == c, 0].mean() for c in val_cfgs}
    ranked = sorted(val_cfgs, key=lambda c: cfg_mean[c])
    picks = [ranked[int(p)] for p in np.linspace(0, len(ranked) - 1, 8)]
    task_name = {v: k for k, v in vocab.task_id.items()}

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, c in zip(axes.flat, picks):
        rows = np.nonzero(ds.config_idx == c)[0]
        rows = rows[np.argsort(ds.t_abs[rows])]
        batch = collate([ds[i] for i in rows]).to(device)
        with torch.no_grad():
            mean, var = model(batch)
        mean = mean.cpu().numpy()
        sig = (var.sqrt().cpu().numpy()) * GSCALE   # calibrated scale
        ep, y = ds.t_abs[rows], ds.y[rows]

        for ch, (nm, col) in enumerate([("val", "C0"), ("train", "C1")]):
            for L, a in zip(LEVELS, ALPHAS):
                hw = TVAL[L] * sig[:, ch]
                ax.fill_between(ep, mean[:, ch] - hw, mean[:, ch] + hw, color=col, alpha=a,
                                label=f"{nm} {int(L*100)}%" if ch < 2 and L == LEVELS[0] else None)
            ax.plot(ep, mean[:, ch], "-", color=col, lw=1.5)
            ax.plot(ep, y[:, ch], "o", ms=3, color=col, mec="k", mew=0.3)

        # y-limits: keep the 80% band + data readable; let the 95% tail clip
        hw80 = TVAL[0.80] * sig
        lo = min(y.min(), (mean - hw80).min())
        hi = max(y.max(), (mean + hw80).max())
        pad = 0.05 * (hi - lo + 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
        tn = task_name[int(ds.task_id[rows[0]])].replace("lcbench/", "")
        ax.set_title(f"{tn} | cfg {c}", fontsize=9)
        ax.set_xlabel("epoch"); ax.set_ylabel("log-loss")

    # shared legend: shading meaning
    handles = [plt.Rectangle((0, 0), 1, 1, color="C0", alpha=a) for a in ALPHAS]
    axes.flat[0].legend(handles, [f"{int(L*100)}% interval" for L in LEVELS], fontsize=8, loc="best")
    fig.suptitle(f"Held-out curves with calibrated predictive intervals "
                 f"[StudentT(nu={NU}, scale={GSCALE}*sigma)] -- shading = 50/80/95% coverage",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(ROOT / "curves_calibrated.png", dpi=110)
    print("saved curves_calibrated.png")


if __name__ == "__main__":
    main()
