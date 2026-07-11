"""Analysis, in the model's native log-loss space (no exp round-trip).

Produces:
  curves_logspace.png  -- held-out curves: real vs predicted mean +/- sigma,
                          plotted directly in log-loss (band symmetric by
                          construction, which is the honest picture).
  var_diagnostic.png   -- does predicted sigma track actual error?
                          (heteroscedastic reliability + z-score calibration)

    /home/andi/venv/bin/python scripts/analyze.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfaker.data.base import collate
from lcfaker.data.lcbench import load_lcbench
from lcfaker.model import LearningCurveModel, ModelConfig

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoint_lcbench.pt"


def load():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, vocab = load_lcbench()
    ckpt = torch.load(CKPT, map_location=device)
    model = LearningCurveModel(vocab, ModelConfig(**ckpt["cfg"])).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    rng = np.random.default_rng(0)
    n_cfg = int(ds.config_idx.max()) + 1
    is_val_cfg = rng.random(n_cfg) < 0.1
    val_cfgs = np.nonzero(is_val_cfg)[0]
    return ds, vocab, model, device, val_cfgs


def predict_rows(model, ds, rows, device):
    batch = collate([ds[i] for i in rows]).to(device)
    with torch.no_grad():
        mean, var = model(batch)
    return mean.cpu().numpy(), var.sqrt().cpu().numpy()


def plot_curves(ds, vocab, model, device, val_cfgs):
    cfg_mean = {c: ds.y[ds.config_idx == c, 0].mean() for c in val_cfgs}
    ranked = sorted(val_cfgs, key=lambda c: cfg_mean[c])
    picks = [ranked[int(p)] for p in np.linspace(0, len(ranked) - 1, 8)]
    task_name = {v: k for k, v in vocab.task_id.items()}

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, c in zip(axes.flat, picks):
        rows = np.nonzero(ds.config_idx == c)[0]
        rows = rows[np.argsort(ds.t_abs[rows])]
        mean, std = predict_rows(model, ds, rows, device)
        ep, y = ds.t_abs[rows], ds.y[rows]
        for ch, (nm, col) in enumerate([("val", "C0"), ("train", "C1")]):
            ax.plot(ep, y[:, ch], "o", ms=3, color=col, label=f"{nm} real")
            ax.plot(ep, mean[:, ch], "-", color=col, label=f"{nm} pred")
            ax.fill_between(ep, mean[:, ch] - std[:, ch], mean[:, ch] + std[:, ch],
                            color=col, alpha=0.2)
        tn = task_name[int(ds.task_id[rows[0]])].replace("lcbench/", "")
        ax.set_title(f"{tn} | cfg {c} | mae={np.abs(mean-y).mean():.3f}", fontsize=9)
        ax.set_xlabel("epoch"); ax.set_ylabel("log-loss")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Held-out curves in native log-loss space (band = mean +/- 1 sigma)", fontsize=13)
    fig.tight_layout(); fig.savefig(ROOT / "curves_logspace.png", dpi=110)
    print("saved curves_logspace.png")


def var_diagnostic(ds, vocab, model, device, val_cfgs):
    val_rows = np.nonzero(np.isin(ds.config_idx, val_cfgs))[0]
    dl = DataLoader(Subset(ds, val_rows.tolist()), batch_size=8192,
                    shuffle=False, collate_fn=collate)
    means, stds, ys = [], [], []
    for b in dl:
        b = b.to(device)
        with torch.no_grad():
            m, v = model(b)
        means.append(m.cpu()); stds.append(v.sqrt().cpu()); ys.append(b.y.cpu())
    mean = torch.cat(means).numpy(); std = torch.cat(stds).numpy(); y = torch.cat(ys).numpy()
    err = np.abs(mean - y).ravel()
    sig = std.ravel()
    z = ((y - mean) / std).ravel()

    from scipy.stats import spearmanr
    rho = spearmanr(sig, err).statistic

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    # (A) predicted sigma vs |error|, hexbin
    hb = ax[0].hexbin(sig, err, gridsize=60, bins="log", mincnt=1, extent=(0, 0.25, 0, 0.25))
    ax[0].plot([0, 0.25], [0, 0.25], "r--", lw=1)
    ax[0].set(xlabel="predicted sigma", ylabel="|mean error|",
              title=f"sigma vs error (Spearman rho={rho:.3f})", xlim=(0, 0.25), ylim=(0, 0.25))
    fig.colorbar(hb, ax=ax[0], label="log count")
    # (B) reliability: bin by predicted sigma, empirical RMS residual per bin
    order = np.argsort(sig)
    nb = 20
    bins = np.array_split(order, nb)
    ps = np.array([sig[b].mean() for b in bins])
    es = np.array([np.sqrt(((y - mean).ravel()[b] ** 2).mean()) for b in bins])
    ax[1].plot(ps, es, "o-"); ax[1].plot([0, ps.max()], [0, ps.max()], "r--", lw=1)
    ax[1].set(xlabel="predicted sigma (bin mean)", ylabel="empirical RMS residual",
              title="reliability: is sigma the right *scale*?")
    # (C) z-score calibration
    ax[2].hist(np.clip(z, -6, 6), bins=80, density=True, alpha=0.7)
    xs = np.linspace(-6, 6, 200)
    ax[2].plot(xs, np.exp(-xs**2/2)/np.sqrt(2*np.pi), "r-", label="N(0,1)")
    ax[2].set(xlabel="z = (y - mean)/sigma", title=f"z calibration: std(z)={z.std():.2f}", xlim=(-6, 6))
    ax[2].legend()
    fig.tight_layout(); fig.savefig(ROOT / "var_diagnostic.png", dpi=110)
    print("saved var_diagnostic.png")

    print(f"\n--- variance diagnostics (val, {len(err)} points) ---")
    print(f"predicted sigma: p10={np.percentile(sig,10):.4f} med={np.median(sig):.4f} "
          f"p90={np.percentile(sig,90):.4f} max={sig.max():.4f}")
    print(f"Spearman(sigma, |err|) = {rho:.3f}   (>0 => sigma tracks error)")
    print(f"std(z) = {z.std():.2f}   (1.0 = calibrated; >1 = overconfident)")
    print(f"single global temperature to recalibrate: sigma *= {z.std():.2f}")


def main():
    ds, vocab, model, device, val_cfgs = load()
    plot_curves(ds, vocab, model, device, val_cfgs)
    var_diagnostic(ds, vocab, model, device, val_cfgs)


if __name__ == "__main__":
    main()
