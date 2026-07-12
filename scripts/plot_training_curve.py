"""Plot training curves from a run's history.json.

Reads runs/<source>/history.json (rewritten every epoch by train.py, so this
works on a run that is still going) and draws four panels: pinball loss
(train vs val, with the best-val epoch marked), val MAE (held-out val vs train
channel), central-interval coverage vs its nominal target, and the LR schedule.

    python scripts/plot_training_curve.py [history.json] [out.png]

Defaults to runs/joint4/history.json -> plots/training_curve_joint4.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("history", nargs="?",
                    default=str(ROOT / "runs" / "joint4" / "history.json"))
    ap.add_argument("out", nargs="?", default=None)
    args = ap.parse_args()

    hist_path = Path(args.history)
    blob = json.loads(hist_path.read_text())
    source = blob.get("source", "run")
    H = blob["history"]
    if not H:
        raise SystemExit(f"no epochs in {hist_path}")
    best_ep = blob.get("best_epoch", -1)
    best_val = blob.get("best_val_pinball", float("nan"))

    out_path = Path(args.out) if args.out else ROOT / "plots" / f"training_curve_{source}.png"

    ep = [r["epoch"] for r in H]
    get = lambda k: [r.get(k) for r in H]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_pin, ax_mae, ax_cov, ax_lr = axes.flat

    # -- pinball loss (the optimized objective) --
    ax_pin.plot(ep, get("train_pinball"), "-o", ms=3, color="C1", label="train")
    ax_pin.plot(ep, get("val_pinball"), "-o", ms=3, color="C0", label="val (held-out)")
    if best_ep >= 0:
        ax_pin.axvline(best_ep, color="k", ls="--", lw=1, alpha=0.5)
        ax_pin.scatter([best_ep], [best_val], color="k", zorder=5,
                       label=f"best val @ ep {best_ep} ({best_val:.4f})")
    ax_pin.set_yscale("log")
    ax_pin.set_title("pinball loss (log scale)")
    ax_pin.set_xlabel("epoch"); ax_pin.set_ylabel("pinball")
    ax_pin.legend(fontsize=8)

    # -- MAE on both channels --
    ax_mae.plot(ep, get("val_mae_val"), "-o", ms=3, color="C0", label="val channel")
    ax_mae.plot(ep, get("val_mae_train"), "-o", ms=3, color="C1", label="train channel")
    ax_mae.set_title("held-out MAE (normalized log-loss)")
    ax_mae.set_xlabel("epoch"); ax_mae.set_ylabel("MAE")
    ax_mae.legend(fontsize=8)

    # -- coverage vs nominal target --
    for k, target, col in [("val_cov50", 0.50, "C2"), ("val_cov80", 0.80, "C3"),
                           ("val_cov90", 0.90, "C4"), ("val_cov96", 0.96, "C5")]:
        vals = get(k)
        if any(v is not None for v in vals):
            ax_cov.plot(ep, vals, "-o", ms=3, color=col, label=f"{int(target*100)}%")
            ax_cov.axhline(target, color=col, ls=":", lw=1, alpha=0.6)
    ax_cov.set_title("central-interval coverage (dotted = nominal target)")
    ax_cov.set_xlabel("epoch"); ax_cov.set_ylabel("empirical coverage")
    ax_cov.legend(fontsize=8, ncol=2)

    # -- lr schedule --
    ax_lr.plot(ep, get("lr"), "-", color="0.3")
    ax_lr.set_title("learning rate")
    ax_lr.set_xlabel("epoch"); ax_lr.set_ylabel("lr")

    fig.suptitle(f"{source}: training curves ({len(H)} epochs logged)", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=110)
    print(f"saved {out_path}  ({len(H)} epochs, best val pinball {best_val:.4f} @ ep {best_ep})")


if __name__ == "__main__":
    main()
