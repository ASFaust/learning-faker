"""Visualize a test-time-fitted task embedding: predictions on train configs
(used to fit it) vs held-out val configs (generalization), val-loss channel.

    /home/andi/venv/bin/python scripts/plot_fitted_embedding.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcfaker.data.base import collate
from lcfaker.losses import pinball_loss
from lcfaker.model import LearningCurveModel, ModelConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TASK = "conv_pooling_family_seed14"
BANDS = [(0.96, 0.12), (0.80, 0.20), (0.50, 0.33)]


def main():
    ds, vocab = pickle.load(open("/home/andi/datasets/joint4.pkl", "rb"))
    ck = torch.load(ROOT / "checkpoint_joint4.pt", map_location=DEV)
    model = LearningCurveModel(vocab, ModelConfig(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["model"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    taus = model.taus.cpu().numpy(); ti = lambda t: int(np.argmin(np.abs(taus - t)))

    tid = next(v for k, v in vocab.task_id.items() if TASK in k)
    learned = model.task_emb.weight[tid].detach()
    rows = np.nonzero(ds.task_id == tid)[0]
    cfgs = np.unique(ds.config_idx[rows])
    rng = np.random.default_rng(0); rng.shuffle(cfgs)
    n_tr = int(0.7 * len(cfgs))
    tr_c, va_c = cfgs[:n_tr], cfgs[n_tr:]
    tr_batch = collate([ds[i] for i in rows if ds.config_idx[i] in set(tr_c)]).to(DEV)

    # fit a random embedding on the train configs
    scale = model.task_emb.weight.std().item()
    emb = torch.nn.Parameter(torch.randn(model.cfg.d_model, device=DEV) * scale)
    opt = torch.optim.Adam([emb], lr=2e-2)
    for _ in range(400):
        pred = model(tr_batch, task_emb=emb)
        loss = pinball_loss(pred, tr_batch.y, model.taus)[0]
        opt.zero_grad(); loss.backward(); opt.step()
    emb = emb.detach()

    def cfg_rows(c):
        rr = rows[ds.config_idx[rows] == c]
        return rr[np.argsort(ds.t_abs[rr])]

    ll = {c: ds.y[ds.config_idx == c, 0].mean() for c in cfgs}
    pick_tr = sorted(tr_c, key=lambda c: ll[c])[:: max(len(tr_c) // 5, 1)][:5]
    pick_va = sorted(va_c, key=lambda c: ll[c])[:: max(len(va_c) // 5, 1)][:5]

    fig, axes = plt.subplots(2, 5, figsize=(22, 8.5))
    for r, (picks, tag) in enumerate([(pick_tr, "TRAIN (fit on these)"),
                                      (pick_va, "VAL (held-out)")]):
        for ax, c in zip(axes[r], picks):
            rr = cfg_rows(c)
            b = collate([ds[i] for i in rr]).to(DEV)
            with torch.no_grad():
                pf = model(b, task_emb=emb)
                pl = model(b, task_emb=learned)
            ep, y = ds.t_abs[rr], ds.y[rr, 0]
            mf = pf.median[:, 0].cpu().numpy()
            for cov, a in BANDS:
                lo = pf.quantiles[:, 0, ti((1 - cov) / 2)].cpu().numpy()
                hi = pf.quantiles[:, 0, ti((1 + cov) / 2)].cpu().numpy()
                ax.fill_between(ep, lo, hi, color="C0", alpha=a)
            ax.plot(ep, mf, "-", color="C0", lw=1.5, label="fitted median")
            ml = pl.median[:, 0].cpu().numpy()
            ax.plot(ep, ml, "--", color="0.35", lw=1.3, label="learned median")
            ax.plot(ep, y, "o", ms=3, color="k", label="real val-loss")
            ax.set_xlabel("step"); ax.set_ylabel("log val-loss")
        axes[r][0].set_ylabel(f"[{tag}]\nlog val-loss", fontsize=10)
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Test-time fitted embedding on {TASK} — top: train configs (fit), "
                 f"bottom: held-out val configs (generalization)", fontsize=13)
    fig.tight_layout(); fig.savefig(ROOT / "plots" / "fitted_embedding.png", dpi=110)
    print("saved plots/fitted_embedding.png")


if __name__ == "__main__":
    main()
