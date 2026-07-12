"""PCA of the learned task embeddings -- do clusters emerge by learning
algorithm / data modality?

Uses the trained joint3 model's task_emb table (57 tasks). Raw embeddings (no
whitening yet -- that's a downstream concern).

    /home/andi/venv/bin/python scripts/pca_tasks.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfaker.model import LearningCurveModel, ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def label(key: str) -> dict:
    src, rest = key.split("/", 1)
    if src in ("lcbench", "fcnet"):
        return {"dataset": src, "arch": "MLP", "modality": "tabular"}
    # pd1: "<data>_<model>_bs<N>"
    if "transformer" in rest or "xformer" in rest:
        arch, mod = "Transformer", "sequence"
    elif "resnet" in rest:
        arch, mod = "ResNet", "image"
    elif "cnn" in rest:
        arch, mod = "CNN", "image"
    else:
        arch, mod = "other", "other"
    return {"dataset": "pd1", "arch": arch, "modality": mod}


def main():
    ds, vocab = pickle.load(open("/home/andi/datasets/joint3.pkl", "rb"))
    ck = torch.load(ROOT / "checkpoint_joint3.pt", map_location="cpu")
    # only the task embedding table is needed (independent of the head arch)
    emb = ck["model"]["task_emb.weight"].detach().numpy()   # (n_tasks, d)

    key = {v: k for k, v in vocab.task_id.items()}
    labs = [label(key[i]) for i in range(len(emb))]

    pca = PCA(n_components=2).fit(emb)
    xy = pca.transform(emb)
    ev = pca.explained_variance_ratio_
    print(f"embeddings: {emb.shape} | PC1/PC2 explained var: {ev[0]:.2f}/{ev[1]:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    for ax, field, title in [(axes[0], "dataset", "by dataset"),
                             (axes[1], "arch", "by architecture"),
                             (axes[2], "modality", "by data modality")]:
        cats = sorted({l[field] for l in labs})
        cmap = plt.cm.tab10(np.linspace(0, 1, len(cats)))
        for cat, c in zip(cats, cmap):
            m = np.array([l[field] == cat for l in labs])
            ax.scatter(xy[m, 0], xy[m, 1], color=c, s=60, label=f"{cat} ({m.sum()})",
                       edgecolor="k", linewidth=0.3, alpha=0.9)
        ax.set_title(f"Task embeddings PCA — {title}", fontsize=11)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)"); ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "pca_tasks.png", dpi=120)
    print("saved pca_tasks.png")


if __name__ == "__main__":
    main()
