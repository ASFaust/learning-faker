"""t-SNE of task embeddings + cluster-compactness stats.

    /home/andi/venv/bin/python scripts/tsne_tasks.py
"""

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def lab(k):
    s, rest = k.split("/", 1)
    if s in ("lcbench", "fcnet"):
        return {"dataset": s, "arch": "MLP", "modality": "tabular"}
    if "transformer" in rest or "xformer" in rest:
        a, m = "Transformer", "sequence"
    elif "resnet" in rest:
        a, m = "ResNet", "image"
    elif "cnn" in rest:
        a, m = "CNN", "image"
    else:
        a, m = "other", "other"
    return {"dataset": "pd1", "arch": a, "modality": m}


def main():
    ds, vocab = pickle.load(open("/home/andi/datasets/joint3.pkl", "rb"))
    ck = torch.load(ROOT / "checkpoint_joint3.pt", map_location="cpu")
    emb = ck["model"]["task_emb.weight"].numpy()
    key = {v: k for k, v in vocab.task_id.items()}
    labs = [lab(key[i]) for i in range(len(emb))]

    # compactness: mean intra-modality vs global cosine distance (lower = tighter)
    D = cosine_distances(emb)
    glob = D[np.triu_indices(len(D), 1)].mean()
    print(f"global mean cosine distance: {glob:.3f}")
    for mod in ["tabular", "image", "sequence"]:
        idx = np.array([i for i, l in enumerate(labs) if l["modality"] == mod])
        if len(idx) < 2:
            print(f"  {mod:9s} n={len(idx)} (too few)"); continue
        intra = D[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()
        print(f"  {mod:9s} n={len(idx):2d}  intra-dist={intra:.3f}  (ratio to global {intra/glob:.2f})")

    # L2-normalize -> euclidean t-SNE ~ cosine
    en = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    xy = TSNE(n_components=2, perplexity=8, init="pca",
              learning_rate="auto", random_state=0).fit_transform(en)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    for ax, field, title in [(axes[0], "dataset", "by dataset"),
                             (axes[1], "arch", "by architecture"),
                             (axes[2], "modality", "by data modality")]:
        cats = sorted({l[field] for l in labs})
        cmap = plt.cm.tab10(np.linspace(0, 1, len(cats)))
        for cat, c in zip(cats, cmap):
            m = np.array([l[field] == cat for l in labs])
            ax.scatter(xy[m, 0], xy[m, 1], color=c, s=70, label=f"{cat} ({m.sum()})",
                       edgecolor="k", linewidth=0.3)
        ax.set_title(f"Task embeddings t-SNE — {title}", fontsize=11)
        ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(ROOT / "tsne_tasks.png", dpi=120)
    print("saved tsne_tasks.png")


if __name__ == "__main__":
    main()
