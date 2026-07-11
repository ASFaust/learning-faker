"""Task-embedding structure at 359 tasks (joint4): does cluster structure emerge
now that the task axis is 6x richer? Full-space metrics + t-SNE.

    /home/andi/venv/bin/python scripts/analyze_embeddings.py
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
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def family(key: str) -> str:
    src, rest = key.split("/", 1)
    r = rest.lower()
    if src in ("lcbench", "fcnet"):
        return "mlp"
    if src == "pd1":
        if "transformer" in r or "xformer" in r:
            return "transformer"
        return "resnet" if "resnet" in r else "conv"
    # taskset
    if "conv" in r or "cnn" in r:
        return "conv"
    if any(x in r for x in ["lstm", "gru", "vrnn", "rnn"]) or r.startswith("fixedlm") or "_lm_" in r:
        return "rnn"
    if "transformer" in r or "attention" in r:
        return "transformer"
    if any(x in r for x in ["twod", "bowl", "quad", "rosenbrock", "logreg", "_norm", "wine", "iris"]):
        return "synthetic/simple"
    if "mlp" in r or "_ae" in r or "fc_" in r or "maf" in r or "nvp" in r or "vae" in r:
        return "mlp"
    return "other"


def main():
    ds, vocab = pickle.load(open("/home/andi/datasets/joint4.pkl", "rb"))
    ck = torch.load(ROOT / "checkpoint_joint4.pt", map_location="cpu")
    emb = ck["model"]["task_emb.weight"].numpy()
    key = {v: k for k, v in vocab.task_id.items()}
    dataset = [key[i].split("/")[0] for i in range(len(emb))]
    fam = [family(key[i]) for i in range(len(emb))]

    # full-space structure
    S = cosine_similarity(emb); np.fill_diagonal(S, -1); nn = S.argmax(1)
    from collections import Counter
    print(f"{len(emb)} tasks | mean pairwise cosine dist "
          f"{1 - S[S > -1].mean():.3f}")
    for name, y in [("dataset", dataset), ("family", fam)]:
        purity = np.mean([y[i] == y[nn[i]] for i in range(len(y))])
        chance = max(Counter(y).values()) / len(y)
        yi = np.array([sorted(set(y)).index(v) for v in y])
        sil = silhouette_score(emb, yi, metric="cosine")
        print(f"  {name:8s}: 1-NN purity {purity*100:4.0f}% (chance {chance*100:.0f}%)  "
              f"silhouette {sil:+.3f}  [{len(set(y))} classes]")

    xy = TSNE(n_components=2, perplexity=15, init="pca",
              learning_rate="auto", random_state=0).fit_transform(
                  emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, field, vals, title in [(axes[0], "dataset", dataset, "by dataset"),
                                    (axes[1], "family", fam, "by architecture family")]:
        cats = sorted(set(vals))
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(cats), 3)))
        for cat, c in zip(cats, cmap):
            m = np.array([v == cat for v in vals])
            ax.scatter(xy[m, 0], xy[m, 1], color=c, s=45, label=f"{cat} ({m.sum()})",
                       edgecolor="k", linewidth=0.25, alpha=0.85)
        ax.set_title(f"359-task embedding t-SNE — {title}", fontsize=11)
        ax.legend(fontsize=8, loc="best"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(ROOT / "emb359.png", dpi=120)
    print("saved emb359.png")


if __name__ == "__main__":
    main()
