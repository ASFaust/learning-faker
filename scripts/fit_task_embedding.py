"""Test-time task-embedding inversion.

Freeze the whole network. For a task, split its configs into train/val. Fit a
single random task-embedding vector by gradient descent on the TRAIN configs'
curves (backprop the prediction loss into the embedding only), then measure how
well it predicts the held-out VAL configs. Compares against:
  learned  - the task's actual trained embedding (reference upper bound)
  mean     - mean of all task embeddings (a prior)
  random   - an unfit random embedding (lower bound)
  fitted   - our gradient-descent-recovered embedding

    /home/andi/venv/bin/python scripts/fit_task_embedding.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lcfaker.data.base import collate
from lcfaker.losses import pinball_loss
from lcfaker.model import LearningCurveModel, ModelConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, batch, emb):
    with torch.no_grad():
        pred = model(batch, task_emb=emb)
    mae = (pred.median - batch.y).abs().mean().item()
    ti = lambda t: int(np.argmin(np.abs(model.taus.cpu().numpy() - t)))
    lo = pred.quantiles[..., ti(0.05)]
    hi = pred.quantiles[..., ti(0.95)]
    cov90 = (((batch.y >= lo) & (batch.y <= hi)).float().mean().item())
    return mae, cov90


def fit_one(model, ds, tid, name, emb_table, steps=400, n_train=None, seed=0):
    rows = np.nonzero(ds.task_id == tid)[0]
    cfgs = np.unique(ds.config_idx[rows])
    rng = np.random.default_rng(seed); rng.shuffle(cfgs)
    n_tr = n_train if n_train is not None else int(0.7 * len(cfgs))
    tr_c, va_c = set(cfgs[:n_tr]), set(cfgs[n_tr:])
    tr = collate([ds[i] for i in rows if ds.config_idx[i] in tr_c]).to(DEV)
    va = collate([ds[i] for i in rows if ds.config_idx[i] in va_c]).to(DEV)

    d = emb_table.shape[1]
    scale = emb_table.std().item()
    learned = emb_table[tid].to(DEV)
    mean_emb = emb_table.mean(0).to(DEV)
    rand_emb = (torch.randn(d, device=DEV) * scale)

    # fit a random embedding on train configs
    emb = torch.nn.Parameter((torch.randn(d, device=DEV) * scale))
    opt = torch.optim.Adam([emb], lr=2e-2)
    traj = []
    for s in range(steps):
        pred = model(tr, task_emb=emb)
        loss = pinball_loss(pred, tr.y, model.taus)[0]
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0 or s == steps - 1:
            traj.append((s, evaluate(model, va, emb.detach())[0]))

    print(f"\n=== {name}  ({len(tr_c)} train / {len(va_c)} val configs) ===")
    print(f"  fit trajectory (step: val_mae): " + "  ".join(f"{s}:{m:.3f}" for s, m in traj))
    print(f"  {'method':8s} {'train_mae':>9} {'val_mae':>8} {'val_cov90':>9}")
    for label, e in [("learned", learned), ("mean", mean_emb),
                     ("random", rand_emb), ("fitted", emb.detach())]:
        tr_mae = evaluate(model, tr, e)[0]
        v_mae, v_cov = evaluate(model, va, e)
        print(f"  {label:8s} {tr_mae:>9.3f} {v_mae:>8.3f} {v_cov*100:>8.0f}%")


def main():
    ds, vocab = pickle.load(open("/home/andi/datasets/joint4.pkl", "rb"))
    ck = torch.load(ROOT / "checkpoint_joint4.pt", map_location=DEV)
    model = LearningCurveModel(vocab, ModelConfig(**ck["cfg"])).to(DEV)
    model.load_state_dict(ck["model"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    emb_table = model.task_emb.weight.detach().cpu()

    key = {v: k for k, v in vocab.task_id.items()}
    want = ["mlp_family", "conv_pooling", "FixedMLP_mnist", "char_rnn_language"]
    for w in want:
        tid = next((v for k, v in vocab.task_id.items()
                    if k.startswith("taskset/") and w in k), None)
        if tid is not None:
            fit_one(model, ds, tid, key[tid].replace("taskset/", ""), emb_table)


if __name__ == "__main__":
    main()
