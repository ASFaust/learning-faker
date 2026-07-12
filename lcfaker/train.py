"""Training entrypoint for the learning-curve simulator.

    python -m lcfaker.train --source lcbench --epochs 20
    python -m lcfaker.train --source synthetic

Splits by *config* (not by point) so a held-out curve is fully unseen, which is
the honest test of the surrogate. Reports mean fit in log-loss space and the
variance-calibration ratio (predicted var vs empirical squared residual) to
watch for the collapse we flagged.
"""

from __future__ import annotations

import argparse
import math
import os

# Enumerate GPUs in PCIe bus order (matches nvidia-smi) rather than CUDA's default
# "fastest first" -- so --gpu N picks the Nth device as nvidia-smi lists it. Must be
# set before torch initializes CUDA, hence up here before `import torch`.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data.base import collate
from .losses import pinball_loss
from .model import LearningCurveModel, ModelConfig


def build_source(name: str):
    if name == "synthetic":
        from .data.synthetic import make_synthetic
        return make_synthetic(n_tasks=24, n_configs=80, seed=0)
    if name == "lcbench":
        from .data.lcbench import load_lcbench
        return load_lcbench()
    from .config import DATA_ROOT
    from .data.build import build_joint
    from .data.fcnet import FCNetSource
    from .data.lcbench import LCBenchSource
    from .data.pd1 import PD1Source
    from .data.taskset import TaskSetSource
    if name == "pd1":
        return build_joint([PD1Source()], cache_path=str(DATA_ROOT / "pd1" / "pd1.parsed.pkl"))
    if name == "fcnet":
        return build_joint([FCNetSource()], cache_path=str(DATA_ROOT / "fcnet" / "fcnet.parsed.pkl"))
    if name == "taskset":
        return build_joint([TaskSetSource()], cache_path=str(DATA_ROOT / "taskset_local" / "taskset.pkl"))
    if name == "joint":  # LCBench + PD1
        return build_joint([LCBenchSource(), PD1Source()],
                           cache_path=str(DATA_ROOT / "joint_lcbench_pd1.pkl"))
    if name == "joint3":  # LCBench + PD1 + FCNet
        return build_joint([LCBenchSource(), PD1Source(), FCNetSource()],
                           cache_path=str(DATA_ROOT / "joint3.pkl"))
    if name == "joint4":  # + TaskSet (hundreds of tasks)
        return build_joint([LCBenchSource(), PD1Source(), FCNetSource(), TaskSetSource()],
                           cache_path=str(DATA_ROOT / "joint4.pkl"))
    raise ValueError(name)


def config_split(ds, val_frac: float, seed: int):
    """Hold out whole configs. Works for IndexedCurveDataset (config_idx) and
    falls back to a point split otherwise."""
    rng = np.random.default_rng(seed)
    if hasattr(ds, "config_idx"):
        n_cfg = int(ds.config_idx.max()) + 1
        is_val_cfg = rng.random(n_cfg) < val_frac
        val_mask = is_val_cfg[ds.config_idx]
        val_idx = np.nonzero(val_mask)[0]
        train_idx = np.nonzero(~val_mask)[0]
    else:
        perm = rng.permutation(len(ds))
        cut = int(len(ds) * val_frac)
        val_idx, train_idx = perm[:cut], perm[cut:]
    return Subset(ds, train_idx.tolist()), Subset(ds, val_idx.tolist())


@torch.no_grad()
def evaluate(model, dl, device):
    """mae + central-interval coverage read straight off the quantile grid.

    Coverage at level c uses the grid quantiles tau=(1-c)/2 and (1+c)/2 (present
    in the default grid for c in {50,80,90,96}%): fraction of y inside
    [mean+q_lo, mean+q_hi]."""
    model.eval()
    taus = model.taus.cpu().numpy()

    def find(t):
        i = int(np.argmin(np.abs(taus - t)))
        return i if abs(taus[i] - t) < 1e-3 else None

    pairs = [(c, find((1 - c) / 2), find((1 + c) / 2)) for c in (0.5, 0.8, 0.9, 0.96)]
    pairs = [(c, lo, hi) for c, lo, hi in pairs if lo is not None and hi is not None]
    mae, cov = [], {c: [] for c, _, _ in pairs}
    pin = []
    for b in dl:
        b = b.to(device)
        pred = model(b)
        mae.append((pred.median - b.y).abs().cpu())
        for c, lo, hi in pairs:
            ql = pred.quantiles[..., lo]
            qh = pred.quantiles[..., hi]
            cov[c].append((((b.y >= ql) & (b.y <= qh)).float()).cpu())
        pin.append(pinball_loss(pred, b.y, model.taus)[0].cpu())
    mae = torch.cat(mae)
    out = {"mae_val": mae[:, 0].mean().item(), "mae_train": mae[:, 1].mean().item(),
           "pinball": torch.stack(pin).mean().item()}
    for c, _, _ in pairs:
        out[f"cov{int(c*100)}"] = torch.cat(cov[c]).mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="lcbench",
                    choices=["lcbench", "pd1", "fcnet", "taskset",
                             "joint", "joint3", "joint4", "synthetic"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0,
                    help="CUDA device index in PCIe/nvidia-smi order (see CUDA_DEVICE_ORDER)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not 0 <= args.gpu < torch.cuda.device_count():
            raise SystemExit(f"--gpu {args.gpu} out of range (found {torch.cuda.device_count()} GPUs)")
        torch.cuda.set_device(args.gpu)
        device = f"cuda:{args.gpu}"
        print(f"using {device}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = "cpu"
        print("no CUDA -> cpu")

    ds, vocab = build_source(args.source)
    train_ds, val_ds = config_split(ds, args.val_frac, args.seed)
    print(f"source={args.source} points={len(ds)} "
          f"(train={len(train_ds)} val={len(val_ds)}) tasks={vocab.n_tasks} device={device}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=4096, shuffle=False,
                        collate_fn=collate, num_workers=2)

    model = LearningCurveModel(vocab, ModelConfig()).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e3:.1f}k")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * len(train_dl)
    warmup = min(500, total_steps // 20)

    def lr_at(step):  # linear warmup then cosine decay
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    for ep in range(args.epochs):
        model.train()
        agg = {}
        for b in train_dl:
            b = b.to(device)
            pred = model(b)
            loss, mq = pinball_loss(pred, b.y, model.taus)
            opt.zero_grad(); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gnorm):
                continue  # skip a bad step rather than let NaNs poison the weights
            opt.step(); sched.step()
            for k, v in mq.items():
                agg[k] = agg.get(k, 0.0) + v.item()
        agg = {k: v / max(len(train_dl), 1) for k, v in agg.items()}
        agg.setdefault("pinball", float("nan"))
        ev = evaluate(model, val_dl, device)
        cov = " ".join(f"c{k[3:]}={ev[k]*100:.0f}" for k in ev if k.startswith("cov"))
        print(f"[ep {ep:2d}] train pinball={agg['pinball']:.4f} "
              f"| val mae={ev['mae_val']:.4f}/{ev['mae_train']:.4f} pinball={ev['pinball']:.4f} {cov}")

    torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__},
               f"checkpoint_{args.source}.pt")
    print(f"saved checkpoint_{args.source}.pt")


if __name__ == "__main__":
    main()
