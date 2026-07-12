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
import json
import math
import os
from pathlib import Path

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


def make_loaders(train_ds, val_ds, batch_size: int, device: str):
    """DataLoaders with the throughput settings shared by training and HPO:
    workers prep+collate ahead of the GPU; pin_memory + non_blocking .to()
    overlaps the H2D copy with compute; persistent_workers keeps the pool alive
    across epochs (and, in HPO, across all trials on a worker) instead of
    re-forking."""
    pin = device != "cpu"
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=collate, num_workers=4, drop_last=True,
                          pin_memory=pin, persistent_workers=True, prefetch_factor=4)
    val_dl = DataLoader(val_ds, batch_size=4096, shuffle=False,
                        collate_fn=collate, num_workers=2,
                        pin_memory=pin, persistent_workers=True, prefetch_factor=4)
    return train_dl, val_dl


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
        b = b.to(device, non_blocking=True)
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


def fit(cfg, hp, device, vocab, train_dl, val_dl, *, epochs,
        source=None, out_dir=None, on_epoch=None, meta=None, log=True):
    """Train one model and return (best_val_pinball, best_epoch).

    Shared by the CLI (`main`) and the HPO objective. `cfg` is a ModelConfig;
    `hp` carries the optimizer knobs {"lr", "weight_decay"}. When `out_dir` is
    given it writes per-epoch checkpoints, best_val_loss.pt and history.json
    (CLI behavior); HPO passes out_dir=None to skip all disk I/O. `on_epoch(ep,
    record)` is called after each epoch's eval -- HPO uses it to report the
    intermediate val pinball and raise optuna.TrialPruned for early stopping."""
    model = LearningCurveModel(vocab, cfg).to(device)
    if log:
        print(f"model params: {sum(p.numel() for p in model.parameters())/1e3:.1f}k")
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"],
                            weight_decay=hp["weight_decay"])
    total_steps = epochs * len(train_dl)
    warmup = min(500, total_steps // 20)

    def lr_at(step):  # linear warmup then cosine decay
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    history, best_val, best_ep = [], float("inf"), -1

    for ep in range(epochs):
        model.train()
        agg = {}
        for b in train_dl:
            b = b.to(device, non_blocking=True)
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
        if log:
            cov = " ".join(f"c{k[3:]}={ev[k]*100:.0f}" for k in ev if k.startswith("cov"))
            print(f"[ep {ep:2d}] train pinball={agg['pinball']:.4f} "
                  f"| val mae={ev['mae_val']:.4f}/{ev['mae_train']:.4f} pinball={ev['pinball']:.4f} {cov}")

        # per-epoch record (history.json) + checkpoint; best tracked by val pinball
        record = {"epoch": ep, "lr": sched.get_last_lr()[0],
                  "train_pinball": agg["pinball"], "val_mae_val": ev["mae_val"],
                  "val_mae_train": ev["mae_train"], "val_pinball": ev["pinball"],
                  **{f"val_{k}": ev[k] for k in ev if k.startswith("cov")}}
        history.append(record)
        is_best = ev["pinball"] < best_val
        if is_best:
            best_val, best_ep = ev["pinball"], ep
        if out_dir is not None:
            ckpt = {"model": model.state_dict(), "cfg": model.cfg.__dict__,
                    "epoch": ep, "source": source, "metrics": record}
            torch.save(ckpt, out_dir / f"epoch_{ep:03d}.pt")
            if is_best:
                torch.save(ckpt, out_dir / "best_val_loss.pt")
            # rewrite history every epoch so a crashed/killed run is reconstructable
            with open(out_dir / "history.json", "w") as f:
                json.dump({"source": source, "args": meta or {},
                           "best_epoch": best_ep, "best_val_pinball": best_val,
                           "history": history}, f, indent=2)

        # after checkpointing so a pruned trial still leaves its history behind
        if on_epoch is not None:
            on_epoch(ep, record)

    if out_dir is not None and source is not None:
        # keep a top-level final checkpoint for the plotting scripts
        torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__},
                   f"checkpoint_{source}.pt")
    return best_val, best_ep


def cfg_and_hp(config_path, *, base_lr, base_wd):
    """Build (ModelConfig, {lr, weight_decay}) from an optional JSON of overrides.
    Accepts a flat param dict or an HPO {'value':..., 'params': {...}} file;
    architecture keys map onto ModelConfig, lr/weight_decay onto the optimizer,
    and anything absent falls back to the ModelConfig/base defaults."""
    if not config_path:
        return ModelConfig(), {"lr": base_lr, "weight_decay": base_wd}
    from dataclasses import fields
    with open(config_path) as f:
        params = json.load(f)
    params = params.get("params", params)  # unwrap an HPO best_config.json
    cfg_names = {fld.name for fld in fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in params.items() if k in cfg_names})
    hp = {"lr": params.get("lr", base_lr),
          "weight_decay": params.get("weight_decay", base_wd)}
    return cfg, hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="lcbench",
                    choices=["lcbench", "pd1", "fcnet", "taskset",
                             "joint", "joint3", "joint4", "synthetic"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0,
                    help="CUDA device index in PCIe/nvidia-smi order (see CUDA_DEVICE_ORDER)")
    ap.add_argument("--out-dir", default=None,
                    help="run dir for per-epoch checkpoints + history.json (default: runs/<source>)")
    ap.add_argument("--config", default=None,
                    help="JSON with ModelConfig fields (+lr/weight_decay) to override the "
                         "defaults, e.g. an HPO best_config.json. Flat dict or {'params': {...}}.")
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

    train_dl, val_dl = make_loaders(train_ds, val_ds, args.batch_size, device)

    cfg, hp = cfg_and_hp(args.config, base_lr=args.lr, base_wd=1e-4)
    print(f"config: {cfg} | lr={hp['lr']:.3e} weight_decay={hp['weight_decay']:.3e}")
    out_dir = Path(args.out_dir or f"runs/{args.source}")
    print(f"run dir: {out_dir}")
    best_val, best_ep = fit(cfg, hp, device, vocab, train_dl, val_dl, epochs=args.epochs,
                            source=args.source, out_dir=out_dir, meta=vars(args))
    print(f"done. best val pinball {best_val:.4f} @ epoch {best_ep} "
          f"({out_dir}/best_val_loss.pt); final -> checkpoint_{args.source}.pt")


if __name__ == "__main__":
    main()
