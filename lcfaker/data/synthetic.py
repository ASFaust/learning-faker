"""Synthetic learning-curve generator matching the exact schema.

Lets us exercise the whole pipeline end-to-end before the multi-GB real
datasets are downloaded. Deliberately built to stress the design:
  - numeric (log + linear) AND categorical params in one config set,
  - conditional / one-off params (adam has beta2; sgd has a nesterov flag) so
    configs are genuinely variable-length,
  - per-task loss scales spanning orders of magnitude (few- vs many-class CE),
  - per-task budgets T that differ, so t_rel is non-trivial.

Curves are structured (exp decay toward a config/task-dependent asymptote with
overfitting gap + log-noise), so there is real signal for the model to fit.
"""

from __future__ import annotations

import numpy as np

from ..schema import CategoricalSpec, NumericSpec
from ..vocab import Vocabulary
from .base import FlatCurveDataset


def _sample_config(rng: np.random.Generator) -> dict:
    opt = rng.choice(["adam", "sgd"])
    cfg = {
        "learning_rate": float(10 ** rng.uniform(-4, -0.5)),
        "weight_decay": float(10 ** rng.uniform(-6, -2)),
        "n_layers": int(rng.integers(1, 6)),
        "optimizer": opt,
        "activation": str(rng.choice(["relu", "tanh"])),
    }
    if opt == "adam":
        cfg["adam_beta2"] = float(rng.uniform(0.9, 0.999))     # one-off (adam only)
    else:
        cfg["momentum"] = float(rng.uniform(0.0, 0.99))        # sgd only
        cfg["nesterov"] = str(bool(rng.integers(0, 2)))        # one-off categorical
    return cfg


def _curve(cfg: dict, task: dict, epochs: np.ndarray, rng: np.random.Generator):
    lr = cfg["learning_rate"]
    # convergence rate: best near lr~1e-2, adam faster, sgd helped by momentum
    lr_factor = np.exp(-0.5 * ((np.log10(lr) + 2.0) / 0.7) ** 2)
    opt_factor = 1.3 if cfg["optimizer"] == "adam" else 1.0 + 0.4 * cfg.get("momentum", 0.0)
    rate = task["base_rate"] * lr_factor * opt_factor + 1e-3
    # asymptote: task floor, hurt by too-large lr (near-divergence) and bad wd
    divergence = np.clip((np.log10(lr) + 0.5), 0, None) * 1.5
    wd_pen = 0.3 * abs(np.log10(cfg["weight_decay"]) + 4) / 4
    asymp = task["asymp"] * (1.0 + divergence + wd_pen)
    init = task["init"]
    frac = epochs / task["budget"]
    val = asymp + (init - asymp) * np.exp(-rate * epochs)
    gap = 0.15 * asymp * frac * (1.0 / cfg["n_layers"] ** 0.5 + 0.5)  # overfitting
    train = np.clip(val - gap, 1e-3, None)
    noise = lambda: np.exp(rng.normal(0, 0.05, size=epochs.shape))
    val = np.clip(val, 1e-3, None) * noise()
    train = train * noise()
    return val, train


def make_synthetic(n_tasks: int = 24, n_configs: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    vocab = Vocabulary()

    # task properties: varied loss scales + budgets
    tasks = []
    for k in range(n_tasks):
        n_classes = int(rng.choice([2, 5, 10, 100, 1000]))
        tasks.append({
            "key": f"synthetic/task_{k:03d}",
            "init": float(np.log(n_classes) * rng.uniform(0.8, 1.2)),
            "asymp": float(np.log(n_classes) * rng.uniform(0.02, 0.15)),
            "base_rate": float(rng.uniform(0.05, 0.25)),
            "budget": int(rng.choice([20, 30, 50])),
        })

    # sample everything first (need it to compute numeric normalization stats)
    rows = []  # (task_idx, cfg, epoch, t_rel, y_val, y_train)
    raw_numeric: dict[str, list[float]] = {}
    for ti, task in enumerate(tasks):
        epochs = np.arange(1, task["budget"] + 1, dtype=np.float64)
        for _ in range(n_configs):
            cfg = _sample_config(rng)
            val, train = _curve(cfg, task, epochs, rng)
            for name, v in cfg.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    raw_numeric.setdefault(name, []).append(float(v))
            for e, tr, yv, yt in zip(epochs, epochs / task["budget"], val, train):
                rows.append((ti, cfg, float(e), float(tr), float(yv), float(yt)))

    # param specs (with normalization stats from the sampled data)
    log_params = {"learning_rate", "weight_decay"}
    numeric_names = set(raw_numeric)
    for name in numeric_names:
        transform = "log" if name in log_params else "linear"
        arr = np.log(raw_numeric[name]) if transform == "log" else np.asarray(raw_numeric[name])
        vocab.register_param(NumericSpec(name, transform, float(arr.mean()), float(arr.std() + 1e-6)))
    vocab.register_param(CategoricalSpec("optimizer", ["adam", "sgd"]))
    vocab.register_param(CategoricalSpec("activation", ["relu", "tanh"]))
    vocab.register_param(CategoricalSpec("nesterov", ["True", "False"]))
    for task in tasks:
        vocab.register_task(task["key"])
    vocab.freeze()

    # encode into flat arrays (targets are log-loss)
    type_ids, num_vals, cat_ids, is_num = [], [], [], []
    task_id, t_abs, t_rel, y = [], [], [], []
    for ti, cfg, e, tr, yv, yt in rows:
        tok = vocab.encode_config(cfg)
        type_ids.append(tok.type_ids)
        num_vals.append(tok.num_vals)
        cat_ids.append(tok.cat_ids)
        is_num.append(tok.is_numeric)
        task_id.append(vocab.task_id[tasks[ti]["key"]])
        t_abs.append(e)
        t_rel.append(tr)
        y.append([np.log(yv), np.log(yt)])

    ds = FlatCurveDataset(
        type_ids, num_vals, cat_ids, is_num,
        np.asarray(task_id), np.asarray(t_abs), np.asarray(t_rel), np.asarray(y),
    )
    return ds, vocab
