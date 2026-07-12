"""LCBench loader.

LCBench (Zimmer et al., Auto-PyTorch): 35 OpenML datasets x 2000 configs x ~50
epochs of funnel-shaped MLPs, SGD + cosine annealing. 7 hyperparameters vary;
everything else is constant across runs, so only the 7 become config tokens.

Targets (log-loss): val = Train/val_cross_entropy, train = Train/train_cross_entropy.
Time axis: the raw `epoch` values (t_abs) and epoch/max_epoch (t_rel).

Data file: data_2k_lw.json (the "lightweight" figshare release), ~930 MB.
    figshare project 74151 -> data_2k_lw.zip
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

from ..config import DATA_ROOT
from ..schema import NumericSpec
from ..vocab import Vocabulary
from .base import IndexedCurveDataset

DEFAULT_PATH = str(DATA_ROOT / "lcbench" / "data_2k_lw.json")

VAL_KEY = "Train/val_cross_entropy"
TRAIN_KEY = "Train/train_cross_entropy"

# the 7 varying hyperparameters and how to normalize their values
PARAM_SPECS = [
    NumericSpec("batch_size", "log"),
    NumericSpec("learning_rate", "log"),
    NumericSpec("momentum", "linear"),
    NumericSpec("weight_decay", "log"),
    NumericSpec("num_layers", "linear"),
    NumericSpec("max_units", "log"),
    NumericSpec("max_dropout", "linear"),
]
PARAM_NAMES = [s.name for s in PARAM_SPECS]


class LCBenchSource:
    """LCBench as a Source for the joint builder (see data/build.py)."""

    name = "lcbench"

    def __init__(self, path: str = DEFAULT_PATH,
                 max_datasets: int | None = None, max_configs: int | None = None):
        self.path = path
        self.max_datasets = max_datasets
        self.max_configs = max_configs

    def param_kinds(self) -> dict[str, str]:
        return {s.name: s.transform for s in PARAM_SPECS}

    def records(self):
        from .build import RawRecord
        with open(self.path) as f:
            data = json.load(f)
        for ds in list(data.keys())[: self.max_datasets or None]:
            for cid in list(data[ds].keys())[: self.max_configs or None]:
                entry = data[ds][cid]
                log = entry["log"]
                yield RawRecord(
                    task_key=f"lcbench/{ds}",
                    config={n: entry["config"][n] for n in PARAM_NAMES},
                    t_abs=np.asarray(log["epoch"], float),
                    y_val=np.asarray(log[VAL_KEY], float),
                    y_train=np.asarray(log[TRAIN_KEY], float),
                )
        del data


def load_lcbench(
    path: str = DEFAULT_PATH,
    max_datasets: int | None = None,
    max_configs: int | None = None,
    cache: bool = True,
):
    """Returns (IndexedCurveDataset, Vocabulary)."""
    cache_path = None
    if cache and max_datasets is None and max_configs is None:
        cache_path = Path(path).with_suffix(".parsed.pkl")
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    t = time.time()
    with open(path) as f:
        data = json.load(f)
    dsets = list(data.keys())[: max_datasets or None]
    print(f"[lcbench] loaded json ({time.time()-t:.1f}s), {len(dsets)} datasets")

    vocab = Vocabulary()

    # pass 1: collect raw param values (for normalization) + task ids
    raw = {n: [] for n in PARAM_NAMES}
    for ds in dsets:
        vocab.register_task(f"lcbench/{ds}")
        cfg_ids = list(data[ds].keys())[: max_configs or None]
        for cid in cfg_ids:
            c = data[ds][cid]["config"]
            for n in PARAM_NAMES:
                raw[n].append(float(c[n]))
    for spec in PARAM_SPECS:
        arr = np.asarray(raw[spec.name], float)
        arr = np.log(arr) if spec.transform == "log" else arr
        spec.mean, spec.std = float(arr.mean()), float(arr.std() + 1e-6)
        vocab.register_param(spec)
    vocab.freeze()

    # pass 2: tokenize configs once + build rows
    cfg_tok = {"type": [], "num": [], "cat": [], "isnum": []}
    config_idx, task_id, t_abs, t_rel, y = [], [], [], [], []
    dropped = 0
    for ds in dsets:
        tid = vocab.task_id[f"lcbench/{ds}"]
        cfg_ids = list(data[ds].keys())[: max_configs or None]
        for cid in cfg_ids:
            entry = data[ds][cid]
            tok = vocab.encode_config({n: entry["config"][n] for n in PARAM_NAMES})
            ci = len(cfg_tok["type"])
            cfg_tok["type"].append(tok.type_ids)
            cfg_tok["num"].append(tok.num_vals)
            cfg_tok["cat"].append(tok.cat_ids)
            cfg_tok["isnum"].append(tok.is_numeric)

            log = entry["log"]
            epochs = np.asarray(log["epoch"], float)
            vloss = np.asarray(log[VAL_KEY], float)
            tloss = np.asarray(log[TRAIN_KEY], float)
            emax = epochs.max() if epochs.size else 1.0
            ok = np.isfinite(vloss) & np.isfinite(tloss) & (vloss > 0) & (tloss > 0)
            dropped += int((~ok).sum())
            for e, yv, yt in zip(epochs[ok], vloss[ok], tloss[ok]):
                config_idx.append(ci)
                task_id.append(tid)
                t_abs.append(e)
                t_rel.append(e / emax if emax > 0 else 0.0)
                y.append((np.log(yv), np.log(yt)))

    ds_obj = IndexedCurveDataset(
        cfg_tok["type"], cfg_tok["num"], cfg_tok["cat"], cfg_tok["isnum"],
        np.asarray(config_idx), np.asarray(task_id),
        np.asarray(t_abs), np.asarray(t_rel), np.asarray(y),
    )
    print(f"[lcbench] {len(ds_obj)} points from {len(cfg_tok['type'])} configs "
          f"({dropped} non-finite/non-positive points dropped)")

    if cache_path is not None:
        with open(cache_path, "wb") as f:
            pickle.dump((ds_obj, vocab), f)
        print(f"[lcbench] cached -> {cache_path}")
    return ds_obj, vocab
