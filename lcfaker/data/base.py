"""Dataset base + collate.

`FlatCurveDataset` stores one row per curve *point* (config, task, t). This is
the simplest correct thing: it recomputes the config latent z per point, which
is a little wasteful but keeps the pipeline trivial. Because the model puts the
transformer *before* the time coordinate (z = f(config, task); output =
head(z, t)), grouping many t's under a shared z is a pure efficiency
optimization we can add later without touching the schema or the model.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..schema import Batch


class FlatCurveDataset(Dataset):
    def __init__(
        self,
        type_ids: list[np.ndarray],
        num_vals: list[np.ndarray],
        cat_ids: list[np.ndarray],
        is_numeric: list[np.ndarray],
        task_id: np.ndarray,
        t_abs: np.ndarray,
        t_rel: np.ndarray,
        y: np.ndarray,          # (N, 2) normalized log-loss [val, train]
    ) -> None:
        self.type_ids = type_ids
        self.num_vals = num_vals
        self.cat_ids = cat_ids
        self.is_numeric = is_numeric
        self.task_id = task_id.astype(np.int64)
        self.t_abs = t_abs.astype(np.float32)
        self.t_rel = t_rel.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.task_id)

    def __getitem__(self, i: int):
        return (
            self.type_ids[i], self.num_vals[i], self.cat_ids[i], self.is_numeric[i],
            self.task_id[i], self.t_abs[i], self.t_rel[i], self.y[i],
        )


class IndexedCurveDataset(Dataset):
    """Like FlatCurveDataset but stores each *config*'s tokens once and points to
    them per row via `config_idx`. Avoids duplicating the config tokenization
    across the ~50 epochs of every curve (a ~50x token-memory saving on real
    datasets). Yields the same tuple as FlatCurveDataset, so `collate` is shared.
    """

    def __init__(
        self,
        cfg_type_ids: list[np.ndarray],
        cfg_num_vals: list[np.ndarray],
        cfg_cat_ids: list[np.ndarray],
        cfg_is_numeric: list[np.ndarray],
        config_idx: np.ndarray,
        task_id: np.ndarray,
        t_abs: np.ndarray,
        t_rel: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.cfg_type_ids = cfg_type_ids
        self.cfg_num_vals = cfg_num_vals
        self.cfg_cat_ids = cfg_cat_ids
        self.cfg_is_numeric = cfg_is_numeric
        self.config_idx = config_idx.astype(np.int64)
        self.task_id = task_id.astype(np.int64)
        self.t_abs = t_abs.astype(np.float32)
        self.t_rel = t_rel.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.task_id)

    def __getitem__(self, i: int):
        c = self.config_idx[i]
        return (
            self.cfg_type_ids[c], self.cfg_num_vals[c], self.cfg_cat_ids[c],
            self.cfg_is_numeric[c], self.task_id[i], self.t_abs[i], self.t_rel[i],
            self.y[i],
        )


def collate(items) -> Batch:
    B = len(items)
    L = max(len(it[0]) for it in items)
    type_ids = np.zeros((B, L), np.int64)
    num_vals = np.zeros((B, L), np.float32)
    cat_ids = np.zeros((B, L), np.int64)
    is_num = np.zeros((B, L), bool)
    pad = np.ones((B, L), bool)  # True = padding
    task_id = np.empty(B, np.int64)
    t_abs = np.empty(B, np.float32)
    t_rel = np.empty(B, np.float32)
    y = np.empty((B, 2), np.float32)
    for b, it in enumerate(items):
        n = len(it[0])
        type_ids[b, :n] = it[0]
        num_vals[b, :n] = it[1]
        cat_ids[b, :n] = it[2]
        is_num[b, :n] = it[3]
        pad[b, :n] = False
        task_id[b], t_abs[b], t_rel[b], y[b] = it[4], it[5], it[6], it[7]
    return Batch(
        type_ids=torch.from_numpy(type_ids),
        num_vals=torch.from_numpy(num_vals),
        cat_ids=torch.from_numpy(cat_ids),
        is_numeric=torch.from_numpy(is_num),
        pad_mask=torch.from_numpy(pad),
        task_id=torch.from_numpy(task_id),
        t_abs=torch.from_numpy(t_abs),
        t_rel=torch.from_numpy(t_rel),
        y=torch.from_numpy(y),
    )
