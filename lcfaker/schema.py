"""Core data schema: the contract every dataset loader and the model build against.

A single training example is one point on one learning curve:

    (config, task_id, t_abs, t_rel) -> (y_val, y_train)   with y = log(loss)

`config` is a variable-length *set* of parameter tokens. Each token is either a
numeric param (learning_rate, weight_decay, ...) or a categorical param
(optimizer, activation, ...). The global id spaces for param types and
categorical values live in `Vocabulary` (see vocab.py); this module only defines
the flat, tokenized representation those ids get packed into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch


@dataclass
class NumericSpec:
    """A continuous/ordinal hyperparameter.

    `transform` maps the raw value into the space we normalize in ("log" for
    lr/weight_decay-style params spanning orders of magnitude, "linear"
    otherwise). `mean`/`std` are the standardization stats *in transformed
    space*; loaders fill them from data (or from known ranges).
    """

    name: str
    transform: str = "linear"  # "log" | "linear"
    mean: float = 0.0
    std: float = 1.0

    def encode(self, value: float) -> float:
        x = np.log(value) if self.transform == "log" else float(value)
        return (x - self.mean) / (self.std + 1e-12)


@dataclass
class CategoricalSpec:
    """A categorical hyperparameter (optimizer type, activation, schedule, ...)."""

    name: str
    categories: list[str]


ParamSpec = Union[NumericSpec, CategoricalSpec]  # runtime alias -> py3.9-safe (not PEP 604)


@dataclass
class TokenizedConfig:
    """A config encoded to flat per-token arrays (variable length L)."""

    type_ids: np.ndarray     # (L,) int   -> Vocabulary param-type id
    num_vals: np.ndarray     # (L,) float -> standardized numeric value (0 if categorical)
    cat_ids: np.ndarray      # (L,) int   -> global (param, category) id (0 if numeric)
    is_numeric: np.ndarray   # (L,) bool  -> selects value path in the embedder


@dataclass
class Batch:
    """A collated minibatch. CLS and task tokens are prepended inside the model,
    so `pad_mask` here covers only the config tokens."""

    type_ids: torch.Tensor   # (B, L) long
    num_vals: torch.Tensor   # (B, L) float
    cat_ids: torch.Tensor    # (B, L) long
    is_numeric: torch.Tensor # (B, L) bool
    pad_mask: torch.Tensor   # (B, L) bool  (True = padding)
    task_id: torch.Tensor    # (B,)  long
    t_abs: torch.Tensor      # (B,)  float  raw step/epoch
    t_rel: torch.Tensor      # (B,)  float  fraction of task budget in [0, 1]
    y: torch.Tensor          # (B, 2) float  normalized log-loss log(loss/ref) [val, train]

    def to(self, device) -> "Batch":
        return Batch(**{k: v.to(device) for k, v in self.__dict__.items()})

    def __len__(self) -> int:
        return self.type_ids.shape[0]
