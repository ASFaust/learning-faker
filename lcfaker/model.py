"""The model.

Config encoder is sum-of-embeddings (DeepSets), NOT a transformer: each param
contributes a learned vector -- type_emb (identity/presence) plus either a
value-scaled direction (numeric) or a categorical embedding -- and these are
summed with the task embedding.

    z = task_emb + Σ_p token_vec(p)          # permutation-invariant, variable-length
    z, time_features(t_rel) --readout MLP--> monotone quantiles (val & train)

Rationale: with few, low-interaction HP tokens the task embedding absorbs
constant-within-task context and the downstream time-MLP models any interactions,
so attention is unnecessary. Time never enters the encoder (only t_rel, into the
head). Only within-task-VARYING params are tokenized, plus scalar tokens (which
carry transferable magnitude even when constant-within-task); constant-within-task
categoricals (e.g. optimizer) are dropped -- the task embedding absorbs them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import Batch
from .vocab import Vocabulary


@dataclass
class ModelConfig:
    d_model: int = 32
    hidden_dim: int = 128
    num_freq_bands: int = 7
    dropout: float = 0.5       # aggressive: between the readout's gated layers
    emb_dropout: float = 0.2   # lighter: on z only (time features left intact)
    block: str = "gated"       # readout block type: "gated" (GLU) or "silu" (Linear+SiLU)
    n_layers: int = 2          # readout depth: blocks before the output head
    # residual-quantile levels (denser in the tails, where divergence lives)
    tau_levels: tuple = (0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98)


def _fourier(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """x: (...,) -> (..., 1 + 2*len(freqs))."""
    xf = x.unsqueeze(-1)
    ang = xf * freqs
    return torch.cat([xf, torch.sin(ang), torch.cos(ang)], dim=-1)


class SumEncoder(nn.Module):
    """Config set -> single d-vector by summing per-param contributions.

    Each token = type_emb[type] (identity/presence) + either num_val*num_dir[type]
    (numeric: a learned direction scaled by the normalized value) or cat_emb[cat]
    (categorical). Padding tokens contribute zero. This is DeepSets sum-pooling.
    """

    def __init__(self, vocab: Vocabulary, d: int) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(vocab.n_param_types, d)
        self.num_dir = nn.Embedding(vocab.n_param_types, d)
        self.cat_emb = nn.Embedding(vocab.n_cat_values, d, padding_idx=0)

    def forward(self, b: Batch) -> torch.Tensor:
        te = self.type_emb(b.type_ids)                              # (B, L, d)
        num = self.num_dir(b.type_ids) * b.num_vals.unsqueeze(-1)   # (B, L, d)
        cat = self.cat_emb(b.cat_ids)                               # (B, L, d)
        tok = te + torch.where(b.is_numeric.unsqueeze(-1), num, cat)
        tok = tok.masked_fill(b.pad_mask.unsqueeze(-1), 0.0)        # drop padding
        return tok.sum(dim=1)                                       # (B, d)


class TimeFeatures(nn.Module):
    """t_rel -> Fourier features. Only relative progress enters the model:
    t_abs is dropped (its per-task absolute scale lives in the task embedding),
    and t_rel is already in [0,1] where the Fourier bands are well-matched."""

    def __init__(self, num_bands: int) -> None:
        super().__init__()
        freqs = (2.0 ** torch.arange(num_bands)) * math.pi
        self.register_buffer("freqs", freqs)
        self.dim = 1 + 2 * num_bands  # [t_rel] Fourier-expanded

    def forward(self, t_rel: torch.Tensor) -> torch.Tensor:
        return _fourier(t_rel, self.freqs)


class GatedLayer(nn.Module):
    """h -> value(h) * sigmoid(gate(h)): a GLU-style multiplicative gate.

    The sigmoid branch lets the layer suppress/pass features per-unit (data-
    dependent), giving the head more expressiveness than a plain GELU dense at
    ~2x the layer params -- cheap now that the backbone is a linear sum-pool."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.value = nn.Linear(d_in, d_out)
        self.gate = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x) * torch.sigmoid(self.gate(x))


class DenseLayer(nn.Module):
    """Plain Linear + SiLU: the non-gated readout block. Half the params of a
    GatedLayer (no gate branch) -- the HPO picks between the two."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.lin(x))


_BLOCKS = {"gated": GatedLayer, "silu": DenseLayer}


class Readout(nn.Module):
    """(z, time_features) -> monotone ABSOLUTE quantiles of log(loss/ref) for BOTH
    channels from ONE shared MLP (no per-channel heads). The final layer emits
    n_q*2 logits, reshaped to (B, 2, Q); each channel's Q values are made
    non-decreasing via q0 + cumsum(softplus) (guaranteed no quantile crossing).

    Pinball-only, no separate mean: the point estimate is the median (tau=0.5)
    quantile, which is L1-optimal, robust to the divergence tail, and -- being the
    band center by construction -- can never fall outside its own percentiles the
    way an MSE mean does on skewed predictions. E[y] is recoverable from the
    quantile integral if a risk-neutral objective ever needs it.

    hidden width is decoupled from d_model: the embedding/backbone width and the
    readout width are set independently (hidden_dim).
    """

    def __init__(self, d_in: int, hidden: int, n_q: int,
                 dropout: float = 0.0, emb_dropout: float = 0.0,
                 block: str = "gated", n_layers: int = 2) -> None:
        super().__init__()
        self.n_q = n_q
        # emb_dropout hits z (embedding + config sum-pool) only, forcing the readout
        # off any single embedding dimension; the time features are left intact so
        # the head keeps a clean monotone-in-t signal. The heavier `dropout` between
        # the blocks regularizes the net's only nonlinear depth (the sum-pool
        # backbone is linear). block/n_layers are HPO-tunable; the defaults
        # (gated, 2) reproduce the original two-GatedLayer readout exactly.
        self.emb_drop = nn.Dropout(emb_dropout)
        if block not in _BLOCKS:
            raise ValueError(f"unknown readout block {block!r}; choose from {list(_BLOCKS)}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        Block = _BLOCKS[block]
        layers: list[nn.Module] = []
        d = d_in
        for _ in range(n_layers):
            layers += [Block(d, hidden), nn.Dropout(dropout)]
            d = hidden
        layers.append(nn.Linear(d, n_q * 2))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, tfeat: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(torch.cat([self.emb_drop(z), tfeat], dim=-1))  # (B, 2*Q)
        raw = raw.view(raw.shape[0], 2, self.n_q)              # (B, 2, Q)  [val, train]
        q0 = raw[..., :1]
        deltas = F.softplus(raw[..., 1:])                      # (B, 2, Q-1) > 0
        return torch.cat([q0, q0 + torch.cumsum(deltas, dim=-1)], dim=-1)  # (B, 2, Q)


class Prediction(NamedTuple):
    quantiles: torch.Tensor  # (B, 2, Q)  monotone absolute quantiles of log(loss/ref)
    median: torch.Tensor     # (B, 2)     point estimate = the tau=0.5 quantile [val, train]


class LearningCurveModel(nn.Module):
    def __init__(self, vocab: Vocabulary, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        d = self.cfg.d_model
        self.encoder = SumEncoder(vocab, d)
        self.task_emb = nn.Embedding(vocab.n_tasks, d)
        self.time = TimeFeatures(self.cfg.num_freq_bands)
        n_q = len(self.cfg.tau_levels)
        # one shared MLP emits both channels' absolute quantiles (pinball-only,
        # median-centric); readout width decoupled from the embedding width d.
        self.readout = Readout(d + self.time.dim, self.cfg.hidden_dim, n_q,
                               self.cfg.dropout, self.cfg.emb_dropout,
                               self.cfg.block, self.cfg.n_layers)
        self.register_buffer("taus", torch.tensor(self.cfg.tau_levels, dtype=torch.float32))
        self.median_idx = int(torch.argmin((self.taus - 0.5).abs()))  # tau closest to 0.5

    def encode(self, b: Batch, task_emb: torch.Tensor | None = None) -> torch.Tensor:
        """(config, task) -> z = Σ param_vec + task_emb. Independent of time.
        If task_emb is given (shape (d,) or (B, d)) it overrides the table lookup
        -- used for test-time embedding fitting on a novel task."""
        z_cfg = self.encoder(b)                                # (B, d)
        B = z_cfg.shape[0]
        if task_emb is None:
            te = self.task_emb(b.task_id)                      # (B, d)
        else:
            te = task_emb.expand(B, -1) if task_emb.dim() == 1 else task_emb
        return z_cfg + te

    def forward(self, b: Batch, task_emb: torch.Tensor | None = None) -> Prediction:
        z = self.encode(b, task_emb)
        tf = self.time(b.t_rel)
        # (B, 2, Q) absolute quantiles; the readout shares the encoder gradient (cheap
        # sum-pool, bounded pinball), which also lets pinball inform the fitted task
        # embedding during test-time inversion.
        q = self.readout(z, tf)                                  # (B, 2, Q)  [val, train]
        median = q[:, :, self.median_idx]                       # (B, 2) point estimate
        return Prediction(q, median)

    @torch.no_grad()
    def sample(self, b: Batch) -> torch.Tensor:
        """Draw a log(loss/ref) point by inverting the absolute quantile function at
        u~U(0,1) (piecewise-linear on the tau grid)."""
        pred = self.forward(b)
        u = torch.rand_like(pred.median)                        # (B, 2)
        taus = self.taus                                        # (Q,)
        q = pred.quantiles                                      # (B, 2, Q)
        idx = torch.searchsorted(taus.expand(*q.shape[:2], -1).contiguous(),
                                 u.unsqueeze(-1)).clamp(1, len(taus) - 1)
        t0 = taus[idx - 1]; t1 = taus[idx]
        q0 = q.gather(-1, idx - 1); q1 = q.gather(-1, idx)
        frac = ((u.unsqueeze(-1) - t0) / (t1 - t0 + 1e-12)).clamp(0, 1)
        return (q0 + frac * (q1 - q0)).squeeze(-1)
