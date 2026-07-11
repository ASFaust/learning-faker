"""Multi-source joint dataset builder.

A `Source` (one HPO dataset) declares its param kinds and yields raw per-config
curve records. `build_joint` merges any set of sources into ONE shared
`Vocabulary` (global param-type ids, categorical value ids, task ids) and one
`IndexedCurveDataset`, with numeric normalization stats pooled across sources.

This is what makes cross-dataset transfer real: a param shared by two datasets
(e.g. learning_rate) gets a single type embedding trained on both; a one-off
param gets its own id and is simply absent from other configs.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from ..schema import CategoricalSpec, NumericSpec
from ..vocab import Vocabulary
from .base import IndexedCurveDataset


@dataclass
class RawRecord:
    task_key: str            # global task id, e.g. "lcbench/APSFailure" or "pd1/cifar10_wide_resnet_bs256"
    config: dict             # {param_name: raw value}
    t_abs: np.ndarray        # raw step/epoch
    y_val: np.ndarray        # raw val loss (>0); builder takes log
    y_train: np.ndarray      # raw train loss (>0)
    # t_rel is NOT supplied by the source: build_joint computes it per TASK
    # (t_abs / task_budget), so a run shorter than the task's longest run tops
    # out below 1.0 -- preserving the "stopped/diverged early" signal.


class Source(Protocol):
    name: str
    def param_kinds(self) -> dict[str, str]: ...   # name -> "log" | "linear" | "categorical"
    def records(self) -> Iterator[RawRecord]: ...   # consumed once; frees its own raw data after


def build_joint(sources: list[Source], cache_path: str | None = None,
                loss_cap_mult: float | None = 20.0, tau_ref: float = 0.05,
                ref_pct: float = 25.0):
    """Returns (IndexedCurveDataset, Vocabulary).

    Targets are PER-TASK-NORMALIZED log-loss: y = log(loss / ref_task), where
    ref_task is the ref_pct-th percentile (p25) over the task's runs of the val
    loss at the first observation with t_rel >= tau_ref (i.e. "val loss after
    ~tau_ref of the budget's worth of initial training" -- past the init spike,
    uniform across epoch- and step-timed datasets). We take p25 rather than the
    median because on tasks where most sampled configs diverge early the median
    is itself a diverged value; p25 tracks a SENSIBLE config's baseline, matching
    the test-time protocol (train a bit with a sensible config -> read val loss
    -> divide by it), and collapses to ~median on well-behaved tight-baseline
    tasks (e.g. VAEs at ~1e4 nats). The SAME ref divides both channels, so the
    train/val gap is preserved, and it only re-centers each task to a common
    ballpark (log ref -> 0 at the reference point) without whitening the dynamic
    range. ref_task is stored on the dataset & vocab so seen-task generation can
    multiply back and test-time embedding inversion can replay the protocol.

    loss_cap_mult caps the OBSERVED loss at loss_cap_mult * ref_task (i.e. y at
    log(loss_cap_mult)) on the diverged (upper) side, so diverged runs -- which
    reach 1e33x baseline -- don't dominate the MSE mean head. At 20x the cap is
    y=log(20)=3.0, still SMALLER than the (unbounded, ~-3..-7) improving side, so
    diverged points can't dominate; and 20x preserves the "bad but stable" 5-20x
    regime as distinct curves (it matters for the simulator use case, not just
    ranking) while still crushing true divergence (>20x baseline, ~4% of points)
    to the ceiling. The low side is left unbounded (a run improving 1000x is real
    signal, and it's the region HPO actually needs resolved).
    """
    if cache_path and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # merge param kinds (shared names must agree on kind)
    kinds: dict[str, str] = {}
    for src in sources:
        for name, kind in src.param_kinds().items():
            if name in kinds and kinds[name] != kind:
                raise ValueError(f"param '{name}': {kinds[name]} vs {kind} across sources")
            kinds[name] = kind

    vocab = Vocabulary()
    records: list[RawRecord] = []
    raw_numeric: dict[str, list[float]] = defaultdict(list)
    categories: dict[str, set] = defaultdict(set)
    task_budget: dict[str, float] = defaultdict(float)  # per-task max t_abs

    # pass 1: materialize records (frees each source's raw data), gather stats
    for src in sources:
        n0 = len(records)
        for rec in src.records():
            records.append(rec)
            vocab.register_task(rec.task_key)
            if rec.t_abs.size:
                task_budget[rec.task_key] = max(task_budget[rec.task_key], float(rec.t_abs.max()))
            for name, val in rec.config.items():
                if kinds[name] == "categorical":
                    categories[name].add(str(val))
                else:
                    raw_numeric[name].append(float(val))
        print(f"[build] {src.name}: {len(records) - n0} configs")

    # ref pass: per-task normalizer = p{ref_pct} over runs of the first val loss
    # at t_rel >= tau_ref (past the init spike; p25 tracks a sensible config, not
    # the diverged tail). Fallback to the first finite val loss for tasks whose
    # every run is too short to reach tau_ref.
    tau_samples: dict[str, list[float]] = defaultdict(list)
    first_samples: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        ok = np.isfinite(rec.y_val) & (rec.y_val > 0)
        if not ok.any():
            continue
        budget = task_budget[rec.task_key] or 1.0
        trel = rec.t_abs / budget
        okidx = np.where(ok)[0]
        first_samples[rec.task_key].append(float(rec.y_val[okidx[np.argmin(trel[okidx])]]))
        cand = np.where(ok & (trel >= tau_ref))[0]
        if cand.size:
            tau_samples[rec.task_key].append(float(rec.y_val[cand[np.argmin(trel[cand])]]))
    task_ref: dict[str, float] = {}
    n_fallback = 0
    for task_key in vocab.task_id:
        if tau_samples.get(task_key):
            task_ref[task_key] = float(np.percentile(tau_samples[task_key], ref_pct))
        elif first_samples.get(task_key):
            task_ref[task_key] = float(np.percentile(first_samples[task_key], ref_pct))
            n_fallback += 1
        else:
            task_ref[task_key] = 1.0  # no finite val loss at all; no-op divisor
            n_fallback += 1
    ref_arr = np.array([task_ref[k] for k, _ in
                        sorted(vocab.task_id.items(), key=lambda kv: kv[1])], float)
    print(f"[build] task refs: median {np.median(ref_arr):.3g}, "
          f"range [{ref_arr.min():.3g}, {ref_arr.max():.3g}] "
          f"({n_fallback}/{len(ref_arr)} via fallback)")

    # build + register specs (numeric stats / categories pooled across sources)
    for name, kind in kinds.items():
        if kind == "categorical":
            vocab.register_param(CategoricalSpec(name, sorted(categories[name])))
        else:
            arr = np.asarray(raw_numeric[name], float)
            arr = np.log(arr) if kind == "log" else arr
            vocab.register_param(NumericSpec(name, kind, float(arr.mean()), float(arr.std() + 1e-6)))
    vocab.freeze()

    # pass 2: encode configs (one config index per record) + build point arrays
    ct, cn, cc, ci_num = [], [], [], []
    config_idx, task_id, t_abs, t_rel, y = [], [], [], [], []
    dropped = 0
    for rec in records:
        tok = vocab.encode_config(rec.config)
        cid = len(ct)
        ct.append(tok.type_ids); cn.append(tok.num_vals)
        cc.append(tok.cat_ids); ci_num.append(tok.is_numeric)
        tid = vocab.task_id[rec.task_key]
        budget = task_budget[rec.task_key] or 1.0     # per-task time normalizer
        trel = rec.t_abs / budget                     # < 1.0 for short/diverged runs
        log_ref = np.log(task_ref[rec.task_key])      # per-task loss normalizer
        ok = (np.isfinite(rec.y_val) & np.isfinite(rec.y_train)
              & (rec.y_val > 0) & (rec.y_train > 0))
        dropped += int((~ok).sum())
        for e, r, yv, yt in zip(rec.t_abs[ok], trel[ok], rec.y_val[ok], rec.y_train[ok]):
            config_idx.append(cid); task_id.append(tid)
            t_abs.append(e); t_rel.append(r)
            y.append((np.log(yv) - log_ref, np.log(yt) - log_ref))  # log(loss / ref)
    y = np.asarray(y)
    if loss_cap_mult is not None:
        y_cap = float(np.log(loss_cap_mult))
        n_capped = int((y > y_cap).sum())
        y = np.minimum(y, y_cap)                        # cap the diverged (upper) side only
        print(f"[build] capped {n_capped} ({100*n_capped/y.size:.2f}%) targets at "
              f"{loss_cap_mult:g}x baseline (log={y_cap:.3f})")

    ds = IndexedCurveDataset(
        ct, cn, cc, ci_num,
        np.asarray(config_idx), np.asarray(task_id),
        np.asarray(t_abs), np.asarray(t_rel), np.asarray(y),
    )
    # per-task loss normalizer (indexed by task_id): targets are log(loss/ref),
    # so consumers add log(ref) back to recover absolute log-loss.
    ds.task_ref = ref_arr
    vocab.task_ref = ref_arr
    print(f"[build] {len(ds)} points, {len(ct)} configs, {vocab.n_tasks} tasks, "
          f"{vocab.n_param_types} param types ({dropped} bad points dropped)")

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump((ds, vocab), f)
        print(f"[build] cached -> {cache_path}")
    return ds, vocab
