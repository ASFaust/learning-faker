"""Pick the loss cap: cap = C * ref_task on observed loss (so diverged runs stop
dominating the MSE mean head). C must sit ABOVE the legitimate early-training
peak (a good config's loss at t~0 exceeds its 5%-baseline, the first 5% being the
steepest part of the curve) but BELOW divergence. This measures both so we can
choose C without clipping real curve shape.

Materializes records once, computes p25 refs (tau=0.05, quadratics already
excluded by the source), then reports:
  * where legit EARLY points (t_rel < tau) land in units of baseline
  * the overall loss/ref tail and how many points each candidate C would clip
  * whether clipped points are early (legit steep descent) or spread (divergence)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lcfaker.data.fcnet import FCNetSource       # noqa: E402
from lcfaker.data.lcbench import LCBenchSource   # noqa: E402
from lcfaker.data.pd1 import PD1Source           # noqa: E402
from lcfaker.data.taskset import TaskSetSource   # noqa: E402

TAU = 0.05
sources = [LCBenchSource(), PD1Source(), FCNetSource(), TaskSetSource()]

records = []
budget: dict[str, float] = defaultdict(float)
for src in sources:
    for rec in src.records():
        records.append(rec)
        if rec.t_abs.size:
            budget[rec.task_key] = max(budget[rec.task_key], float(rec.t_abs.max()))
    print(f"[load] {src.name} done ({len(records)} recs)", file=sys.stderr)

# p25 ref per task (same rule as build_joint)
tau_samples: dict[str, list[float]] = defaultdict(list)
for rec in records:
    ok = np.isfinite(rec.y_val) & (rec.y_val > 0)
    if not ok.any():
        continue
    trel = rec.t_abs / (budget[rec.task_key] or 1.0)
    cand = np.where(ok & (trel >= TAU))[0]
    if cand.size:
        tau_samples[rec.task_key].append(float(rec.y_val[cand[np.argmin(trel[cand])]]))
task_ref = {k: float(np.percentile(v, 25)) for k, v in tau_samples.items()}
refs = np.array(list(task_ref.values()))
print(f"\n[refs] {len(refs)} tasks | p25-ref range [{refs.min():.4g}, {refs.max():.4g}] "
      f"median {np.median(refs):.4g}")

# gather loss/ref ratios for all val points, split early vs later
ratio_all, ratio_early, trel_all = [], [], []
for rec in records:
    if rec.task_key not in task_ref:
        continue
    ref = task_ref[rec.task_key]
    ok = np.isfinite(rec.y_val) & (rec.y_val > 0)
    trel = rec.t_abs[ok] / (budget[rec.task_key] or 1.0)
    r = rec.y_val[ok] / ref
    ratio_all.append(r); trel_all.append(trel)
    ratio_early.append(r[trel < TAU])
ratio_all = np.concatenate(ratio_all)
trel_all = np.concatenate(trel_all)
ratio_early = np.concatenate(ratio_early)

def pct(a, ps): return {p: float(np.percentile(a, p)) for p in ps}

print(f"\n[legit early points] loss/ref for t_rel < {TAU} (n={ratio_early.size}):")
for p, v in pct(ratio_early, [50, 90, 99, 99.9]).items():
    print(f"   p{p:<5} = {v:8.3g}x baseline")

print(f"\n[all points] loss/ref percentiles (n={ratio_all.size}):")
for p, v in pct(ratio_all, [50, 90, 99, 99.9, 100]).items():
    print(f"   p{p:<5} = {v:10.4g}x baseline")

print("\n[candidate caps] fraction clipped, and how much of that is EARLY (legit):")
print(f"   {'C (xbaseline)':>13} {'log(C)':>7} {'%clipped':>9} {'%clipped_early':>15}")
for C in (3, 5, 8, 10, 20, 50):
    clip = ratio_all > C
    n = clip.sum()
    early_frac = float((trel_all[clip] < TAU).mean()) * 100 if n else 0.0
    print(f"   {C:>13} {np.log(C):>7.2f} {100*n/ratio_all.size:>8.3f}% {early_frac:>14.1f}%")
