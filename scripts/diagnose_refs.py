"""Diagnose per-task loss references.

build_joint divides every task by ref = median over its runs of the val loss at
the first observation with t_rel >= tau_ref ("val loss after ~5% of training").
A huge ref (we saw 2.86e8) means we're normalizing by a *diverged* value. This
tells us whether that's:

  * a few runs diverged     -> min@5% small, median(ref) huge   (config-sensitive)
  * baseline genuinely huge -> min@5% ALSO huge                 (weird / broken task)

For each task we report min / median(ref) / max of the at-tau val loss across its
runs, sorted by the ref. Faithful to build: same tau_ref, same per-task budget
(max t_abs over the task's runs) for t_rel.
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

# pass 1: per-task budget (max t_abs), streamed so we never hold curves
budget: dict[str, float] = defaultdict(float)
for src in sources:
    for rec in src.records():
        if rec.t_abs.size:
            budget[rec.task_key] = max(budget[rec.task_key], float(rec.t_abs.max()))
    print(f"[pass1] {src.name} done", file=sys.stderr)

# pass 2: per-task list of the at-tau val loss (one scalar per run)
at: dict[str, list[float]] = defaultdict(list)
for src in sources:
    for rec in src.records():
        ok = np.isfinite(rec.y_val) & (rec.y_val > 0)
        if not ok.any():
            continue
        trel = rec.t_abs / (budget[rec.task_key] or 1.0)
        cand = np.where(ok & (trel >= TAU))[0]
        if cand.size:
            at[rec.task_key].append(float(rec.y_val[cand[np.argmin(trel[cand])]]))
    print(f"[pass2] {src.name} done", file=sys.stderr)

rows = []
for k, vals in at.items():
    a = np.asarray(vals)
    rows.append((k, len(a), float(a.min()), float(np.median(a)), float(a.max())))
rows.sort(key=lambda r: r[3], reverse=True)  # by median == the ref build uses


def show(rs, title):
    print(title)
    print(f"{'task':<46} {'n':>5} {'min@5%':>12} {'median(ref)':>13} "
          f"{'max@5%':>12} {'max/min':>9}")
    for k, n, mn, md, mx in rs:
        ratio = mx / mn if mn > 0 else float("inf")
        print(f"{k:<46} {n:>5} {mn:>12.4g} {md:>13.4g} {mx:>12.4g} {ratio:>9.2g}")


show(rows[:30], "\n=== 30 HIGHEST-ref tasks (the divisors that hurt) ===")
show(rows[-10:], "\n=== 10 LOWEST-ref tasks ===")

# classify the offenders: high ref driven by divergence vs by a high baseline
HIGH = 100.0  # "early val loss this large is not a sane baseline"
high_ref = [r for r in rows if r[3] > HIGH]
baseline_weird = [r for r in high_ref if r[2] > HIGH]   # even the BEST run is huge at 5%
run_diverged = [r for r in high_ref if r[2] <= HIGH]    # best run fine, median dragged up
print(f"\n=== summary (ref > {HIGH:g}) ===")
print(f"tasks with ref > {HIGH:g}: {len(high_ref)} / {len(rows)}")
print(f"  baseline-weird (min@5% also > {HIGH:g}): {len(baseline_weird)}")
print(f"  run-diverged   (min@5% <= {HIGH:g}):     {len(run_diverged)}")
if baseline_weird:
    print("  baseline-weird tasks:", [r[0] for r in baseline_weird])
