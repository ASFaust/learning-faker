"""Diagnose init-loss as the normalization anchor (replacing p25-at-5%).

The RNN failure came from a divergence-CONTAMINATED anchor: at t_rel>=5% most
configs of some tasks have already diverged, so the p25 baseline is itself a
diverged value. The initial loss is divergence-immune (nothing has diverged at
init), so median(init) should be a sane, robust per-task anchor -- IF within-task
init losses are tight enough that the median is representative.

This measures, per task across its runs, the val loss at observation index 0 and
index 1 (index 1 = "1 epoch" for LCBench/FCNet, ~step 1000/200 for PD1/TaskSet):
min / median / max, and the within-task spread max/min. Grouped by family, plus a
per-task breakout of the RNN family that broke.

NOTE: PD1's first logged point is already ~step 1000 (no true step-0), so its
index-0 is NOT a pristine init -- flagged in the table.

    /home/andi/venv/bin/python scripts/diagnose_init_anchor.py
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfaker.data.fcnet import FCNetSource       # noqa: E402
from lcfaker.data.lcbench import LCBenchSource   # noqa: E402
from lcfaker.data.pd1 import PD1Source           # noqa: E402
from lcfaker.data.taskset import TaskSetSource   # noqa: E402


def family_key(task_key: str) -> str:
    src, name = task_key.split("/", 1)
    if src != "taskset":
        return src
    m = re.match(r"(.+?_family)_seed\d+", name)
    if m:
        return "taskset/" + m.group(1)
    return "taskset/" + (re.split(r"\d", name)[0].rstrip("_x") or name)


NO_TRUE_INIT = {"pd1"}   # first logged obs is already ~step 1000

sources = [LCBenchSource(), PD1Source(), FCNetSource(), TaskSetSource()]

# per task: lists of (obs0, obs1) across runs
task_o0: dict[str, list[float]] = defaultdict(list)
task_o1: dict[str, list[float]] = defaultdict(list)
for src in sources:
    for rec in src.records():
        ok = np.where(np.isfinite(rec.y_val) & (rec.y_val > 0))[0]  # time-ordered valid
        if ok.size == 0:
            continue
        o0 = float(rec.y_val[ok[0]])
        o1 = float(rec.y_val[ok[1]]) if ok.size > 1 else o0
        task_o0[rec.task_key].append(o0)
        task_o1[rec.task_key].append(o1)
    print(f"[load] {src.name} done", file=sys.stderr)


def stats(vals):
    a = np.asarray(vals)
    return a.min(), float(np.median(a)), a.max()


# aggregate per family (pool runs across the family's tasks)
fam_o0: dict[str, list[float]] = defaultdict(list)
fam_o1: dict[str, list[float]] = defaultdict(list)
fam_tasks: dict[str, set] = defaultdict(set)
fam_spread0: dict[str, list[float]] = defaultdict(list)  # per-task max/min at idx0
fam_spread1: dict[str, list[float]] = defaultdict(list)
for tk in task_o0:
    fk = family_key(tk)
    fam_o0[fk].extend(task_o0[tk])
    fam_o1[fk].extend(task_o1[tk])
    fam_tasks[fk].add(tk)
    mn0, _, mx0 = stats(task_o0[tk])
    mn1, _, mx1 = stats(task_o1[tk])
    fam_spread0[fk].append(mx0 / mn0 if mn0 > 0 else np.inf)
    fam_spread1[fk].append(mx1 / mn1 if mn1 > 0 else np.inf)

rows = []
for fk in fam_o0:
    rows.append((fk, len(fam_tasks[fk]), *stats(fam_o0[fk]), *stats(fam_o1[fk]),
                 float(np.median(fam_spread0[fk])), float(np.median(fam_spread1[fk]))))
rows.sort(key=lambda r: r[3])  # by median obs0

print(f"\n{'family':<42} {'tsk':>3} | {'o0_min':>8} {'o0_med':>8} {'o0_max':>9} |"
      f" {'o1_min':>8} {'o1_med':>8} {'o1_max':>9} | {'wspr0':>6} {'wspr1':>6}")
print("-" * 118)
for fk, nt, m0, md0, x0, m1, md1, x1, s0, s1 in rows:
    flag = " *no-init" if fk in NO_TRUE_INIT else ""
    print(f"{fk:<42} {nt:>3} | {m0:>8.3g} {md0:>8.3g} {x0:>9.3g} |"
          f" {m1:>8.3g} {md1:>8.3g} {x1:>9.3g} | {s0:>6.1f} {s1:>6.1f}{flag}")

print("\n(wspr0/wspr1 = median over the family's tasks of within-task max/min at idx0/idx1;"
      " ~1 = tight, median is a fair anchor)")

# per-task breakout of the RNN family that broke
print("\n=== rnn_text_classification_family per-task (obs0 | obs1) ===")
print(f"{'task':<46} {'runs':>4} | {'o0_min':>8} {'o0_med':>8} {'o0_max':>9} |"
      f" {'o1_min':>8} {'o1_med':>8} {'o1_max':>9}")
for tk in sorted(task_o0):
    if "rnn_text_classification_family" not in tk:
        continue
    m0, md0, x0 = stats(task_o0[tk])
    m1, md1, x1 = stats(task_o1[tk])
    print(f"{tk:<46} {len(task_o0[tk]):>4} | {m0:>8.3g} {md0:>8.3g} {x0:>9.3g} |"
          f" {m1:>8.3g} {md1:>8.3g} {x1:>9.3g}")
