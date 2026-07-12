"""(1) Confirm which datasets log a true step-0 init; (2) list EVERY task whose
median init loss is a within-family outlier (majority of configs diverge before
the first observation -> even median-init is a diverged value -> no anchor works).

Drop criterion: per-task median(obs0) > OUTLIER_MULT * family-median(median obs0).
Modality-relative, so VAEs (init ~2e4) aren't flagged, only genuine outliers.

    /home/andi/venv/bin/python scripts/find_diverging_tasks.py
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

OUTLIER_MULT = 10.0


def family_key(task_key: str) -> str:
    src, name = task_key.split("/", 1)
    if src != "taskset":
        return src
    m = re.match(r"(.+?_family)_seed\d+", name)
    if m:
        return "taskset/" + m.group(1)
    return "taskset/" + (re.split(r"\d", name)[0].rstrip("_x") or name)


sources = [LCBenchSource(), PD1Source(), FCNetSource(), TaskSetSource()]

print("=== first logged time-step per dataset (does it log a true init?) ===")
task_init: dict[str, list[float]] = defaultdict(list)
first_step = {}
for src in sources:
    for rec in src.records():
        first_step.setdefault(src.name, float(rec.t_abs[0]) if rec.t_abs.size else None)
        ok = np.where(np.isfinite(rec.y_val) & (rec.y_val > 0))[0]
        if ok.size:
            task_init[rec.task_key].append(float(rec.y_val[ok[0]]))
for name, s in first_step.items():
    print(f"  {name:<10} first t_abs = {s}")

# per-task median init, grouped by family
task_med = {tk: float(np.median(v)) for tk, v in task_init.items()}
fam_tasks: dict[str, list[str]] = defaultdict(list)
for tk in task_med:
    fam_tasks[family_key(tk)].append(tk)

print(f"\n=== majority-diverging tasks (median-init > {OUTLIER_MULT}x family norm) ===")
drop = []
for fk, tks in sorted(fam_tasks.items()):
    fam_norm = float(np.median([task_med[t] for t in tks]))
    for t in tks:
        if task_med[t] > OUTLIER_MULT * fam_norm:
            drop.append((t, task_med[t], fam_norm))
drop.sort(key=lambda r: -r[1] / r[2])
for t, m, norm in drop:
    print(f"  {t:<48} median-init={m:>10.3g}  (family norm {norm:.3g}, {m/norm:.0f}x)")
print(f"\n{len(drop)} tasks to drop:")
for t, _, _ in drop:
    print(f"    {t.split('/')[-1]}")
