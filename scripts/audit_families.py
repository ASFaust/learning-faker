"""Audit task families for synthetic-toy behavior, so we can drop them.

A synthetic optimization toy (quadratic bowls, losg surfaces, ...) converges to
near machine-precision, so its val loss plunges FAR below its p25 baseline --
log(loss/ref) reaching -20 or worse. Real NN tasks bottom out much shallower
(fcnet/naval's regression MSE, the deepest real case, only reaches ~-7.8). This
groups every task by family and ranks by convergence depth (per-run minimum of
log(val_loss/ref)), with lcbench/pd1/fcnet shown as single reference families.

Discriminator: median per-run depth < DEEP_FLAG (or min abs val loss < ABS_FLAG)
=> synthetic toy, drop it.

    /home/andi/venv/bin/python scripts/audit_families.py
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

TAU = 0.05
DEEP_FLAG = -12.0     # per-run depth below this = suspiciously deep convergence
ABS_FLAG = 1e-6       # absolute val loss below this = past any real NN floor


def family_key(task_key: str) -> str:
    src, name = task_key.split("/", 1)
    if src != "taskset":
        return src                                    # each other dataset = one family
    m = re.match(r"(.+?_family)_seed\d+", name)
    if m:
        return "taskset/" + m.group(1)
    base = re.split(r"\d", name)[0].rstrip("_x") or name  # cut named tasks at first digit
    return "taskset/" + base


# note: TaskSetSource already excludes quadratic_family + TwoD_Bowl; pass a fresh
# source that includes everything so the audit sees the full picture.
sources = [LCBenchSource(), PD1Source(), FCNetSource(),
           TaskSetSource()]

records = []
budget: dict[str, float] = defaultdict(float)
for src in sources:
    for rec in src.records():
        records.append(rec)
        if rec.t_abs.size:
            budget[rec.task_key] = max(budget[rec.task_key], float(rec.t_abs.max()))
    print(f"[load] {src.name} done", file=sys.stderr)

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

fam_depth: dict[str, list[float]] = defaultdict(list)   # per-run min log(val/ref)
fam_absmin: dict[str, list[float]] = defaultdict(list)  # per-run min abs val loss
fam_tasks: dict[str, set] = defaultdict(set)
fam_ref: dict[str, list[float]] = defaultdict(list)
for rec in records:
    if rec.task_key not in task_ref:
        continue
    ok = np.isfinite(rec.y_val) & (rec.y_val > 0)
    if not ok.any():
        continue
    r = rec.y_val[ok]
    fk = family_key(rec.task_key)
    fam_depth[fk].append(float(np.log(r.min()) - np.log(task_ref[rec.task_key])))
    fam_absmin[fk].append(float(r.min()))
    fam_tasks[fk].add(rec.task_key)
    fam_ref[fk].append(task_ref[rec.task_key])

rows = []
for fk in fam_depth:
    depth = np.asarray(fam_depth[fk])
    rows.append((fk, len(fam_tasks[fk]), len(depth),
                 float(np.median(fam_ref[fk])),
                 float(np.median(depth)), float(depth.min()),
                 float(min(fam_absmin[fk]))))
rows.sort(key=lambda r: r[4])   # by median per-run depth (deepest first)

print(f"\n{'family':<42} {'tasks':>5} {'runs':>6} {'med_ref':>9} "
      f"{'med_depth':>9} {'min_depth':>9} {'abs_min':>10}  flag")
for fk, nt, nr, mref, mdep, mindep, absmin in rows:
    syn = mdep < DEEP_FLAG or absmin < ABS_FLAG
    flag = "  <-- SYNTHETIC?" if syn and fk.startswith("taskset/") else ""
    print(f"{fk:<42} {nt:>5} {nr:>6} {mref:>9.3g} "
          f"{mdep:>9.2f} {mindep:>9.2f} {absmin:>10.2e}{flag}")

drop = [fk for fk, nt, nr, mref, mdep, mindep, absmin in rows
        if (mdep < DEEP_FLAG or absmin < ABS_FLAG) and fk.startswith("taskset/")]
print(f"\nSUGGESTED DROP ({len(drop)} families): "
      f"{[fk.split('/')[-1] for fk in drop]}")
