"""Download + convert TaskSet (GCS) to a local pure-numpy format, reproducibly.

TaskSet's files store only config SEEDS, not hyperparameters; the HPs are recovered
from the seeds via the verbatim adam8p sampler (see data/taskset_samplers.py). For
each task this fetches the adam8p_wide_grid file from GCS over plain HTTP (no gsutil,
no TensorFlow), averages the 5 eval replicas, keeps train(ch0)+valid(ch1) curves,
subsamples configs, and writes one compact .npz per task under
LCFAKER_DATA_ROOT/taskset_local/.

Selection is DETERMINISTIC (tasks sorted, first --n-tasks) so runs are reproducible
and NESTED: --n-tasks 300 is a subset of --n-tasks 800 is a subset of all. Existing
.npz are skipped, so extending the set later is just a larger --n-tasks re-run (no
re-download of what you have). Config subsampling is seeded per-task (crc32 of the
task name), so it's independent of how many tasks you fetch. Bad/synthetic tasks are
filtered at LOAD time (data/taskset.py), not here -- this just mirrors the raw data.

    python scripts/convert_taskset.py                 # all available tasks
    python scripts/convert_taskset.py --n-tasks 300   # first 300 (reproducible)
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcfaker.config import DATA_ROOT                                      # noqa: E402
from lcfaker.data.taskset_samplers import ADAM8P_PARAMS, sample_adam8p_wide_grid  # noqa: E402

GCS = "https://storage.googleapis.com/gresearch/task_set_data"
LIST = "https://www.googleapis.com/storage/v1/b/gresearch/o"
FILE = "adam8p_wide_grid_1k_10000_replica5.npz"


def list_tasks() -> list[str]:
    """All TaskSet task names (sorted), paginating the GCS JSON listing."""
    tasks, token = [], None
    while True:
        params = {"prefix": "task_set_data/", "delimiter": "/", "maxResults": 5000}
        if token:
            params["pageToken"] = token
        r = requests.get(LIST, params=params, timeout=60).json()
        tasks += [p.split("/")[-2] for p in r.get("prefixes", [])]
        token = r.get("nextPageToken")
        if not token:
            break
    return sorted(tasks)


def convert_taskset(out_dir, n_tasks: int | None = None, n_configs: int = 150,
                    verbose: bool = True) -> int:
    """Fetch + convert the first n_tasks (sorted; None = all). Returns #saved."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = list_tasks()
    if n_tasks is not None:
        tasks = tasks[:n_tasks]
    if verbose:
        print(f"[taskset] {len(tasks)} tasks selected (sorted, deterministic)")

    saved = skipped = failed = 0
    for n, task in enumerate(tasks):
        out_path = out_dir / f"{task}.npz"
        if out_path.exists():
            skipped += 1
            continue
        try:
            resp = requests.get(f"{GCS}/{task}/{FILE}", timeout=180)
            if resp.status_code != 200:                 # task lacks an adam8p grid
                failed += 1
                continue
            d = np.load(io.BytesIO(resp.content), allow_pickle=True)
            ys, xs, opt = d["ys"], d["xs"], d["optimizers"]   # (1000,5,51,4),(51,),(1000,)
            m = np.nanmean(ys, axis=1)                        # avg 5 replicas -> (1000,51,4)
            seeds = np.array([int(re.search(rb"seed(\d+)", o).group(1)) for o in opt])
            usable = np.nonzero(np.isfinite(m[:, :, 1]).any(axis=1))[0]
            # deterministic per-task config subsample (independent of task order)
            rng = np.random.default_rng(zlib.crc32(task.encode()))
            keep = np.sort(rng.choice(usable, min(n_configs, len(usable)), replace=False))
            hparams = np.array([[sample_adam8p_wide_grid(int(seeds[j]))[p] for p in ADAM8P_PARAMS]
                                for j in keep], dtype=np.float64)
            curves = m[keep][:, :, [1, 0]].astype(np.float32)  # [valid, train]
            np.savez(out_path, hparams=hparams, curves=curves,
                     steps=xs.astype(np.float32), seeds=seeds[keep])
            saved += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            if verbose:
                print(f"  [{task}] {type(e).__name__}: {e}")
        if verbose and (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(tasks)} (saved {saved}, existing {skipped}, failed {failed})")
    if verbose:
        print(f"[taskset] done: {saved} saved, {skipped} existing, {failed} failed -> {out_dir}")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=None, help="first N sorted tasks (default: all)")
    ap.add_argument("--n-configs", type=int, default=150)
    args = ap.parse_args()
    convert_taskset(DATA_ROOT / "taskset_local", args.n_tasks, args.n_configs)


if __name__ == "__main__":
    main()
