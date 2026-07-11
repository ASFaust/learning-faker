"""One-time converter: TaskSet (GCS .npz) -> local pure-numpy format.

For a subsample of TaskSet's 1162 tasks, downloads the adam8p grid, recovers the
exact HPs from the config seeds (via the verbatim samplers), averages the 5 eval
replicas, keeps train(ch0)+valid(ch1) curves, subsamples configs, and saves one
compact .npz per task. No TensorFlow, no network at train time afterwards.

    /home/andi/venv/bin/python scripts/convert_taskset.py --n-tasks 300 --n-configs 150
"""

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lcfaker.data.taskset_samplers import ADAM8P_PARAMS, sample_adam8p_wide_grid

GCS = "https://storage.googleapis.com/gresearch/task_set_data"
LIST = "https://www.googleapis.com/storage/v1/b/gresearch/o"
FILE = "adam8p_wide_grid_1k_10000_replica5.npz"
OUT = Path("/home/andi/datasets/taskset_local")


def list_tasks() -> list[str]:
    r = requests.get(LIST, params={"prefix": "task_set_data/", "delimiter": "/",
                                    "maxResults": 5000}, timeout=60)
    pre = r.json().get("prefixes", [])
    return [p.split("/")[-2] for p in pre]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=300)
    ap.add_argument("--n-configs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tasks = list_tasks()
    print(f"{len(tasks)} tasks available; sampling {args.n_tasks}")
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(tasks), min(args.n_tasks, len(tasks)), replace=False)

    done = fail = 0
    for n, ti in enumerate(sel):
        task = tasks[int(ti)]
        out_path = OUT / f"{task}.npz"
        if out_path.exists():
            done += 1; continue
        try:
            resp = requests.get(f"{GCS}/{task}/{FILE}", timeout=180)
            if resp.status_code != 200:
                fail += 1; continue
            d = np.load(io.BytesIO(resp.content), allow_pickle=True)
            ys, xs, opt = d["ys"], d["xs"], d["optimizers"]     # (1000,5,51,4),(51,),(1000,)
            m = np.nanmean(ys, axis=1)                          # avg 5 replicas -> (1000,51,4)
            seeds = np.array([int(re.search(rb"seed(\d+)", o).group(1)) for o in opt])
            # keep configs with a usable valid curve (not all-NaN)
            usable = np.nonzero(np.isfinite(m[:, :, 1]).any(axis=1))[0]
            keep = rng.choice(usable, min(args.n_configs, len(usable)), replace=False)
            hparams = np.array([[sample_adam8p_wide_grid(int(seeds[j]))[p] for p in ADAM8P_PARAMS]
                                for j in keep], dtype=np.float64)
            curves = m[keep][:, :, [1, 0]].astype(np.float32)   # [valid, train]
            np.savez(out_path, hparams=hparams, curves=curves,
                     steps=xs.astype(np.float32), seeds=seeds[keep])
            done += 1
        except Exception as e:
            fail += 1
            print(f"  [{task}] {type(e).__name__}: {e}")
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(sel)}  (saved {done}, failed {fail})")
    print(f"done: {done} tasks saved, {fail} failed -> {OUT}")


if __name__ == "__main__":
    main()
