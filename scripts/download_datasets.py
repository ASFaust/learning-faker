"""Download all HPO datasets into LCFAKER_DATA_ROOT -- makes the repo reproducible.

Fetches LCBench, PD1, FCNet (archives) and TaskSet (per-task, via convert_taskset)
and lays them out exactly where the loaders expect them:

    <root>/lcbench/data_2k_lw.json
    <root>/pd1/pd1/*.jsonl.gz
    <root>/fcnet/fcnet_tabular_benchmarks/*.hdf5
    <root>/taskset_local/*.npz

Idempotent: skips datasets already present (TaskSet skips per-task), so it doubles as
a resume. Set LCFAKER_DATA_ROOT first (default /home/andi/datasets).

    python scripts/download_datasets.py                        # all four
    python scripts/download_datasets.py --only lcbench pd1 fcnet
    python scripts/download_datasets.py --only taskset --taskset-n-tasks 800
"""

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))          # for convert_taskset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # for lcfaker
from convert_taskset import convert_taskset                       # noqa: E402
from lcfaker.config import DATA_ROOT                              # noqa: E402

LCBENCH_URL = "https://ndownloader.figshare.com/files/21188598"   # data_2k_lw.zip
PD1_URL = "https://storage.googleapis.com/gresearch/pint/pd1.tar.gz"
FCNET_URLS = [  # ml4aad.org tends to redirect to the uni-freiburg mirror; try both
    "https://ml.informatik.uni-freiburg.de/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz",
    "http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz",
]


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}")
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  {got / 1e6:6.0f} / {total / 1e6:.0f} MB", end="", flush=True)
        print()
    return dest


def _extract(archive: Path, into: Path):
    into.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            t.extractall(into)


def lcbench():
    d = DATA_ROOT / "lcbench"
    if (d / "data_2k_lw.json").exists():
        print("[lcbench] already present, skip")
        return
    print("[lcbench] ~340 MB zip -> ~1 GB json")
    _extract(download(LCBENCH_URL, d / "data_2k_lw.zip"), d)
    print("[lcbench] done")


def pd1():
    d = DATA_ROOT / "pd1"
    if (d / "pd1" / "pd1_matched_phase0_results.jsonl.gz").exists():
        print("[pd1] already present, skip")
        return
    print("[pd1] downloading")
    _extract(download(PD1_URL, d / "pd1.tar.gz"), d)   # tar has a top-level pd1/ dir
    print("[pd1] done")


def fcnet():
    d = DATA_ROOT / "fcnet"
    if (d / "fcnet_tabular_benchmarks").exists():
        print("[fcnet] already present, skip")
        return
    print("[fcnet] ~750 MB")
    tar = d / "fcnet.tar.gz"
    for url in FCNET_URLS:
        try:
            download(url, tar)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  mirror failed ({e}); trying next")
    else:
        raise RuntimeError("all FCNet mirrors failed")
    _extract(tar, d)
    print("[fcnet] done")


def taskset(n_tasks):
    convert_taskset(DATA_ROOT / "taskset_local", n_tasks=n_tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=["lcbench", "pd1", "fcnet", "taskset"],
                    help="subset to fetch (default: all)")
    ap.add_argument("--taskset-n-tasks", type=int, default=None,
                    help="first N TaskSet tasks (default: all ~1162)")
    args = ap.parse_args()
    which = args.only or ["lcbench", "pd1", "fcnet", "taskset"]
    print(f"DATA_ROOT = {DATA_ROOT}")
    if "lcbench" in which:
        lcbench()
    if "pd1" in which:
        pd1()
    if "fcnet" in which:
        fcnet()
    if "taskset" in which:
        taskset(args.taskset_n_tasks)
    print("all requested datasets ready.")


if __name__ == "__main__":
    main()
