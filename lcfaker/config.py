"""Runtime configuration.

All dataset locations (and the build caches) are rooted at LCFAKER_DATA_ROOT,
taken from the environment and defaulting to /home/andi/datasets. To relocate all
data -- e.g. on a server where it must live under /shared/work -- set the env var
once before running:

    export LCFAKER_DATA_ROOT=/mnt/ssd01/home/work/lcfaker_data

The expected layout under the root is:
    <root>/lcbench/data_2k_lw.json
    <root>/pd1/pd1/pd1_matched_phase{0,1}_results.jsonl.gz
    <root>/fcnet/fcnet_tabular_benchmarks/fcnet_*_data.hdf5
    <root>/taskset_local/*.npz
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("LCFAKER_DATA_ROOT", "/home/andi/datasets"))
