"""FCNet loader (Klein & Hutter tabular benchmarks), as a Source.

4 UCI-regression tasks x 2-layer FCNet x 9 explicit hyperparameters (full 62208
grid per task) x 100-epoch curves x 4 seed replicas. We subsample configs and
emit each replica as its own curve -- the 4 reps give genuine same-(config,t)
seed scatter, the aleatoric signal LCBench/PD1 (single-run) lacked.

learning_rate + batch_size share type ids with LCBench/PD1. Targets are MSE
(regression), not cross-entropy -- fine in log space, handled per-task.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

DATASETS = {
    "protein": "fcnet_protein_structure_data.hdf5",
    "slice": "fcnet_slice_localization_data.hdf5",
    "naval": "fcnet_naval_propulsion_data.hdf5",
    "parkinsons": "fcnet_parkinsons_telemonitoring_data.hdf5",
}
# emitted config name -> (hdf5 key, kind)
PARAM_MAP = {
    "learning_rate": ("init_lr", "log"),
    "batch_size":    ("batch_size", "log"),
    "n_units_1":     ("n_units_1", "log"),
    "n_units_2":     ("n_units_2", "log"),
    "dropout_1":     ("dropout_1", "linear"),
    "dropout_2":     ("dropout_2", "linear"),
    "activation_1":  ("activation_fn_1", "categorical"),
    "activation_2":  ("activation_fn_2", "categorical"),
    "lr_schedule":   ("lr_schedule", "categorical"),
}


class FCNetSource:
    name = "fcnet"

    def __init__(self, dir: str = "/home/andi/datasets/fcnet/fcnet_tabular_benchmarks",
                 max_configs: int = 2000, seed: int = 0):
        self.dir = Path(dir)
        self.max_configs = max_configs
        self.seed = seed

    def param_kinds(self) -> dict[str, str]:
        return {name: kind for name, (_, kind) in PARAM_MAP.items()}

    def records(self):
        from .build import RawRecord
        for dsname, fname in DATASETS.items():
            f = h5py.File(self.dir / fname, "r")
            keys = list(f.keys())
            rng = np.random.default_rng(self.seed)
            sel = rng.choice(len(keys), min(self.max_configs, len(keys)), replace=False)
            for i in sel:
                k = keys[int(i)]
                raw = json.loads(k)
                cfg = {name: raw[col] for name, (col, _) in PARAM_MAP.items()}
                g = f[k]
                vl = np.asarray(g["valid_loss"]); tl = np.asarray(g["train_loss"])  # (reps, 100)
                epochs = np.arange(vl.shape[1], dtype=float)
                for rep in range(vl.shape[0]):
                    yield RawRecord(
                        task_key=f"fcnet/{dsname}",
                        config=cfg,
                        t_abs=epochs,
                        y_val=vl[rep],
                        y_train=tl[rep],
                    )
            f.close()
