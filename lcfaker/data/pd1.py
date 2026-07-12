"""PD1 loader (Wang et al., "Automatic prior selection..."), as a Source.

19 architecturally-diverse tasks (transformers, ResNets, CNNs) x Halton-sampled
optimizer configs. Time axis = global_step; targets = valid/ce_loss,
train/ce_loss. 4 tuned dims; learning_rate/momentum/batch_size share type ids
with LCBench (real cross-dataset transfer). Diverged trials are kept -- their
finite high-loss points are exactly the heavy-tail data the histogram head wants
(non-finite points get dropped by the builder).
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from ..config import DATA_ROOT

# config param name -> (pd1 column, transform)
PARAM_MAP = {
    "batch_size":      ("hps.batch_size", "log"),
    "learning_rate":   ("hps.lr_hparams.initial_value", "log"),
    "momentum":        ("hps.opt_hparams.momentum", "linear"),
    "lr_power":        ("hps.lr_hparams.power", "linear"),
    "lr_decay_factor": ("hps.lr_hparams.decay_steps_factor", "linear"),
}
DEFAULT_FILES = ["pd1_matched_phase0_results.jsonl.gz", "pd1_matched_phase1_results.jsonl.gz"]


def _arr(x):
    return np.asarray([v if v is not None else np.nan for v in x], float)


class PD1Source:
    name = "pd1"

    def __init__(self, dir: str = str(DATA_ROOT / "pd1" / "pd1"), files: list[str] | None = None):
        self.dir = Path(dir)
        self.files = files or DEFAULT_FILES

    def param_kinds(self) -> dict[str, str]:
        return {name: kind for name, (_, kind) in PARAM_MAP.items()}

    def records(self):
        from .build import RawRecord
        for fname in self.files:
            with gzip.open(self.dir / fname, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    gs = r.get("global_step")
                    vl, tl = r.get("valid/ce_loss"), r.get("train/ce_loss")
                    if not gs or not vl or not tl:
                        continue
                    cfg = {name: r.get(col) for name, (col, _) in PARAM_MAP.items()}
                    if any(v is None for v in cfg.values()):
                        continue
                    yield RawRecord(
                        task_key=f"pd1/{r['dataset']}_{r['model']}_bs{int(r['hps.batch_size'])}",
                        config=cfg,
                        t_abs=np.asarray(gs, float),
                        y_val=_arr(vl),
                        y_train=_arr(tl),
                    )
