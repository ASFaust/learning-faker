"""TaskSet loader (local converted format), as a Source.

Reads the pure-numpy .npz files produced by scripts/convert_taskset.py -- no
TensorFlow, no network. Each file is one task: adam8p HPs + train/valid curves
over ~10k steps. TaskSet's value is TASK COUNT (hundreds of diverse RNN/CNN/MLP/
transformer problems), the axis our other datasets are starved on.

learning_rate shares a type id with the other datasets; the rest (beta1/beta2/
epsilon/l1/l2/linear_decay/exponential_decay) are adam-specific one-offs.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .taskset_samplers import ADAM8P_PARAMS

# adam8p param -> transform. betas are 1-log so live in (0,1); decays/eps/l1/l2 log.
# batch_size (parsed from task name) is a scalar bridge to the other datasets.
KINDS = {
    "learning_rate": "log", "beta1": "linear", "beta2": "linear",
    "epsilon": "log", "l1": "log", "l2": "log",
    "linear_decay": "log", "exponential_decay": "log",
    "batch_size": "log",
}
_BS = re.compile(r"[Bb][Ss](\d+)")

# Synthetic toy objectives (not neural-net learning curves) to exclude -- they
# converge to machine precision (val loss ~1e-18..1e-22, far below any real NN
# floor) and/or are pathologically bimodal, so no single reference is sane:
#   quadratic_family / TwoD_Bowl -- convex bowls (good LRs -> ~0, bad -> ~1e10)
#   losg_tasks_family            -- synthetic loss surfaces reaching abs_min 1.6e-22
# (identified via scripts/audit_families.py: per-run convergence depth, abs_min
# 17 orders below the deepest real task fcnet/naval at 4.8e-5.)
_EXCLUDE = ("quadratic_family", "TwoD_Bowl", "losg_tasks_family")

# Majority-diverging real tasks: >50% of their sampled HP configs diverge before
# the first observation, so even the MEDIAN initial loss (our per-task anchor) is a
# diverged value and every config gets garbage targets. Excluded per-task (not
# per-family -- most seeds of these families are fine). Identified via
# scripts/find_diverging_tasks.py: median-init > 10x the family-median-init.
# See docs/task_selection.md.
_DIVERGING = frozenset({
    "rnn_text_classification_family_seed58", "rnn_text_classification_family_seed70",
    "rnn_text_classification_family_seed96",
    "mlp_ae_family_seed39", "mlp_ae_family_seed48", "mlp_ae_family_seed62",
    "mlp_family_seed39", "mlp_family_seed43", "mlp_family_seed49", "mlp_family_seed78",
    "conv_fc_family_seed36", "conv_fc_family_seed38", "conv_fc_family_seed40",
    "conv_fc_family_seed70", "conv_pooling_family_seed91",
    "char_rnn_language_model_family_seed38", "char_rnn_language_model_family_seed39",
    "word_rnn_language_model_family_seed4", "word_rnn_language_model_family_seed8",
    "word_rnn_language_model_family_seed39", "nvp_family_seed75",
})


def _excluded(stem: str) -> bool:
    if any(x in stem for x in _EXCLUDE):          # synthetic families (substring)
        return True
    key = re.sub(r"_?[Bb][Ss]\d+", "", stem)      # strip BS suffix -> task key
    return key in _DIVERGING                       # diverging tasks (exact)


class TaskSetSource:
    name = "taskset"

    def __init__(self, dir: str = "/home/andi/datasets/taskset_local",
                 max_configs: int | None = None):
        self.dir = Path(dir)
        self.max_configs = max_configs

    def param_kinds(self) -> dict[str, str]:
        return dict(KINDS)

    def records(self):
        from .build import RawRecord
        for path in sorted(self.dir.glob("*.npz")):
            if _excluded(path.stem):
                continue
            d = np.load(path)
            hp, curves, steps = d["hparams"], d["curves"], d["steps"]  # (N,8),(N,T,2),(T,)
            stem = path.stem
            m = _BS.search(stem)
            bs = int(m.group(1)) if m else None
            # strip batch-size token from the key so bs-variant tasks merge
            # (not a real deflation -- batch size is a config knob, not a task)
            key_stem = re.sub(r"_?[Bb][Ss]\d+", "", stem) if m else stem
            task_key = f"taskset/{key_stem}"
            n = hp.shape[0] if self.max_configs is None else min(self.max_configs, hp.shape[0])
            for i in range(n):
                cfg = {p: float(hp[i, j]) for j, p in enumerate(ADAM8P_PARAMS)}
                if bs is not None:
                    cfg["batch_size"] = bs
                yield RawRecord(
                    task_key=task_key,
                    config=cfg,
                    t_abs=steps.astype(float),
                    y_val=curves[i, :, 0].astype(float),    # valid
                    y_train=curves[i, :, 1].astype(float),  # train
                )
