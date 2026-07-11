"""TaskSet hyperparameter samplers, copied VERBATIM (pure numpy) from
google-research/task_set/optimizers/*.py.

TaskSet's .npz files store only the config seed (e.g. "adam8p_wide_grid_seed42"),
not the hyperparameter values -- but each config's HPs are a deterministic
function of its seed via these samplers. Reproducing them here (no TensorFlow)
recovers the exact HPs with zero correctness risk. The draw ORDER matters (all
params share one RandomState), so keep these exactly as upstream.
"""

from __future__ import annotations

import numpy as np


def sample_log_float(rng, low, high):
    return float(np.exp(rng.uniform(np.log(float(low)), np.log(float(high)))))


# adam8p: learning_rate + 7 (upstream optimizers/adam8p.py)
ADAM8P_PARAMS = ["learning_rate", "beta1", "beta2", "epsilon",
                 "l1", "l2", "linear_decay", "exponential_decay"]


def sample_adam8p_wide_grid(seed: int) -> dict:
    rng = np.random.RandomState(seed)
    return {
        "learning_rate": sample_log_float(rng, 1e-8, 1e1),
        "beta1": 1 - sample_log_float(rng, 1e-4, 1e0),
        "beta2": 1 - sample_log_float(rng, 1e-6, 1e0),
        "epsilon": sample_log_float(rng, 1e-10, 1e3),
        "l1": sample_log_float(rng, 1e-8, 1e1),
        "l2": sample_log_float(rng, 1e-8, 1e1),
        "linear_decay": sample_log_float(rng, 1e-7, 1e-4),
        "exponential_decay": sample_log_float(rng, 1e-3, 1e-6),
    }
