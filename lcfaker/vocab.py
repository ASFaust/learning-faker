"""Global vocabulary shared across datasets.

Datasets *register* their param specs and tasks; the vocabulary assigns global
ids so the model can size its embedding tables once and every dataset speaks the
same token language. This is what makes a param that appears in only one dataset
(a "one-off") harmless: it simply gets its own type id and is absent from other
configs' token sets.

Seen-tasks-only setting: tasks are a learned lookup, so every task (across all
datasets) also gets a global integer id here.
"""

from __future__ import annotations

import numpy as np

from .schema import CategoricalSpec, NumericSpec, ParamSpec, TokenizedConfig


class Vocabulary:
    def __init__(self) -> None:
        self.params: dict[str, ParamSpec] = {}
        self.type_id: dict[str, int] = {}
        # (param_name, category) -> id. Id 0 is reserved "n/a" (used by numeric tokens).
        self.cat_id: dict[tuple[str, str], int] = {"__na__": 0}  # type: ignore[dict-item]
        self._cat_counter = 1
        # global task registry: external key (e.g. "lcbench/APSFailure") -> id
        self.task_id: dict[str, int] = {}
        self.frozen = False

    # -- registration -------------------------------------------------------
    def register_param(self, spec: ParamSpec) -> None:
        assert not self.frozen, "vocabulary is frozen"
        existing = self.params.get(spec.name)
        if existing is None:
            self.params[spec.name] = spec
            self.type_id[spec.name] = len(self.type_id)
        if isinstance(spec, CategoricalSpec):
            for c in spec.categories:
                key = (spec.name, str(c))
                if key not in self.cat_id:
                    self.cat_id[key] = self._cat_counter
                    self._cat_counter += 1

    def register_task(self, key: str) -> int:
        if key not in self.task_id:
            assert not self.frozen, "vocabulary is frozen"
            self.task_id[key] = len(self.task_id)
        return self.task_id[key]

    def freeze(self) -> None:
        self.frozen = True

    # -- sizes (for building the model) ------------------------------------
    @property
    def n_param_types(self) -> int:
        return len(self.type_id)

    @property
    def n_cat_values(self) -> int:
        return self._cat_counter  # includes the reserved 0 slot

    @property
    def n_tasks(self) -> int:
        return len(self.task_id)

    # -- encoding -----------------------------------------------------------
    def encode_config(self, config: dict) -> TokenizedConfig:
        type_ids, num_vals, cat_ids, is_num = [], [], [], []
        for name, value in config.items():
            spec = self.params[name]
            type_ids.append(self.type_id[name])
            if isinstance(spec, NumericSpec):
                num_vals.append(spec.encode(value))
                cat_ids.append(0)
                is_num.append(True)
            else:
                num_vals.append(0.0)
                cat_ids.append(self.cat_id[(name, str(value))])
                is_num.append(False)
        return TokenizedConfig(
            type_ids=np.asarray(type_ids, dtype=np.int64),
            num_vals=np.asarray(num_vals, dtype=np.float32),
            cat_ids=np.asarray(cat_ids, dtype=np.int64),
            is_numeric=np.asarray(is_num, dtype=bool),
        )
