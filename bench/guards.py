"""
Guardrail checks shared by the tests and the bench.run gate (D-13), so CI and the
scoreboard check the same thing. The leakage check lives in bench.plan (leaking_plans).
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

CHECK_BINS, CHECK_CHANNELS = 400, 5
CHECK_TIMES = (0, 1, 150, CHECK_BINS - 2)


def is_causal(make: Callable[[List[np.ndarray]], List[np.ndarray]]) -> bool:
    """
    Future-perturbation check (as in the EDA-06 streaming test): change every sample after t
    in a synthetic session; a causal feature leaves its output up to t unchanged, for every t checked.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(50.0, 10.0, size=(CHECK_BINS, CHECK_CHANNELS))
    future = rng.normal(150.0, 30.0, size=X.shape)
    base = make([X])[0]
    for t in CHECK_TIMES:
        X_changed = X.copy()
        X_changed[t + 1 :] = future[t + 1 :]
        if not np.array_equal(make([X_changed])[0][: t + 1], base[: t + 1]):
            return False
    return True
