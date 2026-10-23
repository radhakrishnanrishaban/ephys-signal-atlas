"""Invariants behind the "strictly causal" claim for gold.streaming (D-11)."""

import numpy as np
import pytest

from neurosignal.gold.streaming import (
    EwmaNormalizer,
    causal_smooth_ewma,
    causal_zscore_ewma,
    causal_zscore_window,
    normalize_sessions_sbp_gold_causal,
    stream_session,
)

T, C = 400, 5


def _sbp(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(50.0, 10.0, size=(T, C))


@pytest.mark.parametrize(
    "fn",
    [
        lambda X: causal_zscore_ewma(X, alpha=0.05),
        lambda X: causal_zscore_window(X, win_bins=30),
        lambda X: causal_smooth_ewma(X, alpha_smooth=0.2),
        lambda X: normalize_sessions_sbp_gold_causal([X], alpha=0.05)[0],
    ],
    ids=["zscore_ewma", "zscore_window", "smooth_ewma", "gold_causal"],
)
@pytest.mark.parametrize("t", [0, 1, 150, T - 2])
def test_output_at_t_ignores_future_samples(fn, t):
    X = _sbp()
    X_future_changed = X.copy()
    X_future_changed[t + 1 :] = _sbp(seed=1)[t + 1 :] * 3.0 + 100.0

    np.testing.assert_array_equal(fn(X)[: t + 1], fn(X_future_changed)[: t + 1])


def test_sample_by_sample_replay_equals_batch():
    X = _sbp()
    normalizer = EwmaNormalizer(n_channels=C, alpha=0.05)
    replayed = np.stack([normalizer.update(x) for x in stream_session(X)])

    np.testing.assert_array_equal(replayed, causal_zscore_ewma(X, alpha=0.05))
