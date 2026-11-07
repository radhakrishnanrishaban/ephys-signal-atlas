"""Invariants behind the "strictly causal" claim for features.causal and the fixed-point claim for features.fixed (D-11)."""

import numpy as np
import pytest

from neurosignal.features.causal import (
    EwmaNormalizer,
    EwmaNormalizerV2,
    causal_zscore_ewma_v2,
    normalize_sessions_sbp_causal_v2,
    causal_smooth_ewma,
    causal_zscore_ewma,
    causal_zscore_window,
    normalize_sessions_sbp_gold_causal,
    stream_session,
)
from neurosignal.features.fixed import fixed_point_sessions, normalize_sessions_sbp_fixed

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
        lambda X: causal_zscore_ewma_v2(X, alpha=0.05, warmup=100)[0],
        lambda X: normalize_sessions_sbp_causal_v2([X], tau_s=0.4, carry=True, warmup=100)[0],
        lambda X: normalize_sessions_sbp_fixed([X], tau_s=0.4, warmup=100)[0],
    ],
    ids=["zscore_ewma", "zscore_window", "smooth_ewma", "gold_causal", "zscore_ewma_v2", "causal_v2", "fixed_point"],
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


@pytest.mark.parametrize("carry", [True, False], ids=["carry", "reset"])
def test_v2_replay_across_sessions_equals_batch(carry):
    sessions = [_sbp(0), _sbp(1) * 2.0 + 30.0]
    normalizer = EwmaNormalizerV2(n_channels=C, alpha=0.05, warmup=50)
    state = None
    for X in sessions:
        normalizer.start_session(carry)
        replayed = np.stack([normalizer.update(x) for x in stream_session(X)])
        batch, end = causal_zscore_ewma_v2(X, alpha=0.05, state=state, warmup=50)
        np.testing.assert_array_equal(replayed, batch)
        state = end if carry else None


def test_fixed_point_matches_float_reference():
    # SBP-like: per-channel gain and offset, a slow drift and rare spikes, in the LINK range (EXP-006 P2, P3)
    rng = np.random.default_rng(0)
    n, c = 3000, 8
    X = rng.uniform(0.5, 10.0, c) + rng.uniform(0.5, 5.0, c) * np.abs(rng.normal(1.0, 0.3, (n, c)))
    X += np.linspace(0.0, 2.0, n)[:, None]
    X[rng.random((n, c)) < 0.002] += 30.0
    out, norm, input_saturated = fixed_point_sessions([X, X[::-1] * 0.5])
    reference = normalize_sessions_sbp_causal_v2([X, X[::-1] * 0.5], tau_s=2.0, carry=False)

    err = np.concatenate([np.abs(o / 2**12 - f)[300:] for o, f in zip(out, reference)])
    assert err.mean() <= 1e-3 and err.max() <= 1e-2
    assert norm.saturated == 0 and input_saturated == 0
