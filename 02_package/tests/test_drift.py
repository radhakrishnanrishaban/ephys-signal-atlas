"""Invariants behind decoder alignment, r_norm drift correction and day labels (D-11)."""

import numpy as np
import pytest

from neurosignal.gold.drift import (
    _make_xy,
    compute_day_offsets,
    robust_zscore_per_channel_blocks,
    robust_zscore_per_channel_day,
    zscore_per_channel_day,
)


@pytest.mark.parametrize("window_bins, lag_bins", [(0, 0), (3, 0), (0, 2), (3, 2)])
def test_make_xy_window_and_lag_alignment_on_ramp(window_bins, lag_bins):
    T, n_ch = 50, 3
    t_all = np.arange(T, dtype=float)
    sbp = t_all[:, None] * 10.0 + np.arange(n_ch)  # sbp[t, c] = 10t + c names its own bin
    vel = t_all                                     # vel[t] = t

    X, y = _make_xy(sbp, vel, window_bins=window_bins, lag_bins=lag_bins)

    base_t = np.arange(window_bins, T - lag_bins)
    assert X.shape == (len(base_t), n_ch * (window_bins + 1))
    np.testing.assert_array_equal(y, base_t + lag_bins)
    for i, t in enumerate(base_t):
        expected = np.concatenate([sbp[t - k] for k in range(window_bins + 1)])  # current first
        np.testing.assert_array_equal(X[i], expected)


@pytest.mark.parametrize(
    "zscore",
    [
        zscore_per_channel_day,
        robust_zscore_per_channel_day,
        lambda X: robust_zscore_per_channel_blocks(X, block_size_bins=100),
    ],
    ids=["zscore", "robust", "robust_blocks"],
)
def test_per_session_zscore_ignores_channel_gain_and_offset(zscore):
    rng = np.random.default_rng(0)
    X = rng.normal(50.0, 10.0, size=(400, 5))
    gain = rng.uniform(0.2, 5.0, size=5)
    offset = rng.uniform(-100.0, 100.0, size=5)

    np.testing.assert_allclose(zscore(X * gain + offset), zscore(X), atol=1e-9)


def test_day_offsets_come_from_dates_not_session_index():
    # First four LINK sessions in 01_data/ (ERR-06): session 3 is 2020-02-05 = day 9, not day 3 or 4.
    dates = ["20200127", "20200130", "20200204", "20200205"]

    offsets = compute_day_offsets(dates)

    np.testing.assert_array_equal(offsets, [0, 3, 8, 9])
    assert not np.array_equal(offsets, np.arange(len(dates)))
