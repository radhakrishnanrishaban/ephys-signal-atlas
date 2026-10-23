"""
Calibration-free days (CFD), the north-star metric (D-08), as defined for bench v0 (D-12).

kept ratio per pair = max(R²_cross, 0) / R²_same(calibration session)
    A negative cross-day R² counts as 0 % kept, so one blown-up pair cannot dominate.
CFD = the longest gap at which a non-increasing (isotonic) fit of kept ratio vs gap
      is still ≥ the threshold. 0 if even the shortest gap fails.
The fit is used because 435 pairs spread over 269 distinct gaps: a per-gap median is noise.

95 % CI (D-13): resample sessions with replacement; a pair's weight is
(draws of its calibration session) × (draws of its test session); refit and recompute CFD.
Pairs share sessions, so resampling pairs as if independent would understate the spread.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def kept_ratio(r2_cross: np.ndarray, r2_same: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(r2_cross, dtype=float), 0.0) / np.asarray(r2_same, dtype=float)


def cfd_days(gap_days: np.ndarray, ratio: np.ndarray, threshold: float, weight: np.ndarray = None) -> int:
    gap_days = np.asarray(gap_days, dtype=float)
    fit = IsotonicRegression(increasing=False).fit(gap_days, ratio, sample_weight=weight).predict(gap_days)
    passing = gap_days[fit >= threshold]
    return int(passing.max()) if passing.size else 0


def cfd_bootstrap(
    cal: np.ndarray,
    test: np.ndarray,
    gap_days: np.ndarray,
    ratio: np.ndarray,
    threshold: float,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """CFD of n_boot session-resampled draws; cal and test are session labels per pair."""
    sessions, idx = np.unique(np.concatenate([cal, test]), return_inverse=True)
    i_cal, i_test = idx[: len(cal)], idx[len(cal):]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=int)
    for b in range(n_boot):
        draws = np.bincount(rng.integers(0, len(sessions), len(sessions)), minlength=len(sessions))
        w = draws[i_cal] * draws[i_test]
        keep = w > 0
        out[b] = cfd_days(gap_days[keep], ratio[keep], threshold, weight=w[keep]) if keep.any() else 0
    return out
