"""
Session-resampled bootstrap CIs (D-13) for the numbers the series publishes (EXP-003).

Pairs share sessions, so resampling pairs as if independent understates the spread. Every draw
resamples sessions with replacement; an ordered pair (i, j) gets weight cᵢ·cⱼ (cᵢ = draws of session i),
a per-session value gets weight cᵢ. Paired differences reuse the same draws for both arms.
"""

from typing import Callable, Optional

import numpy as np

N_BOOT = 2000


def session_draws(n_sessions: int, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """(n_boot, n_sessions) draw counts; row b says how often each session was drawn."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_sessions, size=(n_boot, n_sessions))
    return np.stack([np.bincount(row, minlength=n_sessions) for row in idx])


def ci95(draws: np.ndarray) -> tuple:
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def off_diagonal_mean(R: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean of R over ordered pairs i ≠ j (optionally only where mask is True)."""
    keep = ~np.eye(len(R), dtype=bool) if mask is None else mask & ~np.eye(len(R), dtype=bool)
    return float(np.mean(R[keep]))


def bootstrap_off_diagonal_mean(R: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """Session-resampled draws of the mean off-diagonal entry of a square pairwise matrix (pass R_a − R_b for a paired difference)."""
    M = np.array(R, dtype=float)
    np.fill_diagonal(M, 0.0)
    C = session_draws(len(M), n_boot, seed).astype(float)
    num = np.einsum("bi,ij,bj->b", C, M, C)
    den = C.sum(axis=1) ** 2 - (C**2).sum(axis=1)
    return num / den


def bootstrap_session_mean(values: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """Session-resampled draws of the mean of one value per session (pass a − b for a paired difference)."""
    v = np.asarray(values, dtype=float)
    C = session_draws(len(v), n_boot, seed).astype(float)
    return C @ v / C.sum(axis=1)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum = np.cumsum(w)
    return float(x[np.searchsorted(cum, 0.5 * cum[-1])])


def bootstrap_pair_statistic(
    cal: np.ndarray,
    test: np.ndarray,
    values: np.ndarray,
    stat: Callable[[np.ndarray, np.ndarray], float] = weighted_median,
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> np.ndarray:
    """Session-resampled draws of stat(values, weights) over a list of (cal, test) pairs."""
    sessions, idx = np.unique(np.concatenate([cal, test]), return_inverse=True)
    i_cal, i_test = idx[: len(cal)], idx[len(cal):]
    C = session_draws(len(sessions), n_boot, seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        w = C[b, i_cal] * C[b, i_test]
        keep = w > 0
        out[b] = stat(values[keep], w[keep]) if keep.any() else np.nan
    return out
