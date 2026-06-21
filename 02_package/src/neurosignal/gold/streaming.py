"""
neurosignal.gold.streaming — strictly causal (online) SBP normalization.

The batch helpers in `gold.drift` (zscore_per_channel_day, smooth_sbp_gaussian)
are *acausal*: they use whole-session statistics and a symmetric smoothing
kernel, so the output at time t depends on samples from t+1, t+2, ...  That is
fine for offline analysis but impossible on a real implant, which sees one
sample at a time and must emit an output now using only the past.

This module provides causal siblings:
  - causal_zscore_ewma     : EWMA running mean/var z-score (the main one)
  - causal_zscore_window   : trailing fixed-window z-score (for comparison)
  - causal_smooth_ewma     : one-pole exponential smoother (causal)
  - normalize_sessions_sbp_gold_causal : per-session causal Gold feature
  - stream_session / EwmaNormalizer     : replay harness to *prove* causality

Convention: each session SBP array is shape (T, C) = (time bins, channels).
Normalization runs forward in time within a session; state resets per session
(no cross-session memory yet — that is the job of the later alignment work).
"""

from __future__ import annotations

from typing import Iterator, List

import numpy as np

__all__ = [
    "causal_zscore_ewma",
    "causal_zscore_window",
    "causal_smooth_ewma",
    "normalize_sessions_sbp_gold_causal",
    "EwmaNormalizer",
    "stream_session",
]


# ---------------------------------------------------------------------------
# Core causal normalizers
# ---------------------------------------------------------------------------
def causal_zscore_ewma(
    X: np.ndarray,
    alpha: float = 0.01,
    eps: float = 1e-6,
    warmup: int = 0,
) -> np.ndarray:
    """
    Strictly causal per-channel z-score using exponentially-weighted running
    mean and variance (West's online update).

    X      : (T, C) SBP array.
    alpha  : EWMA update rate in (0, 1]. Small = slow/stable, large = fast/jittery.
    eps    : numerical floor added to the running std.
    warmup : if > 0, the first `warmup` bins are set to NaN so they can be
             excluded from scoring instead of polluting it.

    Returns z : (T, C), same shape as X. z[t] depends only on X[0..t].
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    T, C = X.shape
    z = np.empty((T, C), dtype=np.float64)

    mu = X[0].copy()            # init mean to first sample (cheaper warm-up)
    var = np.zeros(C)           # init variance to 0
    for t in range(T):
        x = X[t]
        # West's online update: update mean first, then variance with new mean.
        mu = (1.0 - alpha) * mu + alpha * x
        var = (1.0 - alpha) * var + alpha * (x - mu) ** 2
        z[t] = (x - mu) / (np.sqrt(var) + eps)

    if warmup > 0:
        z[:warmup] = np.nan
    return z


def causal_zscore_window(
    X: np.ndarray,
    win_bins: int = 300,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Causal per-channel z-score using mean/std of the trailing `win_bins`
    samples only (a sliding window that never looks forward).

    Simpler to explain than EWMA; included as a comparison line.
    """
    if win_bins < 1:
        raise ValueError(f"win_bins must be >= 1, got {win_bins}")
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    T, C = X.shape
    z = np.empty((T, C), dtype=np.float64)
    for t in range(T):
        lo = max(0, t - win_bins + 1)
        window = X[lo : t + 1]                 # past-and-current only
        mu = window.mean(axis=0)
        sd = window.std(axis=0)
        z[t] = (X[t] - mu) / (sd + eps)
    return z


def causal_smooth_ewma(X: np.ndarray, alpha_smooth: float = 0.2) -> np.ndarray:
    """
    One-pole exponential smoother (causal): y[t] = (1-a)*y[t-1] + a*x[t].
    Replaces the symmetric Gaussian in gold.drift, which looks into the future.
    Adds phase delay (output lags input) — the price of causality.
    """
    if not (0.0 < alpha_smooth <= 1.0):
        raise ValueError(f"alpha_smooth must be in (0, 1], got {alpha_smooth}")
    X = np.asarray(X, dtype=np.float64)
    y = np.empty_like(X)
    y[0] = X[0]
    for t in range(1, X.shape[0]):
        y[t] = (1.0 - alpha_smooth) * y[t - 1] + alpha_smooth * X[t]
    return y


def normalize_sessions_sbp_gold_causal(
    sessions_sbp: List[np.ndarray],
    alpha: float = 0.01,
    alpha_smooth: float = 0.2,
    warmup: int = 0,
) -> List[np.ndarray]:
    """
    Causal counterpart of gold.drift.normalize_sessions_sbp_gold:
    per session, causal EWMA z-score then causal one-pole smoothing.
    State resets each session. Returns list of (T, C) arrays.
    """
    out: List[np.ndarray] = []
    for sbp in sessions_sbp:
        z = causal_zscore_ewma(np.asarray(sbp, dtype=np.float64), alpha=alpha, warmup=warmup)
        z = np.nan_to_num(z, nan=0.0)  # neutralise warm-up NaNs before smoothing
        out.append(causal_smooth_ewma(z, alpha_smooth=alpha_smooth))
    return out


# ---------------------------------------------------------------------------
# Replay harness — proves the batch implementation is genuinely causal
# ---------------------------------------------------------------------------
class EwmaNormalizer:
    """
    Stateful, one-sample-at-a-time EWMA z-scorer. This is what a device would
    run: call .update(x) for each incoming sample and get back its z-score.

    Feeding a session through this sample-by-sample must reproduce
    causal_zscore_ewma(...) exactly — that equality is the causality proof.
    """

    def __init__(self, n_channels: int, alpha: float = 0.01, eps: float = 1e-6):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.mu = np.zeros(n_channels)
        self.var = np.zeros(n_channels)
        self._initialised = False

    def update(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if not self._initialised:
            self.mu = x.copy()
            self._initialised = True
        a = self.alpha
        self.mu = (1.0 - a) * self.mu + a * x
        self.var = (1.0 - a) * self.var + a * (x - self.mu) ** 2
        return (x - self.mu) / (np.sqrt(self.var) + self.eps)


def stream_session(sbp: np.ndarray) -> Iterator[np.ndarray]:
    """Yield rows sbp[t] one at a time — the replay stand-in for a live feed."""
    sbp = np.asarray(sbp, dtype=np.float64)
    for t in range(sbp.shape[0]):
        yield sbp[t]
