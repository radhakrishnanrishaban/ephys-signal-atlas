"""
neurosignal.features.causal — strictly causal (online) SBP normalization.

The batch helpers in `features.normalize` (zscore_per_channel_day, smooth_sbp_gaussian)
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
  - causal_zscore_ewma_v2 / normalize_sessions_sbp_causal_v2 / EwmaNormalizerV2 :
        normaliser v2 (EXP-005): exact EW variance, warm-up start, z clipped, state optionally
        carried from one session to the next

Convention: each session SBP array is shape (T, C) = (time bins, channels).
v1 runs forward in time within a session and resets per session; v2 can carry its state
across sessions in date order (no labels are used, so carrying it leaks nothing into a decoder).
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
    "BIN_S",
    "alpha_from_tau",
    "causal_zscore_ewma_v2",
    "normalize_sessions_sbp_causal_v2",
    "EwmaNormalizerV2",
]

BIN_S = 0.02  # LINK SBP bin width (50 Hz)


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
    Replaces the symmetric Gaussian in features.normalize, which looks into the future.
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
    Causal counterpart of features.normalize.normalize_sessions_sbp_gold:
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


# ---------------------------------------------------------------------------
# Normaliser v2 (EDA-10, EXP-005)
# ---------------------------------------------------------------------------
def alpha_from_tau(tau_s: float, bin_s: float = BIN_S) -> float:
    """EWMA rate for a time constant of tau_s seconds: α = bin / τ (τ 2 s at 50 Hz → 0.01, the v1 rate)."""
    if tau_s < bin_s:
        raise ValueError(f"tau_s must be >= one bin ({bin_s} s), got {tau_s}")
    return bin_s / tau_s


def causal_zscore_ewma_v2(
    X: np.ndarray,
    alpha: float,
    state: tuple | None = None,
    warmup: int = 300,
    clip: float = 6.0,
    eps: float = 1e-6,
) -> tuple:
    """
    Strictly causal per-channel z-score, v2. Returns (z, state); state = (mean, variance) after the
    last bin, to hand to the next session for carry-over.

    state None (first session, or reset): the first `warmup` bins use the cumulative mean and
    variance of the samples so far (Welford), so the EWMA starts from the warm-up statistics instead
    of variance 0; from bin `warmup` on, the exact EW update
        δ = x − μ_old,  μ = μ_old + αδ,  var = (1 − α)(var + αδ²).
    state given: the EW update runs from bin 0, starting from that state.
    z = (x − μ) / (√var + eps), clipped to ±clip. z[t] depends only on X[0..t] (and the state).
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    T, C = X.shape
    z = np.empty((T, C), dtype=np.float64)
    if state is None:
        mu, m2, start = np.zeros(C), np.zeros(C), min(warmup, T)
        for t in range(start):
            delta = X[t] - mu
            mu = mu + delta / (t + 1)
            m2 = m2 + delta * (X[t] - mu)
            z[t] = (X[t] - mu) / (np.sqrt(m2 / (t + 1)) + eps)
        var = m2 / max(start, 1)
    else:
        mu, var, start = np.array(state[0], dtype=np.float64), np.array(state[1], dtype=np.float64), 0
    for t in range(start, T):
        delta = X[t] - mu
        mu = mu + alpha * delta
        var = (1.0 - alpha) * (var + alpha * delta**2)
        z[t] = (X[t] - mu) / (np.sqrt(var) + eps)
    np.clip(z, -clip, clip, out=z)
    return z, (mu, var)


def normalize_sessions_sbp_causal_v2(
    sessions_sbp: List[np.ndarray],
    tau_s: float,
    carry: bool,
    alpha_smooth: float = 0.2,
    warmup: int = 300,
    clip: float = 6.0,
) -> List[np.ndarray]:
    """
    Normaliser v2 over sessions given in date order: z-score v2 then the one-pole smoother.
    carry=True starts each session from the previous session's final (mean, variance); only the
    first session estimates them from its warm-up. carry=False does that for every session.
    The smoother restarts each session (its time constant is ~0.1 s, inside the unscored warm-up).
    """
    alpha, state, out = alpha_from_tau(tau_s), None, []
    for sbp in sessions_sbp:
        z, end = causal_zscore_ewma_v2(sbp, alpha, state=state, warmup=warmup, clip=clip)
        state = end if carry else None
        out.append(causal_smooth_ewma(z, alpha_smooth=alpha_smooth))
    return out


class EwmaNormalizerV2:
    """
    One-sample-at-a-time z-scorer v2, the device form of causal_zscore_ewma_v2. Call
    start_session(carry) at each session start, then update(x) per bin. Replaying sessions through it
    must reproduce the batch function with the same carry, which is the causality proof across sessions.
    """

    def __init__(self, n_channels: int, alpha: float, warmup: int = 300, clip: float = 6.0, eps: float = 1e-6):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha, self.warmup, self.clip, self.eps = float(alpha), int(warmup), float(clip), float(eps)
        self.mu, self.var, self._m2 = np.zeros(n_channels), np.zeros(n_channels), np.zeros(n_channels)
        self._n = 0  # bins seen in the warm-up; None once the EW update runs

    def start_session(self, carry: bool) -> None:
        if not carry:
            self.mu, self.var, self._m2 = np.zeros_like(self.mu), np.zeros_like(self.mu), np.zeros_like(self.mu)
            self._n = 0
        elif self._n:  # the last session ended inside its warm-up: carry its statistics, as the batch function does
            self._n = None

    def update(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._n is not None and self._n < self.warmup:
            delta = x - self.mu
            self._n += 1
            self.mu = self.mu + delta / self._n
            self._m2 = self._m2 + delta * (x - self.mu)
            self.var = self._m2 / self._n
            if self._n == self.warmup:
                self._n = None
        else:
            delta = x - self.mu
            self.mu = self.mu + self.alpha * delta
            self.var = (1.0 - self.alpha) * (self.var + self.alpha * delta**2)
        return np.clip((x - self.mu) / (np.sqrt(self.var) + self.eps), -self.clip, self.clip)
