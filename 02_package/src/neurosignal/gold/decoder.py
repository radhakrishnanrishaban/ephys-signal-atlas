"""
The one decoder spec every evaluation uses (D-12, D-13, D-15, D-16, D-18): design matrix, warm-up trim,
Ridge with leave-one-out alpha, and purged blocked folds for held-out same-day R².

bench/ and the gold evaluation functions import from here; nothing else defines a decoder.
The MLP (gold/nonlinear.py) uses this spec with a 20-bin window, MLP_SPEC (D-17).
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from sklearn.metrics import r2_score

N_CHANNELS = 96  # LINK SBP channels


@dataclass(frozen=True)
class DecoderSpec:
    spec_id: str = "v2"
    window_bins: int = 5        # X(t) = SBP(t), SBP(t-1), ..., SBP(t-5)
    lag_bins: int = 0           # v2 (D-18): predict v(t) from SBP up to t; v1 predicted v(t + 3)
    alphas: tuple = field(default_factory=lambda: tuple(np.logspace(-6, 8, 29)))  # v1 (D-16); v0 was 1e-3..1e3, 13
    warmup_bins: int = 300      # first bins of every session are never scored (ERR-09)
    same_day_folds: int = 5     # blocked folds for held-out same-day R² (ERR-01)
    kept_threshold: float = 0.5  # CFD: decoder must keep ≥ 50 % of same-day R²
    seeds: tuple = (0, 1, 2)    # each shifts the same-day fold grid (D-13); ≥ 3 required
    n_boot: int = 1000          # session-resampled CFD draws per seed (D-13)

    @property
    def decoder_macs_per_bin(self) -> int:
        """Multiply-accumulates per 20 ms bin for the linear decoder on 96 SBP channels."""
        return N_CHANNELS * (self.window_bins + 1)

    @property
    def purge_rows(self) -> int:
        """Rows dropped from training on each side of a test block, so no SBP or velocity bin is shared."""
        return self.window_bins + self.lag_bins


SPEC = DecoderSpec()


def make_xy(
    sbp: np.ndarray,
    vel: np.ndarray,
    window_bins: int = 0,
    lag_bins: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build decoder X (all SBP channels, with optional temporal window) and y (velocity).

    By default (window_bins=0, lag_bins=0) this matches the original behaviour:
    predict v(t) from SBP(t) only, aligned by min length.

    If window_bins > 0, X(t) stacks SBP over the last `window_bins` bins:
        X(t) = [SBP(t), SBP(t-1), ..., SBP(t-window_bins)]
    flattened over channels with "current first" ordering.

    If lag_bins > 0, predict v(t + lag_bins) from SBP up to time t.
    """
    if window_bins < 0:
        raise ValueError(f"window_bins must be >= 0, got {window_bins}")
    if lag_bins < 0:
        raise ValueError(f"lag_bins must be >= 0, got {lag_bins}")

    X_raw = np.nan_to_num(
        np.asarray(sbp, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    y_full = np.ravel(np.asarray(vel, dtype=np.float64))

    T_sbp, n_ch = X_raw.shape
    T_y = len(y_full)

    # Valid base time indices t satisfy:
    #   t >= window_bins         (enough history)
    #   t < T_sbp                (within SBP length)
    #   t + lag_bins < T_y       (target within velocity length)
    max_t = min(T_sbp - 1, T_y - 1 - lag_bins)
    if max_t < window_bins:
        X_empty = np.empty((0, n_ch * (window_bins + 1 if window_bins > 0 else 1)))
        y_empty = np.empty((0,), dtype=np.float64)
        return X_empty, y_empty

    t_indices = np.arange(window_bins, max_t + 1, dtype=int)

    if window_bins == 0:
        X = X_raw[t_indices]
    else:
        n_samples = len(t_indices)
        n_time = window_bins + 1
        X = np.empty((n_samples, n_ch * n_time), dtype=np.float64)
        for i, t in enumerate(t_indices):
            # Stack [SBP(t), SBP(t-1), ..., SBP(t-window_bins)] with current first.
            window = X_raw[t - window_bins : t + 1][::-1]
            X[i] = window.reshape(-1)

    y = y_full[t_indices + lag_bins]

    return X, y


def n_rows(n_bins: int, spec: DecoderSpec = SPEC) -> int:
    """Rows make_xy builds from a session of n_bins (SBP and velocity of equal length)."""
    return max(0, n_bins - spec.window_bins - spec.lag_bins)


def session_design(features: np.ndarray, vel: np.ndarray, spec: DecoderSpec = SPEC) -> Tuple[np.ndarray, np.ndarray, int]:
    """(X, y, bins) for one session: features and velocity cut to a common length, warm-up trimmed, then make_xy."""
    T = min(len(features), len(vel))
    X, y = make_xy(features[spec.warmup_bins:T], vel[spec.warmup_bins:T], spec.window_bins, spec.lag_bins)
    assert X.shape[0] == n_rows(T - spec.warmup_bins, spec)
    return X, y, T - spec.warmup_bins


def grid_offset(rows: int, seed: int, spec: DecoderSpec = SPEC) -> int:
    """Rows the fold grid is shifted by for this seed (D-13): one draw u ~ U[0, 1) per seed."""
    u = np.random.default_rng(seed).random()
    return int(u * rows / spec.same_day_folds)


def blocked_folds(rows: int, spec: DecoderSpec = SPEC, offset: int = 0) -> List[tuple]:
    """
    Contiguous test blocks on a grid shifted by `offset` rows; the block crossing the end
    wraps to the start. Training drops every row within purge_rows of a test row, so no bin is shared.
    """
    edges = np.linspace(0, rows, spec.same_day_folds + 1).astype(int)
    all_rows = np.arange(rows)
    shifted = (all_rows - offset) % rows if rows else all_rows
    window = np.ones(2 * spec.purge_rows + 1)
    folds = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        is_test = (shifted >= lo) & (shifted < hi)
        near_test = np.convolve(is_test, window, mode="same") > 0
        folds.append((all_rows[~near_test], all_rows[is_test]))
    return folds


class GramRidge:
    """
    Same model as sklearn RidgeCV(alphas, cv=None): centred X and y, unpenalised intercept,
    alpha with the lowest mean squared leave-one-out error (first one on ties), training rows only.
    One eigendecomposition of XᵀX serves every alpha, instead of an SVD of X per fit:
    with Q = X_c V and d = 1/(s + α), fitted = Q (d ∘ Qᵀy_c) and leverage h = (Q∘Q) d + 1/n.
    """

    def __init__(self, alphas: tuple):
        self.alphas = alphas

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GramRidge":
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        s, V = np.linalg.eigh(Xc.T @ Xc)
        Q = Xc @ V
        Qy, Q2, n = Q.T @ yc, Q * Q, len(y)
        best_err, best_d = np.inf, None
        for alpha in self.alphas:
            d = 1.0 / (s + alpha)
            loo = (yc - Q @ (d * Qy)) / (1.0 - (Q2 @ d + 1.0 / n))
            err = float(np.mean(loo**2))
            if err < best_err:
                best_err, best_d, self.alpha_ = err, d, alpha
        self.coef_ = V @ (best_d * Qy)
        self.intercept_ = y_mean - x_mean @ self.coef_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_


def fit_decoder(X: np.ndarray, y: np.ndarray, spec: DecoderSpec = SPEC) -> GramRidge:
    return GramRidge(spec.alphas).fit(X, y)


def same_day_r2(X: np.ndarray, y: np.ndarray, spec: DecoderSpec = SPEC, offset: int = 0) -> np.ndarray:
    """Held-out R² of each blocked fold of one session (mean = the CFD denominator)."""
    return np.array([
        r2_score(y[test], fit_decoder(X[train], y[train], spec).predict(X[test]))
        for train, test in blocked_folds(len(y), spec, offset)
    ])
