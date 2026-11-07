"""
Acausal per-session SBP normalisers: z-score, robust z-score, Gaussian smoothing (`r_norm`, D-02).

Each session is normalised with its own full-session statistics and a two-sided kernel, so the
output at bin t depends on later bins. Offline reference only; the causal versions are in
`features.causal`.
"""

from typing import List, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROBUST_MAD_SCALE = 1.4826
DEFAULT_ROBUST_BLOCK_SIZE_BINS = 250


def zscore_per_channel_day(X: np.ndarray) -> np.ndarray:
    """Z-score along time (axis=0) per channel (axis=1). Returns same shape."""
    X = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sigma = np.nanstd(X, axis=0, keepdims=True)
    sigma = np.where(sigma > 0, sigma, 1.0)
    return np.nan_to_num((X - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_scale(scale: np.ndarray) -> np.ndarray:
    """Replace invalid or nonpositive scales with 1 so normalization stays finite."""
    return np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)


def robust_zscore_per_channel_day(
    X: np.ndarray,
    mad_scale: float = ROBUST_MAD_SCALE,
) -> np.ndarray:
    """
    Robust z-score along time (axis=0) per channel using median and MAD.

    The scale is `MAD * mad_scale`, where the default factor 1.4826 makes MAD
    consistent with Gaussian standard deviation for easier threshold comparison.
    """
    X = np.asarray(X, dtype=np.float64)
    median = np.nanmedian(X, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(X - median), axis=0, keepdims=True)
    scale = _safe_scale(mad * mad_scale)
    return np.nan_to_num((X - median) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def robust_zscore_per_channel_blocks(
    X: np.ndarray,
    block_size_bins: int,
    mad_scale: float = ROBUST_MAD_SCALE,
) -> np.ndarray:
    """Apply robust z-score independently in contiguous time blocks per channel."""
    if block_size_bins <= 0:
        raise ValueError(f"block_size_bins must be > 0, got {block_size_bins}")
    X = np.asarray(X, dtype=np.float64)
    out = np.empty_like(X, dtype=np.float64)
    for start in range(0, X.shape[0], block_size_bins):
        stop = min(start + block_size_bins, X.shape[0])
        out[start:stop] = robust_zscore_per_channel_day(
            X[start:stop],
            mad_scale=mad_scale,
        )
    return out


def _normalize_single_session(
    sbp: np.ndarray,
    normalization: str = "zscore",
    granularity: str = "session",
    block_size_bins: Optional[int] = None,
    mad_scale: float = ROBUST_MAD_SCALE,
) -> np.ndarray:
    """Normalize one `(T, C)` session using the requested scaling method."""
    sbp = np.asarray(sbp, dtype=np.float64)
    if normalization == "zscore":
        return zscore_per_channel_day(sbp)
    if normalization != "robust":
        raise ValueError(f"Unsupported normalization: {normalization}")
    if granularity == "session":
        return robust_zscore_per_channel_day(sbp, mad_scale=mad_scale)
    if granularity == "block":
        if block_size_bins is None:
            raise ValueError("block_size_bins is required when granularity='block'")
        return robust_zscore_per_channel_blocks(
            sbp,
            block_size_bins=block_size_bins,
            mad_scale=mad_scale,
        )
    raise ValueError(f"Unsupported granularity: {granularity}")


def smooth_sbp_gaussian(sbp: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Smooth SBP along time (axis=0) with Gaussian kernel. sigma_bins in time bins."""
    return gaussian_filter1d(
        np.asarray(sbp, dtype=np.float64), sigma=sigma_bins, axis=0, mode="nearest"
    )


def normalize_sessions_sbp_gold(
    sessions_sbp: List[np.ndarray],
    sigma_bins: float = 2.5,
) -> List[np.ndarray]:
    """
    Gold drift-robust SBP: per-channel per-day z-score then 50 ms Gaussian (≈ 2.5 bins at 20 ms).
    Returns list of arrays same shape as input. Use for pairwise R² to warm off-diagonals.
    """
    out: List[np.ndarray] = []
    for sbp in sessions_sbp:
        z = zscore_per_channel_day(sbp)
        out.append(smooth_sbp_gaussian(z, sigma_bins))
    return out


def normalize_sessions_sbp_zscore_only(sessions_sbp: List[np.ndarray]) -> List[np.ndarray]:
    """
    Per-channel per-day z-score only (no Gaussian smoothing). For ablation: compare
    pairwise R² with Gold (z + smooth) to see if smoothing adds value.
    """
    return [zscore_per_channel_day(np.asarray(s, dtype=np.float64)) for s in sessions_sbp]


def normalize_sessions_sbp_robust(
    sessions_sbp: List[np.ndarray],
    sigma_bins: float = 2.5,
    granularity: str = "session",
    block_size_bins: Optional[int] = None,
    mad_scale: float = ROBUST_MAD_SCALE,
) -> List[np.ndarray]:
    """
    Robust alternative to `normalize_sessions_sbp_gold`.

    Applies MAD-based per-channel normalization either once per session/day or
    separately within contiguous time blocks, then Gaussian smooths over time.
    """
    out: List[np.ndarray] = []
    for sbp in sessions_sbp:
        z = _normalize_single_session(
            sbp,
            normalization="robust",
            granularity=granularity,
            block_size_bins=block_size_bins,
            mad_scale=mad_scale,
        )
        out.append(smooth_sbp_gaussian(z, sigma_bins))
    return out


def _sessions_to_1d_raw_mean(sessions: List[np.ndarray]) -> List[np.ndarray]:
    """(T, C) -> (T, 1) per session: mean over channels."""
    return [np.nanmean(s, axis=1, keepdims=True) for s in sessions]


def _sessions_to_pop_rate_norm(
    sessions: List[np.ndarray],
    sigma_bins: float = 2.5,
    normalization: str = "zscore",
    granularity: str = "session",
    block_size_bins: Optional[int] = None,
    mad_scale: float = ROBUST_MAD_SCALE,
) -> List[np.ndarray]:
    """
    Per session: normalize per channel, mean over channels, then Gaussian smooth → (T, 1).

    `normalization='zscore'` reproduces the original behaviour; `normalization='robust'`
    uses the package's MAD-based robust z-score with session or block granularity.
    """
    out: List[np.ndarray] = []
    for s in sessions:
        z = _normalize_single_session(
            s,
            normalization=normalization,
            granularity=granularity,
            block_size_bins=block_size_bins,
            mad_scale=mad_scale,
        )
        pop = np.nanmean(z, axis=1)
        pop_smooth = gaussian_filter1d(pop, sigma=sigma_bins, mode="nearest")
        out.append(pop_smooth.reshape(-1, 1))
    return out


def sessions_to_pop_rate_norm(
    sessions: List[np.ndarray],
    sigma_bins: float = 2.5,
    normalization: str = "zscore",
    granularity: str = "session",
    block_size_bins: Optional[int] = None,
    mad_scale: float = ROBUST_MAD_SCALE,
) -> List[np.ndarray]:
    """Public wrapper for population-rate-style 1D features built from session matrices."""
    return _sessions_to_pop_rate_norm(
        sessions,
        sigma_bins=sigma_bins,
        normalization=normalization,
        granularity=granularity,
        block_size_bins=block_size_bins,
        mad_scale=mad_scale,
    )
