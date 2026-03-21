"""
Drift analysis and decoder degradation for LINK-style LINK sessions.

Supports paper draft sections:
- §2.2 Gold-Layer Normalization (feature construction)
- §2.3 Evaluation (pairwise R², decoder horizons)
- §3.1 Static decoders fail rapidly (Day‑0 → Day‑K curves)
- §3.2 Gold normalization eliminates negative transfer (normalization ablations)
- §3.3 Decoder comparison by horizon (r_norm vs baselines)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# Figure style constants — enforced across all package plots
AXIS_LABEL_FONTSIZE = 16   # ax.set_xlabel / ax.set_ylabel
TITLE_FONTSIZE = 18        # ax.set_title / fig.suptitle
LEGEND_FONTSIZE = 11       # ax.legend fontsize

# Default alpha grid for RidgeCV in decoders (used for raw / baseline models).
# Narrow range avoids extreme over‑shrinkage but still spans several orders.
DECODER_ALPHAS = np.logspace(-2, 2, 10)

# Slightly softer grid for Gold (r_norm) decoders, biased toward smaller alphas
# so that amplitudes are not overly damped while cross‑day R² is still optimised.
GOLD_DECODER_ALPHAS = np.logspace(-3, 1, 10)
ROBUST_MAD_SCALE = 1.4826
DEFAULT_ROBUST_BLOCK_SIZE_BINS = 250


# ---------------------------------------------------------------------------
# §2.3 (Methods – Evaluation): windowed, lagged design matrix for decoders
# ---------------------------------------------------------------------------

def _make_xy(
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


# ---------------------------------------------------------------------------
# §2.2 (Methods – Gold-Layer Normalization): r_norm feature construction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# §3.2 (Results – Gold normalization vs static): normalization ablations
# ---------------------------------------------------------------------------


def quantify_norm_improvement(
    R_raw: np.ndarray,
    R_norm: np.ndarray,
    session_dates: List[str],
) -> pd.DataFrame:
    """
    Summary table: Day K, Date, R²_raw (train Day 0 → test Day K), R²_norm, Δ R².
    Also includes mean off-diagonal R² for raw and norm (excluding diagonal).
    """
    N = R_raw.shape[0]
    r0_raw = R_raw[0, :]
    r0_norm = R_norm[0, :]
    df = pd.DataFrame({
        "Day K": range(N),
        "Date": session_dates,
        "R²_raw": np.round(r0_raw, 4),
        "R²_norm": np.round(r0_norm, 4),
        "Δ R² (norm − raw)": np.round(r0_norm - r0_raw, 4),
    })
    off_diag = ~np.eye(N, dtype=bool)
    mean_off_raw = np.nanmean(R_raw[off_diag])
    mean_off_norm = np.nanmean(R_norm[off_diag])
    df.attrs["mean_off_diagonal_R2_raw"] = mean_off_raw
    df.attrs["mean_off_diagonal_R2_norm"] = mean_off_norm
    df.attrs["mean_off_diagonal_delta"] = mean_off_norm - mean_off_raw
    return df


# ---------------------------------------------------------------------------
# §3.3 (Results – Yield): frac_active (% channels with finite σ per day)
# ---------------------------------------------------------------------------


def compute_frac_active_per_session(
    sessions_sbp: List[np.ndarray],
    min_sigma: float = 1e-9,
) -> List[float]:
    """
    For each session (T, C), compute per-channel σ (std over time); count channels
    with σ >= min_sigma and finite. Return frac_active[k] = n_active / n_channels.
    """
    out: List[float] = []
    for sbp in sessions_sbp:
        sbp = np.asarray(sbp, dtype=np.float64)
        sigma = np.nanstd(sbp, axis=0)
        n_ch = sigma.size
        n_active = np.sum(np.isfinite(sigma) & (sigma >= min_sigma))
        out.append(n_active / n_ch if n_ch > 0 else 0.0)
    return out


def _robust_z(x: np.ndarray) -> np.ndarray:
    """
    Robust z-score using median and MAD (scaled to match Gaussian σ).

    Returns an array of the same shape as `x` with NaNs preserved where input
    is non-finite. This helper is used for per-session QC so that a single
    corrupted day does not dominate the scale.
    """
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(med):
        return np.full_like(x, np.nan, dtype=np.float64)
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x, dtype=np.float64)
    z = (x - med) / (ROBUST_MAD_SCALE * mad)
    z[~np.isfinite(x)] = np.nan
    return z


def _compute_intra_r2_cv(
    sessions_sbp: Sequence[np.ndarray],
    sessions_vel: Sequence[np.ndarray],
    alphas: np.ndarray = DECODER_ALPHAS,
    cv: int = 5,
) -> List[float]:
    """
    Per-session within-day decoder R² estimated by k-fold cross-validation.

    For each session k, builds (X, y) via _make_xy and runs cross_val_score
    with a nested RidgeCV estimator (inner fold selects alpha, outer fold
    estimates generalisation R²). Returns a list of length N of mean CV R²
    values (one per session).

    This is the scientifically correct alternative to the in-sample diagonal of
    compute_pairwise_r2_matrix, which inflates R² by scoring on training data.
    compute_pairwise_r2_matrix is left unchanged; it is used for cross-day
    pairwise analyses where train != test by construction.
    """
    out: List[float] = []
    for sbp, vel in zip(sessions_sbp, sessions_vel):
        X, y = _make_xy(sbp, vel)
        if X.shape[0] < cv * 2:
            out.append(float("nan"))
            continue
        cv_scores = cross_val_score(
            RidgeCV(alphas=alphas, cv=cv),
            X,
            y,
            cv=cv,
            scoring="r2",
        )
        out.append(float(np.mean(cv_scores)))
    return out


def qc_sbp_sessions(
    sessions_sbp: Sequence[np.ndarray],
    sessions_vel: Sequence[np.ndarray],
    session_dates: Sequence[str],
    z_thresh: float = 3.5,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Per-session QC metrics for SBP/velocity sessions with robust outlier flags.

    Metrics per session k:
    - frac_active: fraction of channels with finite, non-trivial σ over time.
    - r2_intra_raw: within-day decoder R² from raw SBP, estimated by 5-fold
      cross-validation (nested RidgeCV; inner fold selects alpha, outer fold
      estimates generalisation R²). CV is used instead of in-sample scoring
      to avoid inflation from Ridge overfitting within a session.
    - pop_raw_mean / pop_raw_std: mean and std of population-average SBP(t).
    - frac_zero_var_channels: fraction of channels with exactly zero std.
    - frac_all_nan_channels: fraction of channels whose mean is NaN.

    Robust z-scores are computed for `frac_active`, `r2_intra_raw`, and
    `pop_raw_std`. A session is flagged as a QC outlier if any of these
    |z| > z_thresh. Returns (df_qc, bad_dates).
    """
    sessions_sbp = list(sessions_sbp)
    sessions_vel = list(sessions_vel)
    session_dates = list(session_dates)
    if not (len(sessions_sbp) == len(sessions_vel) == len(session_dates)):
        raise ValueError("sessions_sbp, sessions_vel, and session_dates must have the same length")

    N = len(session_dates)
    frac_active = compute_frac_active_per_session(list(sessions_sbp))
    r2_intra = _compute_intra_r2_cv(sessions_sbp, sessions_vel)

    rows: List[Dict[str, Any]] = []
    for k, date in enumerate(session_dates):
        sbp = np.asarray(sessions_sbp[k], dtype=np.float64)
        if sbp.ndim != 2:
            raise ValueError(f"Session {k} SBP must have shape (T, C), got {sbp.shape}")

        pop_mean_t = np.nanmean(sbp, axis=1)
        pop_raw_mean = float(np.nanmean(pop_mean_t))
        pop_raw_std = float(np.nanstd(pop_mean_t))

        ch_means = np.nanmean(sbp, axis=0)
        ch_stds = np.nanstd(sbp, axis=0)
        n_ch = ch_stds.size if ch_stds.ndim == 1 else 0
        frac_zero_var = float(np.mean(ch_stds == 0)) if n_ch > 0 else 0.0
        frac_all_nan = float(np.mean(np.isnan(ch_means))) if n_ch > 0 else 0.0

        # pop_raw_mean, frac_zero_var_channels, and frac_all_nan_channels are
        # diagnostic-only: they appear in the returned DataFrame for inspection
        # but have no z-score columns and do not enter is_qc_outlier. This is
        # intentional — they provide context rather than triggering auto-flags.
        rows.append(
            {
                "k": k,
                "date": date,
                "frac_active": float(frac_active[k]),
                "r2_intra_raw": float(r2_intra[k]),
                "pop_raw_mean": pop_raw_mean,
                "pop_raw_std": pop_raw_std,
                "frac_zero_var_channels": frac_zero_var,
                "frac_all_nan_channels": frac_all_nan,
            }
        )

    df = pd.DataFrame(rows)

    df["z_frac_active"] = _robust_z(df["frac_active"].to_numpy())
    df["z_r2_intra_raw"] = _robust_z(df["r2_intra_raw"].to_numpy())
    df["z_pop_raw_std"] = _robust_z(df["pop_raw_std"].to_numpy())

    z_cols = ["z_frac_active", "z_r2_intra_raw", "z_pop_raw_std"]
    df["is_qc_outlier"] = (df[z_cols].abs() > float(z_thresh)).any(axis=1)

    bad_dates = df.loc[df["is_qc_outlier"], "date"].astype(str).tolist()
    return df, bad_dates


def filter_sessions_by_bad_dates(
    sessions_sbp: Sequence[np.ndarray],
    sessions_vel: Sequence[np.ndarray],
    session_dates: Sequence[str],
    bad_dates: Sequence[str],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[int]]:
    """
    Filter (sbp, vel, dates) by excluding any session whose date is in bad_dates.

    Returns
    -------
    sessions_sbp_f, sessions_vel_f, session_dates_f, keep_idx
        Filtered lists plus the indices of kept sessions in the original ordering.
    """
    bad = set(map(str, bad_dates))
    keep_idx = [i for i, d in enumerate(session_dates) if str(d) not in bad]
    sessions_sbp_f = [np.asarray(sessions_sbp[i]) for i in keep_idx]
    sessions_vel_f = [np.asarray(sessions_vel[i]) for i in keep_idx]
    session_dates_f = [str(session_dates[i]) for i in keep_idx]
    return sessions_sbp_f, sessions_vel_f, session_dates_f, keep_idx


# ---------------------------------------------------------------------------
# §2.3 / §3.2: simple Procrustes baseline (Aligned‑FA–style manifold alignment)
# ---------------------------------------------------------------------------


def align_sessions_sbp_procrustes(
    sessions_sbp: List[np.ndarray],
    ref_day: int = 0,
) -> List[np.ndarray]:
    """
    Align each session's SBP to ref_day via orthogonal Procrustes (rotation only).
    Uses overlapping time length to fit R; applies (X - mu) @ R + mu_ref to full session.
    Middle-panel warmth: better than raw, typically worse than NoMAD.
    """
    ref = np.asarray(sessions_sbp[ref_day], dtype=np.float64)
    n_ref, n_ch = ref.shape
    mu_ref = np.nanmean(ref, axis=0)
    out: List[np.ndarray] = []
    for k, sbp in enumerate(sessions_sbp):
        sbp = np.asarray(sbp, dtype=np.float64)
        if k == ref_day:
            out.append(sbp.copy())
            continue
        n = min(n_ref, sbp.shape[0])
        A = np.nan_to_num(sbp[:n], nan=0.0) - np.nanmean(sbp[:n], axis=0)
        B = np.nan_to_num(ref[:n], nan=0.0) - np.nanmean(ref[:n], axis=0)
        try:
            R, _ = orthogonal_procrustes(A, B)
        except Exception:
            out.append(sbp.copy())
            continue
        mu_k = np.nanmean(sbp, axis=0)
        aligned = (np.nan_to_num(sbp, nan=0.0) - mu_k) @ R + mu_ref
        out.append(aligned)
    return out


# ---------------------------------------------------------------------------
# §3.1 (Results – Static decoders): Day‑0 → Day‑K decoder R² and stability
# ---------------------------------------------------------------------------


def compute_decoder_r2_per_day(
    sessions_sbp: list[np.ndarray],
    sessions_vel: list[np.ndarray],
    mean_correct: bool = False,
    alphas: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Any]:
    """
    Train Ridge on Day 0, score on each day. Returns (r2_day_k, fitted_model).

    Uses `_make_xy` to optionally include temporal context and neural→behaviour lag.
    By default, we use a short history window and small lag that are reasonable for
    velocity decoding from LINK SBP.

    If mean_correct=True, correct predictions: y_pred - mean(y_pred) + mean(y_true)
    per test session.
    """
    N = len(sessions_sbp)

    # Temporal context: number of past SBP bins to include in X.
    WINDOW_BINS = 5
    # Neural → behaviour lag in bins (predict v(t + lag) from SBP up to time t).
    LAG_BINS = 3

    X0, y0 = _make_xy(
        sessions_sbp[0],
        sessions_vel[0],
        window_bins=WINDOW_BINS,
        lag_bins=LAG_BINS,
    )
    alpha_grid = np.asarray(alphas if alphas is not None else DECODER_ALPHAS, dtype=float)
    model = RidgeCV(alphas=alpha_grid, cv=5)
    model.fit(X0, y0)
    r2_day_k = np.zeros(N)
    for k in range(N):
        X_k, y_k = _make_xy(
            sessions_sbp[k],
            sessions_vel[k],
            window_bins=WINDOW_BINS,
            lag_bins=LAG_BINS,
        )
        y_pred = model.predict(X_k)
        if mean_correct:
            y_pred = y_pred - np.mean(y_pred) + np.mean(y_k)
        r2_day_k[k] = r2_score(y_k, y_pred)
    return r2_day_k, model


# ---------------------------------------------------------------------------
# Gap-based summaries: R² vs days-between-sessions
# ---------------------------------------------------------------------------


def compute_day_offsets(
    session_dates: Sequence[str],
    ref_index: int = 0,
) -> np.ndarray:
    """
    Convert session date strings (YYYYMMDD) to integer day offsets relative to ref_index.

    Parameters
    ----------
    session_dates : sequence of str
        Session date labels as YYYYMMDD strings, one per session in order.
    ref_index : int, default 0
        Index of the reference session (typically Day 0). Its offset is 0.

    Returns
    -------
    np.ndarray
        Array of shape (N,) with day_offsets[k] = days(session_k) − days(session_ref).
    """
    if not session_dates:
        return np.asarray([], dtype=float)
    if ref_index < 0 or ref_index >= len(session_dates):
        raise IndexError(f"ref_index {ref_index} is out of bounds for {len(session_dates)} sessions")

    ref_date = datetime.strptime(session_dates[ref_index], "%Y%m%d")
    offsets = [
        (datetime.strptime(d, "%Y%m%d") - ref_date).days
        for d in session_dates
    ]
    return np.asarray(offsets, dtype=float)


def median_r2_by_gap(
    R: np.ndarray,
    day_offsets: np.ndarray,
    include_zero_gap: bool = True,
) -> pd.DataFrame:
    """
    Aggregate pairwise R² by absolute days-between-sessions.

    For each gap g (integer number of days between train and test sessions),
    collect all pairs (i, j) with |day_offsets[j] − day_offsets[i]| == g and
    compute the median R² along with the number of contributing pairs.

    Parameters
    ----------
    R : np.ndarray, shape (N, N)
        Pairwise R² matrix where R[i, j] is the decoder R² when training on
        session i and testing on session j.
    day_offsets : np.ndarray, shape (N,)
        Day offsets (in days) for each session relative to a reference day.
    include_zero_gap : bool, default True
        If False, exclude g == 0 (i.e. same-day pairs) from the result.

    Returns
    -------
    pd.DataFrame
        Columns:
        - gap_days : integer days between sessions
        - median_r2 : median R² across all pairs with that gap
        - n_pairs   : number of contributing (i, j) pairs
    """
    R = np.asarray(R, dtype=float)
    day_offsets = np.asarray(day_offsets, dtype=float)

    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square, got shape {R.shape}")
    N = R.shape[0]
    if day_offsets.shape != (N,):
        raise ValueError(
            f"day_offsets must have shape ({N},), got {day_offsets.shape}"
        )

    # Absolute day gaps between all train–test pairs.
    gap_matrix = np.abs(day_offsets[None, :] - day_offsets[:, None]).astype(int)

    R_flat = R.ravel()
    gaps_flat = gap_matrix.ravel()

    mask = np.isfinite(R_flat)
    if not include_zero_gap:
        mask &= gaps_flat > 0

    if not np.any(mask):
        return pd.DataFrame(
            {"gap_days": np.asarray([], dtype=int),
             "median_r2": np.asarray([], dtype=float),
             "n_pairs": np.asarray([], dtype=int)}
        )

    gaps_used = np.unique(gaps_flat[mask])
    rows = []
    for g in gaps_used:
        m_g = mask & (gaps_flat == g)
        vals = R_flat[m_g]
        if vals.size == 0:
            continue
        median = float(np.nanmedian(vals))
        n_pairs = int(np.sum(np.isfinite(vals)))
        rows.append(
            {
                "gap_days": int(g),
                "median_r2": round(median, 4),
                "n_pairs": n_pairs,
            }
        )

    if not rows:
        return pd.DataFrame(
            {"gap_days": np.asarray([], dtype=int),
             "median_r2": np.asarray([], dtype=float),
             "n_pairs": np.asarray([], dtype=int)}
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("gap_days").reset_index(drop=True)
    return df


# §3.1 (Results – Signal‑level drift): mean SBP per channel across sessions

# §2.3 / §3.2 (Methods/Results – Pairwise R² matrices)

def compute_pairwise_r2_matrix(
    sessions_sbp: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    mean_correct: bool = False,
) -> np.ndarray:
    """
    For each (train session i, test session j), fit Ridge on session i and score on session j.
    Returns R[i, j] = R² when training on i and testing on j. Shape (N, N).
    If mean_correct=True, correct predictions per test session before R².
    """
    N = len(sessions_sbp)
    R = np.zeros((N, N))
    alpha_grid = DECODER_ALPHAS
    for i in range(N):
        X_i, y_i = _make_xy(sessions_sbp[i], sessions_vel[i])
        model = RidgeCV(alphas=alpha_grid, cv=5)
        model.fit(X_i, y_i)
        for j in range(N):
            X_j, y_j = _make_xy(sessions_sbp[j], sessions_vel[j])
            y_pred = model.predict(X_j)
            if mean_correct:
                y_pred = y_pred - np.mean(y_pred) + np.mean(y_j)
            R[i, j] = r2_score(y_j, y_pred)
    return R


def mask_bad_sessions(R: np.ndarray, bad_session_indices: Sequence[int]) -> np.ndarray:
    """
    Return a copy of a pairwise R² matrix with flagged sessions masked to NaN.

    Masking both the corresponding rows and columns excludes a bad session from
    train→test summaries without mutating the raw matrix kept for auditability.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square, got shape {R.shape}")

    N = R.shape[0]
    masked = R.copy()
    for idx in sorted({int(i) for i in bad_session_indices}):
        if idx < 0 or idx >= N:
            raise IndexError(f"bad session index {idx} out of bounds for matrix of shape {R.shape}")
        masked[idx, :] = np.nan
        masked[:, idx] = np.nan
    return masked


# ---------------------------------------------------------------------------
# Visualization-only helpers: linear gain calibration for decoded traces
# ---------------------------------------------------------------------------


def fit_plot_gain(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fit_intercept: bool = True,
) -> Tuple[float, float]:
    """
    Fit a 1D linear map y ≈ a + g * y_pred for plotting.

    This is intended for visualization of decoded traces only; downstream R²
    metrics should always be computed on the *unscaled* predictions.

    Returns
    -------
    a : float
        Intercept term.
    g : float
        Gain (slope) term.
    """
    y_true = np.ravel(np.asarray(y_true, dtype=np.float64))
    y_pred = np.ravel(np.asarray(y_pred, dtype=np.float64))
    if y_true.size == 0 or y_pred.size == 0:
        return 0.0, 1.0
    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(y_pred.reshape(-1, 1), y_true)
    a = float(reg.intercept_) if fit_intercept else 0.0
    g = float(reg.coef_[0])
    return a, g


def apply_plot_gain(
    y_pred: np.ndarray,
    a: float,
    g: float,
) -> np.ndarray:
    """Apply a previously fitted (a, g) map to predictions for plotting."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return a + g * y_pred


def degradation_table(
    r2_day_k: np.ndarray,
    session_dates: list[str],
) -> pd.DataFrame:
    """Return a DataFrame with Day K, Date, and R² (Day 0 → Day K)."""
    return pd.DataFrame({
        "Day K": range(len(r2_day_k)),
        "Date": session_dates,
        "R² (Day 0 → Day K)": np.round(r2_day_k, 4),
    })


# ---------------------------------------------------------------------------
# §3.3 (Results – Decoder comparison by horizon)
# ---------------------------------------------------------------------------

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


def decoder_comparison_table(
    sessions_sbp: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    session_dates: List[str],
    sessions_tcr: Optional[List[np.ndarray]] = None,
    sigma_bins: float = 2.5,
    include_robust: bool = False,
    robust_block_size_bins: int = DEFAULT_ROBUST_BLOCK_SIZE_BINS,
) -> pd.DataFrame:
    """
    R² per horizon for decoder feature sets built from SBP/TCR summaries.

    Baseline rows are `r_norm`, `raw_mean`, and `pop_rate_norm`. If
    `include_robust=True`, the table also includes robust session-level and
    robust block-level variants for both `r_norm` and `pop_rate_norm`.

    Horizons: intra_day (5-fold CV on Day 0), 1_7_days (train 0, test 1..7), 1_month_plus (train 0, test ≥30 d later).
    If sessions_tcr is None, raw_mean and pop_rate_norm use SBP (column labels still distinguish).
    """
    N = len(sessions_sbp)
    ref_date_str = session_dates[0]
    ref_date = datetime.strptime(ref_date_str, "%Y%m%d")

    # Use TCR if provided, else SBP for raw_mean and pop_rate_norm
    raw_sessions = sessions_tcr if sessions_tcr is not None else sessions_sbp

    # Feature sets: list of (name, list of (T, C) or (T, 1) arrays)
    r_norm_sessions = normalize_sessions_sbp_gold(sessions_sbp, sigma_bins=sigma_bins)
    raw_mean_sessions = _sessions_to_1d_raw_mean(raw_sessions)
    pop_rate_sessions = _sessions_to_pop_rate_norm(
        raw_sessions,
        sigma_bins=sigma_bins,
    )

    feature_sets = [
        ("r_norm", r_norm_sessions),
        ("raw_mean", raw_mean_sessions),
        ("pop_rate_norm", pop_rate_sessions),
    ]
    if include_robust:
        feature_sets.extend(
            [
                (
                    "r_norm_robust_session",
                    normalize_sessions_sbp_robust(
                        sessions_sbp,
                        sigma_bins=sigma_bins,
                        granularity="session",
                    ),
                ),
                (
                    "r_norm_robust_block",
                    normalize_sessions_sbp_robust(
                        sessions_sbp,
                        sigma_bins=sigma_bins,
                        granularity="block",
                        block_size_bins=robust_block_size_bins,
                    ),
                ),
                (
                    "pop_rate_norm_robust_session",
                    _sessions_to_pop_rate_norm(
                        raw_sessions,
                        sigma_bins=sigma_bins,
                        normalization="robust",
                        granularity="session",
                    ),
                ),
                (
                    "pop_rate_norm_robust_block",
                    _sessions_to_pop_rate_norm(
                        raw_sessions,
                        sigma_bins=sigma_bins,
                        normalization="robust",
                        granularity="block",
                        block_size_bins=robust_block_size_bins,
                    ),
                ),
            ]
        )

    horizons = ["intra_day", "1_7_days", "1_month_plus"]
    rows = []

    for feat_name, sessions_feat in feature_sets:
        X0, y0 = _make_xy(sessions_feat[0], sessions_vel[0])

        # Intra-day: 5-fold CV on Day 0
        cv_scores = cross_val_score(
            RidgeCV(alphas=DECODER_ALPHAS, cv=5),
            X0,
            y0,
            cv=5,
            scoring="r2",
        )
        intra_r2 = float(np.mean(cv_scores))

        # 1–7 days: train 0, test 1..min(7, N-1)
        model = RidgeCV(alphas=DECODER_ALPHAS, cv=5)
        model.fit(X0, y0)
        r2_1_7 = []
        for k in range(1, min(8, N)):
            X_k, y_k = _make_xy(sessions_feat[k], sessions_vel[k])
            r2_1_7.append(r2_score(y_k, model.predict(X_k)))
        r2_1_7_val = float(np.mean(r2_1_7)) if r2_1_7 else np.nan

        # 1 month+: train 0, test sessions with date >= ref + 30 days
        r2_month = []
        for k in range(1, N):
            d = datetime.strptime(session_dates[k], "%Y%m%d")
            if (d - ref_date).days >= 30:
                X_k, y_k = _make_xy(sessions_feat[k], sessions_vel[k])
                r2_month.append(r2_score(y_k, model.predict(X_k)))
        r2_month_val = float(np.mean(r2_month)) if r2_month else np.nan

        rows.append({
            "feature_set": feat_name,
            "intra_day": round(intra_r2, 4),
            "1_7_days": round(r2_1_7_val, 4),
            "1_month_plus": round(r2_month_val, 4),
        })

    return pd.DataFrame(rows)
