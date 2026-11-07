"""
Session quality screen: fraction of active channels, robust z-scores of session summaries, a quick
intra-session R² check, and exclusion by date. A data-quality check, not a published score; it keeps
its own alpha grid so the frozen splits do not move (D-15).
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

from ..decoders.linear import make_xy
from ..features.normalize import ROBUST_MAD_SCALE

# Alpha grid for the QC screen only (_compute_intra_r2_cv); every evaluation uses decoders.linear.SPEC (D-15).
DECODER_ALPHAS = np.logspace(-2, 2, 10)


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

    For each session k, builds (X, y) via make_xy and runs cross_val_score
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
        X, y = make_xy(sbp, vel)
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
