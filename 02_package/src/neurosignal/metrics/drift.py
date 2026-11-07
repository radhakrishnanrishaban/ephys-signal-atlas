"""
Decoder R² across days: Day-0 decay, held-out same-day R², pairwise matrices, R² by date gap and
the horizon table. Every fit uses the package decoder spec (`decoders.linear.SPEC`, D-15); day
labels come from session dates, never session indices (ERR-06).
"""

from datetime import datetime
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from ..decoders.linear import SPEC, DecoderSpec, blocked_folds, fit_decoder, same_day_r2, session_design
from ..features.normalize import (
    DEFAULT_ROBUST_BLOCK_SIZE_BINS,
    _sessions_to_1d_raw_mean,
    _sessions_to_pop_rate_norm,
    normalize_sessions_sbp_gold,
    normalize_sessions_sbp_robust,
)


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


def compute_decoder_r2_per_day(
    sessions_sbp: list[np.ndarray],
    sessions_vel: list[np.ndarray],
    mean_correct: bool = False,
    spec: DecoderSpec = SPEC,
    heldout_day0: bool = True,
) -> tuple[np.ndarray, Any]:
    """
    Train the spec's decoder on Day 0, score every session. Returns (r2_day_k, fitted_model).

    Every session goes through `session_design` (warm-up trimmed, window and lag from the spec,
    D-15). The returned model is fit on every Day-0 row, so its Day-0 score would be in-sample;
    with heldout_day0=True (default) r2_day_k[0] is the mean held-out R² of the spec's blocked
    folds instead (ERR-01). Later days are held out by construction. heldout_day0=False, or
    mean_correct=True, gives the in-sample Day-0 value.

    If mean_correct=True, correct predictions: y_pred - mean(y_pred) + mean(y_true)
    per test session.
    """
    designs = [session_design(f, v, spec)[:2] for f, v in zip(sessions_sbp, sessions_vel)]
    X0, y0 = designs[0]
    model = fit_decoder(X0, y0, spec)
    r2_day_k = np.zeros(len(designs))
    for k, (X_k, y_k) in enumerate(designs):
        y_pred = model.predict(X_k)
        if mean_correct:
            y_pred = y_pred - np.mean(y_pred) + np.mean(y_k)
        r2_day_k[k] = r2_score(y_k, y_pred)
    if heldout_day0 and not mean_correct:
        r2_day_k[0] = float(same_day_r2(X0, y0, spec).mean())
    return r2_day_k, model


def compute_same_day_r2_blocked(
    sbp: np.ndarray,
    vel: np.ndarray,
    spec: DecoderSpec = SPEC,
    offset: int = 0,
) -> dict:
    """
    Held-out same-session R² for the spec's decoder (ERR-01, EXP-001, D-15).

    Rows come from `session_design` (warm-up trimmed); the spec's blocked folds start at
    `offset` rows and wrap, and training drops every row within window + lag of a test row,
    so no SBP or velocity bin is read by both. Alpha is chosen on training rows only.

    Returns fold R² values, their mean, the pooled R² over all held-out predictions,
    and the in-sample R² of the same decoder fit on every row.
    """
    X, y, _ = session_design(sbp, vel, spec)
    y_pred = np.empty(len(y))
    fold_r2 = []
    for train, test in blocked_folds(len(y), spec, offset):
        y_pred[test] = fit_decoder(X[train], y[train], spec).predict(X[test])
        fold_r2.append(r2_score(y[test], y_pred[test]))
    return {
        "fold_r2": np.asarray(fold_r2),
        "r2_heldout": float(np.mean(fold_r2)),
        "r2_heldout_pooled": float(r2_score(y, y_pred)),
        "r2_in_sample": float(r2_score(y, fit_decoder(X, y, spec).predict(X))),
        "n_rows": len(y),
    }


def compute_pairwise_r2_matrix(
    sessions_sbp: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    mean_correct: bool = False,
    spec: DecoderSpec = SPEC,
) -> np.ndarray:
    """
    For each (train session i, test session j), fit the spec's decoder on session i and score
    on session j (D-15). Returns R[i, j], shape (N, N). The diagonal is the held-out blocked
    same-day R² of session i, never a score on training rows (ERR-01).
    If mean_correct=True, correct predictions per test session before R².
    """
    designs = [session_design(f, v, spec)[:2] for f, v in zip(sessions_sbp, sessions_vel)]
    N = len(designs)
    R = np.zeros((N, N))
    for i, (X_i, y_i) in enumerate(designs):
        model = fit_decoder(X_i, y_i, spec)
        for j, (X_j, y_j) in enumerate(designs):
            if i == j:
                R[i, j] = float(same_day_r2(X_i, y_i, spec).mean())
                continue
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


def degradation_table(
    r2_day_k: np.ndarray,
    session_dates: list[str],
) -> pd.DataFrame:
    """Return a DataFrame with session index, date, day (from dates, not index) and R² (Day 0 → session)."""
    return pd.DataFrame({
        "Session": range(len(r2_day_k)),
        "Date": session_dates,
        "Day": compute_day_offsets(session_dates).astype(int),
        "R² (Day 0 → session)": np.round(r2_day_k, 4),
    })


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


def decoder_comparison_table(
    sessions_sbp: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    session_dates: List[str],
    sessions_tcr: Optional[List[np.ndarray]] = None,
    sigma_bins: float = 2.5,
    include_robust: bool = False,
    robust_block_size_bins: int = DEFAULT_ROBUST_BLOCK_SIZE_BINS,
    spec: DecoderSpec = SPEC,
) -> pd.DataFrame:
    """
    R² per horizon for decoder feature sets built from SBP/TCR summaries.

    Baseline rows are `r_norm`, `raw_mean`, and `pop_rate_norm`. If
    `include_robust=True`, the table also includes robust session-level and
    robust block-level variants for both `r_norm` and `pop_rate_norm`.

    Horizons (D-15): intra_day = held-out blocked same-day R² of Day 0; 1_7_days and 1_month_plus =
    mean R² of the Day-0 decoder on later sessions 1–7 and ≥ 30 days after Day 0, by session date
    (it used to take sessions 1–7 by index, which are days 3–28, ERR-12). Decoder from `spec`.
    If sessions_tcr is None, raw_mean and pop_rate_norm use SBP (column labels still distinguish).
    """
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

    gaps = compute_day_offsets(session_dates)
    rows = []
    for feat_name, sessions_feat in feature_sets:
        designs = [session_design(f, v, spec)[:2] for f, v in zip(sessions_feat, sessions_vel)]
        X0, y0 = designs[0]
        model = fit_decoder(X0, y0, spec)

        def mean_r2(lo: float, hi: float) -> float:
            """Mean R² of the Day-0 model on later sessions lo ≤ gap ≤ hi days from Day 0 (dates, ERR-12)."""
            vals = [r2_score(y, model.predict(X)) for k, (X, y) in enumerate(designs) if k and lo <= gaps[k] <= hi]
            return float(np.mean(vals)) if vals else np.nan

        rows.append({
            "feature_set": feat_name,
            "intra_day": round(float(same_day_r2(X0, y0, spec).mean()), 4),
            "1_7_days": round(mean_r2(1, 7), 4),
            "1_month_plus": round(mean_r2(30, np.inf), 4),
        })

    return pd.DataFrame(rows)
