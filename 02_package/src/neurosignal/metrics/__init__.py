"""Decoder R² across days, session QC and session-bootstrap statistics."""

from .drift import (  # noqa: F401
    compute_day_offsets,
    compute_decoder_r2_per_day,
    compute_pairwise_r2_matrix,
    compute_same_day_r2_blocked,
    decoder_comparison_table,
    degradation_table,
    mask_bad_sessions,
    median_r2_by_gap,
    quantify_norm_improvement,
)
from .qc import compute_frac_active_per_session, filter_sessions_by_bad_dates, qc_sbp_sessions  # noqa: F401
from .stats import (  # noqa: F401
    bootstrap_off_diagonal_mean,
    bootstrap_pair_statistic,
    bootstrap_session_mean,
    ci95,
    off_diagonal_mean,
    session_draws,
)

__all__ = [
    "compute_day_offsets",
    "compute_decoder_r2_per_day",
    "compute_pairwise_r2_matrix",
    "compute_same_day_r2_blocked",
    "decoder_comparison_table",
    "degradation_table",
    "mask_bad_sessions",
    "median_r2_by_gap",
    "quantify_norm_improvement",
    "compute_frac_active_per_session",
    "filter_sessions_by_bad_dates",
    "qc_sbp_sessions",
    "bootstrap_off_diagonal_mean",
    "bootstrap_pair_statistic",
    "bootstrap_session_mean",
    "ci95",
    "off_diagonal_mean",
    "session_draws",
]
