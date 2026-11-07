"""Per-session SBP features: acausal `r_norm` (offline reference), causal streaming normalisers v1 and v2, v2 in fixed point, joint SBP + TCR."""

from .causal import (  # noqa: F401
    EwmaNormalizer,
    EwmaNormalizerV2,
    alpha_from_tau,
    causal_zscore_ewma_v2,
    normalize_sessions_sbp_causal_v2,
    causal_smooth_ewma,
    causal_zscore_ewma,
    causal_zscore_window,
    normalize_sessions_sbp_gold_causal,
    stream_session,
)
from .fixed import FixedPointNormalizer, fixed_point_sessions, normalize_sessions_sbp_fixed  # noqa: F401
from .joint import joint_features  # noqa: F401
from .normalize import (  # noqa: F401
    normalize_sessions_sbp_gold,
    normalize_sessions_sbp_robust,
    normalize_sessions_sbp_zscore_only,
    robust_zscore_per_channel_blocks,
    robust_zscore_per_channel_day,
    sessions_to_pop_rate_norm,
    smooth_sbp_gaussian,
    zscore_per_channel_day,
)

__all__ = [
    "EwmaNormalizer",
    "EwmaNormalizerV2",
    "alpha_from_tau",
    "causal_zscore_ewma_v2",
    "normalize_sessions_sbp_causal_v2",
    "causal_smooth_ewma",
    "causal_zscore_ewma",
    "causal_zscore_window",
    "normalize_sessions_sbp_gold_causal",
    "stream_session",
    "FixedPointNormalizer",
    "fixed_point_sessions",
    "normalize_sessions_sbp_fixed",
    "joint_features",
    "normalize_sessions_sbp_gold",
    "normalize_sessions_sbp_robust",
    "normalize_sessions_sbp_zscore_only",
    "robust_zscore_per_channel_blocks",
    "robust_zscore_per_channel_day",
    "sessions_to_pop_rate_norm",
    "smooth_sbp_gaussian",
    "zscore_per_channel_day",
]
