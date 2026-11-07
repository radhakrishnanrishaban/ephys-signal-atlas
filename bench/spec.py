"""Feature sets scored by the bench; the decoder spec itself lives in the package (D-12, D-15)."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from neurosignal.decoders.linear import SPEC, DecoderSpec  # noqa: F401  (re-exported for bench modules)
from neurosignal.features.normalize import normalize_sessions_sbp_gold
from neurosignal.features.causal import normalize_sessions_sbp_gold_causal
from neurosignal.features.fixed import STATE_BYTES_PER_CH, normalize_sessions_sbp_fixed



# Each feature set maps the full sessions to features; trimming the warm-up happens afterwards.
FEATURES: Dict[str, Callable[[List[np.ndarray]], List[np.ndarray]]] = {
    "raw_sbp": lambda sessions: [np.asarray(s, dtype=np.float64) for s in sessions],
    "r_norm_acausal": lambda sessions: normalize_sessions_sbp_gold(sessions, sigma_bins=2.5),
    "r_norm_causal": lambda sessions: normalize_sessions_sbp_gold_causal(
        sessions, alpha=0.01, alpha_smooth=0.2
    ),
    "r_norm_causal_q": lambda sessions: normalize_sessions_sbp_fixed(sessions, tau_s=2.0, alpha_smooth=0.2),  # EXP-006
}

# State per channel each feature needs to run online (D-13): float64, except the fixed-point row (32-bit integers);
# None = needs the whole session.
FEATURE_STATE_BYTES_PER_CH: Dict[str, Optional[int]] = {
    "raw_sbp": 0,
    "r_norm_acausal": None,
    "r_norm_causal": 3 * 8,  # EWMA mean, EWMA variance, one-pole smoother
    "r_norm_causal_q": STATE_BYTES_PER_CH,  # the same three as int32
}
