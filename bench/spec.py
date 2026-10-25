"""Feature sets scored by the bench; the decoder spec itself lives in the package (D-12, D-15)."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from neurosignal.gold.decoder import SPEC, DecoderSpec  # noqa: F401  (re-exported for bench modules)
from neurosignal.gold.drift import normalize_sessions_sbp_gold
from neurosignal.gold.streaming import normalize_sessions_sbp_gold_causal



# Each feature set maps the full sessions to features; trimming the warm-up happens afterwards.
FEATURES: Dict[str, Callable[[List[np.ndarray]], List[np.ndarray]]] = {
    "raw_sbp": lambda sessions: [np.asarray(s, dtype=np.float64) for s in sessions],
    "r_norm_acausal": lambda sessions: normalize_sessions_sbp_gold(sessions, sigma_bins=2.5),
    "r_norm_causal": lambda sessions: normalize_sessions_sbp_gold_causal(
        sessions, alpha=0.01, alpha_smooth=0.2
    ),
}

# float64 state per channel each feature needs to run online (D-13); None = needs the whole session.
FEATURE_STATE_BYTES_PER_CH: Dict[str, Optional[int]] = {
    "raw_sbp": 0,
    "r_norm_acausal": None,
    "r_norm_causal": 3 * 8,  # EWMA mean, EWMA variance, one-pole smoother
}
