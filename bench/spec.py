"""The single decoder spec and feature sets used by every bench score (D-12)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from neurosignal.gold.drift import normalize_sessions_sbp_gold
from neurosignal.gold.streaming import normalize_sessions_sbp_gold_causal


N_CHANNELS = 96  # LINK SBP channels


@dataclass(frozen=True)
class DecoderSpec:
    spec_id: str = "v0"
    window_bins: int = 5        # X(t) = SBP(t), SBP(t-1), ..., SBP(t-5)
    lag_bins: int = 3           # predict v(t + 3) from SBP up to t
    alphas: tuple = field(default_factory=lambda: tuple(np.logspace(-3, 3, 13)))
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
