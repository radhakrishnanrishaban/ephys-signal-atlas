"""Joint SBP `r_norm` + TCR z-score channels, the MLP's input before PCA (D-17)."""

from typing import List, Tuple

import numpy as np

from .normalize import normalize_sessions_sbp_gold, normalize_sessions_sbp_zscore_only


def _normalize_sessions_tcr(
    sessions_tcr: List[np.ndarray],
) -> List[np.ndarray]:
    """Per-channel z-score for TCR — no smoothing (preserves spike event sparsity)."""
    return normalize_sessions_sbp_zscore_only(sessions_tcr)


def _build_joint_features(
    sessions_sbp_norm: List[np.ndarray],
    sessions_tcr_norm: List[np.ndarray],
) -> List[np.ndarray]:
    """Concatenate SBP r_norm and TCR z_norm along the channel axis: a (T, 2*C) array per session."""
    return [
        np.concatenate([sbp, tcr], axis=1)
        for sbp, tcr in zip(sessions_sbp_norm, sessions_tcr_norm)
    ]


def joint_features(
    sessions_sbp: List[np.ndarray],
    sessions_tcr: List[np.ndarray],
    sigma_bins: float = 2.5,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """(joint SBP r_norm + TCR z-score, SBP r_norm) per session; each session uses its own statistics (acausal)."""
    sbp_norm = normalize_sessions_sbp_gold(sessions_sbp, sigma_bins=sigma_bins)
    return _build_joint_features(sbp_norm, _normalize_sessions_tcr(sessions_tcr)), sbp_norm
