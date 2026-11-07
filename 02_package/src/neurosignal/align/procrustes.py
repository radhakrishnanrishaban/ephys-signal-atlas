"""Orthogonal Procrustes alignment of session SBP onto a reference session."""

from typing import List

import numpy as np
from scipy.linalg import orthogonal_procrustes


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
