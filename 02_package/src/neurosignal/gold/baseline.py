"""
The LINK authors' same-day ridge baseline (Temmar et al., NeurIPS 2025; github.com/chesteklab/LINK_dataset,
data_processing/bci_decoding/train_models_singleday.py), and our spec scored on the same trial split (EXP-003).

Their protocol: first 300 trials train, trials 300 → min(n_trials − 1, 399) test; SBP z-scored with
training statistics; 8 bins of history (current + 7 past, zero-padded at the start of each segment), no lag;
sklearn Ridge(alpha=0.1) with intercept on 4 outputs; R² per output.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from .decoder import SPEC, DecoderSpec, fit_decoder, make_xy

PAPER_TRAIN_TRIALS = 300
PAPER_SEQ_LEN = 8
PAPER_ALPHA = 0.1


def paper_split(trial_start_bins: np.ndarray, num_train_trials: int = PAPER_TRAIN_TRIALS) -> Tuple[slice, slice]:
    """(train bins, test bins) of the paper's split: trials before num_train_trials, then up to trial min(n − 1, 399)."""
    n = len(trial_start_bins)
    if n <= num_train_trials:
        raise ValueError(f"{n} trials, need more than {num_train_trials}")
    test_end = trial_start_bins[min(n - 1, 399)]
    return slice(0, int(trial_start_bins[num_train_trials])), slice(int(trial_start_bins[num_train_trials]), int(test_end))


def add_history(X: np.ndarray, seq_len: int) -> np.ndarray:
    """(T, C) → (T, seq_len * C): row t holds X[t-seq_len+1 … t], earlier bins zero before the segment start (as the paper)."""
    T, C = X.shape
    out = np.zeros((T, seq_len, C))
    for k in range(seq_len):
        out[k:, seq_len - 1 - k, :] = X[: T - k]
    return out.reshape(T, seq_len * C)


def zscore_from(train: np.ndarray, *segments: np.ndarray) -> Tuple[np.ndarray, ...]:
    """z-score every segment with the per-channel mean and std of train (flat channels get std 1)."""
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return tuple((s - mu) / sd for s in (train, *segments))


def paper_ridge_r2(sbp: np.ndarray, kin: np.ndarray, trial_start_bins: np.ndarray) -> np.ndarray:
    """R² per output of the paper's ridge on its own split; kin is (T, n_outputs)."""
    tr, te = paper_split(trial_start_bins)
    x_tr, x_te = zscore_from(sbp[tr], sbp[te])
    model = Ridge(alpha=PAPER_ALPHA, fit_intercept=True).fit(add_history(x_tr, PAPER_SEQ_LEN), kin[tr])
    pred = model.predict(add_history(x_te, PAPER_SEQ_LEN))
    return np.array([r2_score(kin[te][:, k], pred[:, k]) for k in range(kin.shape[1])])


def spec_r2_on_paper_split(
    features: np.ndarray,
    vel: np.ndarray,
    trial_start_bins: np.ndarray,
    spec: DecoderSpec = SPEC,
    zscore_with_train: bool = False,
) -> Dict[str, float]:
    """
    Our decoder spec on the paper's split: design matrices built inside each segment (no row spans the
    boundary), the first spec.warmup_bins of the session left out of training, alpha by leave-one-out
    on training rows. zscore_with_train=True gives it the paper's inputs (SBP z-scored with training statistics).
    """
    tr, te = paper_split(trial_start_bins)
    f_tr, f_te = features[tr], features[te]
    if zscore_with_train:
        f_tr, f_te = zscore_from(f_tr, f_te)
    X_tr, y_tr = make_xy(f_tr[spec.warmup_bins:], vel[tr][spec.warmup_bins:], spec.window_bins, spec.lag_bins)
    X_te, y_te = make_xy(f_te, vel[te], spec.window_bins, spec.lag_bins)
    model = fit_decoder(X_tr, y_tr, spec)
    return {"r2": float(r2_score(y_te, model.predict(X_te))), "alpha": float(model.alpha_), "test_rows": len(y_te)}
