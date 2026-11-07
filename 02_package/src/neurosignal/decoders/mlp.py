"""
Cross-session MLP decoder on joint SBP r_norm + TCR z-score features.

Every fit takes the training sessions explicitly (ERR-02): PCA, the velocity-weight threshold
and both decoders see training sessions only, and scores are computed on whatever sessions the
caller passes. The design matrix is the package spec (D-15) with the MLP's 20-bin window (D-17).

Plots of these fits are in `viz.mlp_plots`.
"""

import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from .linear import SPEC, GramRidge, fit_decoder, session_design


MLP_SPEC = replace(SPEC, spec_id=f"mlp-{SPEC.spec_id}", window_bins=20)  # 400 ms history; warm-up, lag, alphas from SPEC


MLP_ROWS_PER_SESSION = 20_000  # rows drawn from each training session (the Part 3 notebook setting)


def fit_pca_joint(
    sessions_joint: List[np.ndarray],
    train_idx: Sequence[int],
    n_components: int = 5,
    max_bins: int = 50_000,
    random_state: int = 0,
) -> Tuple[PCA, List[np.ndarray]]:
    """
    Fit PCA on a random subsample of the training sessions' post-warm-up bins; transform every session.

    Returns (fitted_pca, sessions_pca) with sessions_pca[i] of shape (T, n_components).
    """
    rng = np.random.default_rng(random_state)
    bins_per = max(1, max_bins // len(train_idx))
    chunks = []
    for i in train_idx:
        arr = sessions_joint[i][MLP_SPEC.warmup_bins:]
        chunks.append(arr[rng.choice(len(arr), size=min(bins_per, len(arr)), replace=False)])
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(np.concatenate(chunks))
    return pca, [pca.transform(arr) for arr in sessions_joint]


def pooled_training_rows(
    sessions_mlp_x: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    train_idx: Sequence[int],
    rng: np.random.Generator,
    rows_per_session: int = MLP_ROWS_PER_SESSION,
    sessions_spec_x: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Stack up to rows_per_session rows from each training session (warm-up trimmed, MLP_SPEC).

    If sessions_spec_x is given, also returns the SPEC rows predicting the same velocity bins,
    so the window-5 Ridge trains on exactly the targets the MLP trains on.
    """
    shift = MLP_SPEC.window_bins - SPEC.window_bins  # MLP row i and SPEC row i + shift share a base bin
    Xs, ys, Vs = [], [], []
    for i in train_idx:
        X, y, _ = session_design(sessions_mlp_x[i], sessions_vel[i], MLP_SPEC)
        rows = np.sort(rng.choice(len(y), size=min(rows_per_session, len(y)), replace=False))
        Xs.append(X[rows])
        ys.append(y[rows])
        if sessions_spec_x is not None:
            X1, y1, _ = session_design(sessions_spec_x[i], sessions_vel[i], SPEC)
            assert np.array_equal(y1[rows + shift], y[rows])
            Vs.append(X1[rows + shift])
    return np.concatenate(Xs), np.concatenate(ys), (np.concatenate(Vs) if Vs else None)


def fit_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (128, 64),
    max_iter: int = 300,
    random_state: int = 0,
    vel_weight_pct: float = 75.0,
    vel_weight_factor: float = 4.0,
) -> MLPRegressor:
    """
    The Part 3 MLP on pooled training rows. Rows with |velocity| above the vel_weight_pct-th
    percentile of the training rows get vel_weight_factor × loss weight, so peaks are not
    averaged toward zero. Early stopping holds out 10 % of the training rows.
    """
    thresh = np.percentile(np.abs(y), vel_weight_pct)
    weights = np.where(np.abs(y) >= thresh, float(vel_weight_factor), 1.0)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        verbose=False,
    )
    model.fit(X, y, sample_weight=weights)
    return model


@dataclass
class PooledDecoders:
    """Every decoder fit on one set of training sessions, plus what it cost."""

    train_idx: Tuple[int, ...]
    pca: PCA
    sessions_pca: List[np.ndarray]
    mlp: MLPRegressor
    ridge_same: GramRidge                 # Ridge on the MLP's exact inputs
    ridge_spec: Optional[GramRidge]       # Ridge, package SPEC, SBP r_norm (None if not fit)
    fit_s: Dict[str, float]

    @property
    def macs_per_bin(self) -> Dict[str, int]:
        """Inference multiply-accumulates per 20 ms bin (PCA once per bin; its outputs are reused across the window)."""
        n_in, n_pc = self.pca.components_.shape[1], self.pca.n_components_
        pca_macs = n_in * n_pc
        layers = [n_pc * (MLP_SPEC.window_bins + 1), *self.mlp.hidden_layer_sizes, 1]
        out = {
            "mlp": pca_macs + sum(a * b for a, b in zip(layers[:-1], layers[1:])),
            "ridge_same": pca_macs + layers[0],
        }
        if self.ridge_spec is not None:
            out["ridge_spec"] = SPEC.decoder_macs_per_bin
        return out


def fit_pooled_decoders(
    sessions_joint: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    train_idx: Sequence[int],
    seed: int,
    sessions_sbp_norm: Optional[List[np.ndarray]] = None,
    n_components: int = 5,
    rows_per_session: int = MLP_ROWS_PER_SESSION,
    **mlp_kwargs,
) -> PooledDecoders:
    """
    Fit PCA, the MLP and Ridge on the MLP's inputs on the training sessions only, from one
    row subsample; with sessions_sbp_norm, also the SPEC Ridge on SBP r_norm on the same target bins.
    The seed sets the PCA subsample, the row subsample and the MLP's initialisation and batches.
    """
    train_idx = tuple(int(i) for i in train_idx)
    t0 = time.time()
    pca, sessions_pca = fit_pca_joint(sessions_joint, train_idx, n_components, random_state=seed)
    rng = np.random.default_rng(seed)
    X, y, X_spec = pooled_training_rows(sessions_pca, sessions_vel, train_idx, rng, rows_per_session, sessions_sbp_norm)
    fit_s = {"features": time.time() - t0}

    t0 = time.time()
    mlp = fit_mlp(X, y, random_state=seed, **mlp_kwargs)
    fit_s["mlp"] = time.time() - t0
    t0 = time.time()
    ridge_same = fit_decoder(X, y, MLP_SPEC)
    fit_s["ridge_same"] = time.time() - t0
    ridge_spec = None
    if X_spec is not None:
        del X
        t0 = time.time()
        ridge_spec = fit_decoder(X_spec, y, SPEC)
        fit_s["ridge_spec"] = time.time() - t0
    return PooledDecoders(train_idx, pca, sessions_pca, mlp, ridge_same, ridge_spec, fit_s)


def score_sessions(
    model,
    sessions_x: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    idx: Sequence[int],
    spec=MLP_SPEC,
) -> np.ndarray:
    """R² of model on every post-warm-up row of each session in idx."""
    r2 = []
    for i in idx:
        X, y, _ = session_design(sessions_x[i], sessions_vel[i], spec)
        r2.append(r2_score(y, model.predict(X)))
    return np.array(r2)


def score_pooled(
    dec: PooledDecoders,
    sessions_vel: List[np.ndarray],
    idx: Sequence[int],
    sessions_sbp_norm: Optional[List[np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Per-session R² of every decoder in dec on the sessions in idx."""
    out = {
        "mlp": score_sessions(dec.mlp, dec.sessions_pca, sessions_vel, idx),
        "ridge_same": score_sessions(dec.ridge_same, dec.sessions_pca, sessions_vel, idx),
    }
    if dec.ridge_spec is not None and sessions_sbp_norm is not None:
        out["ridge_spec"] = score_sessions(dec.ridge_spec, sessions_sbp_norm, sessions_vel, idx, SPEC)
    return out
