"""Cross-session MLP decoder using joint SBP r_norm + TCR z_norm features."""

from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

from .drift import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TITLE_FONTSIZE,
    _make_xy,
    apply_plot_gain,
    compute_day_offsets,
    fit_plot_gain,
    normalize_sessions_sbp_gold,
    normalize_sessions_sbp_zscore_only,
)

# Match compute_decoder_r2_per_day and plot_decoded_traces_early_late defaults.
_WINDOW_BINS = 20  # 400 ms history (was 5 / 100 ms)
_LAG_BINS = 3


def _normalize_sessions_tcr(
    sessions_tcr: List[np.ndarray],
) -> List[np.ndarray]:
    """Per-channel z-score for TCR — no smoothing (preserves spike event sparsity)."""
    return normalize_sessions_sbp_zscore_only(sessions_tcr)


def _build_joint_features(
    sessions_sbp_norm: List[np.ndarray],
    sessions_tcr_norm: List[np.ndarray],
) -> List[np.ndarray]:
    """Concatenate SBP r_norm and TCR z_norm along the channel axis.

    Returns a list of (T, 2*C) arrays — drop-in replacement for sessions_sbp
    in _make_xy and plot_decoded_traces_early_late.
    """
    return [
        np.concatenate([sbp, tcr], axis=1)
        for sbp, tcr in zip(sessions_sbp_norm, sessions_tcr_norm)
    ]


def fit_pca_joint(
    sessions_sbp_norm: List[np.ndarray],
    sessions_tcr_norm: List[np.ndarray],
    n_components: int = 15,
    max_bins: int = 50_000,
    random_state: int = 42,
) -> Tuple[PCA, List[np.ndarray]]:
    """Fit PCA on a random subsample of joint SBP+TCR features, transform all sessions.

    Returns (fitted_pca, sessions_pca) where each sessions_pca[i] is (T, n_components).
    Pass sessions_pca wherever sessions_joint was used in plot/eval functions.
    """
    sessions_joint = _build_joint_features(sessions_sbp_norm, sessions_tcr_norm)
    rng = np.random.default_rng(random_state)
    bins_per = max(1, max_bins // len(sessions_joint))
    chunks = []
    for arr in sessions_joint:
        n = min(bins_per, len(arr))
        idx = rng.choice(len(arr), size=n, replace=False)
        chunks.append(arr[idx])
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(np.concatenate(chunks))
    sessions_pca = [pca.transform(arr) for arr in sessions_joint]
    evr = pca.explained_variance_ratio_.sum()
    print(f"PCA({n_components}) — {evr * 100:.1f}% variance explained")
    return pca, sessions_pca


def train_mlp_cross_session(
    sessions_sbp_norm: List[np.ndarray],
    sessions_tcr_norm: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    hidden_layer_sizes: Tuple[int, ...] = (128, 64),
    max_iter: int = 300,
    random_state: int = 42,
    max_bins_per_session: int = 1000,
    pca: Optional[PCA] = None,
    vel_weight_pct: float = 75.0,
    vel_weight_factor: float = 4.0,
) -> MLPRegressor:
    """Train one MLP on all sessions concatenated using joint SBP+TCR features.

    Uses window_bins=20, lag_bins=3. Pass pca=pca_model (from fit_pca_joint) to
    apply PCA dimensionality reduction before windowing — reduces features from
    (window+1)*192 to (window+1)*n_components and sharpens trace amplitude.

    max_bins_per_session: random subsample each session to this many rows before
    stacking. With PCA(15)+window=20: 315 features, ~0.1 GB at 20k bins/session.

    vel_weight_pct / vel_weight_factor: bins where |velocity| exceeds the
    vel_weight_pct-th percentile are upweighted by vel_weight_factor. This forces
    the model to learn large peaks instead of averaging them toward zero.
    """
    rng = np.random.default_rng(random_state)
    sessions_joint = _build_joint_features(sessions_sbp_norm, sessions_tcr_norm)
    if pca is not None:
        sessions_joint = [pca.transform(arr) for arr in sessions_joint]
    Xs, ys = [], []
    for joint, vel in zip(sessions_joint, sessions_vel):
        X, y = _make_xy(joint, vel, window_bins=_WINDOW_BINS, lag_bins=_LAG_BINS)
        if not len(X):
            continue
        if max_bins_per_session and len(X) > max_bins_per_session:
            idx = rng.choice(len(X), size=max_bins_per_session, replace=False)
            X, y = X[idx], y[idx]
        Xs.append(X)
        ys.append(y)
    X_all = np.concatenate(Xs)
    y_all = np.concatenate(ys)
    # Upweight high-velocity bins so the model learns peaks, not just the baseline.
    thresh = np.percentile(np.abs(y_all), vel_weight_pct)
    weights = np.where(np.abs(y_all) >= thresh, float(vel_weight_factor), 1.0)
    print(f"Training on {X_all.shape[0]:,} rows × {X_all.shape[1]} features "
          f"({X_all.nbytes / 1e9:.2f} GB) — "
          f"{(weights > 1).sum():,} high-vel bins upweighted ×{vel_weight_factor}")
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
    model.fit(X_all, y_all, sample_weight=weights)
    return model


def eval_mlp_per_day(
    model: MLPRegressor,
    sessions_sbp_norm: List[np.ndarray],
    sessions_tcr_norm: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    pca: Optional[PCA] = None,
) -> Tuple[np.ndarray, MLPRegressor]:
    """Score a trained MLP on each session. Returns (r2_array, model)."""
    sessions_joint = _build_joint_features(sessions_sbp_norm, sessions_tcr_norm)
    if pca is not None:
        sessions_joint = [pca.transform(arr) for arr in sessions_joint]
    r2 = np.zeros(len(sessions_joint))
    for k, (joint, vel) in enumerate(zip(sessions_joint, sessions_vel)):
        X_k, y_k = _make_xy(joint, vel, window_bins=_WINDOW_BINS, lag_bins=_LAG_BINS)
        if len(X_k):
            r2[k] = r2_score(y_k, model.predict(X_k))
    return r2, model


def mlp_comparison_row(
    sessions_sbp: List[np.ndarray],
    sessions_tcr: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    session_dates: List[str],
    sigma_bins: float = 2.5,
) -> pd.Series:
    """Return one row matching decoder_comparison_table format for the cross-session MLP.

    Uses window_bins=0, lag_bins=0 (same as decoder_comparison_table) so the
    numbers are directly comparable to Ridge rows in that table.
    Trains on ALL sessions (cross-session), so intra_day reflects in-sample performance.
    """
    sessions_sbp_norm = normalize_sessions_sbp_gold(sessions_sbp, sigma_bins=sigma_bins)
    sessions_tcr_norm = _normalize_sessions_tcr(sessions_tcr)
    sessions_joint = _build_joint_features(sessions_sbp_norm, sessions_tcr_norm)

    N = len(sessions_joint)
    ref_date = datetime.strptime(session_dates[0], "%Y%m%d")

    # Build full cross-session training set (window=0, lag=0 → matches existing table)
    Xs, ys = [], []
    for joint, vel in zip(sessions_joint, sessions_vel):
        X, y = _make_xy(joint, vel)
        if len(X):
            Xs.append(X)
            ys.append(y)
    X_all = np.concatenate(Xs)
    y_all = np.concatenate(ys)

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=False,
    )
    model.fit(X_all, y_all)

    X0, y0 = _make_xy(sessions_joint[0], sessions_vel[0])
    intra_r2 = float(r2_score(y0, model.predict(X0)))

    r2_1_7 = []
    for k in range(1, min(8, N)):
        X_k, y_k = _make_xy(sessions_joint[k], sessions_vel[k])
        r2_1_7.append(r2_score(y_k, model.predict(X_k)))
    r2_1_7_val = float(np.mean(r2_1_7)) if r2_1_7 else np.nan

    r2_month = []
    for k in range(1, N):
        d = datetime.strptime(session_dates[k], "%Y%m%d")
        if (d - ref_date).days >= 30:
            X_k, y_k = _make_xy(sessions_joint[k], sessions_vel[k])
            r2_month.append(r2_score(y_k, model.predict(X_k)))
    r2_month_val = float(np.mean(r2_month)) if r2_month else np.nan

    return pd.Series({
        "feature_set": "mlp_cross_session_sbp+tcr",
        "intra_day": round(intra_r2, 4),
        "1_7_days": round(r2_1_7_val, 4),
        "1_month_plus": round(r2_month_val, 4),
    })


# ── Multi-day visualization ───────────────────────────────────────────────────

def _get_trace_snippet(
    model: MLPRegressor,
    session_joint: np.ndarray,
    session_vel: np.ndarray,
    window_len: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred_raw) for the most movement-rich window_len-bin snippet.

    y_pred is raw (no gain correction) so callers can compute R² and optionally
    apply gain for visualization.
    """
    X, y = _make_xy(session_joint, session_vel, window_bins=_WINDOW_BINS, lag_bins=_LAG_BINS)
    if len(y) == 0:
        return np.array([]), np.array([])
    window_len = min(window_len, len(y))
    half = max(1, window_len // 2)
    rms = np.array([
        np.sqrt(np.mean(y[i : i + window_len] ** 2))
        for i in range(0, max(1, len(y) - window_len + 1), half)
    ])
    start = int(np.argmax(rms)) * half
    return y[start : start + window_len], model.predict(X[start : start + window_len])


def plot_mlp_r2_timeline(
    r2_per_day: np.ndarray,
    session_dates: List[str],
    selected_indices: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "MLP decoder R² — all sessions",
) -> plt.Figure:
    """Option B: R² vs days-since-implant for all sessions.

    All sessions plotted as viridis-colored dots. Selected sessions highlighted
    with larger markers and day-offset text labels.
    """
    day_offsets = compute_day_offsets(session_dates)
    max_offset = float(max(day_offsets)) or 1.0
    cmap = plt.cm.viridis

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure

    colors = [cmap(d / max_offset) for d in day_offsets]

    for d, r2, c in zip(day_offsets, r2_per_day, colors):
        ax.scatter(d, r2, color=c, s=28, alpha=0.55, zorder=2)

    if selected_indices:
        for i in selected_indices:
            d = day_offsets[i]
            ax.scatter(d, r2_per_day[i], color=colors[i], s=90,
                       edgecolors="k", linewidths=0.8, zorder=3)
            ax.text(d, r2_per_day[i] - 0.025, f"D{d}",
                    ha="center", va="top", fontsize=9, color="dimgray")

    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("Days since implant", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("R²", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlim(left=-3)

    if own_fig:
        fig.tight_layout()
    return fig


def plot_mlp_waterfall(
    model: MLPRegressor,
    sessions_joint: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    session_dates: List[str],
    selected_indices: List[int],
    window_len: int = 500,
    apply_gain: bool = True,
    smooth_sigma: float = 0.0,
    ax: Optional[plt.Axes] = None,
    title: str = "Decoded velocity — selected days",
    row_spacing_factor: float = 1.12,
    trace_fill_fraction: float = 0.38,
) -> plt.Figure:
    """Option A: Stacked velocity trace comparison for selected days.

    Each row is one session. Dark gray = actual velocity; colored line = predicted.
    Color keyed to days-since-implant (viridis). R² annotated on right.

    Traces are vertically scaled per row so they fill most of the band between
    row centers (raw velocity units are preserved for R², not for y-axis position).
    """
    day_offsets = compute_day_offsets(session_dates)
    max_offset = float(max(day_offsets)) or 1.0
    cmap = plt.cm.viridis

    snippets = []
    for i in selected_indices:
        y_true, y_pred_raw = _get_trace_snippet(
            model, sessions_joint[i], sessions_vel[i], window_len
        )
        if len(y_true) == 0:
            continue
        if apply_gain and len(y_true) > 1:
            a, g = fit_plot_gain(y_true, y_pred_raw)
            y_pred_vis = apply_plot_gain(y_pred_raw, a, g)
        else:
            y_pred_vis = y_pred_raw.copy()
        if smooth_sigma > 0:
            y_pred_vis = gaussian_filter1d(y_pred_vis, smooth_sigma)
        snippets.append((i, y_true, y_pred_vis, y_pred_raw))

    if not snippets:
        return plt.subplots()[0]

    amp = float(np.percentile(
        np.abs(np.concatenate([yt for _, yt, _, _ in snippets])), 95
    ))
    amp_safe = max(amp, 1e-9)
    row_spacing = max(amp_safe * row_spacing_factor, 0.28)

    own_fig = ax is None
    if own_fig:
        fig_h = max(2.4, len(snippets) * 0.72 + 0.9)
        fig, ax = plt.subplots(figsize=(10.5, fig_h))
    else:
        fig = ax.figure

    n_bins = len(snippets[0][1])
    x_pad = 52

    for row_idx, (sess_idx, y_true, y_pred_vis, y_pred_raw) in enumerate(snippets):
        offset = row_idx * row_spacing
        d = day_offsets[sess_idx]
        color = cmap(d / max_offset)
        t = np.arange(len(y_true))
        m = float(np.median(y_true))
        loc_amp = float(np.percentile(np.abs(y_true - m), 95))
        loc_amp = max(loc_amp, 1e-9)
        row_scale = (trace_fill_fraction * row_spacing) / loc_amp
        y_t = (y_true - m) * row_scale + offset
        y_p = (y_pred_vis - m) * row_scale + offset
        ax.plot(
            t,
            y_t,
            color="#3a3a3a",
            lw=1.25,
            alpha=0.88,
            ls=(0, (1.0, 1.8)),
            zorder=2,
        )
        ax.plot(t, y_p, color=color, lw=2.0, alpha=1.0, zorder=3, solid_capstyle="round")
        ax.text(
            -8,
            offset,
            f"D{int(d)}",
            ha="right",
            va="center",
            fontsize=AXIS_LABEL_FONTSIZE,
            color="#333333",
        )
        r2_ann = float(r2_score(y_true, y_pred_raw))
        ax.text(
            len(y_true) + 8,
            offset,
            f"R²={r2_ann:.2f}",
            ha="left",
            va="center",
            fontsize=LEGEND_FONTSIZE,
            color=color,
        )

    ax.set_xlim(-x_pad, n_bins + x_pad)
    ax.set_yticks([])
    ax.set_xlabel("Time (bins, 20 ms each)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.spines["left"].set_visible(False)

    if own_fig:
        fig.tight_layout()
    return fig


def plot_mlp_error_heatmap(
    model: MLPRegressor,
    sessions_joint: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    session_dates: List[str],
    selected_indices: List[int],
    window_len: int = 500,
    ax: Optional[plt.Axes] = None,
    title: str = "Prediction error — selected days",
) -> plt.Figure:
    """Option C: Signed prediction error heatmap for selected days.

    Rows = selected sessions, columns = time bins within movement-rich snippet.
    Colormap RdBu_r centered at 0 (red = over-predict, blue = under-predict).
    """
    day_offsets = compute_day_offsets(session_dates)
    error_rows, row_labels = [], []
    for i in selected_indices:
        y_true, y_pred = _get_trace_snippet(
            model, sessions_joint[i], sessions_vel[i], window_len
        )
        if len(y_true) == 0:
            continue
        err = (y_pred - y_true)[:window_len]
        if len(err) < window_len:
            err = np.pad(err, (0, window_len - len(err)))
        error_rows.append(err)
        row_labels.append(f"{session_dates[i]}  D{day_offsets[i]}")

    if not error_rows:
        return plt.subplots()[0]

    E = np.stack(error_rows)
    vmax = float(np.percentile(np.abs(E), 97))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, max(2.5, len(error_rows) * 0.45 + 1.0)))
    else:
        fig = ax.figure

    im = ax.imshow(E, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   origin="upper", interpolation="nearest")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Time (bins, 20 ms each)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    plt.colorbar(im, ax=ax, label="Error (pred − actual)", shrink=0.8)

    if own_fig:
        fig.tight_layout()
    return fig


def plot_mlp_all_days(
    model: MLPRegressor,
    sessions_sbp_norm: List[np.ndarray],
    sessions_tcr_norm: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    session_dates: List[str],
    selected_indices: Optional[List[int]] = None,
    window_len: int = 500,
    apply_gain: bool = True,
    smooth_sigma: float = 0.0,
) -> plt.Figure:
    """Combined 2-panel figure: R² timeline (top) + waterfall traces (bottom, full width).

    Top row (full width): R² vs days since implant for all sessions.
    Bottom row (full width): stacked velocity traces for selected days.
      Actual velocity is shown as a dotted gray line; predicted as a solid colored line.

    If selected_indices is None, auto-selects up to 8 evenly-spaced sessions
    spanning Day 0 through the last session.
    """
    sessions_joint = _build_joint_features(sessions_sbp_norm, sessions_tcr_norm)
    N = len(session_dates)

    if selected_indices is None:
        selected_indices = sorted(
            set(np.round(np.linspace(0, N - 1, min(8, N))).astype(int))
        )

    r2_per_day, _ = eval_mlp_per_day(model, sessions_sbp_norm, sessions_tcr_norm, sessions_vel)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 2.5], hspace=0.4)
    ax_b = fig.add_subplot(gs[0])
    ax_a = fig.add_subplot(gs[1])

    plot_mlp_r2_timeline(r2_per_day, session_dates, selected_indices=selected_indices, ax=ax_b)
    plot_mlp_waterfall(
        model, sessions_joint, sessions_vel, session_dates,
        selected_indices, window_len=window_len, apply_gain=apply_gain,
        smooth_sigma=smooth_sigma, ax=ax_a,
    )

    fig.suptitle("Cross-session MLP decoder — all days", fontsize=TITLE_FONTSIZE, y=1.01)
    return fig
