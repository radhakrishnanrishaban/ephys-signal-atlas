"""Plots of the pooled MLP / Ridge fits in `decoders.mlp` (Part 3 figures)."""

from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from ..decoders.linear import session_design
from ..decoders.mlp import MLP_SPEC, PooledDecoders, score_pooled
from ..metrics.drift import compute_day_offsets
from .style import AXIS_LABEL_FONTSIZE, LEGEND_FONTSIZE, TITLE_FONTSIZE, apply_plot_gain, fit_plot_gain


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
    X, y, _ = session_design(session_joint, session_vel, MLP_SPEC)
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
    title: str = "MLP decoder R² — held-out sessions",
    train_idx: Optional[Sequence[int]] = None,
    r2_reference: Optional[np.ndarray] = None,
    reference_label: str = "Ridge, same inputs",
) -> plt.Figure:
    """R² vs days since the first recorded session (Day 0), one dot per session; NaN entries are not drawn.

    train_idx shades the span of the training sessions, so held-out and training scores
    are never read as the same thing (ERR-02). r2_reference overlays a second decoder.
    Selected sessions get larger markers and day labels.
    """
    day_offsets = compute_day_offsets(session_dates)
    max_offset = float(max(day_offsets)) or 1.0
    cmap = plt.cm.viridis
    r2_per_day = np.asarray(r2_per_day, dtype=float)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 3.4))
    else:
        fig = ax.figure

    colors = [cmap(d / max_offset) for d in day_offsets]
    if train_idx is not None and len(train_idx):
        lo, hi = day_offsets[min(train_idx)], day_offsets[max(train_idx)]
        ax.axvspan(lo - 2, hi + 2, color="0.9", zorder=0, label="training sessions")

    ok = ~np.isnan(r2_per_day)
    ax.scatter(day_offsets[ok], r2_per_day[ok], c=[colors[i] for i in np.flatnonzero(ok)],
               s=36, alpha=0.8, zorder=2, label="MLP")
    if r2_reference is not None:
        r2_reference = np.asarray(r2_reference, dtype=float)
        ok_ref = ~np.isnan(r2_reference)
        ax.scatter(day_offsets[ok_ref], r2_reference[ok_ref], marker="x", color="0.35",
                   s=30, zorder=2, label=reference_label)

    for i in selected_indices or []:
        if np.isnan(r2_per_day[i]):
            continue
        d = day_offsets[i]
        ax.scatter(d, r2_per_day[i], color=colors[i], s=90, edgecolors="k", linewidths=0.8, zorder=3)
        ax.text(d, r2_per_day[i] - 0.025, f"D{int(d)}", ha="center", va="top", fontsize=9, color="dimgray")

    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("Days since first session (Day 0)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("R²", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlim(left=-5)
    ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.75, loc="best")

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
    Color keyed to days since Day 0, the first recorded session (viridis). R² annotated on right.

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
        row_labels.append(f"{session_dates[i]}  D{int(day_offsets[i])}")

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
    dec: PooledDecoders,
    sessions_vel: List[np.ndarray],
    session_dates: List[str],
    selected_indices: Optional[List[int]] = None,
    window_len: int = 500,
    apply_gain: bool = True,
    smooth_sigma: float = 0.0,
) -> plt.Figure:
    """Combined 2-panel figure: held-out R² timeline (top) + waterfall traces (bottom).

    Only sessions outside dec.train_idx are scored and drawn. If selected_indices is None,
    up to 8 evenly spaced held-out sessions are picked.
    """
    held_out = [i for i in range(len(session_dates)) if i not in set(dec.train_idx)]
    if selected_indices is None:
        pick = np.round(np.linspace(0, len(held_out) - 1, min(8, len(held_out)))).astype(int)
        selected_indices = sorted({held_out[k] for k in pick})

    scores = score_pooled(dec, sessions_vel, held_out)
    r2_per_day = np.full(len(session_dates), np.nan)
    r2_ridge = np.full(len(session_dates), np.nan)
    r2_per_day[held_out], r2_ridge[held_out] = scores["mlp"], scores["ridge_same"]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, height_ratios=[1, 2.5], hspace=0.4)
    ax_b = fig.add_subplot(gs[0])
    ax_a = fig.add_subplot(gs[1])

    plot_mlp_r2_timeline(r2_per_day, session_dates, selected_indices=selected_indices, ax=ax_b,
                         train_idx=dec.train_idx, r2_reference=r2_ridge)
    plot_mlp_waterfall(
        dec.mlp, dec.sessions_pca, sessions_vel, session_dates,
        selected_indices, window_len=window_len, apply_gain=apply_gain,
        smooth_sigma=smooth_sigma, ax=ax_a,
    )

    fig.suptitle("Cross-session MLP decoder — held-out sessions", fontsize=TITLE_FONTSIZE, y=1.01)
    return fig
