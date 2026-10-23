"""
Plotting helpers for drift analysis.

Numeric APIs live in `drift.py`; this module contains only visualization helpers.
"""

from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from matplotlib import pyplot as plt

from .drift import (
    AXIS_LABEL_FONTSIZE,
    TITLE_FONTSIZE,
    LEGEND_FONTSIZE,
    DEFAULT_ROBUST_BLOCK_SIZE_BINS,
    GOLD_DECODER_ALPHAS,
    _make_xy,
    align_sessions_sbp_procrustes,
    apply_plot_gain,
    compute_day_offsets,
    compute_decoder_r2_per_day,
    compute_frac_active_per_session,
    compute_pairwise_r2_matrix,
    fit_plot_gain,
    median_r2_by_gap,
    normalize_sessions_sbp_gold,
    normalize_sessions_sbp_robust,
    normalize_sessions_sbp_zscore_only,
)


def plot_session_channel_heatmap(
    mean_per_session: Sequence[np.ndarray],
    session_dates: Sequence[str],
    *,
    cmap: str = "viridis",
    robust_clip: Optional[tuple[float, float]] = (1.0, 99.0),
    days_on_x: bool = False,
    ax: Any = None,
) -> Any:
    """
    Heatmap: sessions (rows) × channels (cols) showing mean SBP per channel.

    Y-axis is labeled by Day K (0..N-1). `session_dates` is accepted for
    consistency with other plot helpers and for future annotation, but the
    default y tick labels are Day indices.

    Parameters
    ----------
    mean_per_session:
        Sequence of arrays shaped (n_ch,), one per session.
    session_dates:
        Sequence of date strings (YYYYMMDD) aligned with mean_per_session.
    cmap:
        Matplotlib colormap name.
    robust_clip:
        If provided, clip color scale to these percentiles (low, high) computed
        over all finite matrix entries. Use None to disable.
    days_on_x:
        If False (default), x-axis is Channel and y-axis is Day K.
        If True, x-axis is Day K and y-axis is Channel (transposed view).
    ax:
        Optional matplotlib Axes to plot into.
    """
    if len(mean_per_session) != len(session_dates):
        raise ValueError("mean_per_session and session_dates must have the same length")
    if len(mean_per_session) == 0:
        raise ValueError("mean_per_session is empty")

    M = np.vstack([np.ravel(np.asarray(v, dtype=np.float64)) for v in mean_per_session])
    if M.ndim != 2:
        raise ValueError(f"Expected stacked mean matrix to be 2D, got shape {M.shape}")

    # Orientation: (sessions, channels) or transposed (channels, sessions)
    M_plot = M.T if days_on_x else M

    vmin = vmax = None
    finite = np.isfinite(M_plot)
    if robust_clip is not None and np.any(finite):
        lo, hi = robust_clip
        if not (0 <= lo < hi <= 100):
            raise ValueError("robust_clip must be percentiles (0 <= lo < hi <= 100)")
        vmin = float(np.nanpercentile(M_plot[finite], lo))
        vmax = float(np.nanpercentile(M_plot[finite], hi))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = vmax = None

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
    else:
        fig = ax.figure

    im = ax.imshow(
        M_plot,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    if days_on_x:
        ax.set_xlabel("Day K", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Channel", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title("Mean SBP (Channel×Session)", fontsize=TITLE_FONTSIZE)

        # Sparse x ticks for days
        N_days = M_plot.shape[1]
        step = 1
        if N_days > 40:
            step = 10
        elif N_days > 20:
            step = 5
        xticks = np.arange(0, N_days, step)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"Day {k}" for k in xticks], rotation=0)

        # Sparse y ticks for channels
        n_ch = M_plot.shape[0]
        ch_step = 10 if n_ch >= 40 else 5
        yticks = np.arange(0, n_ch, ch_step)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(c)) for c in yticks])
    else:
        ax.set_xlabel("Channel", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Day K", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title("Mean SBP per channel (Session×Channel)", fontsize=TITLE_FONTSIZE)

        # Sparse y ticks to keep labels readable
        N_days = M_plot.shape[0]
        step = 1
        if N_days > 40:
            step = 10
        elif N_days > 20:
            step = 5
        yticks = np.arange(0, N_days, step)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"Day {k}" for k in yticks])

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean SBP (20 ms bins)")

    fig.tight_layout()
    return fig


def _flatten_r2_vs_gap(
    R: np.ndarray,
    day_offsets: Sequence[float],
    include_zero_gap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite pairwise gap/R² samples for scatter and binned summaries."""
    day_offsets = np.asarray(day_offsets, dtype=float)
    R = np.asarray(R, dtype=float)

    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square, got shape {R.shape}")
    if day_offsets.shape != (R.shape[0],):
        raise ValueError(
            f"day_offsets must have shape ({R.shape[0]},), got {day_offsets.shape}"
        )

    gap_matrix = np.abs(day_offsets[None, :] - day_offsets[:, None]).astype(float)
    gaps_flat = gap_matrix.ravel()
    r2_flat = R.ravel()

    mask = np.isfinite(gaps_flat) & np.isfinite(r2_flat)
    if not include_zero_gap:
        mask &= gaps_flat > 0
    return gaps_flat[mask], r2_flat[mask]


def _binned_r2_summary(
    gaps_flat: np.ndarray,
    r2_flat: np.ndarray,
    *,
    bin_width_days: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Summarize pairwise R² in gap bins using median and IQR."""
    if gaps_flat.size == 0 or r2_flat.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty

    if bin_width_days <= 0:
        raise ValueError("bin_width_days must be positive")

    max_gap = float(np.nanmax(gaps_flat))
    edges = np.arange(0.0, max_gap + bin_width_days, bin_width_days, dtype=float)
    if edges.size < 2:
        edges = np.asarray([0.0, max_gap + bin_width_days], dtype=float)
    elif edges[-1] <= max_gap:
        edges = np.append(edges, edges[-1] + bin_width_days)

    centers: list[float] = []
    medians: list[float] = []
    q25s: list[float] = []
    q75s: list[float] = []

    for left, right in zip(edges[:-1], edges[1:]):
        if right == edges[-1]:
            mask = (gaps_flat >= left) & (gaps_flat <= right)
        else:
            mask = (gaps_flat >= left) & (gaps_flat < right)
        vals = r2_flat[mask]
        gaps_bin = gaps_flat[mask]
        if vals.size == 0:
            continue
        centers.append(float(np.nanmedian(gaps_bin)))
        medians.append(float(np.nanmedian(vals)))
        q25s.append(float(np.nanpercentile(vals, 25)))
        q75s.append(float(np.nanpercentile(vals, 75)))

    return (
        np.asarray(centers, dtype=float),
        np.asarray(medians, dtype=float),
        np.asarray(q25s, dtype=float),
        np.asarray(q75s, dtype=float),
    )


def plot_signal_degradation_overlaid(
    mean_per_session: list[np.ndarray],
    session_dates: list[str],
    n_ch: int,
    ax: Any = None,
) -> Any:
    """
    One plot: mean SBP vs channel with one line per session (overlaid).
    Returns the figure; caller can plt.show() or save.
    """
    N = len(mean_per_session)
    colors = plt.cm.tab10(np.linspace(0, 0.9, N))
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure
    for k in range(N):
        if k == 0 or k == N - 1 or k % 2 == 0:
            label = f"Day {k}"
        else:
            label = "_nolegend_"
        ax.plot(
            np.arange(n_ch),
            mean_per_session[k],
            alpha=0.8,
            label=label,
            color=colors[k],
            linewidth=1.5,
        )
    ax.set_xlabel("Channel", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Mean SBP (20 ms bins)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Mean SBP per channel", fontsize=TITLE_FONTSIZE)
    # Put legend fully below x-axis label, centered
    ax.legend(
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.25),
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        framealpha=0.75,
    )
    ax.set_xlim(0, n_ch - 1)
    fig.tight_layout(rect=(0, 0.14, 1, 1))  # extra bottom margin for legend
    return fig


def plot_signal_degradation_mean_sem(
    mean_per_session: List[np.ndarray],
    session_dates: List[str],
    n_ch: int,
    ax: Any = None,
) -> Any:
    """
    Publication-style: one line = mean SBP vs channel, shaded band = SEM across sessions.
    Returns the figure; caller can plt.show() or save.
    """
    stacked = np.array(mean_per_session)
    n_sessions = stacked.shape[0]
    mean_sbp = np.mean(stacked, axis=0)
    sem_sbp = np.std(stacked, axis=0) / np.sqrt(n_sessions) if n_sessions > 1 else np.zeros_like(mean_sbp)
    channel_ix = np.arange(n_ch)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure
    ax.plot(channel_ix, mean_sbp, color="steelblue", linewidth=2, label="Mean SBP")
    ax.fill_between(
        channel_ix,
        mean_sbp - sem_sbp,
        mean_sbp + sem_sbp,
        alpha=0.3,
        color="steelblue",
    )
    ax.set_xlabel("Channel", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Mean SBP (20 ms bins)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Signal-level degradation: Mean SBP ± SEM across sessions", fontsize=TITLE_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.75)
    ax.set_xlim(0, n_ch - 1)
    plt.tight_layout()
    return fig


def _scatter_day0_vs_k(
    mean_0: np.ndarray,
    mean_k: np.ndarray,
    day_k_label: str,
    date_k: str,
    ax: Any,
) -> None:
    """One panel: Day 0 (x) vs Day k (y). Coral = decay, teal = increase. Modifies ax in place."""
    diff = mean_k - mean_0
    tol = 0.1 * (float(np.nanmax(np.abs(diff))) if np.any(np.isfinite(diff)) else 1.0)
    mask_below = diff < -tol
    mask_above = diff > tol
    mask_same = ~mask_below & ~mask_above
    ax.scatter(mean_0[mask_same], mean_k[mask_same], c="gray", alpha=0.5, s=15, label="No change")
    ax.scatter(mean_0[mask_below], mean_k[mask_below], c="coral", alpha=0.7, s=20, label="Decay")
    ax.scatter(mean_0[mask_above], mean_k[mask_above], c="teal", alpha=0.7, s=20, label="Increase")
    lim = max(float(np.nanmax(mean_0)), float(np.nanmax(mean_k)), 1e-6) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1)
    ax.set_xlabel("Day 0", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(f"{day_k_label}\n{date_k}", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_aspect("equal")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)


def plot_per_channel_drift_small_multiples(
    mean_per_session: List[np.ndarray],
    session_dates: List[str],
    day_indices: Optional[List[int]] = None,
    out_path: Optional[Union[Path, str]] = None,
) -> Any:
    """
    Small multiples: each panel is Day 0 (x) vs a later session (y) at increasing gaps.
    Points scatter further from the diagonal as the gap grows. Coral = decay, teal = increase.
    If out_path is set, saves figure there. Returns the figure.
    """
    N = len(mean_per_session)
    if N < 2:
        raise ValueError("Need at least two sessions for per-channel drift small multiples.")
    if day_indices is None:
        if N <= 4:
            day_indices = list(range(1, N))
        else:
            day_indices = [1, max(1, N // 4), N // 2, max(N // 2 + 1, 3 * N // 4), N - 1]
            day_indices = sorted(set(day_indices))
    day_indices = [k for k in day_indices if 0 < k < N]
    if not day_indices:
        day_indices = [N - 1] if N > 1 else []
    n_panels = len(day_indices)
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_panels == 1:
        axs = np.array([axs])
    axs = axs.ravel()
    mean_0 = mean_per_session[0]
    for i, k in enumerate(day_indices):
        _scatter_day0_vs_k(
            mean_0,
            mean_per_session[k],
            f"Day {k}",
            session_dates[k],
            axs[i],
        )
    for j in range(n_panels, len(axs)):
        axs[j].set_visible(False)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_day0_vs_last_scatter(
    mean_0: np.ndarray,
    mean_last: np.ndarray,
    session_dates: list[str],
    N: int,
    ax: Any = None,
) -> Any:
    """
    Per-channel mean SBP: Day 0 vs last. Each point = one channel.
    Legend uses dynamic last day (Day N-1), not hardcoded Day 3.
    Returns the figure.
    """
    last_label = f"Day {N-1}"
    diff = mean_last - mean_0
    tol = 0.1 * (float(np.nanmax(np.abs(diff))) if np.any(np.isfinite(diff)) else 1.0)
    mask_below = diff < -tol
    mask_above = diff > tol
    mask_same = ~mask_below & ~mask_above
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    ax.scatter(mean_0[mask_same], mean_last[mask_same], c="gray", alpha=0.5, s=25, label="No change")
    ax.scatter(
        mean_0[mask_below],
        mean_last[mask_below],
        c="coral",
        alpha=0.7,
        s=30,
        label=f"Decay/dropout ({last_label} < Day 0)",
    )
    ax.scatter(
        mean_0[mask_above],
        mean_last[mask_above],
        c="seagreen",
        alpha=0.7,
        s=30,
        label=f"Increase ({last_label} > Day 0)",
    )
    lim = max(float(np.nanmax(mean_0)), float(np.nanmax(mean_last))) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y = x (no change)")
    ax.set_xlabel(f"Mean SBP — Day 0 ({session_dates[0]})", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(f"Mean SBP — {last_label} ({session_dates[-1]})", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Per-channel mean SBP: Day 0 vs last\n(each point = one channel; below line = decay/dropout)", fontsize=TITLE_FONTSIZE)
    ax.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=0.75)
    ax.set_aspect("equal")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    plt.tight_layout()
    return fig


def plot_decoder_degradation(
    r2_day_k: np.ndarray,
    session_dates: list[str],
    out_path: Union[Path, str, None] = None,
    ax: Any = None,
    clip_floor: Optional[float] = None,
    clip_ceiling: float = 1.0,
    ylim: Optional[tuple[float, float]] = None,
) -> Any:
    """
    Bar chart: R² per day (trained on Day 0). Adaptive ylim when R² are near zero.
    If out_path is set, saves figure there. Returns the figure.
    """
    r2_day_k = np.asarray(r2_day_k, dtype=np.float64)
    r2_plot = np.clip(r2_day_k, clip_floor, clip_ceiling) if clip_floor is not None else r2_day_k
    N = len(r2_plot)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    if N <= 8:
        tick_positions = np.arange(N)
    else:
        # Keep the axis readable for longer session series while always showing Day 0 and the last day.
        tick_positions = np.unique(np.linspace(0, N - 1, num=6, dtype=int))
    x_labels = [f"Day {k}" for k in tick_positions]
    ax.bar(range(N), r2_plot, color=["steelblue"] + ["coral"] * (N - 1), alpha=0.8)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("R²", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Decoder degradation on Day K (trained on Day 0 only)", fontsize=TITLE_FONTSIZE)
    ax.axhline(r2_plot[0], color="gray", linestyle="--", alpha=0.7, label="Day 0 baseline")
    ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.75)

    if ylim is not None:
        # Caller requests explicit y-range (e.g. [-10, 0.5]); skip adaptive logic.
        ax.set_ylim(*ylim)
        if ylim[0] < 0 < ylim[1]:
            ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    else:
        r2_range = np.ptp(r2_plot)
        if r2_range < 0.1:
            pad = max(0.005, r2_range * 0.5)
            ax.set_ylim(r2_plot.min() - pad, r2_plot.max() + pad)
            ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
        else:
            y_min = clip_floor if clip_floor is not None else r2_plot.min() - 0.05
            ax.set_ylim(min(0, y_min), clip_ceiling)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_decoder_r2_boxplot_by_gap(
    r2_day_k: np.ndarray,
    session_dates: Sequence[str],
    *,
    gap_bins: Optional[Sequence[tuple[int, int, str]]] = None,
    ylim: Optional[tuple[float, float]] = None,
    ax: Any = None,
) -> Any:
    """
    Boxplot of Day-0→Day-K decoder R² grouped by time-gap bins.

    Parameters
    ----------
    r2_day_k :
        Array of shape (N,) with R² for train Day 0 → test Day K.
    session_dates :
        Sequence of YYYYMMDD strings, one per session in the same order as r2_day_k.
    gap_bins :
        Optional sequence of (lo_days, hi_days, label). Default:
        (0, 7, "0–7 d"), (8, 30, "8–30 d"), (31, 365, ">30 d").
    ylim :
        Optional explicit y-axis limits (ymin, ymax).
    ax :
        Optional matplotlib Axes to plot into.
    """
    r2_day_k = np.asarray(r2_day_k, dtype=np.float64)
    if r2_day_k.ndim != 1:
        raise ValueError(f"r2_day_k must be 1D, got shape {r2_day_k.shape}")
    if len(session_dates) != r2_day_k.shape[0]:
        raise ValueError("session_dates length must match r2_day_k")

    # Compute day gaps relative to Day 0 using existing helper.
    gaps_days = compute_day_offsets(list(session_dates), ref_index=0)

    if gap_bins is None:
        gap_bins = [
            (0, 7, "0–7 d"),
            (8, 30, "8–30 d"),
            (31, 365, ">30 d"),
        ]

    data_per_bin: list[np.ndarray] = []
    labels: list[str] = []
    for lo, hi, label in gap_bins:
        mask = (gaps_days >= lo) & (gaps_days <= hi)
        vals = r2_day_k[mask]
        if vals.size == 0:
            continue
        data_per_bin.append(vals)
        labels.append(label)

    if not data_per_bin:
        raise ValueError("No data available for any gap bin; check gap_bins or inputs.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4))
    else:
        fig = ax.figure

    ax.boxplot(
        data_per_bin,
        labels=labels,
        showfliers=True,
        patch_artist=True,
    )
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylabel("R² (train Day 0 → test K)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("Time gap bin (days from Day 0)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Decoder R² by time-gap bin", fontsize=TITLE_FONTSIZE)

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        # Adaptive y-limits: wrap the observed R² values with a small margin so
        # the axis is tighter than a fixed [-10, 0.5] while still leaving headroom.
        all_vals = np.concatenate(data_per_bin)
        finite = all_vals[np.isfinite(all_vals)]
        if finite.size > 0:
            y_min = float(np.nanmin(finite))
            y_max = float(np.nanmax(finite))
            # Ensure we always show at least modest positive headroom to be
            # consistent with other R² plots in this module.
            y_max = max(y_max, 0.5)
            # Allow for small negatives even if data are mostly ≥ 0.
            y_min = min(y_min, -0.1)
            span = y_max - y_min if y_max > y_min else 1.0
            pad = 0.03 * span
            ax.set_ylim(y_min - pad, y_max + pad)

    plt.tight_layout()
    return fig

def plot_pairwise_r2_heatmap(
    R: np.ndarray,
    session_dates: List[str],
    ax: Any = None,
    out_path: Optional[Union[Path, str]] = None,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    clip_floor: Optional[float] = None,
    clip_ceiling: float = 1.0,
   cmap: str = "hot", 
) -> Any:
    """
    Heatmap of pairwise R²: rows = train session, cols = test session.
    If out_path is set, saves figure there. Returns the figure.
    vmin/vmax: shared scale for side-by-side plots (e.g. -0.4, 1.0).
    """
    N = R.shape[0]
    labels = [f"Day {k}" for k in range(N)]
    if N <= 8:
        tick_positions = np.arange(N)
    else:
        # Keep both axes readable for longer session series while always
        # showing Day 0 and the last day.
        tick_positions = np.unique(np.linspace(0, N - 1, num=6, dtype=int))
    tick_labels = [f"Day {k}" for k in tick_positions]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    R_plot = np.clip(R, clip_floor, clip_ceiling) if clip_floor is not None else np.asarray(R, dtype=np.float64)
    r_min, r_max = np.nanmin(R_plot), np.nanmax(R_plot)
    vmin = vmin if vmin is not None else min(0.0, r_min - 0.05)
    vmax = vmax if vmax is not None else max(0.5, r_max + 0.05)
    im = ax.imshow(R_plot, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Test session", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Train session", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title or "Pairwise decoder R² (train → test)", fontsize=TITLE_FONTSIZE)
    if show_colorbar:
        plt.colorbar(im, ax=ax, label="R²")
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig

def _shared_pairwise_r2_limits(
    *matrices: np.ndarray,
    clip_floor: Optional[float] = None,
    clip_ceiling: float = 1.0,
) -> tuple[float, float]:
    """Choose a common visual range for one or more pairwise R² heatmaps."""
    finite = []
    for m in matrices:
        if m is None:
            continue
        arr = np.asarray(m, dtype=np.float64)
        if clip_floor is not None:
            arr = np.clip(arr, clip_floor, clip_ceiling)
        finite.append(arr)
    if not finite:
        return -0.05, 0.5
    vmin = min(np.nanmin(m) for m in finite)
    vmax = max(np.nanmax(m) for m in finite)
    if vmin > -0.5:
        vmin = min(vmin, -0.05)
    if vmax < 0.5:
        vmax = max(vmax, 0.5)
    return float(vmin), float(vmax)


def plot_pairwise_r2_raw_vs_norm(
    R_raw: np.ndarray,
    R_norm: np.ndarray,
    session_dates: List[str],
    out_path: Optional[Union[Path, str]] = None,
    title_raw: str = "Static (raw SBP)",
    title_norm: str = "Gold (z-score + 50 ms smooth)",
    clip_floor: Optional[float] = -5.0,
    clip_ceiling: float = 1.0,
    cmap: str = "hot",
) -> Any:
    """
    Side-by-side heatmaps: Raw pairwise R² vs Gold-normalized pairwise R².
    Shared color scale (vmin/vmax from both matrices) for visual comparison.
    """
    vmin, vmax = _shared_pairwise_r2_limits(
        R_raw,
        R_norm,
        clip_floor=clip_floor,
        clip_ceiling=clip_ceiling,
    )
    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(14, 6))
    plot_pairwise_r2_heatmap(
        R_raw,
        session_dates,
        ax=ax_raw,
        title=title_raw,
        vmin=vmin,
        vmax=vmax,
        clip_floor=clip_floor,
        clip_ceiling=clip_ceiling,
        cmap=cmap,
    )
    plot_pairwise_r2_heatmap(
        R_norm,
        session_dates,
        ax=ax_norm,
        title=title_norm,
        vmin=vmin,
        vmax=vmax,
        clip_floor=clip_floor,
        clip_ceiling=clip_ceiling,
        cmap=cmap,
    )
    fig.suptitle("Pairwise decoder R²: Raw vs drift-robust features", y=1.02, fontsize=TITLE_FONTSIZE)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pairwise_r2_feature_sets(
    R_raw: np.ndarray,
    R_gold: np.ndarray,
    R_robust_session: np.ndarray,
    R_robust_block: np.ndarray,
    session_dates: List[str],
    out_path: Optional[Union[Path, str]] = None,
    title_raw: str = "Static (raw SBP)",
    title_gold: str = "Gold r_norm\n(mean/std + 50 ms smooth)",
    title_robust_session: str = "Robust Gold\n(session median/MAD + 50 ms smooth)",
    title_robust_block: str = "Robust Gold\n(block median/MAD + 50 ms smooth)",
    clip_floor: Optional[float] = None,
    clip_ceiling: float = 1.0,
) -> Any:
    """
    Four-panel heatmap comparison for raw, Gold, and both robust Gold variants.

    All panels share the same color range so apparent warmth differences reflect
    actual pairwise R² changes rather than autoscaling.
    """
    matrices = [R_raw, R_gold, R_robust_session, R_robust_block]
    titles = [title_raw, title_gold, title_robust_session, title_robust_block]
    vmin, vmax = _shared_pairwise_r2_limits(
        *matrices,
        clip_floor=clip_floor,
        clip_ceiling=clip_ceiling,
    )

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)
    for ax, R, title in zip(axes, matrices, titles):
        plot_pairwise_r2_heatmap(
            R,
            session_dates,
            ax=ax,
            title=title,
            vmin=vmin,
            vmax=vmax,
            show_colorbar=False,
            clip_floor=clip_floor,
            clip_ceiling=clip_ceiling,
        )

    mappable = axes[-1].images[0]
    fig.colorbar(mappable, ax=list(axes), label="R²", shrink=0.9, pad=0.02)
    fig.suptitle("Pairwise decoder R² across feature sets", y=1.02, fontsize=TITLE_FONTSIZE)
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def _select_movement_window(y: np.ndarray, window_len: int = 500) -> tuple[int, int]:
    """
    Choose a movement-rich window based on variance of the velocity trace.

    Returns (start, end) indices for a contiguous window of length window_len
    (or shorter if the trace is shorter).
    """
    y = np.asarray(y, dtype=np.float64)
    T = y.shape[0]
    if T <= window_len or window_len <= 0:
        return 0, T

    # Center for stability, then use sliding window energy as proxy for movement.
    y_c = y - np.nanmean(y)
    sq = y_c ** 2
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    window_energy = csum[window_len:] - csum[:-window_len]
    best_start = int(np.argmax(window_energy))
    return best_start, best_start + window_len


def plot_decoded_traces_early_late(
    model: Any,
    sessions_sbp: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    session_dates: List[str],
    win: Optional[tuple[int, int]] = None,
    out_path: Optional[Union[Path, str]] = None,
    suptitle: Optional[str] = None,
    window_bins: int = 5,
    lag_bins: int = 3,
    apply_gain: bool = True,
) -> Any:
    """
    Two panels: actual vs predicted velocity for Day 0 (early) and last day (late)
    over a time window. Uses the same temporal context (window_bins, lag_bins) as
    compute_decoder_r2_per_day so the model receives matching feature dimension.

    For legacy callers that only pass a single model (typically Gold), this helper
    now applies an optional visualization-only gain so the orange trace amplitude
    roughly matches the blue trace while leaving R² computations unchanged.
    """
    X0, y0 = _make_xy(
        sessions_sbp[0],
        sessions_vel[0],
        window_bins=window_bins,
        lag_bins=lag_bins,
    )
    X_last, y_last = _make_xy(
        sessions_sbp[-1],
        sessions_vel[-1],
        window_bins=window_bins,
        lag_bins=lag_bins,
    )

    if win is None:
        start, end = _select_movement_window(y0, window_len=500)
    else:
        start, end = win

    if end > min(len(y0), len(y_last)):
        end = min(len(y0), len(y_last))
    if start >= end:
        start, end = 0, min(500, len(y0), len(y_last))

    t = np.arange(start, end)
    y_early_act = y0[start:end]
    y_early_pred = model.predict(X0[start:end])
    y_late_act = y_last[start:end]
    y_late_pred = model.predict(X_last[start:end])

    if apply_gain:
        # Fit gain on the full Day 0 data, then apply it to both early and late
        # predictions for plotting only.
        y0_pred_full = model.predict(X0)
        a, g = fit_plot_gain(y0, y0_pred_full, fit_intercept=True)
        y_early_pred_plot = apply_plot_gain(y_early_pred, a, g)
        y_late_pred_plot = apply_plot_gain(y_late_pred, a, g)
    else:
        y_early_pred_plot = y_early_pred
        y_late_pred_plot = y_late_pred

    N = len(sessions_sbp)
    fig, (ax_early, ax_late) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    ax_early.plot(t, y_early_act, color="steelblue", alpha=0.8, label="Actual")
    ax_early.plot(t, y_early_pred_plot, color="coral", alpha=0.8, label="Predicted")
    ax_early.set_ylabel("Velocity", fontsize=AXIS_LABEL_FONTSIZE)
    ax_early.set_title(f"Day 0 (early) — {session_dates[0]}", fontsize=TITLE_FONTSIZE)
    ax_early.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.75)

    ax_late.plot(t, y_late_act, color="steelblue", alpha=0.8, label="Actual")
    ax_late.plot(t, y_late_pred_plot, color="coral", alpha=0.8, label="Predicted")
    ax_late.set_xlabel("Time bin", fontsize=AXIS_LABEL_FONTSIZE)
    ax_late.set_ylabel("Velocity", fontsize=AXIS_LABEL_FONTSIZE)
    ax_late.set_title(f"Day {N-1} (late) — {session_dates[-1]}", fontsize=TITLE_FONTSIZE)
    ax_late.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.75)

    fig.suptitle(
        suptitle or "Decoded velocity: actual vs predicted (model trained on Day 0)",
        fontsize=TITLE_FONTSIZE,
    )
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_frac_active_vs_day(
    frac_active: List[float],
    session_dates: List[str],
    ax: Any = None,
    out_path: Optional[Union[Path, str]] = None,
) -> Any:
    """
    Plot fraction of active channels vs day (x = day index or date). Shows yield drop over time.
    """
    N = len(frac_active)
    x = np.arange(N)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    ax.bar(x, frac_active, color="steelblue", alpha=0.8, edgecolor="navy", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([session_dates[i] for i in range(N)], rotation=45, ha="right")
    ax.set_ylabel("Fraction active channels", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("Session (date)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Yield: % channels with finite σ per session", fontsize=TITLE_FONTSIZE)
    ax.set_ylim(0, 1.05)
    if N > 1:
        z = np.polyfit(x, frac_active, 1)
        ax.plot(x, np.polyval(z, x), "r--", alpha=0.8, label="Linear trend")
        ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.75)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_median_r2_vs_gap(
    R_raw: np.ndarray,
    R_z: np.ndarray,
    R_gold: np.ndarray,
    R_pro: np.ndarray,
    session_dates: List[str],
    R_robust_session: Optional[np.ndarray] = None,
    R_robust_block: Optional[np.ndarray] = None,
    feature_labels: Optional[dict[str, str]] = None,
    ax: Any = None,
    out_path: Optional[Union[Path, str]] = None,
) -> Any:
    """
    Plot decoder R² vs days-between-sessions for multiple feature sets.

    Each condition is shown as faint pairwise scatter plus a binned median trend
    with an interquartile error band, which is easier to read than connecting
    every exact gap value.
    """
    if feature_labels is None:
        feature_labels = {
            "raw": "Raw SBP",
            "zscore": "Z-score only",
            "gold": "Gold (r_norm)",
            "robust_session": "Robust session",
            "robust_block": "Robust block",
            "procrustes": "Procrustes",
        }

    day_offsets = compute_day_offsets(session_dates)
    R_by_feature = {
        "raw": R_raw,
        "zscore": R_z,
        "gold": R_gold,
        "procrustes": R_pro,
    }
    if R_robust_session is not None:
        R_by_feature["robust_session"] = R_robust_session
    if R_robust_block is not None:
        R_by_feature["robust_block"] = R_robust_block
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    # Use a stable palette ordering to mirror other plots.
    order = ["raw", "zscore", "gold", "robust_session", "robust_block", "procrustes"]
    colors = {
        "raw": "dimgray",
        "zscore": "tab:blue",
        "gold": "tab:orange",
        "robust_session": "tab:red",
        "robust_block": "tab:purple",
        "procrustes": "tab:green",
    }
    rng_seeds = {
        "raw": 11,
        "zscore": 23,
        "gold": 37,
        "robust_session": 41,
        "robust_block": 53,
        "procrustes": 67,
    }
    y_summary_values: list[np.ndarray] = []

    for feat in order:
        R_feat = R_by_feature.get(feat)
        if R_feat is None:
            continue
        label = feature_labels.get(feat, feat)
        color = colors.get(feat, None)
        gaps_flat, r2_flat = _flatten_r2_vs_gap(R_feat, day_offsets, include_zero_gap=True)
        if gaps_flat.size == 0:
            continue
        jitter = (np.random.default_rng(rng_seeds[feat]).random(gaps_flat.size) - 0.5) * 0.9
        ax.scatter(
            gaps_flat + jitter,
            r2_flat,
            s=16 if feat in ("raw", "gold") else 12,
            alpha=0.08,
            color=color,
            edgecolors="none",
            zorder=1,
        )

        x_bin, y_med, y_q25, y_q75 = _binned_r2_summary(gaps_flat, r2_flat, bin_width_days=10)
        if x_bin.size == 0:
            continue
        y_summary_values.extend([y_med, y_q25, y_q75])
        ax.fill_between(
            x_bin,
            y_q25,
            y_q75,
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            x_bin,
            y_med,
            marker="o",
            linewidth=2.2 if feat in ("raw", "gold") else 1.7,
            markersize=4,
            label=label,
            color=color,
            alpha=0.95,
            zorder=3,
        )

    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Days between sessions", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Median decoder R²", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Median decoder R² vs days-between-sessions", fontsize=TITLE_FONTSIZE)

    # Adaptive y-limits: ensure we show negatives if present and at least up to ~0.5.
    if y_summary_values:
        y_all = np.concatenate(y_summary_values)
        y_min = float(np.nanmin(y_all))
        y_max = float(np.nanmax(y_all))
        if y_max < 0.5:
            y_max = 0.5
        y_min = min(y_min, -0.1)
        pad = 0.03 * (y_max - y_min if y_max > y_min else 1.0)
        ax.set_ylim(y_min - pad, y_max + pad)

    ax.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.75)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_r2_scatter_raw_vs_gold(
    R_raw: np.ndarray,
    R_gold: np.ndarray,
    session_dates: List[str],
    out_path: Optional[Union[Path, str]] = None,
) -> Any:
    """
    Two-panel comparison of R² vs days-between-sessions: raw vs Gold (r_norm).

    Left: Raw SBP; right: Gold-normalized features. Each point is a train–test pair.
    """
    # Shared y-limits across panels for direct comparison.
    all_min = float(np.nanmin([np.nanmin(R_raw), np.nanmin(R_gold)]))
    all_max = float(np.nanmax([np.nanmax(R_raw), np.nanmax(R_gold)]))
    if all_max < 0.5:
        all_max = 0.5
    if all_min > -0.5:
        all_min = min(all_min, -0.1)
    pad = 0.03 * (all_max - all_min if all_max > all_min else 1.0)
    y_limits = (all_min - pad, all_max + pad)

    fig, (ax_raw, ax_gold) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    plot_r2_scatter_vs_gap(
        R_raw,
        session_dates,
        title="Raw SBP",
        ax=ax_raw,
        y_limits=y_limits,
    )
    plot_r2_scatter_vs_gap(
        R_gold,
        session_dates,
        title="Gold (r_norm)",
        ax=ax_gold,
        y_limits=y_limits,
    )

    fig.suptitle("Decoder R² vs days-between-sessions: Raw vs Gold", y=1.02, fontsize=TITLE_FONTSIZE)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_r2_scatter_feature_grid(
    R_by_feature: dict[str, np.ndarray],
    session_dates: List[str],
    feature_labels: Optional[dict[str, str]] = None,
    out_path: Optional[Union[Path, str]] = None,
) -> Any:
    """Grid of decoder R² vs days-between-sessions scatter plots for multiple features."""
    if feature_labels is None:
        feature_labels = {
            "raw": "Raw SBP",
            "zscore": "Z-score only",
            "gold": "Gold (r_norm)",
            "robust_session": "Robust session",
            "robust_block": "Robust block",
        }
    feature_items = list(R_by_feature.items())
    if not feature_items:
        raise ValueError("R_by_feature must contain at least one feature matrix")

    all_values = np.concatenate(
        [np.ravel(np.asarray(R, dtype=float)) for _, R in feature_items]
    )
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        y_limits = (-0.1, 0.5)
    else:
        all_min = float(np.nanmin(finite))
        all_max = float(np.nanmax(finite))
        if all_max < 0.5:
            all_max = 0.5
        if all_min > -0.5:
            all_min = min(all_min, -0.1)
        pad = 0.03 * (all_max - all_min if all_max > all_min else 1.0)
        y_limits = (all_min - pad, all_max + pad)

    fig, axes = plt.subplots(1, len(feature_items), figsize=(5 * len(feature_items), 5), sharey=True)
    if len(feature_items) == 1:
        axes = [axes]

    for ax, (feat, R) in zip(axes, feature_items):
        plot_r2_scatter_vs_gap(
            R,
            session_dates,
            title=feature_labels.get(feat, feat),
            ax=ax,
            y_limits=y_limits,
        )

    fig.suptitle("Decoder R² vs days-between-sessions by feature set", y=1.02, fontsize=TITLE_FONTSIZE)
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_decoded_traces_raw_vs_gold_early_late(
    model_raw: Any,
    model_gold: Any,
    sbp_raw: List[np.ndarray],
    sbp_gold: List[np.ndarray],
    sessions_vel: List[np.ndarray],
    session_dates: List[str],
    window_bins: int = 5,
    lag_bins: int = 3,
    window_len: int = 500,
    apply_gain: bool = True,
    out_path: Optional[Union[Path, str]] = None,
) -> Any:
    """
    Four-panel comparison: raw vs Gold, early vs late decoded velocity traces.

    Layout:
        Row 1: Day 0 raw (left), Day 0 Gold (right)
        Row 2: Late day raw (left), late day Gold (right)

    Each panel overlays actual (blue) and predicted (orange) velocity for a
    movement-rich window. Optional visualization-only gain is applied based on
    Day 0 data for each feature set so amplitudes are comparable.
    """
    # Build design matrices for Day 0 and last day for both feature sets.
    X0_raw, y0 = _make_xy(
        sbp_raw[0],
        sessions_vel[0],
        window_bins=window_bins,
        lag_bins=lag_bins,
    )
    X_last_raw, y_last = _make_xy(
        sbp_raw[-1],
        sessions_vel[-1],
        window_bins=window_bins,
        lag_bins=lag_bins,
    )
    X0_gold, _ = _make_xy(
        sbp_gold[0],
        sessions_vel[0],
        window_bins=window_bins,
        lag_bins=lag_bins,
    )
    X_last_gold, _ = _make_xy(
        sbp_gold[-1],
        sessions_vel[-1],
        window_bins=window_bins,
        lag_bins=lag_bins,
    )

    # Choose a deterministic movement-rich window based on Day 0 velocity.
    start, end = _select_movement_window(y0, window_len=window_len)
    if end > min(len(y0), len(y_last)):
        end = min(len(y0), len(y_last))
    if start >= end:
        start, end = 0, min(window_len, len(y0), len(y_last))

    t = np.arange(start, end)
    y0_seg = y0[start:end]
    y_last_seg = y_last[start:end]

    # Raw predictions (unscaled) for metrics.
    y0_pred_raw_full = model_raw.predict(X0_raw)
    y_last_pred_raw_full = model_raw.predict(X_last_raw)
    y0_pred_raw = y0_pred_raw_full[start:end]
    y_last_pred_raw = y_last_pred_raw_full[start:end]

    # Gold predictions (unscaled) for metrics.
    y0_pred_gold_full = model_gold.predict(X0_gold)
    y_last_pred_gold_full = model_gold.predict(X_last_gold)
    y0_pred_gold = y0_pred_gold_full[start:end]
    y_last_pred_gold = y_last_pred_gold_full[start:end]

    # Optional visualization-only gain per feature set (fit on full Day 0).
    if apply_gain:
        a_raw, g_raw = fit_plot_gain(y0, y0_pred_raw_full, fit_intercept=True)
        a_gold, g_gold = fit_plot_gain(y0, y0_pred_gold_full, fit_intercept=True)
        y0_pred_raw_plot = apply_plot_gain(y0_pred_raw, a_raw, g_raw)
        y_last_pred_raw_plot = apply_plot_gain(y_last_pred_raw, a_raw, g_raw)
        y0_pred_gold_plot = apply_plot_gain(y0_pred_gold, a_gold, g_gold)
        y_last_pred_gold_plot = apply_plot_gain(y_last_pred_gold, a_gold, g_gold)
    else:
        y0_pred_raw_plot = y0_pred_raw
        y_last_pred_raw_plot = y_last_pred_raw
        y0_pred_gold_plot = y0_pred_gold
        y_last_pred_gold_plot = y_last_pred_gold

    # Panel-level R² (computed on unscaled predictions for transparency).
    from sklearn.metrics import r2_score  # local import to avoid circularity

    r2_day0_raw = float(r2_score(y0_seg, y0_pred_raw))
    r2_day0_gold = float(r2_score(y0_seg, y0_pred_gold))
    r2_last_raw = float(r2_score(y_last_seg, y_last_pred_raw))
    r2_last_gold = float(r2_score(y_last_seg, y_last_pred_gold))

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(12, 6))
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    # Row 1: Day 0 raw vs Gold.
    ax00.plot(t, y0_seg, color="steelblue", alpha=0.8, label="Actual")
    ax00.plot(t, y0_pred_raw_plot, color="coral", alpha=0.8, label="Predicted")
    ax00.set_ylabel("Velocity", fontsize=AXIS_LABEL_FONTSIZE)
    ax00.set_title(f"Day 0 raw — R² = {r2_day0_raw:.2f}", fontsize=TITLE_FONTSIZE)
    ax00.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.75)

    ax01.plot(t, y0_seg, color="steelblue", alpha=0.8, label="Actual")
    ax01.plot(t, y0_pred_gold_plot, color="coral", alpha=0.8, label="Predicted")
    ax01.set_title(f"Day 0 Gold — R² = {r2_day0_gold:.2f}", fontsize=TITLE_FONTSIZE)
    ax01.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.75)

    # Row 2: Late day raw vs Gold.
    ax10.plot(t, y_last_seg, color="steelblue", alpha=0.8, label="Actual")
    ax10.plot(t, y_last_pred_raw_plot, color="coral", alpha=0.8, label="Predicted")
    ax10.set_xlabel("Time bin", fontsize=AXIS_LABEL_FONTSIZE)
    ax10.set_ylabel("Velocity", fontsize=AXIS_LABEL_FONTSIZE)
    ax10.set_title(f"Day {len(sbp_raw) - 1} raw — R² = {r2_last_raw:.2f}", fontsize=TITLE_FONTSIZE)
    ax10.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.75)

    ax11.plot(t, y_last_seg, color="steelblue", alpha=0.8, label="Actual")
    ax11.plot(t, y_last_pred_gold_plot, color="coral", alpha=0.8, label="Predicted")
    ax11.set_xlabel("Time bin", fontsize=AXIS_LABEL_FONTSIZE)
    ax11.set_title(f"Day {len(sbp_gold) - 1} Gold — R² = {r2_last_gold:.2f}", fontsize=TITLE_FONTSIZE)
    ax11.legend(fontsize=LEGEND_FONTSIZE, framealpha=0.75)

    fig.suptitle(
        "Decoded velocity: raw vs Gold, early vs late (models trained on Day 0)",
        y=1.02,
        fontsize=TITLE_FONTSIZE,
    )
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_decoded_traces_feature_sets_early_late(
    models_by_feature: dict[str, Any],
    sessions_by_feature: dict[str, List[np.ndarray]],
    sessions_vel: List[np.ndarray],
    session_dates: List[str],
    feature_labels: Optional[dict[str, str]] = None,
    window_bins: int = 5,
    lag_bins: int = 3,
    window_len: int = 500,
    apply_gain: bool = True,
    out_path: Optional[Union[Path, str]] = None,
) -> Any:
    """Compare early/late decoded traces across multiple feature sets."""
    if feature_labels is None:
        feature_labels = {
            "raw": "Raw SBP",
            "zscore": "Z-score only",
            "gold": "Gold (r_norm)",
            "robust_session": "Robust session",
            "robust_block": "Robust block",
        }
    feature_items = [
        (feat, models_by_feature[feat], sessions_by_feature[feat])
        for feat in models_by_feature
        if feat in sessions_by_feature
    ]
    if not feature_items:
        raise ValueError("Need at least one shared feature key in models_by_feature and sessions_by_feature")

    first_sessions = feature_items[0][2]
    _, y0 = _make_xy(
        first_sessions[0],
        sessions_vel[0],
        window_bins=window_bins,
        lag_bins=lag_bins,
    )
    _, y_last = _make_xy(
        first_sessions[-1],
        sessions_vel[-1],
        window_bins=window_bins,
        lag_bins=lag_bins,
    )
    start, end = _select_movement_window(y0, window_len=window_len)
    if end > min(len(y0), len(y_last)):
        end = min(len(y0), len(y_last))
    if start >= end:
        start, end = 0, min(window_len, len(y0), len(y_last))
    t = np.arange(start, end)
    y0_seg = y0[start:end]
    y_last_seg = y_last[start:end]

    n_cols = len(feature_items)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.5 * n_cols, 7), sharex=True)
    if n_cols == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, (feat, model, sessions_feat) in enumerate(feature_items):
        X0, _ = _make_xy(
            sessions_feat[0],
            sessions_vel[0],
            window_bins=window_bins,
            lag_bins=lag_bins,
        )
        X_last, _ = _make_xy(
            sessions_feat[-1],
            sessions_vel[-1],
            window_bins=window_bins,
            lag_bins=lag_bins,
        )
        y0_pred_full = model.predict(X0)
        y_last_pred_full = model.predict(X_last)
        y0_pred = y0_pred_full[start:end]
        y_last_pred = y_last_pred_full[start:end]

        if apply_gain:
            a, g = fit_plot_gain(y0, y0_pred_full)
            y0_pred_plot = apply_plot_gain(y0_pred, a, g)
            y_last_pred_plot = apply_plot_gain(y_last_pred, a, g)
        else:
            y0_pred_plot = y0_pred
            y_last_pred_plot = y_last_pred

        ax_early = axes[0, col]
        ax_late = axes[1, col]
        label = feature_labels.get(feat, feat)

        ax_early.plot(t, y0_seg, label="True velocity", color="tab:blue", linewidth=1.5)
        ax_early.plot(t, y0_pred_plot, label="Decoded", color="tab:orange", linewidth=1.5)
        ax_early.set_title(f"Day 0: {label}", fontsize=TITLE_FONTSIZE)
        if col == 0:
            ax_early.set_ylabel("Velocity", fontsize=AXIS_LABEL_FONTSIZE)
        ax_early.grid(alpha=0.2)
        if col == n_cols - 1:
            ax_early.legend(loc="upper right", fontsize=LEGEND_FONTSIZE, framealpha=0.75)

        ax_late.plot(t, y_last_seg, label="True velocity", color="tab:blue", linewidth=1.5)
        ax_late.plot(t, y_last_pred_plot, label="Decoded", color="tab:orange", linewidth=1.5)
        ax_late.set_title(f"Last day: {label}", fontsize=TITLE_FONTSIZE)
        if col == 0:
            ax_late.set_ylabel("Velocity", fontsize=AXIS_LABEL_FONTSIZE)
        ax_late.set_xlabel("Time bin", fontsize=AXIS_LABEL_FONTSIZE)
        ax_late.grid(alpha=0.2)

    fig.suptitle(
        f"Decoded velocity traces across feature sets ({session_dates[0]} vs {session_dates[-1]})",
        y=1.02,
        fontsize=TITLE_FONTSIZE,
    )
    plt.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig
