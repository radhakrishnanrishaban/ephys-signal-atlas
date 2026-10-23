"""
Figure for the lab note: kept ratio vs calibration→test gap, with the isotonic fit CFD reads
and the 50 % line, one panel per causal feature. Data: results/bench_v0_pairs.csv (seed 0) and
the matching results/scoreboard.csv row for CFD and its CI. Nothing here is typed by hand (D-08).

    python -m bench.figures results/notes/2026-W43_kept_ratio_vs_gap.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression

from bench.scoreboard import REPO, RESULTS, SCOREBOARD_CSV
from bench.spec import SPEC
from neurosignal.gold.drift import AXIS_LABEL_FONTSIZE, LEGEND_FONTSIZE, TITLE_FONTSIZE

PAIRS_CSV = RESULTS / "bench_v0_pairs.csv"
PANELS = {"raw_sbp": ("Raw SBP", "#2a78d6"), "r_norm_causal": ("Causal r_norm", "#eb6834")}  # validated pair
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def _rows(path: Path) -> list:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def plot(out: Path, seed: int = SPEC.seeds[0]) -> None:
    pairs = _rows(PAIRS_CSV)
    commit = pairs[0]["commit"]
    board = {r["feature"]: r for r in _rows(SCOREBOARD_CSV) if r["commit"] == commit}

    fig, axes = plt.subplots(1, len(PANELS), figsize=(11, 4.8), sharey=True)
    for ax, (feature, (label, color)) in zip(axes, PANELS.items()):
        rows = [r for r in pairs if r["feature"] == feature and int(r["seed"]) == seed]
        gap = np.array([float(r["gap_days"]) for r in rows])
        ratio = np.array([float(r["kept_ratio"]) for r in rows])
        order = np.argsort(gap)
        fit = IsotonicRegression(increasing=False).fit(gap, ratio).predict(gap[order])
        row = board[feature]
        cfd, lo, hi = (int(row[k]) for k in ("cfd_days", "cfd_ci95_lo", "cfd_ci95_hi"))

        if hi > 0:
            ax.axvspan(max(lo, 1), hi, color=color, alpha=0.10, linewidth=0)
        ax.scatter(gap, ratio, s=10, color=color, alpha=0.45, linewidths=0)
        ax.step(gap[order], fit, where="post", color=color, linewidth=2)
        ax.axhline(SPEC.kept_threshold, color=MUTED, linestyle="--", linewidth=1)
        ax.text(1.05, SPEC.kept_threshold + 0.03, "50 % kept", color=MUTED, fontsize=9)
        ax.text(
            0.97, 0.95, f"CFD {cfd} d (95 % CI {lo}–{hi})", transform=ax.transAxes,
            ha="right", va="top", color=INK, fontsize=9,
        )
        ax.set_xscale("log")
        ax.set_xlim(1, 500)
        ax.set_xticks([1, 3, 10, 30, 100, 300])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(-0.03, 1.6)
        ax.set_title(label, fontsize=TITLE_FONTSIZE, color=INK)
        ax.set_xlabel("Calibration → test gap (days)", fontsize=AXIS_LABEL_FONTSIZE, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Kept share of same-day R²", fontsize=AXIS_LABEL_FONTSIZE, color=INK)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=MUTED, alpha=0.6, label="One session pair"),
        plt.Line2D([], [], color=MUTED, linewidth=2, label="Non-increasing fit"),
        plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.15, label="CFD 95 % CI"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=LEGEND_FONTSIZE, framealpha=0.75,
               bbox_to_anchor=(0.5, 0.04))
    fig.text(
        0.01, 0.005, f"bench v0 · commit {commit} · seed {seed} of {len(SPEC.seeds)} · "
        f"{len(rows)} forward pairs, 30 LINK sessions",
        fontsize=8, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    print(f"wrote {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    plot(parser.parse_args().out.resolve())
