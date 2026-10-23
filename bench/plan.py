"""
Which rows train and which rows test, for every bench score.

Kept free of data loading so the leakage test can check the plan on synthetic
session lengths. Row i of a session is the _make_xy row at base bin t = window + i
(after the warm-up trim); it reads SBP bins t-window..t and velocity bin t+lag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from bench.spec import DecoderSpec


@dataclass(frozen=True)
class ScorePlan:
    kind: str                # "same_day" or "cross_day"
    cal: str                 # calibration session date
    test: str                # test session date
    fold: int                # same-day fold index; -1 for cross-day
    train_rows: np.ndarray   # row indices in the calibration session
    test_rows: np.ndarray    # row indices in the test session


def n_rows(n_bins: int, spec: DecoderSpec) -> int:
    """Rows _make_xy builds from a session of n_bins (SBP and velocity of equal length)."""
    return max(0, n_bins - spec.window_bins - spec.lag_bins)


def row_bins(rows: np.ndarray, spec: DecoderSpec) -> np.ndarray:
    """Every SBP or velocity bin the given rows read, as a sorted unique array."""
    t = np.asarray(rows) + spec.window_bins
    spans = t[:, None] + np.arange(-spec.window_bins, spec.lag_bins + 1)[None, :]
    return np.unique(spans)


def grid_offset(rows: int, seed: int, spec: DecoderSpec) -> int:
    """Rows the fold grid is shifted by for this seed (D-13): one draw u ~ U[0, 1) per seed."""
    u = np.random.default_rng(seed).random()
    return int(u * rows / spec.same_day_folds)


def blocked_folds(rows: int, spec: DecoderSpec, offset: int = 0) -> List[tuple]:
    """
    Contiguous test blocks on a grid shifted by `offset` rows; the block crossing the end
    wraps to the start. Training drops every row within purge_rows of a test row, so no bin is shared.
    """
    edges = np.linspace(0, rows, spec.same_day_folds + 1).astype(int)
    all_rows = np.arange(rows)
    shifted = (all_rows - offset) % rows if rows else all_rows
    window = np.ones(2 * spec.purge_rows + 1)
    folds = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        is_test = (shifted >= lo) & (shifted < hi)
        near_test = np.convolve(is_test, window, mode="same") > 0
        folds.append((all_rows[~near_test], all_rows[is_test]))
    return folds


def plan_scores(splits: Dict, session_bins: Dict[str, int], spec: DecoderSpec, seed: int = 0) -> List[ScorePlan]:
    """
    Same-day scores: blocked folds within each calibration session (the CFD denominator),
    on the grid this seed shifts to. Cross-day scores: fit on the whole calibration session,
    test on the whole later session; they do not depend on the seed.
    session_bins maps date -> bins left after the warm-up trim.
    """
    plans: List[ScorePlan] = []
    for date in splits["sessions"]:
        rows = n_rows(session_bins[date], spec)
        for k, (train, test) in enumerate(blocked_folds(rows, spec, grid_offset(rows, seed, spec))):
            plans.append(ScorePlan("same_day", date, date, k, train, test))
    for pair in splits["pairs"]:
        plans.append(
            ScorePlan(
                "cross_day",
                pair["cal"],
                pair["test"],
                -1,
                np.arange(n_rows(session_bins[pair["cal"]], spec)),
                np.arange(n_rows(session_bins[pair["test"]], spec)),
            )
        )
    return plans


def leaking_plans(plans: List[ScorePlan], spec: DecoderSpec, allowed_pairs: set) -> List[str]:
    """
    Every plan that trains on what it tests (ERR-01, ERR-02): a same-day plan whose training
    and test rows read a shared bin, or a cross-day plan that is not a frozen forward pair.
    Used by the leakage test and by the bench.run gate (D-13).
    """
    bad = []
    for p in plans:
        if not (len(p.train_rows) and len(p.test_rows)):
            bad.append(f"{p.kind} {p.cal}->{p.test} fold {p.fold}: empty rows")
        elif p.kind == "same_day":
            shared = np.intersect1d(row_bins(p.train_rows, spec), row_bins(p.test_rows, spec))
            if shared.size:
                bad.append(f"{p.cal} fold {p.fold}: trains on {shared.size} bins it tests")
        elif not (p.cal < p.test and (p.cal, p.test) in allowed_pairs):
            bad.append(f"{p.cal}->{p.test}: not a frozen forward pair")
    return bad
