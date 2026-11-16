"""
Which rows train and which rows test, for every bench score.

Kept free of data loading so the leakage test can check the plan on synthetic
session lengths. Row i of a session is the make_xy row at base bin t = window + i
(after the warm-up trim); it reads SBP bins t-window..t and velocity bin t+lag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from neurosignal.decoders.linear import DecoderSpec, blocked_folds, grid_offset, n_rows  # noqa: F401


@dataclass(frozen=True)
class ScorePlan:
    kind: str                # "same_day" or "cross_day"
    cal: str                 # calibration session date
    test: str                # test session date
    fold: int                # same-day fold index; -1 for cross-day
    train_rows: np.ndarray   # row indices in the calibration session
    test_rows: np.ndarray    # row indices in the test session


def row_bins(rows: np.ndarray, spec: DecoderSpec) -> np.ndarray:
    """Every SBP or velocity bin the given rows read, as a sorted unique array."""
    t = np.asarray(rows) + spec.window_bins
    spans = t[:, None] + np.arange(-spec.window_bins, spec.lag_bins + 1)[None, :]
    return np.unique(spans)


def session_plans(date: str, bins: int, spec: DecoderSpec, seed: int = 0) -> List[ScorePlan]:
    """Same-day plans of one session: blocked folds on the grid this seed shifts to (the CFD denominator)."""
    rows = n_rows(bins, spec)
    return [
        ScorePlan("same_day", date, date, k, train, test)
        for k, (train, test) in enumerate(blocked_folds(rows, spec, grid_offset(rows, seed, spec)))
    ]


def cross_plan(cal: str, test: str, all_rows: Dict[str, np.ndarray]) -> ScorePlan:
    """Fit on the whole calibration session, test on the whole later session; seed-independent."""
    return ScorePlan("cross_day", cal, test, -1, all_rows[cal], all_rows[test])


def plan_scores(splits: Dict, session_bins: Dict[str, int], spec: DecoderSpec, seed: int = 0) -> List[ScorePlan]:
    """
    Same-day scores: blocked folds within each calibration session (the CFD denominator),
    on the grid this seed shifts to. Cross-day scores: fit on the whole calibration session,
    test on the whole later session; they do not depend on the seed.
    session_bins maps date -> bins left after the warm-up trim. Cross-day plans share one
    row array per session, so memory grows with sessions, not pairs.
    """
    plans: List[ScorePlan] = []
    for date in splits["sessions"]:
        plans += session_plans(date, session_bins[date], spec, seed)
    all_rows = {d: np.arange(n_rows(session_bins[d], spec)) for d in splits["sessions"]}
    plans += [cross_plan(pair["cal"], pair["test"], all_rows) for pair in splits["pairs"]]
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
