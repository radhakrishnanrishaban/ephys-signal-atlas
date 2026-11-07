"""
Shared plumbing for analyses/ (D-19): one script per question, each writing

    results/analyses/<question>.json      every number the question publishes, with its CI and provenance
    results/analyses/<question>/*.png     its figures
    results/analyses/<question>.md        the Q&A report, rendered from the JSON alone

A report never computes or types a number: it reads the JSON. Re-render without recomputing with
`python -m analyses.<question> --report-only`.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from bench.run import git_state, load_frozen_sessions
from bench.scoreboard import REPO
from bench.splits import load_splits
from neurosignal.decoders.linear import SPEC
from neurosignal.metrics.stats import ci95

OUT = REPO / "results" / "analyses"


def provenance() -> Dict:
    commit, dirty = git_state()
    return {"commit": commit, "dirty": dirty, "spec": SPEC.spec_id, "run_date": date.today().isoformat()}


def frozen_sessions() -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """SBP, velocity and dates of the frozen session list (bench/splits_v0.json)."""
    splits = load_splits()
    sbp, vel = load_frozen_sessions(splits)
    return sbp, vel, splits["sessions"]


def estimate(value: float, draws: np.ndarray, n: str) -> Dict:
    """A published number: point value, 95 % session-bootstrap CI, whether it excludes 0, and its n."""
    lo, hi = ci95(np.asarray(draws)[~np.isnan(draws)])
    return {"value": round(float(value), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n": n}


def write_json(name: str, result: Dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    return path


def read_json(name: str) -> Dict:
    return json.loads((OUT / f"{name}.json").read_text())


def fmt(e: Dict, digits: int = 3) -> str:
    """`+0.030 (95 % CI −0.012 to +0.079)` from an estimate dict."""
    lo, hi = e["ci95"]
    s = lambda x: f"{x:+.{digits}f}".replace("-", "−")  # noqa: E731
    return f"{s(e['value'])} (95 % CI {s(lo)} to {s(hi)})"


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.time() - self.t0, 1)
