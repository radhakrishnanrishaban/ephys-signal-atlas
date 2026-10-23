"""
Frozen session list and calibration→test pairs.

splits_v0.json is written once and committed; it never changes. A new split is a new
file (splits_v1.json) and a new decision entry.

    python -m bench.splits    # writes splits_v0.json only if it does not exist yet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from neurosignal.gold.drift import compute_day_offsets

SPLITS_PATH = Path(__file__).with_name("splits_v0.json")


def forward_pairs(dates: List[str]) -> List[Dict]:
    """Every (calibration, test) pair with the test session strictly later, gap from dates (ERR-06)."""
    offsets = compute_day_offsets(dates).astype(int)
    return [
        {"cal": dates[i], "test": dates[j], "gap_days": int(offsets[j] - offsets[i])}
        for i in range(len(dates))
        for j in range(i + 1, len(dates))
    ]


def load_splits(path: Path = SPLITS_PATH) -> Dict:
    splits = json.loads(Path(path).read_text())
    dates = splits["sessions"]
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise ValueError(f"{path}: sessions must be unique and sorted by date")
    if splits["pairs"] != forward_pairs(dates):
        raise ValueError(f"{path}: pairs do not match the forward pairs of its sessions")
    return splits


def _write_v0() -> None:
    if SPLITS_PATH.exists():
        sys.exit(f"{SPLITS_PATH.name} already exists and is frozen; a new split needs a new file.")

    from neurosignal.gold.drift import filter_sessions_by_bad_dates, qc_sbp_sessions
    from neurosignal.io.loaders import list_link_nwb_paths, load_link_sessions

    sbp, vel, dates = load_link_sessions(list_link_nwb_paths())
    _, bad = qc_sbp_sessions(sbp, vel, dates, z_thresh=3.5)
    _, _, dates, _ = filter_sessions_by_bad_dates(sbp, vel, dates, bad)
    splits = {
        "version": "v0",
        "source": "01_data/*.nwb (LINK, Monkey N), qc_sbp_sessions z_thresh=3.5",
        "qc_dropped": sorted(bad),
        "sessions": dates,
        "pairs": forward_pairs(dates),
    }
    head = json.dumps({k: v for k, v in splits.items() if k != "pairs"}, indent=1)[:-2]
    pairs = ",\n  ".join(json.dumps(p) for p in splits["pairs"])
    SPLITS_PATH.write_text(f'{head},\n "pairs": [\n  {pairs}\n ]\n}}\n')  # one pair per line
    gaps = np.array([p["gap_days"] for p in splits["pairs"]])
    print(f"wrote {SPLITS_PATH.name}: {len(dates)} sessions, {len(gaps)} pairs, gaps {gaps.min()}–{gaps.max()} d")


if __name__ == "__main__":
    _write_v0()
