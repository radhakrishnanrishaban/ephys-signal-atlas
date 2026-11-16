"""
Frozen session list and calibration→test pairs.

Each splits_<version>.json is written once and committed; it never changes. A new split is a
new file and a new decision entry (v0: D-12; v1, v2: D-22).

    python -m bench.splits v2    # writes splits_v2.json from every session on disk, only if it does not exist yet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from neurosignal.metrics.drift import compute_day_offsets

SPLITS_SOURCES = {
    "v0": "01_data/*.nwb (LINK, Monkey N), qc_sbp_sessions z_thresh=3.5",
    "v1": "01_data/*.nwb (LINK, Monkey N; 30 local + 47 symlinked from the archive), qc_sbp_sessions z_thresh=3.5",
    "v2": "DANDI 001201 version 0.261015.2153, all 312 sessions: 77 NWB in 01_data + 235 extracted by "
          "scripts/fetch_link.py into 01_data/link/*.npz; qc_sbp_sessions z_thresh=3.5 per session, "
          "flagged together; same-date pairs dropped",
}
SPLITS_EXPECTED_SESSIONS = {"v2": 312}  # write_splits refuses to freeze a split with sessions missing


def splits_path(version: str = "v0") -> Path:
    return Path(__file__).with_name(f"splits_{version}.json")


SPLITS_PATH = splits_path("v0")


def forward_pairs(sessions: List[str]) -> List[Dict]:
    """
    Every (calibration, test) pair with the test session on a strictly later date, gap from dates
    (ERR-06). Sessions are keys whose first 8 characters are the date; two sessions on one date
    (v2) never pair with each other (D-22).
    """
    offsets = compute_day_offsets([s[:8] for s in sessions]).astype(int)
    return [
        {"cal": sessions[i], "test": sessions[j], "gap_days": int(offsets[j] - offsets[i])}
        for i in range(len(sessions))
        for j in range(i + 1, len(sessions))
        if offsets[j] > offsets[i]
    ]


def load_splits(path: Path = SPLITS_PATH) -> Dict:
    splits = json.loads(Path(path).read_text())
    dates = splits["sessions"]
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise ValueError(f"{path}: sessions must be unique and sorted by date")
    if splits["pairs"] != forward_pairs(dates):
        raise ValueError(f"{path}: pairs do not match the forward pairs of its sessions")
    return splits


def qc_dropped(paths: Dict[str, Path], z_thresh: float = 3.5) -> List[str]:
    """QC metrics one session at a time (one session in memory), robust-z flagged across all of them (D-12)."""
    import pandas as pd

    from neurosignal.io.loaders import load_link_session
    from neurosignal.metrics.qc import flag_qc_outliers, qc_sbp_sessions

    rows = []
    for i, (key, path) in enumerate(paths.items()):
        sbp, vel = load_link_session(path)
        rows.append(qc_sbp_sessions([sbp], [vel], [key], z_thresh=z_thresh)[0])
        print(f"  QC {i + 1}/{len(paths)} {key}", end="\r", flush=True)
    print()
    return sorted(flag_qc_outliers(pd.concat(rows), z_thresh)[1])


def write_splits(version: str) -> None:
    path = splits_path(version)
    if path.exists():
        sys.exit(f"{path.name} already exists and is frozen; a new split needs a new file.")
    if version not in SPLITS_SOURCES:
        sys.exit(f"no source recorded for splits {version}: add it to SPLITS_SOURCES with its D-entry first")

    from neurosignal.io.loaders import list_link_session_paths

    paths = list_link_session_paths()
    expected = SPLITS_EXPECTED_SESSIONS.get(version)
    if expected and len(paths) != expected:
        sys.exit(f"splits {version} needs {expected} sessions on disk, found {len(paths)}: run scripts/fetch_link.py first")
    bad = qc_dropped(paths)
    dates = [k for k in paths if k not in set(bad)]
    splits = {
        "version": version,
        "source": SPLITS_SOURCES[version],
        "qc_dropped": sorted(bad),
        "sessions": dates,
        "pairs": forward_pairs(dates),
    }
    head = json.dumps({k: v for k, v in splits.items() if k != "pairs"}, indent=1)[:-2]
    pairs = ",\n  ".join(json.dumps(p) for p in splits["pairs"])
    path.write_text(f'{head},\n "pairs": [\n  {pairs}\n ]\n}}\n')  # one pair per line
    gaps = np.array([p["gap_days"] for p in splits["pairs"]])
    print(f"wrote {path.name}: {len(dates)} sessions, {len(gaps)} pairs, gaps {gaps.min()}–{gaps.max()} d, QC dropped {sorted(bad)}")


if __name__ == "__main__":
    write_splits(sys.argv[1] if len(sys.argv) > 1 else "v0")
