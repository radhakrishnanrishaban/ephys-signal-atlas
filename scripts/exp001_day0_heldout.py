"""
EXP-001 (EDA-14): score the discovery notebook's Day-0 decoder on held-out rows (ERR-01).

    python scripts/exp001_day0_heldout.py

Loads every LINK session as `sbp-drift-discovery_v1` does, then scores session 0 with
`compute_same_day_r2_blocked`: 5 contiguous blocks, offset 0 plus seeds 0, 1, 2.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

from neurosignal.gold.drift import _make_xy, compute_decoder_r2_per_day, compute_same_day_r2_blocked
from neurosignal.io.loaders import list_link_nwb_paths, load_link_sessions, session_date_from_path

REPO = Path(__file__).resolve().parents[1]
N_FOLDS = 5
SEEDS = (0, 1, 2)


def main() -> None:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    paths = sorted(list_link_nwb_paths(), key=session_date_from_path)
    sbp, vel, dates = load_link_sessions(paths)
    print(f"commit {sha} · {len(dates)} sessions · Day 0 = {dates[0]}")

    t0 = time.time()
    r2_day_k, _ = compute_decoder_r2_per_day(sbp, vel)
    print(f"notebook path: Day 0 in-sample R² {r2_day_k[0]:.4f} · last session {dates[-1]} R² {r2_day_k[-1]:.4f}")

    n_rows = len(_make_xy(sbp[0], vel[0], window_bins=5, lag_bins=3)[1])
    grids = {"offset 0": 0}
    for seed in SEEDS:
        grids[f"seed {seed}"] = int(np.random.default_rng(seed).integers(0, n_rows // N_FOLDS))

    heldout = []
    for name, offset in grids.items():
        r = compute_same_day_r2_blocked(sbp[0], vel[0], n_folds=N_FOLDS, offset=offset)
        heldout.append(r["r2_heldout"])
        folds = " ".join(f"{v:.4f}" for v in r["fold_r2"])
        print(
            f"{name:8s} offset {offset:5d} · held-out {r['r2_heldout']:.4f} (pooled {r['r2_heldout_pooled']:.4f})"
            f" · in-sample {r['r2_in_sample']:.4f} · folds {folds}"
        )
    h = np.asarray(heldout)
    print(f"held-out over {len(h)} grids: mean {h.mean():.4f} · sd {h.std(ddof=1):.4f} · min {h.min():.4f} · max {h.max():.4f}")
    print(f"in-sample − held-out: {r2_day_k[0] - h.mean():.4f}")
    print(f"drop to {dates[-1]}: in-sample {r2_day_k[0] - r2_day_k[-1]:.4f} · held-out {h.mean() - r2_day_k[-1]:.4f}")
    print(f"wall time {time.time() - t0:.0f} s · {len(grids) * (N_FOLDS + 1) + 1} RidgeCV fits")


if __name__ == "__main__":
    main()
