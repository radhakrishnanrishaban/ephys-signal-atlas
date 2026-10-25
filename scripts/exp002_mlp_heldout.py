"""
EXP-002 (EDA-08): score the Part 3 cross-session MLP on sessions it never trained on (ERR-02).

    python scripts/exp002_mlp_heldout.py              # writes results/exp002_mlp_heldout.csv
    python scripts/exp002_mlp_heldout.py --dry-run    # prints only

Splits (D-17), over the 30 sessions of bench/splits_v0.json, by date:
  forward   train 2020-01-27 → 2020-03-20 (days 0–53), test 2020-06-26 → 2021-04-14 (days 151–443)
  grouped   5 contiguous date groups of 6 sessions; each held out once, trained on the other 24
  in-sample train and score all 30 (the notebook protocol, kept as the reference for the inflation)

Decoders per fit (gold/nonlinear.fit_pooled_decoders, identical training rows): the MLP,
Ridge on the MLP's exact inputs, and on the forward split the SPEC Ridge on SBP r_norm.
Seeds 0, 1, 2. Writes one row per (split, fold, seed, decoder, session); refuses to write
with uncommitted changes outside results/, so every row traces to a commit.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import date

import numpy as np

from bench.run import git_state
from bench.scoreboard import REPO, RESULTS
from bench.splits import load_splits
from neurosignal.gold.decoder import SPEC
from neurosignal.gold.drift import compute_day_offsets
from neurosignal.gold.nonlinear import fit_pooled_decoders, joint_features, score_pooled
from neurosignal.io.loaders import list_link_nwb_paths, load_link_sessions, session_date_from_path

OUT_CSV = RESULTS / ("exp002_mlp_heldout.csv" if SPEC.spec_id == "v1" else f"exp002_mlp_heldout_{SPEC.spec_id}.csv")
FORWARD_LAST_TRAIN = "20200320"
N_GROUPS = 5
COLUMNS = ["run_date", "commit", "split", "fold", "seed", "decoder", "session", "day", "held_out",
           "r2", "train_sessions", "fit_s", "macs_per_bin"]


def splits_d17(dates: list) -> list:
    """(split, fold, train_idx, test_idx) for every fit, sessions chosen by date (D-17)."""
    n = len(dates)
    train_fwd = [i for i, d in enumerate(dates) if d <= FORWARD_LAST_TRAIN]
    test_fwd = [i for i in range(n) if i not in train_fwd]
    out = [("forward", 0, train_fwd, test_fwd), ("in_sample", 0, list(range(n)), list(range(n)))]
    for g, block in enumerate(np.array_split(np.arange(n), N_GROUPS)):
        out.append(("grouped", g, [i for i in range(n) if i not in set(block)], [int(i) for i in block]))
    return out


def load_sessions(splits: dict) -> tuple:
    by_date = {session_date_from_path(p): p for p in list_link_nwb_paths()}
    sbp, tcr, vel, dates = load_link_sessions([by_date[d] for d in splits["sessions"]], include_tcr=True)
    assert dates == splits["sessions"]
    return sbp, tcr, vel, dates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="score and print; write nothing")
    args = parser.parse_args()

    commit, dirty = git_state()
    splits = load_splits()
    t0 = time.time()
    sbp, tcr, vel, dates = load_sessions(splits)
    joint, sbp_norm = joint_features(sbp, tcr)
    del sbp, tcr
    days = compute_day_offsets(dates).astype(int)
    print(f"loaded {len(dates)} sessions in {time.time() - t0:.0f} s · commit {commit}{' (dirty)' if dirty else ''}")

    rows, t_all, n_fits = [], time.time(), 0
    for split, fold, train_idx, test_idx in splits_d17(dates):
        if split != "in_sample":
            assert not set(train_idx) & set(test_idx), f"{split} fold {fold}: test sessions in training (ERR-02)"
        with_spec = split == "forward"
        # Score held-out sessions, and the training sessions as the in-sample comparison.
        scored = sorted(set(test_idx) | set(train_idx))
        for seed in SPEC.seeds:
            dec = fit_pooled_decoders(joint, vel, train_idx, seed, sessions_sbp_norm=sbp_norm if with_spec else None)
            n_fits += 1
            scores = score_pooled(dec, vel, scored, sbp_norm if with_spec else None)
            macs = dec.macs_per_bin
            for name, r2 in scores.items():
                for i, v in zip(scored, r2):
                    rows.append({
                        "run_date": date.today().isoformat(), "commit": commit, "split": split, "fold": fold,
                        "seed": seed, "decoder": name, "session": dates[i], "day": int(days[i]),
                        "held_out": "no" if i in set(train_idx) else "yes", "r2": f"{v:.4f}",
                        "train_sessions": len(train_idx), "fit_s": f"{dec.fit_s[name]:.1f}",
                        "macs_per_bin": macs[name],
                    })
            held = [i for i in test_idx if i not in set(train_idx)] or test_idx
            pos = [scored.index(i) for i in held]
            summary = "  ".join(f"{k} {np.mean(v[pos]):+.4f}" for k, v in scores.items())
            print(f"{split:9s} fold {fold} seed {seed} · {len(held)} sessions mean R²: {summary}"
                  f" · MLP fit {dec.fit_s['mlp']:.0f} s, {dec.mlp.n_iter_} iters", flush=True)

    print(f"{n_fits} pooled fits in {time.time() - t_all:.0f} s")
    if args.dry_run:
        print("dry run: nothing written")
    elif dirty:
        print("not written: uncommitted changes outside results/; commit first so rows trace to a commit")
    else:
        with OUT_CSV.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {OUT_CSV.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
