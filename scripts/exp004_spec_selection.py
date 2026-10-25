"""
EXP-004 (EDA-09, D-18): pick the decoder spec's window and lag on the 17 early sessions, confirm on the 13 later.

    PYTHONPATH=. python scripts/exp004_spec_selection.py              # writes results/exp004_*.csv
    PYTHONPATH=. python scripts/exp004_spec_selection.py --dry-run    # prints only

Selection: mean held-out same-day R² (5 blocked folds, unshifted grid) over the selection sessions for
causal r_norm, window ∈ {2, 5, 7, 10} × lag ∈ {0, 1, 2, 3}; the smallest window within 0.005 of the best
wins, then the highest score. Confirmation on the later sessions, chosen spec vs v1: same-day R² (seeds 0–2),
causal ÷ acausal, and mean off-diagonal cross-session R², each with a session-bootstrap CI (gold/stats.py).
Refuses to write with uncommitted changes outside results/, so every row traces to a commit.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from datetime import date

import numpy as np

from bench.run import git_state, load_frozen_sessions
from bench.scoreboard import REPO, RESULTS
from bench.splits import load_splits
from neurosignal.gold.decoder import SPEC as _CURRENT, DecoderSpec, grid_offset, same_day_r2, session_design
from neurosignal.gold.drift import compute_pairwise_r2_matrix, normalize_sessions_sbp_gold
from neurosignal.gold.stats import (
    bootstrap_off_diagonal_mean,
    bootstrap_session_mean,
    ci95,
    off_diagonal_mean,
    session_draws,
    weighted_median,
)
from neurosignal.gold.streaming import normalize_sessions_sbp_gold_causal

GRID_CSV = RESULTS / "exp004_selection.csv"
CONFIRM_CSV = RESULTS / "exp004_confirmation.csv"
SELECTION_LAST = "20200320"
WINDOWS = (2, 5, 7, 10)
LAGS = (0, 1, 2, 3)
TOLERANCE = 0.005
SPEC = replace(_CURRENT, spec_id="v1", lag_bins=3)  # EXP-004 compares against spec v1 (D-16), the spec in force when it ran


def same_day(features, vel, idx, spec: DecoderSpec, seeds=(None,)) -> np.ndarray:
    """Held-out same-day R² per session in idx (mean of the folds, mean over seeds; None = unshifted grid)."""
    out = []
    for i in idx:
        X, y, _ = session_design(features[i], vel[i], spec)
        per_seed = [same_day_r2(X, y, spec, 0 if s is None else grid_offset(len(y), s, spec)).mean() for s in seeds]
        out.append(np.mean(per_seed))
    return np.array(out)


def pick(scores: dict) -> tuple:
    """Smallest window within TOLERANCE of the best score; then the highest score."""
    best = max(scores.values())
    near = [k for k, v in scores.items() if v >= best - TOLERANCE]
    w_min = min(w for w, _ in near)
    return max((k for k in near if k[0] == w_min), key=lambda k: scores[k])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="compute and print; write nothing")
    args = parser.parse_args()

    commit, dirty = git_state()
    splits = load_splits()
    dates = splits["sessions"]
    t0 = time.time()
    sbp, vel = load_frozen_sessions(splits)
    feats = {
        "r_norm_causal": normalize_sessions_sbp_gold_causal(sbp, alpha=0.01, alpha_smooth=0.2),
        "r_norm_acausal": normalize_sessions_sbp_gold(sbp, sigma_bins=2.5),
        "raw_sbp": [np.asarray(s, dtype=np.float64) for s in sbp],
    }
    sel = [i for i, d in enumerate(dates) if d <= SELECTION_LAST]
    con = [i for i in range(len(dates)) if i not in sel]
    print(f"loaded {len(dates)} sessions in {time.time() - t0:.0f} s · commit {commit}{' (dirty)' if dirty else ''} · "
          f"selection {len(sel)} ({dates[sel[0]]} → {dates[sel[-1]]}) · confirmation {len(con)} ({dates[con[0]]} → {dates[con[-1]]})")

    # ── selection on the early sessions ──────────────────────────────────────
    grid_rows, scores, n_fits = [], {}, 0
    for w in WINDOWS:
        for lag in LAGS:
            spec = replace(SPEC, spec_id=f"w{w}l{lag}", window_bins=w, lag_bins=lag)
            t1 = time.time()
            row = {"window": w, "lag": lag, "macs_per_bin": spec.decoder_macs_per_bin}
            for name in ("r_norm_causal", "r_norm_acausal"):
                r = same_day(feats[name], vel, sel, spec)
                row[name] = float(r.mean())
                n_fits += len(sel) * spec.same_day_folds
            scores[(w, lag)] = row["r_norm_causal"]
            grid_rows.append(row)
            print(f"  window {w:2d} lag {lag} · causal {row['r_norm_causal']:.4f} · acausal {row['r_norm_acausal']:.4f} · "
                  f"{time.time() - t1:.0f} s", flush=True)
    chosen = pick(scores)
    best_acausal = max(grid_rows, key=lambda r: r["r_norm_acausal"])
    v2 = replace(SPEC, spec_id="candidate", window_bins=chosen[0], lag_bins=chosen[1])
    print(f"chosen (causal r_norm): window {chosen[0]} lag {chosen[1]} · score {scores[chosen]:.4f} "
          f"(best {max(scores.values()):.4f} at {max(scores, key=scores.get)}) · acausal best: "
          f"window {best_acausal['window']} lag {best_acausal['lag']}", flush=True)

    # ── confirmation on the later sessions ───────────────────────────────────
    conf, sd = [], {}
    stamp = {"run_date": date.today().isoformat(), "commit": commit}

    def put(cid, quantity, value, draws=None, n=""):
        lo, hi = ci95(draws) if draws is not None else ("", "")
        excl = "" if draws is None else ("yes" if (lo > 0 or hi < 0) else "no")
        conf.append({"id": cid, "quantity": quantity, "value": value, "ci_lo": lo, "ci_hi": hi, "excludes_zero": excl, "n": n})
        ci = f"[{lo:+.4f}, {hi:+.4f}] excl. 0: {excl}" if draws is not None else ""
        print(f"  {cid:5s} {quantity:62s} {value:+.4f} {ci}", flush=True)

    print("\nconfirmation, same-day R² (mean over seeds 0–2), 13 later sessions")
    for name in feats:
        for tag, spec in (("v1", SPEC), ("chosen", v2)):
            sd[(name, tag)] = same_day(feats[name], vel, con, spec, seeds=SPEC.seeds)
            n_fits += len(con) * spec.same_day_folds * len(SPEC.seeds)
        d = sd[(name, "chosen")] - sd[(name, "v1")]
        put(f"C1-{name}", f"{name}: same-day mean, v1 → chosen ({sd[(name, 'v1')].mean():.4f} → "
            f"{sd[(name, 'chosen')].mean():.4f}), difference", float(d.mean()), bootstrap_session_mean(d), f"{len(con)} sessions")

    C = session_draws(len(con)).astype(float)
    ratio = {}
    for tag in ("v1", "chosen"):
        c, a = sd[("r_norm_causal", tag)], sd[("r_norm_acausal", tag)]
        ratio[tag] = (np.median(c) / np.median(a),
                      np.array([weighted_median(c, w) / weighted_median(a, w) for w in C]))
        put(f"C2-{tag}", f"causal ÷ acausal same-day (medians), {tag}", float(ratio[tag][0]))
    put("C2-d", "causal ÷ acausal, chosen − v1", float(ratio["chosen"][0] - ratio["v1"][0]),
        ratio["chosen"][1] - ratio["v1"][1], f"{len(con)} sessions")

    print("\nconfirmation, cross-session R² among the later sessions")
    R = {}
    for name in ("r_norm_causal", "r_norm_acausal"):
        for tag, spec in (("v1", SPEC), ("chosen", v2)):
            R[(name, tag)] = compute_pairwise_r2_matrix([feats[name][i] for i in con], [vel[i] for i in con], spec=spec)
            n_fits += len(con) * (1 + spec.same_day_folds)
            put(f"C3-{name[7:]}-{tag}", f"{name}: mean off-diag R², {tag}", off_diagonal_mean(R[(name, tag)]),
                bootstrap_off_diagonal_mean(R[(name, tag)]), f"{len(con) * (len(con) - 1)} pairs")
        dR = R[(name, "chosen")] - R[(name, "v1")]
        put(f"C3-{name[7:]}-d", f"{name}: mean off-diag R², chosen − v1", off_diagonal_mean(dR),
            bootstrap_off_diagonal_mean(dR), f"{len(con) * (len(con) - 1)} pairs")

    print(f"\nchosen spec: window {v2.window_bins}, lag {v2.lag_bins}, {v2.decoder_macs_per_bin} MACs/bin (v1 {SPEC.decoder_macs_per_bin})"
          f" · {n_fits} fits · wall time {time.time() - t0:.0f} s")
    if args.dry_run:
        print("dry run: nothing written")
    elif dirty:
        print("not written: uncommitted changes outside results/; commit first so rows trace to a commit")
    else:
        with GRID_CSV.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["run_date", "commit", "window", "lag", "macs_per_bin",
                                               "r_norm_causal", "r_norm_acausal", "chosen"])
            w.writeheader()
            for r in grid_rows:
                w.writerow({**stamp, **{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()},
                            "chosen": "yes" if (r["window"], r["lag"]) == chosen else ""})
        with CONFIRM_CSV.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["run_date", "commit", "id", "quantity", "value", "ci_lo", "ci_hi", "excludes_zero", "n"])
            w.writeheader()
            for r in conf:
                w.writerow({**stamp, **{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}})
        print(f"wrote {GRID_CSV.relative_to(REPO)}, {CONFIRM_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
