"""
Score every feature set on the frozen splits and write the scoreboard.

    python -m bench.run              # writes results/ (refuses on uncommitted changes outside results/)
    python -m bench.run --dry-run    # prints only

Writes results/scoreboard.csv (one row per commit × spec × feature; the source of truth),
results/SCOREBOARD.md (rendered from the CSV) and results/bench_v0_pairs.csv (every pair × seed).
Nothing is written unless every guardrail passes (D-13): no plan shares a bin between training
and test, ≥ 3 seeds, cost columns filled. CFD and its CI are written only for causal features.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import date
from typing import Dict, List

import numpy as np
from sklearn.metrics import r2_score

from bench.cfd import cfd_bootstrap, cfd_days, kept_ratio
from bench.guards import is_causal
from bench.plan import leaking_plans, plan_scores
from bench.scoreboard import GAP_BINS, REPO, RESULTS, SCOREBOARD_COLUMNS, SCOREBOARD_CSV, SCOREBOARD_MD, render
from bench.spec import FEATURE_STATE_BYTES_PER_CH, FEATURES, SPEC, DecoderSpec
from bench.splits import SPLITS_PATH, load_splits
from neurosignal.gold.decoder import GramRidge, fit_decoder, session_design
from neurosignal.io.loaders import list_link_nwb_paths, load_link_sessions, session_date_from_path

PAIRS_CSV = RESULTS / "bench_v0_pairs.csv"
MONTH_PLUS_DAYS = 30
COST_COLUMNS = ("decoder_macs_per_bin", "fits", "runtime_s")
MIN_SEEDS = 3


def git_state() -> tuple:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", ":(exclude)results"],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    return sha.stdout.strip(), bool(dirty.stdout.strip())


def load_frozen_sessions(splits: Dict) -> tuple:
    by_date = {session_date_from_path(p): p for p in list_link_nwb_paths()}
    missing = [d for d in splits["sessions"] if d not in by_date]
    if missing:
        raise SystemExit(f"sessions in {SPLITS_PATH.name} missing from the data dir: {missing}")
    sbp, vel, dates = load_link_sessions([by_date[d] for d in splits["sessions"]])
    assert dates == splits["sessions"]
    return sbp, vel


def designs(features: List[np.ndarray], vel: List[np.ndarray], dates: List[str], spec: DecoderSpec) -> Dict:
    """(X, y, bins) per session from the package's session_design (warm-up trimmed, D-15)."""
    return {d: session_design(f, v, spec) for d, f, v in zip(dates, features, vel)}


def score_cross_day(design: Dict, plans: List, spec: DecoderSpec) -> Dict[tuple, float]:
    """Fit once per calibration session on all its rows; score every later session in its pairs."""
    cross: Dict[tuple, float] = {}
    models: Dict[str, GramRidge] = {}
    for p in plans:
        if p.kind != "cross_day":
            continue
        if p.cal not in models:
            X_cal, y_cal, _ = design[p.cal]
            models[p.cal] = fit_decoder(X_cal[p.train_rows], y_cal[p.train_rows], spec)
        X_t, y_t, _ = design[p.test]
        cross[(p.cal, p.test)] = r2_score(y_t[p.test_rows], models[p.cal].predict(X_t[p.test_rows]))
    return cross


def score_same_day(design: Dict, plans: List, spec: DecoderSpec) -> Dict[str, float]:
    """Mean held-out R² over the blocked folds of each session (the CFD denominator)."""
    same: Dict[str, List[float]] = {}
    for p in plans:
        if p.kind != "same_day":
            continue
        X, y, _ = design[p.cal]
        model = fit_decoder(X[p.train_rows], y[p.train_rows], spec)
        same.setdefault(p.cal, []).append(r2_score(y[p.test_rows], model.predict(X[p.test_rows])))
    return {d: float(np.mean(v)) for d, v in same.items()}


def summarise(r2_same: Dict, cross: Dict, splits: Dict, spec: DecoderSpec, causal: bool, seed: int) -> tuple:
    pairs = splits["pairs"]
    gaps = np.array([p["gap_days"] for p in pairs])
    r2_cross = np.array([cross[(p["cal"], p["test"])] for p in pairs])
    r2_cal = np.array([r2_same[p["cal"]] for p in pairs])
    ratio = kept_ratio(r2_cross, r2_cal)
    summary = {
        "same_day_r2_median": float(np.median(list(r2_same.values()))),
        **{k: float(np.median(ratio[(gaps >= lo) & (gaps <= hi)])) for k, (lo, hi) in GAP_BINS.items()},
        "month_plus_r2_median": float(np.median(r2_cross[gaps >= MONTH_PLUS_DAYS])),
    }
    boot = np.array([], dtype=int)
    if causal:
        summary["cfd_days"] = cfd_days(gaps, ratio, spec.kept_threshold)
        cal = np.array([p["cal"] for p in pairs])
        test = np.array([p["test"] for p in pairs])
        boot = cfd_bootstrap(cal, test, gaps, ratio, spec.kept_threshold, spec.n_boot, seed)
    rows = [
        {**p, "seed": seed, "r2_same_cal": r2_cal[i], "r2_cross": r2_cross[i], "kept_ratio": ratio[i]}
        for i, p in enumerate(pairs)
    ]
    return summary, boot, rows


def combine_seeds(per_seed: List[Dict], boots: List[np.ndarray], causal: bool) -> Dict:
    """Medians over seeds; CFD also gets its seed range and the CI of all draws pooled (D-13)."""
    out = {k: float(np.median([s[k] for s in per_seed])) for k in per_seed[0] if k != "cfd_days"}
    if causal:
        cfds = [s["cfd_days"] for s in per_seed]
        pooled = np.concatenate(boots)
        out.update({
            "cfd_days": int(np.median(cfds)),
            "cfd_ci95_lo": int(np.percentile(pooled, 2.5)),
            "cfd_ci95_hi": int(np.percentile(pooled, 97.5)),
            "cfd_seed_min": int(min(cfds)),
            "cfd_seed_max": int(max(cfds)),
        })
    return out


def guardrail_failures(summary: Dict) -> List[str]:
    """What stops a row from being written (D-13); leakage is checked on the plans before scoring."""
    failed = []
    if summary["seeds"] < MIN_SEEDS:
        failed.append(f"{summary['seeds']} seeds < {MIN_SEEDS}")
    failed += [f"cost column {c} empty" for c in COST_COLUMNS if summary.get(c) in (None, "")]
    if summary["causal"] == "yes" and summary.get("feature_state_bytes_per_ch") is None:
        failed.append("causal feature without a state size")
    return failed


def _fmt(v) -> str:
    if v is None:
        return ""
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def write_results(results: Dict, pair_rows: Dict, commit: str, spec: DecoderSpec, splits: Dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    key = (commit, spec.spec_id)
    old = []
    if SCOREBOARD_CSV.exists():
        with SCOREBOARD_CSV.open() as fh:
            old = [r for r in csv.DictReader(fh) if (r["commit"], r["spec"]) != key]
    new = [
        {"run_date": date.today().isoformat(), "commit": commit, "spec": spec.spec_id,
         "splits": splits["version"], "feature": name, **{k: _fmt(v) for k, v in s.items()}}
        for name, s in results.items()
    ]
    with SCOREBOARD_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCOREBOARD_COLUMNS, restval="")
        w.writeheader()
        w.writerows(old + new)

    with PAIRS_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["commit", "spec", "feature", "seed", "cal", "test", "gap_days",
                                           "r2_same_cal", "r2_cross", "kept_ratio"])
        w.writeheader()
        for name, rows in pair_rows.items():
            w.writerows({"commit": commit, "spec": spec.spec_id, "feature": name,
                         **{k: _fmt(v) for k, v in r.items()}} for r in rows)

    render(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="score and print; write nothing")
    args = parser.parse_args()

    commit, dirty = git_state()
    splits = load_splits()
    t0 = time.time()
    sbp, vel = load_frozen_sessions(splits)
    print(f"loaded {len(sbp)} sessions in {time.time() - t0:.0f} s · commit {commit}{' (dirty)' if dirty else ''}")

    results, pair_rows = {}, {}
    allowed = {(p["cal"], p["test"]) for p in splits["pairs"]}
    for name, make in FEATURES.items():
        t1 = time.time()
        causal = is_causal(make)
        design = designs(make(sbp), vel, splits["sessions"], SPEC)
        bins = {d: b for d, (_, _, b) in design.items()}
        plans = {seed: plan_scores(splits, bins, SPEC, seed) for seed in SPEC.seeds}
        leaks = [msg for seed_plans in plans.values() for msg in leaking_plans(seed_plans, SPEC, allowed)]
        if leaks:
            raise SystemExit(f"{name}: {len(leaks)} plans train on what they test, nothing written: {leaks[:3]}")

        cross = score_cross_day(design, plans[SPEC.seeds[0]], SPEC)  # the same for every seed
        per_seed, boots, rows = [], [], []
        for seed in SPEC.seeds:
            summary, boot, seed_rows = summarise(score_same_day(design, plans[seed], SPEC), cross, splits, SPEC, causal, seed)
            per_seed.append(summary)
            boots.append(boot)
            rows += seed_rows
        fits = len({p.cal for p in plans[SPEC.seeds[0]] if p.kind == "cross_day"}) + sum(
            p.kind == "same_day" for seed_plans in plans.values() for p in seed_plans
        )
        state = FEATURE_STATE_BYTES_PER_CH[name]
        results[name] = {
            "causal": "yes" if causal else "no",
            "seeds": len(SPEC.seeds),
            "sessions": len(splits["sessions"]),
            "pairs": len(splits["pairs"]),
            **combine_seeds(per_seed, boots, causal),
            "decoder_macs_per_bin": SPEC.decoder_macs_per_bin,
            "feature_state_bytes_per_ch": state,
            "fits": fits,
            "runtime_s": round(time.time() - t1, 1),
        }
        pair_rows[name] = rows
        failed = guardrail_failures(results[name])
        if failed:
            raise SystemExit(f"{name}: guardrail failed, nothing written: {failed}")
        print(f"{name:15s} " + "  ".join(f"{k}={_fmt(v)}" for k, v in results[name].items()))

    if args.dry_run:
        print("dry run: nothing written")
    elif dirty:
        print("not written: uncommitted changes outside results/; commit first so rows trace to a commit")
    else:
        write_results(results, pair_rows, commit, SPEC, splits)
        print(f"wrote {SCOREBOARD_CSV.relative_to(REPO)}, {SCOREBOARD_MD.relative_to(REPO)}, {PAIRS_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
