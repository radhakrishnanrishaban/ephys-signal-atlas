"""
EXP-003 (EDA-09): is our Ridge baseline handicapped against the LINK paper's, which published
differences survive a 95 % CI, and does picking the causal α on separate sessions change it?

    PYTHONPATH=. python scripts/exp003_baseline_and_cis.py              # writes results/exp003_*.csv
    PYTHONPATH=. python scripts/exp003_baseline_and_cis.py --dry-run    # prints only

A  per session, on the paper's trial split: A1 the paper's ridge (gold/baseline.py), A2 our spec on the
   paper's inputs, A3 our spec on acausal r_norm.
B  CIs (gold/stats.py, session bootstrap, 2,000 draws) on the differences listed in EXP-003, recomputed
   with compute_pairwise_r2_matrix (current SPEC), plus the bench pair rows and EXP-002 rows of the same spec.
C  causal EWMA α picked on the 17 sessions up to 2020-03-20, reported on the 13 later sessions.
Refuses to write with uncommitted changes outside results/, so every row traces to a commit.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import date

import numpy as np
import pandas as pd

from bench.run import git_state
from bench.scoreboard import GAP_BINS, REPO, RESULTS
from bench.splits import load_splits
from neurosignal.gold.baseline import paper_ridge_r2, spec_r2_on_paper_split
from neurosignal.gold.decoder import SPEC
from neurosignal.gold.drift import (
    compute_day_offsets,
    compute_pairwise_r2_matrix,
    normalize_sessions_sbp_gold,
    normalize_sessions_sbp_zscore_only,
)
from neurosignal.gold.stats import (
    bootstrap_off_diagonal_mean,
    bootstrap_pair_statistic,
    bootstrap_session_mean,
    ci95,
    off_diagonal_mean,
    session_draws,
    weighted_median,
)
from neurosignal.gold.streaming import causal_smooth_ewma, causal_zscore_window, normalize_sessions_sbp_gold_causal
from neurosignal.io.loaders import LINK_KINEMATICS, list_link_nwb_paths, load_link_trial_session, session_date_from_path

SUFFIX = "" if SPEC.spec_id == "v1" else f"_{SPEC.spec_id}"  # EXP-003 ran on v1; later specs write new files
BASELINE_CSV = RESULTS / f"exp003_baseline{SUFFIX}.csv"
CIS_CSV = RESULTS / f"exp003_cis{SUFFIX}.csv"
BENCH_PAIRS = RESULTS / "bench_v0_pairs.csv"
EXP002_CSV = RESULTS / f"exp002_mlp_heldout{SUFFIX}.csv"
ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
ALPHA_SMOOTH = 0.2
WIN_BINS = 300
SELECTION_LAST = "20200320"
MONTH_PLUS_DAYS = 30


def median_ci(values: np.ndarray, seed: int = 0) -> tuple:
    """Median over sessions and the session-bootstrap CI of the median."""
    C = session_draws(len(values), seed=seed)
    draws = np.array([weighted_median(values, c) for c in C.astype(float)])
    return float(np.median(values)), *ci95(draws)


def ci_row(rows: list, rid: str, quantity: str, value: float, draws: np.ndarray, n: str, published: str) -> None:
    lo, hi = ci95(draws[~np.isnan(draws)])
    survives = "yes" if (lo > 0 or hi < 0) else "no"
    rows.append({"id": rid, "quantity": quantity, "value": value, "ci_lo": lo, "ci_hi": hi,
                 "excludes_zero": survives, "n": n, "published": published})
    print(f"{rid:4s} {quantity:58s} {value:+.4f}  [{lo:+.4f}, {hi:+.4f}]  excl. 0: {survives:3s}  (published {published})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="compute and print; write nothing")
    args = parser.parse_args()

    commit, dirty = git_state()
    splits = load_splits()
    dates = splits["sessions"]
    by_date = {session_date_from_path(p): p for p in list_link_nwb_paths()}
    t0 = time.time()
    sessions = [load_link_trial_session(by_date[d]) for d in dates]
    sbp = [s["sbp"] for s in sessions]
    vel = [s["kin"][:, LINK_KINEMATICS.index("index_velocity")] for s in sessions]
    days = compute_day_offsets(dates).astype(int)
    print(f"loaded {len(dates)} sessions in {time.time() - t0:.0f} s · commit {commit}{' (dirty)' if dirty else ''}")

    # ── A · baseline on the paper's split ────────────────────────────────────
    r_norm = normalize_sessions_sbp_gold(sbp, sigma_bins=2.5)
    base_rows = []
    for d, day, s, v, rn in zip(dates, days, sessions, vel, r_norm):
        a1 = paper_ridge_r2(s["sbp"], s["kin"], s["trial_start_bins"])
        a2 = spec_r2_on_paper_split(s["sbp"], v, s["trial_start_bins"], zscore_with_train=True)
        a3 = spec_r2_on_paper_split(rn, v, s["trial_start_bins"])
        base_rows.append({
            "session": d, "day": int(day), "trials": len(s["trial_start_bins"]),
            **{f"a1_{k}": a1[i] for i, k in enumerate(LINK_KINEMATICS)},
            "a1_mean4": float(a1.mean()), "a2_spec_zscored": a2["r2"], "a2_alpha": a2["alpha"],
            "a3_spec_rnorm": a3["r2"], "a3_alpha": a3["alpha"], "test_rows": a2["test_rows"],
        })
    base = pd.DataFrame(base_rows)
    a1v, a2v, a3v = base["a1_index_velocity"].to_numpy(), base["a2_spec_zscored"].to_numpy(), base["a3_spec_rnorm"].to_numpy()
    print("\nA · same-day R² on the paper's split, median over sessions [session-bootstrap CI of the median]")
    for name, x in [("A1 paper ridge, index velocity", a1v), ("A1 paper ridge, mean of 4 outputs", base["a1_mean4"].to_numpy()),
                    ("A2 our spec, z-scored SBP", a2v), ("A3 our spec, acausal r_norm", a3v),
                    ("A2 − A1 (paired)", a2v - a1v), ("A3 − A1 (paired)", a3v - a1v)]:
        m, lo, hi = median_ci(x)
        print(f"  {name:36s} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    for k in LINK_KINEMATICS:
        print(f"  A1 {k:16s} median {base['a1_' + k].median():+.4f}")

    cis: list = []
    for rid, name, x in [("A1", "paper ridge, index velocity (median)", a1v),
                         ("A1m", "paper ridge, mean of 4 outputs (median)", base["a1_mean4"].to_numpy()),
                         ("A2", "our spec on z-scored SBP (median)", a2v),
                         ("A3", "our spec on acausal r_norm (median)", a3v),
                         ("A2-1", "A2 − A1, paired (median)", a2v - a1v),
                         ("A3-1", "A3 − A1, paired (median)", a3v - a1v)]:
        C = session_draws(len(x)).astype(float)
        ci_row(cis, rid, name, float(np.median(x)), np.array([weighted_median(x, c) for c in C]), "30 sessions", "")

    # ── B · pairwise matrices, the package code behind every published number ─
    print(f"\nB · pairwise matrices (spec {SPEC.spec_id})", flush=True)
    feats = {
        "raw": [np.asarray(s, dtype=np.float64) for s in sbp],
        "r_norm": r_norm,
        "z_only": normalize_sessions_sbp_zscore_only(sbp),
        "causal_window": [causal_smooth_ewma(causal_zscore_window(s, win_bins=WIN_BINS), alpha_smooth=ALPHA_SMOOTH) for s in sbp],
        **{f"ewma_{a}": normalize_sessions_sbp_gold_causal(sbp, alpha=a, alpha_smooth=ALPHA_SMOOTH) for a in ALPHAS},
    }
    R, n_fits = {}, 0
    for name, f in feats.items():
        t1 = time.time()
        R[name] = compute_pairwise_r2_matrix(f, vel)
        n_fits += len(dates) * 6  # one whole-session fit + 5 same-day folds per session
        print(f"  {name:14s} mean off-diagonal {off_diagonal_mean(R[name]):+.4f} · Day-0 row mean "
              f"{R[name][0, 1:].mean():+.4f} · {time.time() - t1:.0f} s", flush=True)

    print("\nB · CIs")
    ewma = R["ewma_0.01"]
    for rid, q, A, B, pub in [("B1", "mean off-diag R², r_norm − raw", R["r_norm"], R["raw"], "+0.662"),
                              ("B2", "mean off-diag R², r_norm − z-score only (smoothing)", R["r_norm"], R["z_only"], "+0.004"),
                              ("B3", "mean off-diag R², causal EWMA − acausal r_norm", ewma, R["r_norm"], "−0.0061"),
                              ("B4", "mean off-diag R², causal window − causal EWMA", R["causal_window"], ewma, "−0.0008")]:
        ci_row(cis, rid, q, off_diagonal_mean(A - B), bootstrap_off_diagonal_mean(A - B), "870 pairs / 30 sessions", pub)
    for name in ("raw", "r_norm", "z_only", "causal_window", "ewma_0.01"):
        ci_row(cis, f"L-{name}", f"mean off-diag R², {name} (level)", off_diagonal_mean(R[name]),
               bootstrap_off_diagonal_mean(R[name]), "870 pairs / 30 sessions", "")

    bp = pd.read_csv(BENCH_PAIRS, dtype={"cal": str, "test": str})
    bp = bp[bp.spec == SPEC.spec_id]
    assert bp.commit.nunique() == 1, f"bench pairs for spec {SPEC.spec_id}: {bp.commit.unique()}"
    kept = bp.groupby(["feature", "cal", "test", "gap_days"]).agg(kept=("kept_ratio", "mean"), cross=("r2_cross", "first")).reset_index()
    piv = kept.pivot_table(index=["cal", "test", "gap_days"], columns="feature", values=["kept", "cross"]).reset_index()
    cal, test, gap = piv["cal"].to_numpy(), piv["test"].to_numpy(), piv["gap_days"].to_numpy()
    diff_of_medians = lambda v, w: weighted_median(v[:, 0], w) - weighted_median(v[:, 1], w)  # noqa: E731
    for rid, (lo_d, hi_d), pub in [("B5a", GAP_BINS["kept_1_7d"], "+0.0022"), ("B5b", GAP_BINS["kept_8_30d"], "+0.0105")]:
        m = (gap >= lo_d) & (gap <= hi_d)
        v = np.column_stack([piv[("kept", "r_norm_causal")], piv[("kept", "r_norm_acausal")]])[m]
        ci_row(cis, rid, f"bench kept ratio {lo_d}–{hi_d} d, causal − acausal (medians)",
               float(np.median(v[:, 0]) - np.median(v[:, 1])),
               bootstrap_pair_statistic(cal[m], test[m], v, diff_of_medians), f"{m.sum()} pairs", pub)
    m = gap >= MONTH_PLUS_DAYS
    v = np.column_stack([piv[("cross", "r_norm_causal")], piv[("cross", "r_norm_acausal")]])[m]
    ci_row(cis, "B6", "bench month+ R², causal − acausal (medians)", float(np.median(v[:, 0]) - np.median(v[:, 1])),
           bootstrap_pair_statistic(cal[m], test[m], v, diff_of_medians), f"{m.sum()} pairs", "+0.0075")

    row0 = {a: R[f"ewma_{a}"][0, 1:] for a in ALPHAS}
    for rid, other, pub in [("B7a", 0.003, "+0.0045"), ("B7b", 0.03, "+0.0012")]:
        d = row0[0.01] - row0[other]
        ci_row(cis, rid, f"α sweep, Day-0 decoder: α 0.01 − α {other}", float(d.mean()), bootstrap_session_mean(d),
               "29 test sessions", pub)

    e2 = pd.read_csv(EXP002_CSV, dtype={"session": str})
    fw = e2[(e2.split == "forward") & (e2.held_out == "yes")].groupby(["session", "decoder"]).r2.mean().unstack()
    d = (fw["mlp"] - fw["ridge_same"]).to_numpy()
    ci_row(cis, "B8", "EXP-002 forward: MLP − Ridge same inputs", float(d.mean()), bootstrap_session_mean(d),
           f"{len(d)} test sessions", "−0.070")
    ci_row(cis, "B8sp", f"EXP-002 forward: Ridge spec {SPEC.spec_id} pooled (level)", float(fw["ridge_spec"].mean()),
           bootstrap_session_mean(fw["ridge_spec"].to_numpy()), f"{len(d)} test sessions", "−0.003")

    month = days[1:] >= MONTH_PLUS_DAYS
    v = R["r_norm"][0, 1:][month]
    ci_row(cis, "B9", "r_norm Day-0 decoder, month+ mean R² (level)", float(v.mean()), bootstrap_session_mean(v),
           f"{month.sum()} test sessions", "−0.0465")
    print(f"  (r_norm Day-0 decoder: intra held out {R['r_norm'][0, 0]:.4f}; 1–7 d n = {int(((days >= 1) & (days <= 7)).sum())} "
          f"session(s): {R['r_norm'][0, 1:][(days[1:] <= 7)].round(4).tolist()})")

    # ── C · α picked on early sessions, reported on later ones ───────────────
    sel = np.array([d <= SELECTION_LAST for d in dates])
    held = ~sel
    print(f"\nC · α picked on {sel.sum()} early sessions ({sel.sum() * (sel.sum() - 1)} pairs), "
          f"reported on {held.sum()} later sessions ({held.sum() * (held.sum() - 1)} pairs)")
    sel_score = {a: off_diagonal_mean(R[f"ewma_{a}"][np.ix_(sel, sel)]) for a in ALPHAS}
    held_score = {a: off_diagonal_mean(R[f"ewma_{a}"][np.ix_(held, held)]) for a in ALPHAS}
    for a in ALPHAS:
        print(f"  α {a:<6} selection {sel_score[a]:+.4f} · held out {held_score[a]:+.4f}")
    picked = max(ALPHAS, key=lambda a: sel_score[a])
    d = R[f"ewma_{picked}"][np.ix_(held, held)] - R["ewma_0.01"][np.ix_(held, held)]
    ci_row(cis, "C", f"held-out mean off-diag R², picked α {picked} − α 0.01", off_diagonal_mean(d),
           bootstrap_off_diagonal_mean(d), f"{held.sum() * (held.sum() - 1)} pairs / {held.sum()} sessions", "")
    for a in ALPHAS:
        cis.append({"id": f"C-{a}", "quantity": f"α {a}: selection / held-out mean off-diag R²",
                    "value": held_score[a], "ci_lo": "", "ci_hi": "", "excludes_zero": "",
                    "n": f"selection {sel_score[a]:+.4f}", "published": "picked" if a == picked else ""})

    print(f"\n{n_fits} decoder fits for B and C, {len(dates) * 3} for A · wall time {time.time() - t0:.0f} s")
    if args.dry_run:
        print("dry run: nothing written")
    elif dirty:
        print("not written: uncommitted changes outside results/; commit first so rows trace to a commit")
    else:
        stamp = {"run_date": date.today().isoformat(), "commit": commit}
        base.insert(0, "commit", commit)
        base.insert(0, "run_date", stamp["run_date"])
        base.round(4).to_csv(BASELINE_CSV, index=False)
        with CIS_CSV.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["run_date", "commit", "id", "quantity", "value", "ci_lo", "ci_hi",
                                               "excludes_zero", "n", "published"])
            w.writeheader()
            for r in cis:
                w.writerow({**stamp, **{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}})
        print(f"wrote {BASELINE_CSV.relative_to(REPO)}, {CIS_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()
