"""
EXP-005 (EDA-10): normaliser v2, its time constant and whether to carry state across sessions,
picked on the 17 early sessions and confirmed on the 13 later.

    PYTHONPATH=02_package/src:. python scripts/exp005_normaliser_v2.py              # writes results/exp005_*
    PYTHONPATH=02_package/src:. python scripts/exp005_normaliser_v2.py --dry-run    # prints only

Grid: τ ∈ {2 s … 30 min} × {reset, carry} (features.causal.normalize_sessions_sbp_causal_v2); the
normaliser runs over all 30 sessions in date order, only the scored pairs are split. Selection score:
mean cross-session R² over the ordered pairs among the selection sessions; variants whose same-day
median is more than 0.01 below normaliser v1's are dropped; reset wins if within 0.005 of the best.
Confirmation, chosen v2 vs v1 vs acausal r_norm: same-day R² (seeds 0–2), mean cross-session R² per
gap bin with session-bootstrap CIs, the M2 exit criterion, bench CFD, clip rate.
Refuses to write with uncommitted changes outside results/, so every row traces to a commit.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import date

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from bench.guards import is_causal  # noqa: E402
from bench.plan import leaking_plans, plan_scores  # noqa: E402
from bench.run import combine_seeds, designs, git_state, load_frozen_sessions, score_cross_day, score_same_day, summarise  # noqa: E402
from bench.scoreboard import REPO, RESULTS  # noqa: E402
from bench.splits import load_splits  # noqa: E402
from neurosignal.decoders.linear import SPEC, grid_offset, same_day_r2, session_design  # noqa: E402
from neurosignal.features.causal import (  # noqa: E402
    alpha_from_tau,
    causal_zscore_ewma_v2,
    normalize_sessions_sbp_causal_v2,
    normalize_sessions_sbp_gold_causal,
)
from neurosignal.features.normalize import normalize_sessions_sbp_gold  # noqa: E402
from neurosignal.metrics.drift import compute_day_offsets, compute_pairwise_r2_matrix  # noqa: E402
from neurosignal.metrics.stats import bootstrap_off_diagonal_mean, bootstrap_session_mean, ci95, off_diagonal_mean  # noqa: E402
from neurosignal.viz.drift_plots import plot_r2_by_gap_bin_with_ci  # noqa: E402

GRID_CSV = RESULTS / "exp005_selection.csv"
CONFIRM_CSV = RESULTS / "exp005_confirmation.csv"
FIGURE = RESULTS / "exp005_r2_by_gap.png"
SELECTION_LAST = "20200320"
TAUS_S = (2, 10, 30, 60, 180, 600, 1800)
SAME_DAY_TOLERANCE = 0.01   # a variant may lose at most this much selection-set same-day median vs v1
RESET_TOLERANCE = 0.005     # reset wins if within this of the best (no state kept between sessions)
EXIT_FRACTION = 0.02        # M2: v2 − acausal CI lower bound ≥ −0.02 × acausal same-day R², every gap bin
CLIP, ALPHA_SMOOTH = 6.0, 0.2
GAP_BINS = ((1, 7, "1–7 d"), (8, 30, "8–30 d"), (31, 10**6, "> 30 d"))


def v2(sbp, tau_s, carry):
    return normalize_sessions_sbp_causal_v2(sbp, tau_s=tau_s, carry=carry, alpha_smooth=ALPHA_SMOOTH,
                                            warmup=SPEC.warmup_bins, clip=CLIP)


def tau_label(tau_s: int) -> str:
    return f"{tau_s} s" if tau_s < 60 else f"{tau_s // 60} min"


def same_day(features, vel, idx) -> np.ndarray:
    """Held-out same-day R² per session in idx: mean of the folds, mean over seeds 0–2 (fold grid shifted per seed)."""
    out = []
    for i in idx:
        X, y, _ = session_design(features[i], vel[i], SPEC)
        out.append(np.mean([same_day_r2(X, y, SPEC, grid_offset(len(y), s, SPEC)).mean() for s in SPEC.seeds]))
    return np.array(out)


def pick(rows: list) -> dict:
    """Best eligible score; the best reset variant instead if it is within RESET_TOLERANCE."""
    eligible = [r for r in rows if r["eligible"]]
    best = max(eligible, key=lambda r: r["score"])
    resets = [r for r in eligible if not r["carry"]]
    best_reset = max(resets, key=lambda r: r["score"]) if resets else None
    return best_reset if best_reset and best_reset["score"] >= best["score"] - RESET_TOLERANCE else best


def bench_cfd(feats, vel, splits) -> dict:
    """The bench's CFD row for one feature set (same plans, seeds and bootstrap as bench.run)."""
    design = designs(feats, vel, splits["sessions"], SPEC)
    bins = {d: b for d, (_, _, b) in design.items()}
    plans = {seed: plan_scores(splits, bins, SPEC, seed) for seed in SPEC.seeds}
    allowed = {(p["cal"], p["test"]) for p in splits["pairs"]}
    leaks = [m for ps in plans.values() for m in leaking_plans(ps, SPEC, allowed)]
    if leaks:
        raise SystemExit(f"{len(leaks)} plans train on what they test: {leaks[:3]}")
    cross = score_cross_day(design, plans[SPEC.seeds[0]], SPEC)
    per_seed, boots = [], []
    for seed in SPEC.seeds:
        summary, boot, _ = summarise(score_same_day(design, plans[seed], SPEC), cross, splits, SPEC, True, seed)
        per_seed.append(summary)
        boots.append(boot)
    return combine_seeds(per_seed, boots, True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="compute and print; write nothing")
    args = parser.parse_args()

    commit, dirty = git_state()
    splits = load_splits()
    dates = splits["sessions"]
    t0 = time.time()
    sbp, vel = load_frozen_sessions(splits)
    sel = [i for i, d in enumerate(dates) if d <= SELECTION_LAST]
    con = [i for i in range(len(dates)) if i not in sel]
    print(f"loaded {len(dates)} sessions in {time.time() - t0:.0f} s · commit {commit}{' (dirty)' if dirty else ''} · "
          f"selection {len(sel)} ({dates[sel[0]]} → {dates[sel[-1]]}) · confirmation {len(con)} ({dates[con[0]]} → {dates[con[-1]]})")
    n_fits = 0

    # ── selection on the early sessions ──────────────────────────────────────
    v1 = normalize_sessions_sbp_gold_causal(sbp, alpha=0.01, alpha_smooth=ALPHA_SMOOTH)
    R = compute_pairwise_r2_matrix([v1[i] for i in sel], [vel[i] for i in sel], spec=SPEC)
    n_fits += len(sel) * (1 + SPEC.same_day_folds)
    v1_sel = {"score": off_diagonal_mean(R), "same_day_median": float(np.median(np.diag(R)))}
    print(f"  normaliser v1 (τ 2 s, reset)  · cross {v1_sel['score']:+.4f} · same-day median {v1_sel['same_day_median']:.4f}")
    grid = []
    for carry in (False, True):
        for tau in TAUS_S:
            t1 = time.time()
            f = v2(sbp[: sel[-1] + 1], tau, carry)  # carry runs in date order; nothing after the selection sessions is needed
            R = compute_pairwise_r2_matrix([f[i] for i in sel], [vel[i] for i in sel], spec=SPEC)
            n_fits += len(sel) * (1 + SPEC.same_day_folds)
            row = {"tau_s": tau, "carry": carry, "alpha": alpha_from_tau(tau), "score": off_diagonal_mean(R),
                   "same_day_median": float(np.median(np.diag(R)))}
            row["eligible"] = row["same_day_median"] >= v1_sel["same_day_median"] - SAME_DAY_TOLERANCE
            grid.append(row)
            print(f"  τ {tau_label(tau):>6s} {'carry' if carry else 'reset'} · cross {row['score']:+.4f} · same-day median "
                  f"{row['same_day_median']:.4f}{'' if row['eligible'] else ' (dropped)'} · {time.time() - t1:.0f} s", flush=True)
    chosen = pick(grid)
    resets = [r for r in grid if not r["carry"]]
    best_reset = max(resets, key=lambda r: r["score"])
    print(f"chosen: τ {tau_label(chosen['tau_s'])} {'carry' if chosen['carry'] else 'reset'} · score {chosen['score']:+.4f} · "
          f"best reset τ {tau_label(best_reset['tau_s'])} {best_reset['score']:+.4f}", flush=True)

    # ── confirmation on the later sessions ───────────────────────────────────
    conf = []

    def put(cid, quantity, value, draws=None, n=""):
        lo, hi = ci95(draws[~np.isnan(draws)]) if draws is not None else ("", "")
        excl = "" if draws is None else ("yes" if (lo > 0 or hi < 0) else "no")
        conf.append({"id": cid, "quantity": quantity, "value": value, "ci_lo": lo, "ci_hi": hi, "excludes_zero": excl, "n": n})
        ci = f"[{lo:+.4f}, {hi:+.4f}] excl. 0: {excl}" if draws is not None else ""
        print(f"  {cid:14s} {quantity:58s} {value:+.4f} {ci}", flush=True)

    feats = {"v2": v2(sbp, chosen["tau_s"], chosen["carry"]), "v1": v1,
             "acausal": normalize_sessions_sbp_gold(sbp, sigma_bins=2.5)}
    print("\nK1 same-day R² (mean over seeds 0–2), later sessions")
    sd = {k: same_day(f, vel, con) for k, f in feats.items()}
    n_fits += len(feats) * len(con) * SPEC.same_day_folds * len(SPEC.seeds)
    for k in feats:
        put(f"K1-{k}", f"same-day median, {k}", float(np.median(sd[k])), n=f"{len(con)} sessions")
    for other in ("v1", "acausal"):
        d = sd["v2"] - sd[other]
        put(f"K1-v2-{other}", f"same-day, v2 − {other} (mean of paired differences)", float(d.mean()),
            bootstrap_session_mean(d), f"{len(con)} sessions")

    # The later sessions have no pair 1–7 d apart (shortest gap 9 d), so that bin is reported on the
    # selection sessions, which chose the variant: not held out, and not part of the exit criterion.
    print("\nK2 mean cross-session R² by gap bin, ordered pairs")
    floor = -EXIT_FRACTION * float(np.median(sd["acausal"]))
    estimates, exit_ok = {k: {} for k in feats}, {}
    for scope, idx in (("later", con), ("selection", sel)):
        Rs = {k: compute_pairwise_r2_matrix([f[i] for i in idx], [vel[i] for i in idx], spec=SPEC) for k, f in feats.items()}
        n_fits += len(feats) * len(idx) * (1 + SPEC.same_day_folds)
        days = compute_day_offsets([dates[i] for i in idx])
        gap, off = np.abs(days[:, None] - days[None, :]), ~np.eye(len(idx), dtype=bool)
        masks = {"all": off, **{label: (gap >= lo) & (gap <= hi) & off for lo, hi, label in GAP_BINS}}
        for label, m in masks.items():
            if scope == "selection" and label != GAP_BINS[0][2]:
                continue
            if not m.any():
                print(f"  {scope} sessions, {label}: no pairs")
                continue
            tag = f"{scope}, {label}" + (" (not held out)" if scope == "selection" else "")
            n = f"{int(m.sum())} pairs / {len(idx)} sessions"
            for k in feats:
                draws = bootstrap_off_diagonal_mean(Rs[k], mask=m)
                value = off_diagonal_mean(Rs[k], mask=m)
                put(f"K2-{k}-{scope}-{label}", f"cross-session mean, {k}, {tag}", value, draws, n)
                estimates[k][tag] = (value, *ci95(draws[~np.isnan(draws)]))
            for other in ("v1", "acausal"):
                D = Rs["v2"] - Rs[other]
                draws = bootstrap_off_diagonal_mean(D, mask=m)
                put(f"K2-v2-{other}-{scope}-{label}", f"cross-session, v2 − {other}, {tag}", off_diagonal_mean(D, mask=m), draws, n)
                if other == "acausal" and scope == "later" and label != "all":
                    exit_ok[label] = ci95(draws[~np.isnan(draws)])[0] >= floor
    put("M2-floor", f"exit floor −{EXIT_FRACTION} × acausal same-day median", floor)
    for label, ok in exit_ok.items():
        put(f"M2-{label}", f"exit criterion holds, later sessions, {label} (1 = yes)", float(ok))

    names = {"v2": f"live v2 (τ {tau_label(chosen['tau_s'])}, {'carry' if chosen['carry'] else 'reset'})",
             "v1": "live v1 (τ 2 s, reset)", "acausal": "offline (whole session)"}
    order = [t for t in estimates["v2"] if t.startswith("selection")] + [t for t in estimates["v2"] if t.startswith("later") and "all" not in t]
    ticks = {t: t.replace("selection, ", "").replace(" (not held out)", "\n(early sessions,\nnot held out)").replace("later, ", "") for t in order}
    fig = plot_r2_by_gap_bin_with_ci({names[k]: {ticks[t]: estimates[k][t] for t in order} for k in feats},
                                     out_path=None if args.dry_run or dirty else FIGURE,
                                     title="Decoder R² by calibration → test gap")
    fig.clf()

    print("\nK4 bench CFD (all 30 sessions, 435 forward pairs)")
    t1 = time.time()
    for k in ("v2", "v1"):
        row = bench_cfd(feats[k], vel, splits)
        put(f"K4-{k}", f"CFD days, {k} (CI {row['cfd_ci95_lo']}–{row['cfd_ci95_hi']}, seeds {row['cfd_seed_min']}–{row['cfd_seed_max']})",
            float(row["cfd_days"]), n="435 pairs / 30 sessions")
        put(f"K4-{k}-same", f"bench same-day median, {k}", row["same_day_r2_median"])
        n_fits += 30 + 30 * SPEC.same_day_folds * len(SPEC.seeds)
    print(f"  bench {time.time() - t1:.0f} s")
    causal = is_causal(lambda s: v2(s, chosen["tau_s"], chosen["carry"]))
    put("causal", "future-perturbation check passes (1 = yes)", float(causal))

    state, scored, clipped = None, 0, 0
    for s in sbp:
        z, end = causal_zscore_ewma_v2(s, alpha_from_tau(chosen["tau_s"]), state=state, warmup=SPEC.warmup_bins, clip=CLIP)
        state = end if chosen["carry"] else None
        scored += z[SPEC.warmup_bins:].size
        clipped += int((np.abs(z[SPEC.warmup_bins:]) >= CLIP).sum())
    put("clip-rate", f"fraction of scored bins with |z| at the ±{CLIP:g} clip", clipped / scored, n=f"{scored} bin × channel")
    put("state-bytes", "state per channel, float64 (float32 = half)", 24.0)

    wall = time.time() - t0
    print(f"\n{n_fits} fits · wall time {wall:.0f} s")
    if args.dry_run:
        print("dry run: nothing written")
    elif dirty:
        print("not written: uncommitted changes outside results/; commit first so rows trace to a commit")
    else:
        stamp = {"run_date": date.today().isoformat(), "commit": commit}
        with GRID_CSV.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["run_date", "commit", "tau_s", "carry", "alpha", "score",
                                               "same_day_median", "eligible", "chosen"])
            w.writeheader()
            w.writerow({**stamp, "tau_s": 2, "carry": "v1", "alpha": 0.01, "score": f"{v1_sel['score']:.4f}",
                        "same_day_median": f"{v1_sel['same_day_median']:.4f}", "eligible": "", "chosen": ""})
            for r in grid:
                w.writerow({**stamp, **{k: (f"{v:.6g}" if k == "alpha" else f"{v:.4f}" if isinstance(v, float) else
                                        ("carry" if v else "reset") if k == "carry" else ("yes" if v else "no") if k == "eligible"
                                        else v) for k, v in r.items()}, "chosen": "yes" if r is chosen else ""})
        with CONFIRM_CSV.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["run_date", "commit", "id", "quantity", "value", "ci_lo", "ci_hi", "excludes_zero", "n"])
            w.writeheader()
            for r in conf:
                w.writerow({**stamp, **{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}})
        print(f"wrote {GRID_CSV.relative_to(REPO)}, {CONFIRM_CSV.relative_to(REPO)}, {FIGURE.relative_to(REPO)} ({wall:.0f} s)")


if __name__ == "__main__":
    main()
