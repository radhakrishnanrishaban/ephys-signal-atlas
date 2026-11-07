"""
EXP-006 (EDA-11): the τ 2 s reset normaliser in int32 fixed point (features.fixed) against its float64
reference (features.causal.normalize_sessions_sbp_causal_v2, carry=False), on all 30 sessions and the
bench's 435 forward pairs.

    PYTHONPATH=02_package/src:. python scripts/exp006_fixed_point.py              # writes results/exp006_*
    PYTHONPATH=02_package/src:. python scripts/exp006_fixed_point.py --dry-run    # prints only

E1 feature error over scored bins; E2 saturations and clip rate; K1 same-day R² (seeds 0–2); K2 cross-session
R² over the forward pairs; K3 bench CFD; C1 cost table from the code's own op counts, with sqrt and div as one
op each and expanded into their digit-by-digit steps, totals at 96 and 1024 channels × 50 Hz. Only the feature
changes; the decoder is spec v2 in float64. Refuses to write with uncommitted changes outside results/.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import date

import numpy as np

from bench.guards import is_causal
from bench.plan import leaking_plans, plan_scores
from bench.run import combine_seeds, designs, git_state, load_frozen_sessions, score_cross_day, score_same_day, summarise
from bench.scoreboard import REPO, RESULTS
from bench.splits import load_splits
from neurosignal.decoders.linear import SPEC, grid_offset, same_day_r2, session_design
from neurosignal.features.causal import alpha_from_tau, causal_zscore_ewma_v2, normalize_sessions_sbp_causal_v2
from neurosignal.features.fixed import (
    KINDS,
    Q_V,
    Q_X,
    Q_Z,
    STATE_BYTES_PER_CH,
    FixedPointNormalizer,
    Ops,
    div_round,
    div_round_bitwise,
    fixed_point_sessions,
    isqrt,
    isqrt_bitwise,
    normalize_sessions_sbp_fixed,
    to_q16,
)
from neurosignal.metrics.drift import compute_pairwise_r2_matrix
from neurosignal.metrics.stats import bootstrap_off_diagonal_mean, bootstrap_session_mean, ci95, off_diagonal_mean

RESULTS_CSV = RESULTS / "exp006_results.csv"
COST_CSV = RESULTS / "exp006_cost.csv"
TAU_S, CLIP, ALPHA_SMOOTH = 2.0, 6.0, 0.2
CHANNELS = (96, 1024)
BINS_PER_S = 50
OPS_BUDGET = 10**5  # charter north star: ops per 20 ms bin


def same_day(features, vel) -> np.ndarray:
    """Held-out same-day R² per session: mean of the folds, mean over seeds 0–2 (fold grid shifted per seed)."""
    out = []
    for f, v in zip(features, vel):
        X, y, _ = session_design(f, v, SPEC)
        out.append(np.mean([same_day_r2(X, y, SPEC, grid_offset(len(y), s, SPEC)).mean() for s in SPEC.seeds]))
    return np.array(out)


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


def bitwise_check(sbp: np.ndarray) -> tuple:
    """
    Replay one session, keeping every σ argument (var · 2²⁰) and z numerator ((x − μ) · 2¹⁶) the normaliser
    divides; the digit-by-digit sqrt and division must give the same integers as the fast paths (after the
    ±clip, which is all the saturated division keeps). Returns (equal, sqrt step counts, div step counts).
    """
    x, _ = to_q16(sbp)
    norm = FixedPointNormalizer(x.shape[1], tau_s=TAU_S, warmup=SPEC.warmup_bins, clip=CLIP, alpha_smooth=ALPHA_SMOOTH)
    args, nums = [], []
    for row in x:
        norm.update(row)
        args.append(norm.var << Q_V)
        nums.append((row - norm.mu) << (Q_V + Q_Z - Q_X))
    args, nums = np.concatenate(args), np.concatenate(nums)
    sq_ops, div_ops = Ops(), Ops()
    sigma = isqrt(args)
    ok_sqrt = np.array_equal(isqrt_bitwise(args, sq_ops), sigma)
    sigma = np.maximum(sigma, 1)
    zmax = norm.z_max
    ok_div = np.array_equal(np.clip(div_round_bitwise(nums, sigma, div_ops), -zmax, zmax),
                            np.clip(div_round(nums, sigma), -zmax, zmax))
    return ok_sqrt and ok_div, sq_ops.count, div_ops.count


def expanded(base: dict, sq: dict, dv: dict) -> dict:
    """Per-channel counts with the one-op sqrt and the one-op rounded division replaced by their digit-by-digit steps."""
    out = dict(base)
    out["sqrt"] -= 1
    out["div"] -= 1
    out["shift"] -= 1  # Ops.div_round's rounding shift and add; div_round_bitwise counts its own
    out["add"] -= 1
    for k in KINDS:
        out[k] += sq[k] + dv[k]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="compute and print; write nothing")
    args = parser.parse_args()

    commit, dirty = git_state()
    splits = load_splits()
    t0 = time.time()
    sbp, vel = load_frozen_sessions(splits)
    n_ch = sbp[0].shape[1]
    print(f"loaded {len(sbp)} sessions in {time.time() - t0:.0f} s · commit {commit}{' (dirty)' if dirty else ''}")
    rows, cost_rows, n_fits = [], [], 0

    def put(cid, quantity, value, draws=None, n=""):
        lo, hi = ci95(draws[~np.isnan(draws)]) if draws is not None else ("", "")
        rows.append({"id": cid, "quantity": quantity, "value": value, "ci_lo": lo, "ci_hi": hi, "n": n})
        ci = f"[{lo:+.5f}, {hi:+.5f}]" if draws is not None else ""
        print(f"  {cid:16s} {quantity:62s} {value:+.6g} {ci}", flush=True)

    t1 = time.time()
    out_q, norm, input_sat = fixed_point_sessions(sbp, tau_s=TAU_S, warmup=SPEC.warmup_bins, clip=CLIP, alpha_smooth=ALPHA_SMOOTH)
    q = [o.astype(np.float64) / 2**Q_Z for o in out_q]
    print(f"fixed point {time.time() - t1:.0f} s")
    f = normalize_sessions_sbp_causal_v2(sbp, tau_s=TAU_S, carry=False, alpha_smooth=ALPHA_SMOOTH,
                                         warmup=SPEC.warmup_bins, clip=CLIP)

    print("\nE1 feature error |q − f| (z units)")
    w = SPEC.warmup_bins
    scored = np.concatenate([np.abs(a - b)[w:].ravel() for a, b in zip(q, f)])
    every = np.concatenate([np.abs(a - b).ravel() for a, b in zip(q, f)])
    n_scored = f"{scored.size} bin × channel"
    put("E1-max", "max, scored bins", float(scored.max()), n=n_scored)
    put("E1-mean", "mean, scored bins", float(scored.mean()), n=n_scored)
    put("E1-p9999", "99.99th percentile, scored bins", float(np.percentile(scored, 99.99)), n=n_scored)
    put("E1-max-all", "max, warm-up included", float(every.max()), n=f"{every.size} bin × channel")
    put("E1-mean-all", "mean, warm-up included", float(every.mean()), n=f"{every.size} bin × channel")
    put("E1-lsb", "output LSB, 2^-12", 2.0**-Q_Z)

    print("\nE2 saturations and clip rate")
    put("E2-saturated", "saturation events in the normaliser (other than the ±6 clip)", float(norm.saturated))
    put("E2-input-saturated", "input samples saturated converting to Q16.16", float(input_sat))
    put("E2-clip-q", "clip rate, fixed point (scored bins at ±6)", norm.clipped / scored.size, n=n_scored)
    clipped = 0
    for s in sbp:
        z, _ = causal_zscore_ewma_v2(s, alpha_from_tau(TAU_S), warmup=w, clip=CLIP)
        clipped += int((np.abs(z[w:]) >= CLIP).sum())
    put("E2-clip-f", "clip rate, float reference", clipped / scored.size, n=n_scored)

    feats = {"q": q, "f": f}
    print("\nK1 same-day R² (mean over seeds 0–2), all sessions")
    sd = {k: same_day(v, vel) for k, v in feats.items()}
    n_fits += 2 * len(sbp) * SPEC.same_day_folds * len(SPEC.seeds)
    for k in feats:
        put(f"K1-{k}", f"same-day median, {k}", float(np.median(sd[k])), n=f"{len(sbp)} sessions")
    d = sd["q"] - sd["f"]
    put("K1-q-f", "same-day, q − f (mean of paired differences)", float(d.mean()), bootstrap_session_mean(d), f"{len(sbp)} sessions")

    print("\nK2 cross-session R², forward pairs")
    Rs = {k: compute_pairwise_r2_matrix(v, vel, spec=SPEC) for k, v in feats.items()}
    n_fits += 2 * len(sbp) * (1 + SPEC.same_day_folds)
    forward = np.triu(np.ones((len(sbp), len(sbp)), dtype=bool), k=1)  # R[i, j]: fit on i, score on later j
    n_pairs = f"{int(forward.sum())} pairs / {len(sbp)} sessions"
    for k in feats:
        put(f"K2-{k}", f"cross-session mean, {k}", off_diagonal_mean(Rs[k], mask=forward),
            bootstrap_off_diagonal_mean(Rs[k], mask=forward), n_pairs)
    D = Rs["q"] - Rs["f"]
    put("K2-q-f", "cross-session, q − f", off_diagonal_mean(D, mask=forward), bootstrap_off_diagonal_mean(D, mask=forward), n_pairs)
    put("K2-q-f-maxabs", "largest |q − f| over the pairs", float(np.abs(D[forward]).max()), n=n_pairs)

    print("\nK3 bench CFD")
    t1 = time.time()
    for k in feats:
        row = bench_cfd(feats[k], vel, splits)
        put(f"K3-{k}", f"CFD days, {k} (CI {row['cfd_ci95_lo']}–{row['cfd_ci95_hi']}, seeds {row['cfd_seed_min']}–{row['cfd_seed_max']})",
            float(row["cfd_days"]), n="435 pairs / 30 sessions")
        put(f"K3-{k}-same", f"bench same-day median, {k}", row["same_day_r2_median"])
        n_fits += 30 + 30 * SPEC.same_day_folds * len(SPEC.seeds)
    print(f"  bench {time.time() - t1:.0f} s")
    put("causal", "future-perturbation check passes, fixed point (1 = yes)", float(is_causal(normalize_sessions_sbp_fixed)))

    print("\nC1 cost")
    ok, sq, dv = bitwise_check(sbp[0])
    put("C1-bitwise-equal", "digit-by-digit sqrt and division equal the fast paths, session 1 (1 = yes)", float(ok),
        n=f"{sbp[0].size} bin × channel")
    per = norm.ops_per_channel_bin()
    shared_div_per_warmup_bin = norm.ops_shared.count["div"] / norm.bins["warmup"]
    for phase in ("steady", "warmup"):
        for counting, counts in (("hardware sqrt/div", per[phase]), ("expanded sqrt/div", expanded(per[phase], sq, dv))):
            total = sum(counts.values())
            cost_rows.append({"phase": phase, "counting": counting, **{k: counts[k] for k in KINDS}, "total_per_ch_bin": total})
            print(f"  {phase:6s} {counting:18s} " + "  ".join(f"{k} {counts[k]:g}" for k in KINDS) + f"  total {total:g}")
    put("C1-shared-div", "shared divisions per warm-up bin (1/n, all channels)", shared_div_per_warmup_bin)
    put("C1-state", "state bytes per channel (μ, var, y as 32-bit)", float(STATE_BYTES_PER_CH))
    steady_hw = sum(per["steady"].values())
    steady_ex = sum(expanded(per["steady"], sq, dv).values())
    put("C1-norm-ch-hw", "normaliser ops per channel per bin, steady, sqrt and div = 1 op", float(steady_hw))
    put("C1-norm-ch-ex", "normaliser ops per channel per bin, steady, sqrt and div expanded", float(steady_ex))
    put("C1-sqrtdiv-share", "share of sqrt and div steps in the expanded count", (sum(sq.values()) + sum(dv.values())) / steady_ex)
    for c in CHANNELS:
        macs = c * (SPEC.window_bins + 1)
        for label, per_ch in (("hw", steady_hw), ("ex", steady_ex)):
            total = per_ch * c + 2 * macs
            put(f"C1-{c}-{label}-bin", f"{c} ch, {'sqrt/div = 1 op' if label == 'hw' else 'sqrt/div expanded'}: normaliser + decoder ops per bin",
                float(total), n=f"decoder {macs} MACs = {2 * macs} ops; budget {OPS_BUDGET}")
            put(f"C1-{c}-{label}-s", f"{c} ch, {'sqrt/div = 1 op' if label == 'hw' else 'sqrt/div expanded'}: ops per second at {BINS_PER_S} Hz",
                float(total * BINS_PER_S))
        put(f"C1-{c}-sqrt-div-s", f"{c} ch: sqrt and divisions per second (each)", float(c * BINS_PER_S))
        put(f"C1-{c}-state", f"{c} ch: normaliser state bytes", float(c * STATE_BYTES_PER_CH))
        put(f"C1-{c}-weights", f"{c} ch: decoder coefficients (float64, not converted)", float(macs + 1))

    wall = time.time() - t0
    print(f"\n{n_fits} fits · wall time {wall:.0f} s · {n_ch} channels")
    if args.dry_run:
        print("dry run: nothing written")
    elif dirty:
        print("not written: uncommitted changes outside results/; commit first so rows trace to a commit")
    else:
        stamp = {"run_date": date.today().isoformat(), "commit": commit}
        with RESULTS_CSV.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["run_date", "commit", "id", "quantity", "value", "ci_lo", "ci_hi", "n"])
            wr.writeheader()
            for r in rows:
                wr.writerow({**stamp, **{k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()}})
        with COST_CSV.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["run_date", "commit", "phase", "counting", *KINDS, "total_per_ch_bin"])
            wr.writeheader()
            for r in cost_rows:
                wr.writerow({**stamp, **{k: (f"{v:g}" if isinstance(v, float) else v) for k, v in r.items()}})
        print(f"wrote {RESULTS_CSV.relative_to(REPO)}, {COST_CSV.relative_to(REPO)} ({wall:.0f} s)")


if __name__ == "__main__":
    main()
