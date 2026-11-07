"""
Question: can the per-session normalisation run live, using only past samples, and what does that cost?

    PYTHONPATH=02_package/src:. python -m analyses.causal_norm                # compute, write JSON, figure, report
    PYTHONPATH=02_package/src:. python -m analyses.causal_norm --report-only  # re-render the report from the JSON

Acausal r_norm (whole-session z-score + Gaussian) vs the causal EWMA version the bench scores
(α 0.01, smoother 0.2) and a trailing-window variant. Causality: the bench's future-perturbation check
plus batch == sample-by-sample replay. Same-day R²: the bench's purged blocked folds, seeds 0-2 (D-13).
Cross-day: pairwise matrices from the package spec, session-bootstrap CIs (EXP-003 B3, B4). α sweep:
Day-0 decoder on later sessions (EXP-003 B7). State per channel from the bench cost column.
Same computation as sbp-causal-streaming_v1 cells 6-17, EXP-003 B and the scoreboard rows, now one command.
With uncommitted changes outside results/, the JSON records dirty: true and the report says Draft.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from analyses.common import OUT, Timer, estimate, fmt, frozen_sessions, provenance, read_json, write_json  # noqa: E402
from bench.guards import is_causal  # noqa: E402
from bench.plan import plan_scores  # noqa: E402
from bench.run import designs, score_same_day  # noqa: E402
from bench.spec import FEATURE_STATE_BYTES_PER_CH, FEATURES  # noqa: E402
from bench.splits import load_splits  # noqa: E402
from neurosignal.decoders.linear import SPEC  # noqa: E402
from neurosignal.features.causal import (  # noqa: E402
    EwmaNormalizer,
    causal_smooth_ewma,
    causal_zscore_ewma,
    causal_zscore_window,
    normalize_sessions_sbp_gold_causal,
    stream_session,
)
from neurosignal.metrics.drift import compute_decoder_r2_per_day, compute_pairwise_r2_matrix  # noqa: E402
from neurosignal.metrics.stats import (  # noqa: E402
    bootstrap_off_diagonal_mean,
    bootstrap_session_mean,
    off_diagonal_mean,
    session_draws,
    weighted_median,
)
from neurosignal.viz.drift_plots import plot_pairwise_r2_raw_vs_norm  # noqa: E402

NAME = "causal_norm"
QUESTION = "Can the per-session normalisation run live, using only past samples, and what does that cost?"
ALPHA, ALPHA_SMOOTH, WIN_BINS = 0.01, 0.2, 300  # bench r_norm_causal; window variant as in EXP-003
ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
STATE_TARGET_BYTES_PER_CH = 16  # charter R2

MAKE = {
    "acausal": FEATURES["r_norm_acausal"],
    "causal": FEATURES["r_norm_causal"],
    "causal_window": lambda sessions: [causal_smooth_ewma(causal_zscore_window(s, win_bins=WIN_BINS), alpha_smooth=ALPHA_SMOOTH)
                                       for s in sessions],
}


def same_day_by_seed(feats: list, vel: list, splits: dict) -> dict:
    """Per seed, held-out same-day R² of every session (bench plans, warm-up never scored)."""
    design = designs(feats, vel, splits["sessions"], SPEC)
    bins = {d: b for d, (_, _, b) in design.items()}
    return {seed: score_same_day(design, plan_scores(splits, bins, SPEC, seed), SPEC) for seed in SPEC.seeds}


def ratio_of_medians(num: dict, den: dict, dates: list) -> tuple:
    """Ratio of the scoreboard medians (median over seeds of the session median); CI from session draws pooled over seeds (D-13)."""
    medians, draws = {"num": [], "den": []}, []
    for seed in SPEC.seeds:
        a, b = (np.array([d[seed][s] for s in dates]) for d in (num, den))
        medians["num"].append(np.median(a))
        medians["den"].append(np.median(b))
        C = session_draws(len(dates), seed=seed)
        draws.append([weighted_median(a[c > 0], c[c > 0]) / weighted_median(b[c > 0], c[c > 0]) for c in C])
    return float(np.median(medians["num"]) / np.median(medians["den"])), np.concatenate(draws)


def compute() -> dict:
    with Timer() as t:
        sbp, vel, dates = frozen_sessions()
        splits = load_splits()
        feats = {k: make(sbp) for k, make in MAKE.items()}

        x0 = np.asarray(sbp[0], dtype=np.float64)
        norm = EwmaNormalizer(n_channels=x0.shape[1], alpha=ALPHA)
        replay = np.array([norm.update(x) for x in stream_session(x0)])
        causality = {k: is_causal(make) for k, make in MAKE.items()}
        causality["batch_equals_replay"] = bool(np.allclose(causal_zscore_ewma(x0, alpha=ALPHA), replay))

        same = {k: same_day_by_seed(feats[k], vel, splits) for k in ("acausal", "causal")}
        R = {k: compute_pairwise_r2_matrix(f, vel) for k, f in feats.items()}
        sweep = {a: compute_decoder_r2_per_day(normalize_sessions_sbp_gold_causal(sbp, alpha=a, alpha_smooth=ALPHA_SMOOTH), vel)[0][1:]
                 for a in ALPHAS}

    n_s = f"{len(dates)} sessions × {len(SPEC.seeds)} seeds"
    pairs = f"{len(dates) * (len(dates) - 1)} pairs / {len(dates)} sessions"
    later = f"{len(dates) - 1} later sessions"
    medians = {k: {str(s): round(float(np.median(list(v[s].values()))), 4) for s in SPEC.seeds} for k, v in same.items()}
    ratio, ratio_draws = ratio_of_medians(same["causal"], same["acausal"], dates)

    OUT.joinpath(NAME).mkdir(parents=True, exist_ok=True)
    fig = plot_pairwise_r2_raw_vs_norm(R["acausal"], R["causal"], dates, out_path=OUT / NAME / "pairwise_r2_acausal_vs_causal.png",
                                       title_raw="Offline (whole-session statistics)", title_norm="Live (past samples only)",
                                       clip_floor=-1.0, cmap="viridis")
    fig.clf()

    return {
        "question": QUESTION,
        **provenance(),
        "sessions": {"n": len(dates), "first": dates[0], "last": dates[-1]},
        "settings": {"alpha": ALPHA, "alpha_smooth": ALPHA_SMOOTH, "window_bins": WIN_BINS, "warmup_bins": SPEC.warmup_bins},
        "causality": causality,
        "same_day_r2": {
            "median_by_seed": medians,
            "median": {k: round(float(np.median(list(m.values()))), 4) for k, m in medians.items()},
            "causal_over_acausal": estimate(ratio, ratio_draws, n_s),
        },
        "cross_session_r2": {k: estimate(off_diagonal_mean(R[k]), bootstrap_off_diagonal_mean(R[k]), pairs) for k in R},
        "differences": {
            "causal_minus_acausal": estimate(off_diagonal_mean(R["causal"] - R["acausal"]),
                                             bootstrap_off_diagonal_mean(R["causal"] - R["acausal"]), pairs),
            "window_minus_causal": estimate(off_diagonal_mean(R["causal_window"] - R["causal"]),
                                            bootstrap_off_diagonal_mean(R["causal_window"] - R["causal"]), pairs),
        },
        "alpha_sweep_day0_decoder": {
            "mean_r2": {str(a): round(float(v.mean()), 4) for a, v in sweep.items()},
            "alpha_0.01_minus_0.003": estimate((sweep[0.01] - sweep[0.003]).mean(), bootstrap_session_mean(sweep[0.01] - sweep[0.003]), later),
            "alpha_0.01_minus_0.03": estimate((sweep[0.01] - sweep[0.03]).mean(), bootstrap_session_mean(sweep[0.01] - sweep[0.03]), later),
        },
        "state_bytes_per_channel": {"causal": FEATURE_STATE_BYTES_PER_CH["r_norm_causal"], "target": STATE_TARGET_BYTES_PER_CH,
                                    "acausal": None},
        "figure": f"{NAME}/pairwise_r2_acausal_vs_causal.png",
        "cost": {"decoder_macs_per_bin": SPEC.decoder_macs_per_bin,
                 "decoder_fits": 2 * len(dates) * SPEC.same_day_folds * len(SPEC.seeds)
                                 + 3 * len(dates) * (1 + SPEC.same_day_folds) + len(ALPHAS),
                 "runtime_s": t.seconds},
    }


def render(r: dict) -> str:
    """The Q&A report. Every number comes from the JSON; the wording follows the CIs."""
    c, sd, cs, d, sw = r["causality"], r["same_day_r2"], r["cross_session_r2"], r["differences"], r["alpha_sweep_day0_decoder"]
    st, s = r["state_bytes_per_channel"], r["settings"]
    live = c["causal"] and c["causal_window"] and c["batch_equals_replay"]
    rat = sd["causal_over_acausal"]
    pct = lambda x: f"{100 * x:.0f} %"  # noqa: E731
    within = f"{pct(rat['value'])} of the offline version's (95 % CI {pct(rat['ci95'][0])} to {pct(rat['ci95'][1])})"
    cm = d["causal_minus_acausal"]
    if not cm["excludes_zero"]:
        between = f"no detectable difference between days: {fmt(cm)}"
    elif cm["value"] > 0:
        between = f"slightly better transfer between days: {fmt(cm)}"
    else:
        between = f"worse transfer between days: {fmt(cm)}"
    wm = d["window_minus_causal"]
    window = ("makes no detectable difference" if not wm["excludes_zero"] else
              "transfers better" if wm["value"] > 0 else "transfers worse")
    best = max(sw["mean_r2"], key=lambda a: sw["mean_r2"][a])
    verdict = "no" if not live else ("yes, at a cost" if cm["excludes_zero"] and cm["value"] < 0 else "yes")
    fits_budget = st["causal"] <= st["target"]
    status = "Answered" if not r["dirty"] else "Draft (uncommitted code)"
    iso = lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}"  # noqa: E731
    yn = lambda b: "passes" if b else "**fails**"  # noqa: E731
    return f"""# {r['question']}

**Status:** {status} · decoder spec `{r['spec']}` · commit `{r['commit']}` · {r['run_date']}

**Short answer: {verdict}.** A version that only uses past samples (running mean and variance, α {s['alpha']}) {'passes' if live else 'fails'} the causality checks. Within a session, its decoder reaches a median R² of {sd['median']['causal']:.3f} against {sd['median']['acausal']:.3f} offline: {within}. Across sessions it shows {between}. It keeps {st['causal']} bytes of state per channel, {'within' if fits_budget else 'above'} the {st['target']}-byte target.

![Pairwise decoder R², offline vs live normalisation: train session on the vertical axis, test session on the horizontal]({r['figure']})

<details><summary>Details</summary>

| Quantity | Value |
| --- | --- |
| Causality: future-perturbation check, EWMA · window · offline | {yn(c['causal'])} · {yn(c['causal_window'])} · {'fails (expected)' if not c['acausal'] else 'passes (unexpected)'} |
| Causality: batch output == sample-by-sample replay | {yn(c['batch_equals_replay'])} |
| Same-day R², median over sessions, live (seeds) | {sd['median']['causal']:.4f} ({' · '.join(f'{v:.4f}' for v in sd['median_by_seed']['causal'].values())}) |
| Same-day R², median over sessions, offline (seeds) | {sd['median']['acausal']:.4f} ({' · '.join(f'{v:.4f}' for v in sd['median_by_seed']['acausal'].values())}) |
| Live ÷ offline, same day | {rat['value']:.3f} (95 % CI {rat['ci95'][0]:.3f} to {rat['ci95'][1]:.3f}) |
| Mean cross-session R², offline | {fmt(cs['acausal'])} |
| Mean cross-session R², live EWMA | {fmt(cs['causal'])} |
| Mean cross-session R², live trailing window ({s['window_bins']} bins) | {fmt(cs['causal_window'])} |
| Live EWMA − offline | {fmt(cm)} |
| Trailing window − EWMA | {fmt(wm)}: the window {window} |
| α sweep, Day-0 decoder mean R² on later sessions | {' · '.join(f'{a}: ' + f'{v:+.4f}'.replace('-', '−') for a, v in sw['mean_r2'].items())} |
| α 0.01 − 0.003 · α 0.01 − 0.03 | {fmt(sw['alpha_0.01_minus_0.003'], 4)} · {fmt(sw['alpha_0.01_minus_0.03'], 4)} |
| State per channel (float64) | live {st['causal']} B (EWMA mean, variance, smoother); offline needs the whole session; target ≤ {st['target']} B |

- **Data:** {r['sessions']['n']} LINK sessions, {iso(r['sessions']['first'])} → {iso(r['sessions']['last'])}, index-finger velocity.
- **Features:** offline = each session z-scored with its own mean and SD, then a 50 ms Gaussian (sees the future); live = running mean and variance (α {s['alpha']}, a time constant of about 2 s) then a one-pole smoother ({s['alpha_smooth']}), state reset every session; trailing window = z-score over the last {s['window_bins']} bins. The first {s['warmup_bins']} bins (6 s) of every session are never scored, for both.
- **Decoder:** ridge on the last 100 ms of features predicting velocity at the current bin (spec `{r['spec']}`). Same day: 5 purged held-out blocks, seeds shift the block grid. Cross-session: fit on one whole session, score another.
- **Uncertainty:** 95 % CIs from 2,000 session-resampled draws (pooled over seeds for the same-day ratio). A difference counts only if its CI excludes zero.
- **α sweep:** one Day-0 decoder per α, so it is a cheap screen, not a selection; α {best} scores highest here. Selecting α properly (on early sessions, confirmed on later ones) and carrying state across sessions is the next step (streaming v2).
- **Limits:** one animal, one task; with α {s['alpha']} the running statistics forget within seconds, so this is close to an adaptive high-pass filter rather than a slow drift tracker; state resets each session; ops per bin and fixed-point size are not yet counted.
- **Cost:** {r['cost']['decoder_fits']} decoder fits, {r['cost']['runtime_s']:.0f} s; the decoder costs {r['cost']['decoder_macs_per_bin']} multiply-accumulates per 20 ms bin.

</details>

*Generated by `analyses/causal_norm.py` from `results/analyses/causal_norm.json`. Do not edit by hand.*
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-only", action="store_true", help="re-render the report from the existing JSON")
    args = parser.parse_args()
    if args.report_only:
        result = read_json(NAME)
    else:
        result = compute()
        if result["dirty"]:
            print("working tree dirty outside results/: writing a draft (status says so); commit and re-run for a traceable result")
        print(f"wrote {write_json(NAME, result).relative_to(OUT.parents[1])}")
    path = OUT / f"{NAME}.md"
    path.write_text(render(result))
    print(f"wrote {path.relative_to(OUT.parents[1])}")


if __name__ == "__main__":
    main()
