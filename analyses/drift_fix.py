"""
Question: does per-session normalisation stop a decoder trained on one day from failing on another?

    PYTHONPATH=02_package/src:. python -m analyses.drift_fix                # compute, write JSON, figure, report
    PYTHONPATH=02_package/src:. python -m analyses.drift_fix --report-only  # re-render the report from the JSON

Pairwise cross-session R² (train on session i, test on session j, package decoder spec) for raw SBP,
per-session z-score, and r_norm (z-score + 50 ms Gaussian); a Day-0 decoder by gap from Day 0;
95 % session-bootstrap CIs (D-13). Same computation as sbp-drift-fix_v1 and EXP-003, now one command.
With uncommitted changes outside results/, the JSON records dirty: true and the report says Draft.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from analyses.common import OUT, Timer, estimate, fmt, frozen_sessions, provenance, read_json, write_json  # noqa: E402
from neurosignal.decoders.linear import SPEC  # noqa: E402
from neurosignal.features.normalize import normalize_sessions_sbp_gold, normalize_sessions_sbp_zscore_only  # noqa: E402
from neurosignal.metrics.drift import compute_day_offsets, compute_pairwise_r2_matrix  # noqa: E402
from neurosignal.viz.drift_plots import plot_pairwise_r2_raw_vs_norm  # noqa: E402
from neurosignal.metrics.stats import bootstrap_off_diagonal_mean, bootstrap_session_mean, off_diagonal_mean  # noqa: E402

NAME = "drift_fix"
QUESTION = "Does per-session normalisation stop a decoder trained on one day from failing on another?"
MONTH_PLUS_DAYS = 30


def compute() -> dict:
    with Timer() as t:
        sbp, vel, dates = frozen_sessions()
        days = compute_day_offsets(dates).astype(int)
        feats = {
            "raw": [np.asarray(s, dtype=np.float64) for s in sbp],
            "z_only": normalize_sessions_sbp_zscore_only(sbp),
            "r_norm": normalize_sessions_sbp_gold(sbp, sigma_bins=2.5),
        }
        R = {k: compute_pairwise_r2_matrix(f, vel) for k, f in feats.items()}

    pairs = f"{len(dates) * (len(dates) - 1)} pairs / {len(dates)} sessions"
    later = slice(1, None)
    month = days[later] >= MONTH_PLUS_DAYS
    week = (days[later] >= 1) & (days[later] <= 7)
    row = {k: R[k][0, later] for k in R}
    neg_dates = [dates[1:][i] for i in np.flatnonzero(row["r_norm"] < 0)]

    OUT.joinpath(NAME).mkdir(parents=True, exist_ok=True)
    fig = plot_pairwise_r2_raw_vs_norm(R["raw"], R["r_norm"], dates, out_path=OUT / NAME / "pairwise_r2_raw_vs_r_norm.png",
                                       title_norm="r_norm (z-score + 50 ms smooth)", clip_floor=-5.0, cmap="viridis")
    fig.clf()

    return {
        "question": QUESTION,
        **provenance(),
        "sessions": {"n": len(dates), "first": dates[0], "last": dates[-1], "span_days": int(days[-1])},
        "cross_session_r2": {k: estimate(off_diagonal_mean(R[k]), bootstrap_off_diagonal_mean(R[k]), pairs) for k in R},
        "differences": {
            "r_norm_minus_raw": estimate(off_diagonal_mean(R["r_norm"] - R["raw"]),
                                         bootstrap_off_diagonal_mean(R["r_norm"] - R["raw"]), pairs),
            "smoothing": estimate(off_diagonal_mean(R["r_norm"] - R["z_only"]),
                                  bootstrap_off_diagonal_mean(R["r_norm"] - R["z_only"]), pairs),
        },
        "day0_decoder_r_norm": {
            "same_day_heldout": round(float(R["r_norm"][0, 0]), 4),
            "days_1_7": {"values": [round(float(v), 4) for v in row["r_norm"][week]],
                         "days": [int(d) for d in days[later][week]]},
            "month_plus": estimate(row["r_norm"][month].mean(), bootstrap_session_mean(row["r_norm"][month]),
                                   f"{int(month.sum())} sessions"),
        },
        "day0_later_sessions_below_zero": {
            "raw": int((row["raw"] < 0).sum()), "r_norm": int((row["r_norm"] < 0).sum()),
            "out_of": len(dates) - 1, "r_norm_first_failure": min(neg_dates) if neg_dates else None,
        },
        "figure": f"{NAME}/pairwise_r2_raw_vs_r_norm.png",
        "cost": {"decoder_macs_per_bin": SPEC.decoder_macs_per_bin, "decoder_fits": 3 * len(dates) * (1 + SPEC.same_day_folds),
                 "runtime_s": t.seconds},
    }


def render(r: dict) -> str:
    """The Q&A report. Every number comes from the JSON; the wording follows the CIs."""
    cs, d, d0, neg = r["cross_session_r2"], r["differences"], r["day0_decoder_r_norm"], r["day0_later_sessions_below_zero"]
    level = ("above zero" if cs["r_norm"]["excludes_zero"] and cs["r_norm"]["value"] > 0
             else "not distinguishable from zero")
    if d["smoothing"]["excludes_zero"]:
        smooth = ("The 50 ms smoothing step **lowers** transfer" if d["smoothing"]["value"] < 0
                  else "The 50 ms smoothing step adds a little") + f": {fmt(d['smoothing'])}."
    else:
        smooth = f"The 50 ms smoothing step makes no detectable difference: {fmt(d['smoothing'])}."
    mp = d0["month_plus"]
    month = ("below zero, worse than predicting the mean" if mp["excludes_zero"] and mp["value"] < 0
             else "indistinguishable from zero: no useful transfer")
    wk = d0["days_1_7"]
    wk_txt = (f"{wk['values'][0]:+.3f} on day {wk['days'][0]} (one session)" if len(wk["values"]) == 1
              else f"{np.mean(wk['values']):+.3f} over {len(wk['values'])} sessions")
    s = r["sessions"]
    iso = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}" if d else "none"  # noqa: E731
    helps = d["r_norm_minus_raw"]["excludes_zero"] and d["r_norm_minus_raw"]["value"] > 0
    lasts = mp["excludes_zero"] and mp["value"] > 0
    verdict = "yes" if helps and lasts else ("partly" if helps else "no")
    status = "Answered" if not r["dirty"] else "Draft (uncommitted code)"
    return f"""# {r['question']}

**Status:** {status} · decoder spec `{r['spec']}` · commit `{r['commit']}` · {r['run_date']}

**Short answer: {verdict}.** Z-scoring each session's channels removes the large failures of a frozen decoder: mean cross-day R² goes from {fmt(cs['raw'])} on raw spike-band power to {fmt(cs['r_norm'])} with `r_norm`. The gain, {fmt(d['r_norm_minus_raw'])}, is real; the level it reaches is {level}. {smooth} Beyond a month, a decoder trained on Day 0 scores {fmt(mp)}, {month}.

![Pairwise decoder R², raw SBP vs r_norm: train session on the vertical axis, test session on the horizontal]({r['figure']})

<details><summary>Details</summary>

| Quantity | Value |
| --- | --- |
| Mean cross-session R², raw SBP | {fmt(cs['raw'])} |
| Mean cross-session R², z-score only | {fmt(cs['z_only'])} |
| Mean cross-session R², `r_norm` | {fmt(cs['r_norm'])} |
| `r_norm` − raw | {fmt(d['r_norm_minus_raw'])} |
| Smoothing (`r_norm` − z-score only) | {fmt(d['smoothing'])} |
| Day-0 decoder on `r_norm`, same day (held out) | {d0['same_day_heldout']:.3f} |
| Same, 1–7 days later | {wk_txt} |
| Same, ≥ {MONTH_PLUS_DAYS} days later | {fmt(mp)}, {mp['n']} |
| Later sessions below zero, Day-0 decoder | raw {neg['raw']} of {neg['out_of']} · `r_norm` {neg['r_norm']} of {neg['out_of']} (first on {iso(neg['r_norm_first_failure'])}) |

- **Data:** {s['n']} LINK sessions, {iso(s['first'])} → {iso(s['last'])} ({s['span_days']} days), index-finger velocity; Day 0 is the first recorded session.
- **Decoder:** ridge on spike-band power over the last 100 ms, predicting velocity at the current bin (spec `{r['spec']}`); trained on one whole session, tested on another; the same-day value is scored on 5 held-out blocks; the first 6 s of each session are never scored.
- **Uncertainty:** 95 % CIs from 2,000 session-resampled draws (pairs share sessions, so sessions are resampled, not pairs). A difference counts only if its CI excludes zero.
- **Limits:** one animal, one task; normalisation uses each session's own statistics (it needs the whole session), so it is a reference, not a deployable method; recording period and gap are partly confounded (most short gaps are in Jan–Mar 2020).
- **Cost:** {r['cost']['decoder_fits']} decoder fits, {r['cost']['runtime_s']:.0f} s; the decoder costs {r['cost']['decoder_macs_per_bin']} multiply-accumulates per 20 ms bin.

</details>

*Generated by `analyses/drift_fix.py` from `results/analyses/drift_fix.json`. Do not edit by hand.*
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
