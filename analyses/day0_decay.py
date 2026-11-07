"""
Question: how badly does a decoder trained on Day 0 fail on later days, and what changes in the signal?

    PYTHONPATH=02_package/src:. python -m analyses.day0_decay                # compute, write JSON, figures, report
    PYTHONPATH=02_package/src:. python -m analyses.day0_decay --report-only  # re-render the report from the JSON

One decoder (package spec) fit on all of Day 0's raw SBP, scored on every later session; Day 0 itself
scored on held-out blocks (ERR-01), seeds 0-2 shifting the block grid (D-13). Medians and shares over
later sessions get 95 % session-bootstrap CIs. Day labels from session dates (ERR-06). Signal layer:
per-channel mean SBP, Day 0 vs the last session (figure only). Same computation as
sbp-drift-discovery_v1 cells 15-22, now one command.
With uncommitted changes outside results/, the JSON records dirty: true and the report says Draft.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from analyses.common import OUT, Timer, estimate, fmt, frozen_sessions, provenance, read_json, write_json  # noqa: E402
from neurosignal.decoders.linear import SPEC, grid_offset, same_day_r2, session_design  # noqa: E402
from neurosignal.metrics.drift import compute_day_offsets, compute_decoder_r2_per_day  # noqa: E402
from neurosignal.metrics.stats import bootstrap_session_mean, bootstrap_session_median  # noqa: E402
from neurosignal.viz.drift_plots import plot_day0_vs_last_scatter, plot_decoder_degradation  # noqa: E402

NAME = "day0_decay"
QUESTION = "How badly does a decoder trained on Day 0 fail on later days, and what changes in the signal?"
SEEDS = (0, 1, 2)
GAP_BINS = ((1, 7, "1–7 days"), (8, 30, "8–30 days"), (31, None, "over 30 days"))
R2_PLOT_FLOOR = -20.0


def compute() -> dict:
    with Timer() as t:
        sbp, vel, dates = frozen_sessions()
        days = compute_day_offsets(dates).astype(int)
        r2, _ = compute_decoder_r2_per_day(sbp, vel)  # r2[0]: held out, unshifted grid (the notebook's value)
        X0, y0, _ = session_design(sbp[0], vel[0], SPEC)
        day0_seeds = {s: float(same_day_r2(X0, y0, SPEC, grid_offset(len(y0), s)).mean()) for s in SEEDS}

    later, later_days, later_dates = r2[1:], days[1:], dates[1:]
    n_later = f"{later.size} later sessions"
    k = int(np.argmin(later))
    gap_bins = {}
    for lo, hi, label in GAP_BINS:
        keep = (later_days >= lo) & (later_days <= (hi if hi is not None else later_days.max()))
        v = later[keep]
        gap_bins[label] = {"days": [lo, hi], "values": [round(float(x), 4) for x in v],
                           "median": estimate(np.median(v), bootstrap_session_median(v), f"{v.size} sessions")}

    OUT.joinpath(NAME).mkdir(parents=True, exist_ok=True)
    fig = plot_decoder_degradation(r2, dates, out_path=OUT / NAME / "decoder_r2_by_session.png",
                                   clip_floor=R2_PLOT_FLOOR, ylim=(-4.0, 0.5))
    fig.clf()
    means = [np.nanmean(s, axis=0) for s in sbp]
    fig = plot_day0_vs_last_scatter(means[0], means[-1], dates, len(dates))
    fig.savefig(OUT / NAME / "channel_sbp_day0_vs_last.png", dpi=150, bbox_inches="tight")
    fig.clf()

    return {
        "question": QUESTION,
        **provenance(),
        "sessions": {"n": len(dates), "first": dates[0], "last": dates[-1], "span_days": int(days[-1])},
        "day0_heldout": {"unshifted_grid": round(float(r2[0]), 4),
                         "seeds": {str(s): round(v, 4) for s, v in day0_seeds.items()},
                         "seed_median": round(float(np.median(list(day0_seeds.values()))), 4)},
        "later_sessions": {
            "below_zero": int((later < 0).sum()), "out_of": int(later.size),
            "share_below_zero": estimate((later < 0).mean(), bootstrap_session_mean((later < 0).astype(float)), n_later),
            "median": estimate(np.median(later), bootstrap_session_median(later), n_later),
            "lowest": {"r2": round(float(later[k]), 4), "date": later_dates[k], "day": int(later_days[k])},
            "first": {"r2": round(float(later[0]), 4), "date": later_dates[0], "day": int(later_days[0])},
            "last": {"r2": round(float(later[-1]), 4), "date": later_dates[-1], "day": int(later_days[-1])},
        },
        "by_gap": gap_bins,
        "figures": {"decoder": f"{NAME}/decoder_r2_by_session.png", "signal": f"{NAME}/channel_sbp_day0_vs_last.png"},
        "cost": {"decoder_macs_per_bin": SPEC.decoder_macs_per_bin,
                 "decoder_fits": 1 + SPEC.same_day_folds * (1 + len(SEEDS)), "runtime_s": t.seconds},
    }


def render(r: dict) -> str:
    """The Q&A report. Every number comes from the JSON; the wording follows the CIs."""
    d0, ls, s = r["day0_heldout"], r["later_sessions"], r["sessions"]
    med, share = ls["median"], ls["share_below_zero"]
    level = ("not distinguishable from zero" if not med["excludes_zero"]
             else "below zero" if med["value"] < 0 else "above zero")
    sg = lambda x: f"{x:+.3f}".replace("-", "−")  # noqa: E731
    pc = lambda x: f"{100 * x:.0f} %"  # noqa: E731
    pct = lambda e: f"{pc(e['value'])} (95 % CI {pc(e['ci95'][0])} to {pc(e['ci95'][1])})"  # noqa: E731
    fails = med["excludes_zero"] and med["value"] < 0
    verdict = ("badly, from the first later session on" if fails and ls["first"]["r2"] < 0
               else "badly" if fails else "not clearly")
    iso = lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}"  # noqa: E731

    def gap_txt(g: dict) -> str:
        if len(g["values"]) == 1:
            return f"{sg(g['values'][0])} (one session)"
        return f"median {fmt(g['median'])}, {g['median']['n']}"

    seeds = " · ".join(f"seed {k} {v:.3f}" for k, v in d0["seeds"].items())
    status = "Answered" if not r["dirty"] else "Draft (uncommitted code)"
    return f"""# {r['question']}

**Status:** {status} · decoder spec `{r['spec']}` · commit `{r['commit']}` · {r['run_date']}

**Short answer: {verdict}.** A decoder trained on Day 0 scores R² {d0['seed_median']:.3f} on held-out Day-0 data. On later sessions it does worse than predicting the average movement on {ls['below_zero']} of {ls['out_of']}; its median R² over those sessions is {fmt(med)}, {level}. The first later session (day {ls['first']['day']}) is already at {sg(ls['first']['r2'])}. The signal it reads has moved: per channel, the average spike-band power on the last session differs from Day 0, mostly downwards, and by different amounts on different channels, so the Day-0 weights no longer match.

![Day-0 decoder R² on every session, by days since Day 0]({r['figures']['decoder']})

<details><summary>Details</summary>

| Quantity | Value |
| --- | --- |
| Day-0 decoder, Day 0 held out | {d0['seed_median']:.3f} (median of {seeds}; unshifted grid {d0['unshifted_grid']:.3f}) |
| Later sessions with R² < 0 | {ls['below_zero']} of {ls['out_of']}; share {pct(share)} |
| Median R², later sessions | {fmt(med)} |
| First later session | {sg(ls['first']['r2'])} on day {ls['first']['day']} ({iso(ls['first']['date'])}) |
| Lowest | {sg(ls['lowest']['r2'])} on day {ls['lowest']['day']} ({iso(ls['lowest']['date'])}) |
| Last session | {sg(ls['last']['r2'])} on day {ls['last']['day']} ({iso(ls['last']['date'])}) |
| By gap, 1–7 days | {gap_txt(r['by_gap']['1–7 days'])} |
| By gap, 8–30 days | {gap_txt(r['by_gap']['8–30 days'])} |
| By gap, over 30 days | {gap_txt(r['by_gap']['over 30 days'])} |

![Per-channel mean spike-band power, Day 0 vs the last session: points below the line lost power]({r['figures']['signal']})

- **Data:** {s['n']} LINK sessions, {iso(s['first'])} → {iso(s['last'])} ({s['span_days']} days), index-finger velocity; Day 0 is the first recorded session, about day 349 after implant. Raw spike-band power, no normalisation.
- **Decoder:** ridge on spike-band power over the last 100 ms, predicting velocity at the current bin (spec `{r['spec']}`); fit once on all of Day 0 and frozen. Day 0 itself is scored on 5 held-out blocks; seeds shift where the blocks start. The first 6 s of each session are never scored.
- **Uncertainty:** 95 % CIs from 2,000 session-resampled draws over the later sessions. Later-session scores do not depend on the seed (one fit, whole sessions).
- **Limits:** one animal, one task, one calibration day (every calibration day against every later one is in the drift-fix report and the scoreboard); recording period and gap are partly confounded; the channel figure compares two sessions only and is not a test of which channels drift most.
- **Cost:** {r['cost']['decoder_fits']} decoder fits, {r['cost']['runtime_s']:.0f} s; the decoder costs {r['cost']['decoder_macs_per_bin']} multiply-accumulates per 20 ms bin.

</details>

*Generated by `analyses/day0_decay.py` from `results/analyses/day0_decay.json`. Do not edit by hand.*
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
