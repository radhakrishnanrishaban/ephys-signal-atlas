"""
Question: does calibration-free days (CFD) measure the calibration→test gap, or the recording period? (EXP-008)

    PYTHONPATH=02_package/src:. python -m analyses.cfd_period --splits v1                # compute, write JSON, figure, report
    PYTHONPATH=02_package/src:. python -m analyses.cfd_period --splits v1 --report-only  # re-render the report from the JSON

Reads the per-pair rows the bench wrote for that split (results/bench_<split>_pairs.csv) and the matching
scoreboard rows; it fits no decoder. Pairs are grouped by the calibration session's year and by gap
(1-7, 8-30, 31-52 d). Per cell: pairs and the median kept share over pairs × seeds 0-2; a cell with fewer
than 15 pairs is "too few". Primary contrast per gap bin: median kept share, calibration year 2020 minus
2021-2023 pooled, with a 95 % CI from 2,000 session-resampled draws (a pair's weight = draws of its
calibration session × draws of its test session, D-13). Secondary: CFD fitted on the 2021-2023-calibrated
pairs only, as the bench fits it. Every threshold here was fixed in EXP-008 before any v1 or v2 run.
Target style (EXP-008 amendment 1, approved 2026-11-08 before any v2 data): a pair is same-style if its
calibration and test sessions share a target style (CO or RD, from the NWB identifier). Per gap bin, the
2020 − 2021-2023 contrast on same-style pairs only, and same-style − cross-style with all years pooled,
same bootstrap and "too few" rule. P8-P9 are checked on v2 only; on v1 the tables are descriptive.
With uncommitted changes outside results/, the JSON records dirty: true and the report says Draft.
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyses.common import OUT, Timer, estimate, fmt, provenance, read_json, write_json  # noqa: E402
from bench.cfd import cfd_bootstrap, cfd_days  # noqa: E402
from bench.run import pairs_csv  # noqa: E402
from bench.scoreboard import SCOREBOARD_CSV  # noqa: E402
from neurosignal.decoders.linear import SPEC  # noqa: E402
from neurosignal.io.loaders import list_link_session_paths, load_link_target_style  # noqa: E402
from neurosignal.metrics.stats import N_BOOT, session_draws, weighted_median  # noqa: E402
from neurosignal.viz.style import AXIS_LABEL_FONTSIZE, LEGEND_FONTSIZE, TITLE_FONTSIZE  # noqa: E402

NAME = "cfd_period"
QUESTION = "Does calibration-free days measure the gap between sessions, or the recording period?"
GAP_BINS = {"1-7 d": (1, 7), "8-30 d": (8, 30), "31-52 d": (31, 52)}
YEARS = ("2020", "2021", "2022", "2023")
EARLY, LATE = ("2020",), ("2021", "2022", "2023")
MIN_PAIRS = 15
MARGIN = 0.10         # EXP-008: a period effect is a contrast ≥ +0.10 whose CI excludes 0
CFD_OF_RECORD = 52    # scoreboard 1554f00, splits v0 (FACTS → Metrics)
FEATURES = ("r_norm_causal", "raw_sbp")


def board_rows(version: str) -> dict:
    """Scoreboard rows of the newest commit scored on this split, by feature."""
    rows = [r for r in csv.DictReader(SCOREBOARD_CSV.open()) if (r["splits"] or "v0") == version]
    if not rows:
        raise SystemExit(f"no scoreboard rows for splits {version}: run python -m bench.run --splits {version} first")
    commit = rows[-1]["commit"]
    return {r["feature"]: r for r in rows if r["commit"] == commit}


def pair_arrays(version: str, commit: str, feature: str) -> dict:
    rows = [r for r in csv.DictReader(pairs_csv(version).open()) if r["commit"] == commit and r["feature"] == feature]
    if not rows:
        raise SystemExit(f"{pairs_csv(version).name} has no {feature} rows for commit {commit}")
    return {
        "cal": np.array([r["cal"] for r in rows]),
        "test": np.array([r["test"] for r in rows]),
        "gap": np.array([int(r["gap_days"]) for r in rows]),
        "kept": np.array([float(r["kept_ratio"]) for r in rows]),
        "seed": np.array([int(r["seed"]) for r in rows]),
    }


def contrast(p: dict, early: np.ndarray, late: np.ndarray, C: np.ndarray, i_cal: np.ndarray, i_test: np.ndarray) -> dict:
    """Median kept share early − late with session-resampled draws; same draws for both groups."""
    value = float(np.median(p["kept"][early]) - np.median(p["kept"][late]))
    draws = np.full(len(C), np.nan)
    for b, c in enumerate(C):
        w = c[i_cal] * c[i_test]
        we, wl = w[early], w[late]
        if we.sum() and wl.sum():
            draws[b] = (weighted_median(p["kept"][early][we > 0], we[we > 0])
                        - weighted_median(p["kept"][late][wl > 0], wl[wl > 0]))
    return estimate(value, draws, f"{early.sum() // len(SPEC.seeds)} vs {late.sum() // len(SPEC.seeds)} pairs × {len(SPEC.seeds)} seeds")


def reading(e: dict) -> str:
    """EXP-008's rule for one gap bin."""
    lo, hi = e["ci95"]
    if e["value"] >= MARGIN and lo > 0:
        return "period"
    if lo >= -MARGIN and hi <= MARGIN:
        return "gap"
    return "inconclusive"


def late_cfd(p: dict, late_rows: np.ndarray) -> dict:
    """CFD on the 2021-2023-calibrated pairs only: median of seed points, CI from all draws pooled (D-13)."""
    points, draws = [], []
    for seed in SPEC.seeds:
        m = late_rows & (p["seed"] == seed)
        points.append(cfd_days(p["gap"][m], p["kept"][m], SPEC.kept_threshold))
        draws.append(cfd_bootstrap(p["cal"][m], p["test"][m], p["gap"][m], p["kept"][m], SPEC.kept_threshold, SPEC.n_boot, seed))
    pooled = np.concatenate(draws)
    return {"cfd_days": int(np.median(points)), "ci95": [int(np.percentile(pooled, 2.5)), int(np.percentile(pooled, 97.5))],
            "seed_points": [int(x) for x in points], "pairs": int(late_rows.sum() // len(SPEC.seeds))}


def session_styles(keys) -> dict:
    """Target style of every session key, read from the files on disk."""
    paths = list_link_session_paths()
    missing = sorted(set(keys) - set(paths))
    if missing:
        raise SystemExit(f"target style needs the session files on disk; missing: {missing[:5]}")
    return {k: load_link_target_style(paths[k]) for k in sorted(set(keys))}


def guarded_contrast(p: dict, a: np.ndarray, b: np.ndarray, C, i_cal, i_test, with_reading: bool = True) -> dict:
    """contrast() with the "too few" rule (< MIN_PAIRS pairs on either side)."""
    n_seeds = len(SPEC.seeds)
    if min(a.sum(), b.sum()) // n_seeds < MIN_PAIRS:
        return {"too_few": True, "pairs": [int(a.sum() // n_seeds), int(b.sum() // n_seeds)]}
    e = contrast(p, a, b, C, i_cal, i_test)
    return {**e, "reading": reading(e)} if with_reading else e


def analyse(p: dict, styles: dict) -> dict:
    year = np.array([c[:4] for c in p["cal"]])
    same = np.array([styles[c] == styles[t] for c, t in zip(p["cal"], p["test"])])
    sessions, idx = np.unique(np.concatenate([p["cal"], p["test"]]), return_inverse=True)
    i_cal, i_test = idx[: len(p["cal"])], idx[len(p["cal"]):]
    C = session_draws(len(sessions), N_BOOT, seed=0)
    n_seeds = len(SPEC.seeds)
    cells, contrasts, within_style, style_gap = {}, {}, {}, {}
    for label, (lo, hi) in GAP_BINS.items():
        in_bin = (p["gap"] >= lo) & (p["gap"] <= hi)
        cells[label] = {}
        for y in YEARS:
            m = in_bin & (year == y)
            n = int(m.sum() // n_seeds)
            cells[label][y] = {"pairs": n, "median_kept": round(float(np.median(p["kept"][m])), 4) if n >= MIN_PAIRS else None}
        early, late = in_bin & np.isin(year, EARLY), in_bin & np.isin(year, LATE)
        contrasts[label] = guarded_contrast(p, early, late, C, i_cal, i_test)
        within_style[label] = guarded_contrast(p, early & same, late & same, C, i_cal, i_test)
        style_gap[label] = guarded_contrast(p, in_bin & same, in_bin & ~same, C, i_cal, i_test, with_reading=False)
    return {"cells": cells, "early_minus_late": contrasts, "late_only_cfd": late_cfd(p, np.isin(year, LATE)),
            "style": {"same_style_early_minus_late": within_style, "same_minus_cross": style_gap,
                      "sessions": {s: int(sum(1 for v in styles.values() if v == s)) for s in ("CO", "RD")}}}


def predictions(version: str, board: dict, causal: dict) -> dict:
    """EXP-008 P1-P7, checked against this split's rows (stage 1 = v1, stage 2 = v2)."""
    num = lambda r, k: int(r[k])  # noqa: E731
    c, q, raw = board["r_norm_causal"], board["r_norm_causal_q"], board["raw_sbp"]
    con = causal["early_minus_late"]
    period = lambda label: con[label].get("reading") == "period"  # noqa: E731
    if version == "v1":
        return {
            "P1 causal CFD < 52 d": num(c, "cfd_days") < CFD_OF_RECORD,
            "P2 period effect at 8-30 d and 31-52 d": period("8-30 d") and period("31-52 d"),
            "P3 raw CFD ≤ 7 d": num(raw, "cfd_days") <= 7,
            "P4 fixed-point and float CFD within 1 d": abs(num(q, "cfd_days") - num(c, "cfd_days")) <= 1,
        }
    if version == "v2":
        v1 = board_rows("v1")["r_norm_causal"]
        width = lambda r: num(r, "cfd_ci95_hi") - num(r, "cfd_ci95_lo")  # noqa: E731
        late = causal["late_only_cfd"]
        return {
            "P5 causal CFD < 52 d, CI width ≤ half of stage 1": num(c, "cfd_days") < CFD_OF_RECORD and 2 * width(c) <= width(v1),
            "P6 period effect at 1-7 d": period("1-7 d"),
            "P7 2021-2023-only CFD < 52 d, CI upper bound < 52 d": late["cfd_days"] < CFD_OF_RECORD and late["ci95"][1] < CFD_OF_RECORD,
            **style_checks(causal),
        }
    return {}


def style_checks(causal: dict) -> dict:
    """EXP-008 amendment 1: P8, P9. None = untested (fewer than 2 bins with enough pairs, or none for P9)."""
    within = {b: e for b, e in causal["style"]["same_style_early_minus_late"].items() if not e.get("too_few")}
    p8 = all(e["reading"] == "period" for e in within.values()) if len(within) >= 2 else None
    both = [b for b, e in causal["style"]["same_minus_cross"].items()
            if not e.get("too_few") and not causal["early_minus_late"][b].get("too_few")]
    p9 = all(causal["style"]["same_minus_cross"][b]["value"] < causal["early_minus_late"][b]["value"] for b in both) if both else None
    return {"P8 period effect within same-style pairs": p8, "P9 style contrast < period contrast in every bin": p9}


def verdict(version: str, board: dict, causal: dict, checks: dict) -> str:
    con = [e for e in causal["early_minus_late"].values() if not e.get("too_few")]
    lo, hi = int(board["r_norm_causal"]["cfd_ci95_lo"]), int(board["r_norm_causal"]["cfd_ci95_hi"])
    if checks and all(v for k, v in checks.items() if k[:2] in ("P1", "P2", "P5", "P6", "P7")):
        p8 = next((v for k, v in checks.items() if k.startswith("P8")), "absent")
        return {"absent": "period effect", True: "period effect, not explained by target style",
                False: "period effect not separable from target style",
                None: "period effect (target style not controlled)"}[p8]
    if con and all(e["reading"] == "gap" for e in con) and lo <= CFD_OF_RECORD <= hi:
        return "gap explains it"
    return "inconclusive"


def figure(causal: dict, version: str, n_sessions: int) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(GAP_BINS))
    for k, y in enumerate(YEARS):
        vals = [causal["cells"][b][y]["median_kept"] for b in GAP_BINS]
        xs = [xi + (k - 1.5) * 0.12 for xi, v in zip(x, vals) if v is not None]
        ys = [v for v in vals if v is not None]
        if ys:
            ax.plot(xs, ys, marker="o", linestyle="-", label=f"Calibrated in {y}")
    ax.axhline(SPEC.kept_threshold, color="0.5", linewidth=1, linestyle="--")
    ax.set_xticks(x, list(GAP_BINS))
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Calibration → test gap", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Median kept share of same-day R²", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f"Kept share by calibration year ({n_sessions} sessions)", fontsize=TITLE_FONTSIZE)
    ax.legend(loc="best", fontsize=LEGEND_FONTSIZE, framealpha=0.75)
    ax.grid(axis="y", color="0.9")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    rel = f"{NAME}/kept_by_year_{version}.png"
    OUT.joinpath(NAME).mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / rel, dpi=150)
    plt.close(fig)
    return rel


def compute(version: str) -> dict:
    with Timer() as t:
        board = board_rows(version)
        commit = next(iter(board.values()))["commit"]
        pairs = {f: pair_arrays(version, commit, f) for f in FEATURES}
        styles = session_styles(np.concatenate([pairs["r_norm_causal"]["cal"], pairs["r_norm_causal"]["test"]]))
        by_feature = {f: analyse(pairs[f], styles) for f in FEATURES}
    n_sessions = int(board["r_norm_causal"]["sessions"])
    checks = predictions(version, board, by_feature["r_norm_causal"])
    return {
        "question": QUESTION,
        **provenance(),
        "splits": version,
        "bench_commit": commit,
        "sessions": n_sessions,
        "pairs": int(board["r_norm_causal"]["pairs"]),
        "settings": {"gap_bins": GAP_BINS, "early": EARLY, "late": LATE, "min_pairs": MIN_PAIRS, "margin": MARGIN,
                     "n_boot": N_BOOT, "seeds": list(SPEC.seeds), "kept_threshold": SPEC.kept_threshold},
        "cfd": {f: {k: board[f][k] for k in ("cfd_days", "cfd_ci95_lo", "cfd_ci95_hi", "cfd_seed_min", "cfd_seed_max")}
                for f in ("raw_sbp", "r_norm_causal", "r_norm_causal_q")},
        "by_feature": by_feature,
        "exp008": checks,
        "verdict": verdict(version, board, by_feature["r_norm_causal"], checks),
        "figure": figure(by_feature["r_norm_causal"], version, n_sessions),
        "cost": {"decoder_fits": 0, "runtime_s": t.seconds},
    }


def render(r: dict) -> str:
    """The report. Every number comes from the JSON."""
    c = r["by_feature"]["r_norm_causal"]
    cfd = r["cfd"]["r_norm_causal"]
    status = "Answered" if not r["dirty"] else "Draft (uncommitted code)"
    cell = lambda e: "too few" if e["median_kept"] is None else f"{e['median_kept']:.2f} ({e['pairs']})"  # noqa: E731
    con = lambda e: (f"too few ({e['pairs'][0]} vs {e['pairs'][1]} pairs)" if e.get("too_few")  # noqa: E731
                     else f"{fmt(e, 2)} · {e['reading']}")
    rows = "\n".join(f"| {b} | " + " | ".join(cell(c["cells"][b][y]) for y in YEARS) + f" | {con(c['early_minus_late'][b])} |"
                     for b in GAP_BINS)
    raw = r["by_feature"]["raw_sbp"]
    raw_rows = "\n".join(f"| {b} | " + " | ".join(cell(raw["cells"][b][y]) for y in YEARS) + f" | {con(raw['early_minus_late'][b])} |"
                         for b in GAP_BINS)
    late = c["late_only_cfd"]
    word = lambda v: "untested" if v is None else ("pass" if v else "**fail**")  # noqa: E731
    checks = "\n".join(f"| {k} | {word(v)} |" for k, v in r["exp008"].items()) or "| — | — |"
    st = c.get("style")
    est = lambda e: f"too few ({e['pairs'][0]} vs {e['pairs'][1]} pairs)" if e.get("too_few") else fmt(e, 2)  # noqa: E731
    style_block = "" if st is None else (
        "**Target style (EXP-008 amendment 1"
        + (", descriptive on this split: P8–P9 are stage 2 only" if r["splits"] != "v2" else "")
        + f"): causal `r_norm`; sessions CO {st['sessions']['CO']} · RD {st['sessions']['RD']}**\n\n"
        "| Gap | 2020 − 2021–2023, same-style pairs | Same-style − cross-style, all years |\n| --- | --- | --- |\n"
        + "\n".join(f"| {b} | {con(st['same_style_early_minus_late'][b])} | {est(st['same_minus_cross'][b])} |" for b in GAP_BINS)
        + "\n\n")
    return f"""# {r['question']}

**Status:** {status} · splits `{r['splits']}` ({r['sessions']} sessions, {r['pairs']} pairs) · bench commit `{r['bench_commit']}` · decoder spec `{r['spec']}` · {r['run_date']}

**Reading (EXP-008): {r['verdict']}.** On this split, causal CFD is {cfd['cfd_days']} d (95 % CI {cfd['cfd_ci95_lo']}–{cfd['cfd_ci95_hi']}), against {CFD_OF_RECORD} d on the 30-session split. Fitted on pairs calibrated in 2021–2023 only, it is {late['cfd_days']} d (95 % CI {late['ci95'][0]}–{late['ci95'][1]}, {late['pairs']} pairs).

![Median kept share by gap and calibration year]({r['figure']})

**Causal `r_norm`: median kept share (pairs) by calibration year, and 2020 − 2021–2023**

| Gap | {' | '.join(YEARS)} | 2020 − 2021–2023 |
| --- | {' | '.join('---' for _ in YEARS)} | --- |
{rows}

<details><summary>Details</summary>

**Raw SBP, same layout**

| Gap | {' | '.join(YEARS)} | 2020 − 2021–2023 |
| --- | {' | '.join('---' for _ in YEARS)} | --- |
{raw_rows}

{style_block}**EXP-008 predictions on this split**

| Prediction | Result |
| --- | --- |
{checks}

| CFD (d, 95 % CI, seed range) | raw SBP | causal `r_norm` | fixed-point causal |
| --- | --- | --- | --- |
| this split | {r['cfd']['raw_sbp']['cfd_days']} ({r['cfd']['raw_sbp']['cfd_ci95_lo']}–{r['cfd']['raw_sbp']['cfd_ci95_hi']}) | {cfd['cfd_days']} ({cfd['cfd_ci95_lo']}–{cfd['cfd_ci95_hi']}, {cfd['cfd_seed_min']}–{cfd['cfd_seed_max']}) | {r['cfd']['r_norm_causal_q']['cfd_days']} ({r['cfd']['r_norm_causal_q']['cfd_ci95_lo']}–{r['cfd']['r_norm_causal_q']['cfd_ci95_hi']}) |

- **Kept share** of a pair = max(cross-day R², 0) ÷ the calibration session's held-out same-day R² (bench, D-12). Medians pool seeds {', '.join(map(str, r['settings']['seeds']))}; a cell needs ≥ {r['settings']['min_pairs']} pairs.
- **Contrast CI:** {r['settings']['n_boot']:,} draws resampling sessions; a pair's weight is the draws of its calibration session × the draws of its test session. "period" = contrast ≥ +{r['settings']['margin']:.2f} with a CI that excludes 0; "gap" = CI inside ±{r['settings']['margin']:.2f}; else "inconclusive" (EXP-008, fixed before running).
- **Limits:** the calibration year stands in for the recording period and cannot separate it from the implant's age. Target style (CO, RD) is read from the NWB identifier; the file does not say what the codes mean. One animal, one task.
- **Source:** `results/{('bench_' + r['splits'] + '_pairs.csv')}` and `results/scoreboard.csv` at commit `{r['bench_commit']}`; this script fits no decoder.
</details>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", default="v1", help="split whose bench rows to read")
    parser.add_argument("--report-only", action="store_true", help="re-render the report from the JSON")
    args = parser.parse_args()
    name = f"{NAME}_{args.splits}"
    r = read_json(name) if args.report_only else compute(args.splits)
    if not args.report_only:
        write_json(name, r)
    (OUT / f"{name}.md").write_text(render(r))
    print(f"{name}: verdict {r['verdict']} · {r['exp008']}")


if __name__ == "__main__":
    main()
