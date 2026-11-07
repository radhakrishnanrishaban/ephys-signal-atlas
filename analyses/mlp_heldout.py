"""
Question: does a nonlinear decoder (the Part 3 MLP) transfer to later days better than a linear one, held out?

    PYTHONPATH=02_package/src:. python -m analyses.mlp_heldout                # compute, write JSON, figure, report
    PYTHONPATH=02_package/src:. python -m analyses.mlp_heldout --report-only  # re-render the report from the JSON

The D-17 splits (forward: train days 0-53, score 13 later sessions; grouped: 5 date groups held out
in turn; in-sample: the old protocol, kept to measure the inflation), seeds 0-2. Every fit also fits
Ridge on the MLP's exact inputs, and on the forward split the package-spec Ridge on SBP r_norm.
Per-session R² is averaged over seeds before the session-bootstrap CI (EXP-003 B8). Same computation as
EXP-002 (scripts/exp002_mlp_heldout.py, whose splits and loader it reuses), EXP-003 B8 and
sbp-mlp-decoder_v1, now one command. About 25 min.
With uncommitted changes outside results/, the JSON records dirty: true and the report says Draft.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from analyses.common import OUT, Timer, estimate, fmt, provenance, read_json, write_json  # noqa: E402
from bench.splits import load_splits  # noqa: E402
from neurosignal.decoders.linear import SPEC  # noqa: E402
from neurosignal.decoders.mlp import fit_pooled_decoders, score_pooled  # noqa: E402
from neurosignal.features.joint import joint_features  # noqa: E402
from neurosignal.metrics.drift import compute_day_offsets  # noqa: E402
from neurosignal.metrics.stats import bootstrap_session_mean  # noqa: E402
from neurosignal.viz.mlp_plots import plot_mlp_r2_timeline  # noqa: E402
from scripts.exp002_mlp_heldout import load_sessions, splits_d17  # noqa: E402

NAME = "mlp_heldout"
QUESTION = "Does a nonlinear decoder (MLP) transfer to later days better than a linear one, held out?"


def run_splits(joint: list, sbp_norm: list, vel: list, dates: list) -> dict:
    """{split: {decoder: (seeds, sessions) R² array, NaN where not scored}} plus train sets, PCA variance and cost."""
    n = len(dates)
    r2, train, var_kept, fit_s, macs = {}, {}, [], [], {}
    for split, fold, train_idx, test_idx in splits_d17(dates):
        if split != "in_sample":
            assert not set(train_idx) & set(test_idx), f"{split} fold {fold}: test sessions in training (ERR-02)"
        with_spec = split == "forward"
        scored = sorted(set(test_idx) | set(train_idx)) if split == "forward" else list(test_idx)
        parts = ({("forward", "held"): test_idx, ("forward", "train"): train_idx} if split == "forward"
                 else {(split, "held"): test_idx})
        train.setdefault(split, []).append(train_idx)
        for s, seed in enumerate(SPEC.seeds):
            dec = fit_pooled_decoders(joint, vel, train_idx, seed, sessions_sbp_norm=sbp_norm if with_spec else None)
            scores = score_pooled(dec, vel, scored, sbp_norm if with_spec else None)
            fit_s.append(dec.fit_s["mlp"])
            macs = dec.macs_per_bin if with_spec else macs
            if split == "forward":
                var_kept.append(float(dec.pca.explained_variance_ratio_.sum()))
            for name, v in scores.items():
                for part, idx in parts.items():
                    arr = r2.setdefault(part, {}).setdefault(name, np.full((len(SPEC.seeds), n), np.nan))
                    for i in idx:
                        arr[s, i] = v[scored.index(i)]
    return {"r2": r2, "train": train, "pca_variance_kept": var_kept, "mlp_fit_s": fit_s, "macs": macs}


def seed_mean(a: np.ndarray, idx: list) -> np.ndarray:
    """Per-session R² averaged over seeds, for the sessions in idx."""
    return a[:, idx].mean(axis=0)


def compute() -> dict:
    with Timer() as t:
        splits = load_splits()
        sbp, tcr, vel, dates = load_sessions(splits)
        joint, sbp_norm = joint_features(sbp, tcr)
        del sbp, tcr
        out = run_splits(joint, sbp_norm, vel, dates)

    days = compute_day_offsets(dates).astype(int)
    r2 = out["r2"]
    fw, tr, gr, ins = r2[("forward", "held")], r2[("forward", "train")], r2[("grouped", "held")], r2[("in_sample", "held")]
    test = [i for i in range(len(dates)) if not np.isnan(fw["mlp"][0, i])]
    train = out["train"]["forward"][0]
    allx = list(range(len(dates)))
    n_test, n_all = f"{len(test)} held-out sessions", f"{len(dates)} held-out sessions"
    mlp_t, rs_t, sp_t = (seed_mean(fw[k], test) for k in ("mlp", "ridge_same", "ridge_spec"))
    ins_t = seed_mean(ins["mlp"], test)
    g_mlp, g_rs = seed_mean(gr["mlp"], allx), seed_mean(gr["ridge_same"], allx)

    OUT.joinpath(NAME).mkdir(parents=True, exist_ok=True)
    mlp_plot, ref_plot = np.full(len(dates), np.nan), np.full(len(dates), np.nan)
    mlp_plot[test], ref_plot[test] = mlp_t, rs_t
    fig = plot_mlp_r2_timeline(mlp_plot, dates, train_idx=train, r2_reference=ref_plot,
                               title=f"Decoders frozen after day {days[train[-1]]}, scored on later sessions (mean of 3 seeds)")
    fig.savefig(OUT / NAME / "r2_timeline_forward.png", dpi=150, bbox_inches="tight")
    fig.clf()

    return {
        "question": QUESTION,
        **provenance(),
        "spec_mlp": f"mlp-{SPEC.spec_id}",
        "forward_split": {
            "train": {"n": len(train), "first": dates[train[0]], "last": dates[train[-1]], "last_day": int(days[train[-1]])},
            "test": {"n": len(test), "first": dates[test[0]], "last": dates[test[-1]],
                     "days": [int(days[test[0]]), int(days[test[-1]])]},
            "mlp": estimate(mlp_t.mean(), bootstrap_session_mean(mlp_t), n_test),
            "mlp_by_seed": [round(float(fw["mlp"][s, test].mean()), 4) for s in range(len(SPEC.seeds))],
            "ridge_same_inputs": estimate(rs_t.mean(), bootstrap_session_mean(rs_t), n_test),
            "ridge_spec": estimate(sp_t.mean(), bootstrap_session_mean(sp_t), n_test),
            "mlp_minus_ridge_same": estimate((mlp_t - rs_t).mean(), bootstrap_session_mean(mlp_t - rs_t), n_test),
            "mlp_sessions_at_or_below_zero_by_seed": [int((fw["mlp"][s, test] <= 0).sum()) for s in range(len(SPEC.seeds))],
        },
        "in_sample": {
            "mlp_same_sessions": estimate(ins_t.mean(), bootstrap_session_mean(ins_t), f"the same {len(test)} sessions, trained on"),
            "mlp_by_seed": [round(float(ins["mlp"][s, test].mean()), 4) for s in range(len(SPEC.seeds))],
            "inflation": estimate((ins_t - mlp_t).mean(), bootstrap_session_mean(ins_t - mlp_t), n_test),
        },
        "grouped_split": {
            "mlp": estimate(g_mlp.mean(), bootstrap_session_mean(g_mlp), n_all),
            "ridge_same_inputs": estimate(g_rs.mean(), bootstrap_session_mean(g_rs), n_all),
            "mlp_minus_ridge_same": estimate((g_mlp - g_rs).mean(), bootstrap_session_mean(g_mlp - g_rs), n_all),
        },
        "input_compression": {
            "pca_components": 5,
            "pca_variance_kept": round(float(np.mean(out["pca_variance_kept"])), 4),
            "ridge_same_inputs_on_training_sessions": round(float(seed_mean(tr["ridge_same"], train).mean()), 4),
            "ridge_spec_on_training_sessions": round(float(seed_mean(tr["ridge_spec"], train).mean()), 4),
        },
        "figure": f"{NAME}/r2_timeline_forward.png",
        "cost": {"macs_per_bin": out["macs"], "mlp_fits": len(out["mlp_fit_s"]),
                 "mlp_fit_s": [round(min(out["mlp_fit_s"])), round(max(out["mlp_fit_s"]))], "runtime_s": t.seconds},
    }


def render(r: dict) -> str:
    """The Q&A report. Every number comes from the JSON; the wording follows the CIs."""
    f, ins, g, comp, c = r["forward_split"], r["in_sample"], r["grouped_split"], r["input_compression"], r["cost"]
    d = f["mlp_minus_ridge_same"]
    if not d["excludes_zero"]:
        vs = f"no better and no worse than Ridge on the same inputs: {fmt(d)}"
    elif d["value"] < 0:
        vs = f"worse than Ridge on the same inputs: {fmt(d)}"
    else:
        vs = f"better than Ridge on the same inputs: {fmt(d)}"
    verdict = "yes" if d["excludes_zero"] and d["value"] > 0 else "no"
    below = f["mlp_sessions_at_or_below_zero_by_seed"]
    below_txt = f"{min(below)}" + (f"–{max(below)}" if max(below) != min(below) else "")
    gd = g["mlp_minus_ridge_same"]
    grouped = ("no detectable difference" if not gd["excludes_zero"] else
               "the MLP ahead" if gd["value"] > 0 else "the MLP behind")
    sg = lambda x: f"{x:+.3f}".replace("-", "−")  # noqa: E731
    seeds = lambda v: " · ".join(sg(x) for x in v)  # noqa: E731
    iso = lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}"  # noqa: E731
    status = "Answered" if not r["dirty"] else "Draft (uncommitted code)"
    tr, te = f["train"], f["test"]
    lin_pca, lin_full = comp["ridge_same_inputs_on_training_sessions"], comp["ridge_spec_on_training_sessions"]
    bottleneck = (f"Its inputs are the bottleneck: Ridge on the same {comp['pca_components']} principal components reaches only "
                  f"{lin_pca:.3f} even on the training sessions, against {lin_full:.3f} for Ridge on the full features."
                  if lin_pca < 0.5 * lin_full else
                  f"Ridge on its {comp['pca_components']} principal components reaches {lin_pca:.3f} on the training sessions, "
                  f"against {lin_full:.3f} on the full features.")
    return f"""# {r['question']}

**Status:** {status} · decoder spec `{r['spec']}`, MLP `{r['spec_mlp']}` · commit `{r['commit']}` · {r['run_date']}

**Short answer: {verdict}.** Trained on the first {tr['n']} sessions (days 0–{tr['last_day']}) and frozen, the MLP scores {fmt(f['mlp'])} on the {te['n']} later sessions (days {te['days'][0]}–{te['days'][1]}), {vs}. It does no better than predicting the average movement on {below_txt} of {te['n']} later sessions, depending on the seed. Scored on sessions it was trained on, the same model gets {fmt(ins['mlp_same_sessions'])}: the old protocol inflated it by {fmt(ins['inflation'])}. {bottleneck}

![Held-out R² per later session, MLP vs Ridge on the same inputs; grey band = training sessions]({r['figure']})

<details><summary>Details</summary>

| Quantity | Value |
| --- | --- |
| Forward split: train | {tr['n']} sessions, {iso(tr['first'])} → {iso(tr['last'])} (days 0–{tr['last_day']}) |
| Forward split: test | {te['n']} sessions, {iso(te['first'])} → {iso(te['last'])} (days {te['days'][0]}–{te['days'][1]}) |
| MLP, held out (seeds) | {fmt(f['mlp'])} ({seeds(f['mlp_by_seed'])}) |
| Ridge, same inputs, held out | {fmt(f['ridge_same_inputs'])} |
| Ridge, full spec on `r_norm`, pooled, held out | {fmt(f['ridge_spec'])} |
| MLP − Ridge same inputs | {fmt(d)} |
| Later sessions with MLP R² ≤ 0, by seed | {' · '.join(str(b) for b in below)} of {te['n']} |
| MLP on the same sessions, trained on them (seeds) | {fmt(ins['mlp_same_sessions'])} ({seeds(ins['mlp_by_seed'])}) |
| In-sample − held out | {fmt(ins['inflation'])} |
| Grouped split (5 date groups, each held out once): MLP · Ridge same inputs | {fmt(g['mlp'])} · {fmt(g['ridge_same_inputs'])} |
| Grouped: MLP − Ridge same inputs | {fmt(gd)}: {grouped} |
| Variance kept by {comp['pca_components']} principal components (training fit) | {100 * comp['pca_variance_kept']:.1f} % |
| Training sessions: Ridge on the {comp['pca_components']} components · Ridge on full features | {comp['ridge_same_inputs_on_training_sessions']:.3f} · {comp['ridge_spec_on_training_sessions']:.3f} |

- **Model:** spike-band power (`r_norm`) and threshold-crossing z-scores per channel (192 inputs), compressed by PCA fit on the training sessions only to {comp['pca_components']} components, 400 ms of history; MLP with layers of 128 and 64 units, fast movements weighted ×4, early stopping on 10 % of training rows. Ridge is fit on exactly the same rows. The first 6 s of each session are never scored.
- **Uncertainty:** per-session R² is averaged over seeds 0–2, then 95 % CIs from 2,000 session-resampled draws over the held-out sessions. A difference counts only if its CI excludes zero.
- **Limits:** one animal, one task; the features are normalised with each session's own statistics, so none of this can run live, and it gets no calibration-free-days score; PCA size and velocity weighting were not re-tuned held out. An MLP on the full inputs without PCA is a separate hypothesis, not tested here.
- **Cost:** inference per 20 ms bin: MLP {c['macs_per_bin']['mlp']:,} multiply-accumulates, Ridge on the same inputs {c['macs_per_bin']['ridge_same']:,}, Ridge full spec {c['macs_per_bin']['ridge_spec']:,}. {c['mlp_fits']} MLP fits of {c['mlp_fit_s'][0]}–{c['mlp_fit_s'][1]} s each; {c['runtime_s'] / 60:.0f} min in all.

</details>

*Generated by `analyses/mlp_heldout.py` from `results/analyses/mlp_heldout.json`. Do not edit by hand.*
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
