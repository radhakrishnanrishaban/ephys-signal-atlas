"""Guards for the bench harness (D-11, D-13): no score trains on what it tests (ERR-01, ERR-02),
no scored bin falls in the warm-up (ERR-09), and only causal features get a CFD."""

import numpy as np
import pytest

from bench.guards import is_causal
from bench.plan import leaking_plans, plan_scores
from bench.run import designs
from bench.spec import FEATURES, SPEC
from bench.splits import forward_pairs
from neurosignal.gold.nonlinear import fit_pooled_decoders

DATES = ["20200127", "20200130", "20200204", "20200205"]
LENGTHS = [1000, 1203, 997, 1500]


@pytest.mark.parametrize("seed", SPEC.seeds)
def test_no_score_shares_a_bin_or_session_between_train_and_test(seed):
    splits = {"sessions": DATES, "pairs": forward_pairs(DATES)}
    session_bins = dict(zip(DATES, LENGTHS))

    plans = plan_scores(splits, session_bins, SPEC, seed)

    assert any(p.kind == "same_day" for p in plans) and any(p.kind == "cross_day" for p in plans)
    assert leaking_plans(plans, SPEC, {(p["cal"], p["test"]) for p in splits["pairs"]}) == []


def test_no_scored_bin_falls_in_the_warmup():
    # Every SBP and velocity sample holds its own bin index, so X and y name the bins each row reads.
    sbp = [np.repeat(np.arange(T, dtype=float)[:, None], 3, axis=1) for T in LENGTHS]
    vel = [np.arange(T, dtype=float) for T in LENGTHS]
    design = designs(sbp, vel, DATES, SPEC)
    splits = {"sessions": DATES, "pairs": forward_pairs(DATES)}
    session_bins = {d: b for d, (_, _, b) in design.items()}

    for seed in SPEC.seeds:
        for p in plan_scores(splits, session_bins, SPEC, seed):
            X, y, _ = design[p.test]
            assert X[p.test_rows].min() >= SPEC.warmup_bins
            assert y[p.test_rows].min() >= SPEC.warmup_bins


@pytest.mark.parametrize("name, causal", [("raw_sbp", True), ("r_norm_causal", True), ("r_norm_acausal", False)])
def test_causality_check_passes_causal_features_and_catches_acausal(name, causal):
    assert is_causal(FEATURES[name]) is causal


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_held_out_sessions_never_reach_the_pooled_fit():
    """ERR-02: PCA, the MLP and both Ridge decoders change when a training session changes, never when a held-out one does."""
    rng = np.random.default_rng(0)
    T, n_sessions, train_idx = 700, 4, (0, 1)
    joint = [rng.normal(size=(T, 192)) for _ in range(n_sessions)]
    sbp_norm = [j[:, :96].copy() for j in joint]
    vel = [rng.normal(size=T) for _ in range(n_sessions)]

    def fitted(joint, sbp_norm):
        dec = fit_pooled_decoders(joint, vel, train_idx, seed=0, sessions_sbp_norm=sbp_norm,
                                  rows_per_session=200, hidden_layer_sizes=(4,), max_iter=5)
        return [dec.pca.components_, *dec.mlp.coefs_, dec.ridge_same.coef_, dec.ridge_spec.coef_]

    base = fitted(joint, sbp_norm)
    for session, should_change in [(2, False), (0, True)]:
        j2, s2 = list(joint), list(sbp_norm)
        j2[session] = joint[session] * 50.0 + 3.0
        s2[session] = sbp_norm[session] * 50.0 + 3.0
        same = all(np.array_equal(a, b) for a, b in zip(base, fitted(j2, s2)))
        assert same != should_change, f"perturbing session {session}: fit {'unchanged' if same else 'changed'}"
