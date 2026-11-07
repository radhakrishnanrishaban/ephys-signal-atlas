# Notebooks (v1, frozen)

These four notebooks are the exploratory record of the first round of work. They are frozen: not edited, not re-run on `main`, and they import the old `neurosignal.gold` layout, which the package no longer has. Their numbers are spec v2 with 95 % CIs, as of release `v0.4.0`.

To re-run one, check out the tag it was last run on:

```bash
git checkout v0.4.0
```

Each research question now runs as one script, `analyses/<question>.py`, which writes every number with its CI to `results/analyses/<question>.json` and renders the report `results/analyses/<question>.md` from that JSON. Cite the report, not the notebook.

| Notebook | Question | Script |
| --- | --- | --- |
| `sbp-drift-discovery_v1` | Why does a decoder stop working after a few days? (Day-0 decoder on later sessions) | `analyses/day0_decay.py` |
| `sbp-drift-fix_v1` | Can a per-session normalisation keep it working without re-tuning? | `analyses/drift_fix.py` |
| `sbp-causal-streaming_v1` | Can that normalisation run live, using only past samples? | `analyses/causal_norm.py` |
| `sbp-mlp-decoder_v1` | Does a nonlinear decoder (MLP) transfer better, held out? | `analyses/mlp_heldout.py` |

The `.html` files are the rendered notebooks with outputs.
