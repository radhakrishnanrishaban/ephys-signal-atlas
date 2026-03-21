# Research figure and table output

This directory is the canonical base (`OUT_DIR`) for notebooks that call `neurosignal.viz.init_figure_export` and for CSV exports written next to figures.

## Layout

- **Figures:** `figures_supporting/<notebook_id>/` — PNGs (or other formats) named `NN_section_gist.png` with serial prefix `01_`, `02_`, …
- **Registry:** The same notebook may write `figure_registry.csv` and `figure_registry.md` under that folder after a run.
- **Tables:** Some notebooks save CSV files directly under this directory (e.g. decoder comparison summaries).

Resolve `OUT_DIR` in notebooks as:

`ROOT / "04_output" / "research" / "output"`

where `ROOT` is the repo root (whether you launch Jupyter from the repo root or from `03_notebooks`).

## Notebook IDs

| Notebook | `notebook_id` |
|----------|----------------|
| `sbp-drift-discovery_v1.ipynb` | `N01_sbp_drift_discovery` |
| `sbp-drift-fix_v1.ipynb` | `N02_gold_drift_robust_features` |
| `sbp-mlp-decoder_v1.ipynb` | `N03_mlp_decoder` |

Generated binaries and CSVs under `04_output/` are listed in `.gitignore`; this README can remain versioned for contributors.
