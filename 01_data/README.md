# 01_data — Local data directory

This folder holds **LINK** session files (NWB) used by notebooks and the neurosignal pipeline. The package resolves this path via `neurosignal.io.loaders.get_data_dir()` (prefers `01_data`, falls back to `02_data` if present).

## Source and provenance

- **Dataset:** [LINK: Long-Term Intracortical Neural Activity and Kinematics](https://dandiarchive.org/dandiset/001201/0.251023.2336)
- **DANDI:** 001201 · **Version:** 0.251023.2336  
- **DOI:** [10.48324/dandi.001201/0.251023.2336](https://doi.org/10.48324/dandi.001201/0.251023.2336)  
- **License:** CC-BY-4.0

Files here are copies of assets from the DANDI Archive. Always cite the dataset when publishing or presenting results (see full citation in the data spec).

## Content and naming

- **Format:** NWB (Neurodata Without Borders) — `.nwb` files.
- **Naming:** e.g. `sub-Monkey-N_ses-YYYYMMDD_ecephys.nwb`.
- **Usage:** Notebooks under `03_notebooks/` load all `*.nwb` in this directory (or in legacy `02_data/`), sorted by session date.

## Download

- **DANDI CLI:** `dandi download https://dandiarchive.org/dandiset/001201/0.251023.2336`
- **Python (notebook):** See the optional DANDI download cell in `05_gold_drift_robust_features.ipynb` (e.g. 2 sessions ≈ 88 MB).

## Full metadata and standardization

**Full data specification (metadata, DANDI fields, NWB layout, why it’s scientifically standard):**  
→ [00_resources-hub-main/data/01_LINK_DANDI_001201_data_spec.md](../00_resources-hub-main/data/01_LINK_DANDI_001201_data_spec.md)
