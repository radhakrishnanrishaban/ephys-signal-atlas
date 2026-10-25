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

## `odoherty_broadband/` — raw broadband for the silver layer

Not LINK, and not read by the LINK loaders (they only glob `01_data/*.nwb`).

- **Dataset:** O'Doherty, J. E., Cardoso, M. M. B., Makin, J. G., & Sabes, P. N. (2017). *Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology.* Zenodo, [10.5281/zenodo.583331](https://doi.org/10.5281/zenodo.583331), plus one "broadband" Zenodo record per session. **License:** CC-BY-4.0.
- **Content:** `indy_YYYYMMDD_NN.nwb`, raw broadband from the Utah array, 24,414.0625 Hz, NWB 1.0.6 (HDF5; read with `h5py`, not `pynwb`). `indy_YYYYMMDD_NN.mat`, the same session's fingertip/target position and spike times from the main record.
- **Sessions here:** 2016-09-15, 2016-09-21, 2016-10-05, 2016-10-24, 2016-12-07, 2017-01-24 (6 of the 30 broadband sessions, 7.8 GB). Chosen to span the recording period.
- **Download:** `python scripts/download_odoherty_broadband.py` (files, md5 checksums and URLs from the Zenodo API; resumes, and checks every md5). `--list` shows all 30 broadband sessions.
