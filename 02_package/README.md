# neurosignal

Electrophysiology toolkit for **LINK-style NWB** workflows: loaders, SBP normalization, drift analysis, linear and MLP decoders, and figure export helpers used by the project notebooks.

## Goals

- Provide a **single, reusable package** that notebooks import instead of duplicating NWB and analysis code.
- Keep notebooks **thin**: orchestration and parameters in the notebook; reusable logic in `neurosignal`.
- Support **consistent, publication-style figures** via shared plotting helpers.

## Install (from this repo)

From the project root, in an activated virtualenv:

```bash
pip install -e 02_package
```

In a Jupyter notebook (top cell), use:

```python
%pip install -e ../02_package --quiet
import neurosignal
print(f"neurosignal v{neurosignal.__version__}")
```

## Package layout

```text
02_package/
  pyproject.toml
  README.md
  src/
    neurosignal/
      io/
      bronze/
      silver/
      gold/
      viz/
      utils/
```

### Subpackages (intended roles)

- `neurosignal.io` — NWB readers and related I/O (LINK loaders live here).
- `neurosignal.bronze` — reserved for raw ingestion (stub for v0.1).
- `neurosignal.silver` — filtering / artefact / spike-detection API (**stubs**; use `from neurosignal.silver import bandpass_filter` or submodules `silver.filters`, etc.).
- `neurosignal.gold` — SBP QC, normalization, drift metrics, decoders, and drift plots.
- `neurosignal.viz` — figure export and registry utilities.
- `neurosignal.utils` — configuration and validation helpers.

Example entrypoints for the drift / decoder notebooks: `neurosignal.io.loaders`, `neurosignal.gold`, `neurosignal.viz`.
