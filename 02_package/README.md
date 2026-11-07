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
      features/
      decoders/
      align/
      metrics/
      viz/
```

### Subpackages, by role

Dependencies point one way: `features` ← `decoders` ← `metrics` ← `viz`; `align` and `io` stand alone.

- `neurosignal.io` — LINK NWB loaders and file inspection.
- `neurosignal.features` — per-session SBP normalisers: acausal `r_norm` (`normalize`, offline reference), the causal streaming normaliser (`causal`), joint SBP + TCR channels for the MLP (`joint`).
- `neurosignal.decoders` — the one decoder spec every evaluation uses (`linear`: `SPEC`, `make_xy`, `GramRidge`, purged blocked folds), the pooled MLP (`mlp`), the LINK-paper ridge baseline (`baseline`).
- `neurosignal.align` — cross-session alignment (Procrustes).
- `neurosignal.metrics` — decoder R² across days and date gaps (`drift`), the session QC screen (`qc`), session-bootstrap CIs (`stats`).
- `neurosignal.viz` — figure style constants (`style`), export (`fig_export`), drift and MLP plots.

Raw-broadband processing (filters, threshold crossings, SBP) returns as its own subpackage when it is built on the O'Doherty broadband data (D-07); there are no placeholder stubs.
