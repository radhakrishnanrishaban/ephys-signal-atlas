"""
Figure export helper for notebooks: serial numbering, section-based filenames, and registry.

Use from a notebook after creating a figure:

    from neurosignal.viz.fig_export import init_figure_export, save_figure, get_figure_registry

    init_figure_export(out_dir=OUT_DIR, notebook_id="N02_gold_drift_robust_features")
    # ... in each plotting cell, after fig is created:
    save_figure(fig, section="signal_decoder_demo", gist="sbp_overview", role="supporting")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

# Module-level state: set by init_figure_export(), used by save_figure()
_FIG_COUNTER: int = 0
_OUT_DIR: Optional[Path] = None
_SUBDIR: str = "figures_supporting"
_NOTEBOOK_SUBFOLDER: Optional[str] = None
_SAVED_FIGURES: List[Dict[str, Any]] = []


def init_figure_export(
    out_dir: Path,
    notebook_id: Optional[str] = None,
    subdir: str = "figures_supporting",
    reset: bool = True,
) -> None:
    """
    Initialise figure export: output directory, optional notebook subfolder, and optional reset of counter/registry.

    out_dir: base output directory (e.g. 04_output/research/output).
    notebook_id: if set, figures go under out_dir / subdir / notebook_id (e.g. N02_gold_drift_robust_features).
    subdir: typically "figures_main" or "figures_supporting" (any path segment is allowed).
    reset: if True, set counter to 0 and clear the saved-figures registry.
    """
    global _FIG_COUNTER, _OUT_DIR, _SUBDIR, _NOTEBOOK_SUBFOLDER, _SAVED_FIGURES
    _OUT_DIR = Path(out_dir)
    _SUBDIR = subdir
    _NOTEBOOK_SUBFOLDER = notebook_id
    if reset:
        _FIG_COUNTER = 0
        _SAVED_FIGURES = []


def save_figure(
    fig: Any,
    section: str,
    gist: str,
    role: str = "supporting",
    dpi: int = 150,
    fig_format: str = "png",
) -> Path:
    """
    Save the current or given figure with serial number and section-based filename; append to registry.

    fig: matplotlib figure (or None to use current figure via plt.gcf()).
    section: short section slug (e.g. signal_decoder_demo, gold_layer).
    gist: short description (e.g. sbp_overview, drift_vs_time). Will be normalised to safe filename.
    role: one of "supporting", "candidate_supporting", "exploratory", "main".
    dpi: passed to savefig.
    fig_format: file extension without dot (png, pdf, etc.).

    Returns the path where the figure was saved.
    """
    import re
    import matplotlib.pyplot as plt

    global _FIG_COUNTER, _OUT_DIR, _SUBDIR, _NOTEBOOK_SUBFOLDER, _SAVED_FIGURES

    if _OUT_DIR is None:
        raise RuntimeError("Call init_figure_export(out_dir=...) before save_figure()")

    _FIG_COUNTER += 1
    nn = f"{_FIG_COUNTER:02d}"
    safe_section = re.sub(r"[^\w\-]", "_", section).strip("_") or "section"
    safe_gist = re.sub(r"[^\w\-]", "_", gist).strip("_") or "figure"
    filename = f"{nn}_{safe_section}_{safe_gist}.{fig_format}"
    dest_dir = _OUT_DIR / _SUBDIR
    if _NOTEBOOK_SUBFOLDER:
        dest_dir = dest_dir / _NOTEBOOK_SUBFOLDER
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename

    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    try:
        rel_path = path.relative_to(_OUT_DIR)
    except ValueError:
        rel_path = path

    _SAVED_FIGURES.append({
        "serial": _FIG_COUNTER,
        "filename": str(rel_path),
        "section": section,
        "gist": gist,
        "role": role,
    })
    return path


def get_figure_registry() -> List[Dict[str, Any]]:
    """Return the list of saved-figure metadata dicts (serial, filename, section, gist, role)."""
    return list(_SAVED_FIGURES)


def get_figure_counter() -> int:
    """Return the current figure counter (number of figures saved this session)."""
    return _FIG_COUNTER
