"""
neurosignal: electrophysiology analysis toolkit (LINK NWB, SBP drift, decoders).

Subpackages, by role (D-21):

- neurosignal.io        NWB loaders and inspection
- neurosignal.features  per-session normalisers: acausal `r_norm`, causal (streaming), joint SBP + TCR
- neurosignal.decoders  the decoder spec (Ridge), the pooled MLP, the LINK-paper baseline
- neurosignal.align     cross-session alignment (Procrustes)
- neurosignal.metrics   decoder R² across days, session QC, session-bootstrap statistics
- neurosignal.viz       figure style, export and plots
"""

__all__ = [
    "io",
    "features",
    "decoders",
    "align",
    "metrics",
    "viz",
]

__version__ = "0.2.0"
