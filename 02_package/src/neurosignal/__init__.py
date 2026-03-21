"""
neurosignal: electrophysiology analysis toolkit (LINK NWB, SBP drift, decoders).

This package is designed to be imported once per notebook:

    import neurosignal

Subpackages expose the concrete functions:

- neurosignal.io
- neurosignal.bronze
- neurosignal.silver
- neurosignal.gold
- neurosignal.viz
- neurosignal.utils
"""

__all__ = [
    "io",
    "bronze",
    "silver",
    "gold",
    "viz",
    "utils",
]

__version__ = "0.1.0"
