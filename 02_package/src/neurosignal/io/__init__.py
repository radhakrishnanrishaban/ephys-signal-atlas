"""
I/O utilities for neurosignal.

Responsibility:
- Provide loaders for raw electrophysiology and behavioural data.
- Provide exporters for cleaned / Gold-layer tables and metadata.
"""

from . import loaders, exporters, nwb_inspect  # noqa: F401

__all__ = ["loaders", "exporters", "nwb_inspect"]
