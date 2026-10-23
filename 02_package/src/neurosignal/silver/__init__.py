"""Silver layer: filtering, artefact handling, and spike detection.

LINK-style NWB workflows in this package start at :mod:`neurosignal.gold` (SBP/TCR
are already binned in the files). The functions below are **stubs** reserved for
future raw-trace pipelines (e.g. human micro-ECoG); they raise
:class:`NotImplementedError` until implemented.

Import either from the package or from submodules:

    from neurosignal.silver import bandpass_filter
    from neurosignal.silver import filters
"""

from . import artefact, filters, spike_detect
from .artefact import detect_artefacts
from .filters import bandpass_filter, notch_filter
from .spike_detect import detect_spikes

__all__ = [
    "artefact",
    "filters",
    "spike_detect",
    "detect_spikes",
    "detect_artefacts",
    "bandpass_filter",
    "notch_filter",
]
