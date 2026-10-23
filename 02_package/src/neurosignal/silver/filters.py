"""
Filtering utilities for neurosignal.

Bandpass and notch filters at the Silver layer.
"""

from typing import Any


def bandpass_filter(signal: Any, *, low_hz: float, high_hz: float, fs: float) -> Any:
    """Stub for applying a bandpass filter to a signal."""
    raise NotImplementedError("bandpass_filter is not implemented yet.")


def notch_filter(signal: Any, *, freq_hz: float, fs: float) -> Any:
    """Stub for applying a notch filter to a signal."""
    raise NotImplementedError("notch_filter is not implemented yet.")

