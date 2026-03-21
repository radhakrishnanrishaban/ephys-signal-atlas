"""
NWB discovery and inspection helpers for neurosignal.

Functions take an opened pynwb NWB object and return data or print summaries.
Used by discovery notebooks (e.g. N01) to stay thin.
"""

import sys
from typing import Any, Optional

import pandas as pd


def _get_fields(node: Any) -> dict:
    """Get attribute/field dict from an NWB node (handles callable .fields)."""
    if not hasattr(node, "fields"):
        return {}
    f = node.fields
    if callable(f):
        try:
            f = f()
        except Exception:
            return {}
    return f if isinstance(f, dict) else {}


def _get_keys(node: Any) -> list:
    """Get iterable keys from an NWB node (keys() or fields keys)."""
    if hasattr(node, "keys") and callable(node.keys):
        return list(node.keys())
    return list(_get_fields(node).keys())


def print_nwb_tree(nwb: Any, stream: Optional[Any] = None) -> None:
    """Print full NWB structure (recursive tree). stream defaults to sys.stdout."""
    if stream is None:
        stream = sys.stdout

    def _print_tree(node: Any, prefix: str) -> None:
        name = getattr(node, "name", None) or str(node)
        stream.write(prefix + name + "\n")
        keys = _get_keys(node)
        for k in keys:
            try:
                child = node[k]
            except (TypeError, KeyError):
                child = getattr(node, k, None)
            if child is None:
                continue
            try:
                _print_tree(child, prefix + "  ")
            except Exception as e:
                stream.write(prefix + "  " + k + " -> " + str(e) + "\n")

    _print_tree(nwb, "")


def nwb_top_level_fields(nwb: Any) -> dict[str, str]:
    """Return dict of top-level attribute name -> type name (e.g. for Step 3)."""
    fields = _get_fields(nwb)
    if not fields:
        # NWBFile may use callable .fields
        if hasattr(nwb, "fields"):
            f = nwb.fields
            if callable(f):
                try:
                    fields = f()
                except Exception:
                    fields = {}
            else:
                fields = f if isinstance(f, dict) else {}
    result = {}
    for key in fields:
        try:
            obj = nwb[key]
        except (TypeError, KeyError):
            obj = getattr(nwb, key, None)
        if obj is None:
            continue
        result[key] = type(obj).__name__
    return result


def nwb_session_metadata(nwb: Any) -> dict[str, Any]:
    """Return small dict: session_id, session_start_time, experiment_description, general_keys."""
    out: dict[str, Any] = {}
    if hasattr(nwb, "session_id") and nwb.session_id:
        out["session_id"] = nwb.session_id
    if hasattr(nwb, "session_start_time"):
        out["session_start_time"] = nwb.session_start_time
    if hasattr(nwb, "experiment_description"):
        out["experiment_description"] = nwb.experiment_description
    if hasattr(nwb, "general") and nwb.general is not None:
        out["general_keys"] = (
            list(nwb.general.keys()) if hasattr(nwb.general, "keys") else []
        )
    return out


def nwb_electrodes_dataframe(nwb: Any) -> Optional[pd.DataFrame]:
    """Return electrodes table as DataFrame, or None if absent."""
    if not hasattr(nwb, "electrodes") or nwb.electrodes is None:
        return None
    return nwb.electrodes.to_dataframe()


def nwb_analysis_timeseries_summary(nwb: Any) -> list[tuple[str, tuple, int]]:
    """
    Prefer analysis, fallback acquisition. Return list of (name, data_shape, timestamps_len).
    """
    ts_container = getattr(nwb, "acquisition", None)
    if not ts_container or (hasattr(ts_container, "__len__") and len(ts_container) == 0):
        ts_container = getattr(nwb, "analysis", None)
    if not ts_container:
        return []
    result = []
    for name, ts in ts_container.items():
        if not hasattr(ts, "data") or not hasattr(ts.data, "shape"):
            continue
        shape = tuple(ts.data.shape)
        tlen = 0
        if hasattr(ts, "timestamps") and ts.timestamps is not None:
            t = ts.timestamps
            if hasattr(t, "__getitem__"):
                try:
                    tlen = len(t[:])
                except Exception:
                    tlen = -1
        result.append((name, shape, tlen))
    return result


def nwb_timeseries_dataframe(ts: Any, channel_prefix: str = "ch") -> pd.DataFrame:
    """
    Build DataFrame from a PyNWB TimeSeries: columns ch_0, ch_1, ... or single column if 1D.
    """
    data = ts.data[:]
    t = ts.timestamps
    if t is not None and hasattr(t, "__getitem__"):
        timestamps = t[:]
    else:
        import numpy as np

        rate = getattr(ts, "rate", None) or 1.0
        timestamps = np.arange(data.shape[0], dtype=float) / rate
    data = data[:]
    if data.ndim == 1:
        df = pd.DataFrame({"value": data}, index=pd.Index(timestamps, name="time"))
    else:
        columns = [f"{channel_prefix}_{i}" for i in range(data.shape[1])]
        df = pd.DataFrame(data, columns=columns, index=pd.Index(timestamps, name="time"))
    return df


def nwb_trials_dataframe(nwb: Any, interval_name: str = "trials") -> Optional[pd.DataFrame]:
    """Return trials (or named interval) as DataFrame, or None if absent."""
    intervals = getattr(nwb, "intervals", None)
    if not intervals:
        return None
    interval = None
    if hasattr(intervals, "get"):
        interval = intervals.get(interval_name)
    if interval is None and hasattr(intervals, "values"):
        vals = list(intervals.values())
        if len(vals) == 1:
            interval = vals[0]
    if interval is None or not hasattr(interval, "to_dataframe"):
        return None
    return interval.to_dataframe()


def nwb_units_dataframe(nwb: Any) -> Optional[pd.DataFrame]:
    """Return units table as DataFrame, or None if absent."""
    if not hasattr(nwb, "units") or nwb.units is None:
        return None
    return nwb.units.to_dataframe()


__all__ = [
    "print_nwb_tree",
    "nwb_top_level_fields",
    "nwb_session_metadata",
    "nwb_electrodes_dataframe",
    "nwb_analysis_timeseries_summary",
    "nwb_timeseries_dataframe",
    "nwb_trials_dataframe",
    "nwb_units_dataframe",
]
