"""
Data loading entrypoints for neurosignal.

Provides LINK (DANDI 001201) loaders for SBP and kinematics.
Other loaders are stubs until implementation.
"""

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
from pynwb import NWBHDF5IO


def get_repo_root() -> Path:
    """Return ephys-signal repo root (directory that contains ``02_package/src``)."""
    cwd = Path.cwd()
    if (cwd / "02_package" / "src").is_dir():
        return cwd
    if (cwd.parent / "02_package" / "src").is_dir():
        return cwd.parent
    raise FileNotFoundError(
        "Could not find repo root (expected 02_package/src). "
        "Run Jupyter from the repo root or from 03_notebooks."
    )


def get_data_dir(root: Optional[Union[Path, str]] = None) -> Path:
    """Return LINK NWB directory: ``01_data`` under repo root, or ``02_data`` if that is the only tree present."""
    base = Path(root) if root is not None else get_repo_root()
    preferred = base / "01_data"
    legacy = base / "02_data"
    if preferred.is_dir():
        return preferred
    if legacy.is_dir():
        return legacy
    return preferred


def list_link_nwb_paths(data_dir: Optional[Union[Path, str]] = None) -> list[Path]:
    """List LINK NWB files in data_dir (default: get_data_dir()), sorted by session date."""
    data_dir = data_dir or get_data_dir()
    paths = sorted(Path(data_dir).glob("*.nwb"), key=session_date_from_path)
    if not paths:
        raise FileNotFoundError(
            "No .nwb in 01_data (or 02_data). Add LINK sessions from DANDI 001201 or run a DANDI download cell."
        )
    return paths


def get_link_nwb_path(
    data_dir: Optional[Union[Path, str]] = None, index: int = 0
) -> Path:
    """Return path to one LINK NWB file (by index in date-sorted list)."""
    return list_link_nwb_paths(data_dir)[index]


@contextmanager
def open_nwb(path: Union[Path, str]) -> Iterator[Any]:
    """Context manager: open NWB file and yield the read NWB object. Closes on exit."""
    with NWBHDF5IO(str(path), mode="r") as io:
        yield io.read()


def session_date_from_path(p: Path) -> str:
    """Extract session date from LINK-style path name (e.g. ses-20200127)."""
    m = re.search(r"ses-(\d{8})", p.name)
    return m.group(1) if m else "00000000"


def load_link_sbp_velocity(nwb_path: Union[Path, str]) -> tuple:
    """Load SBP and index_velocity from LINK NWB. Returns (sbp, vel); warns if rates differ."""
    with NWBHDF5IO(str(nwb_path), mode="r") as io:
        nwb = io.read()
        sbp_ts = nwb.analysis["SpikingBandPower"]
        vel_ts = nwb.analysis["index_velocity"]
        sbp_rate = getattr(sbp_ts, "rate", None)
        vel_rate = getattr(vel_ts, "rate", None)
        if sbp_rate and vel_rate and not np.isclose(sbp_rate, vel_rate, rtol=0.01):
            print(
                f"[WARNING] {Path(nwb_path).name}: SBP rate ({sbp_rate} Hz) ≠ "
                f"velocity rate ({vel_rate} Hz) — bins are misaligned; R² may be suppressed."
            )
        sbp = np.asarray(sbp_ts.data[:])
        vel = np.asarray(vel_ts.data[:]).ravel()
    return sbp, vel


def load_link_sbp_tcr_velocity(nwb_path: Union[Path, str]) -> tuple:
    """Load SBP, TCR (ThresholdCrossings), and index_velocity from LINK NWB. Returns (sbp, tcr, vel)."""
    with NWBHDF5IO(str(nwb_path), mode="r") as io:
        nwb = io.read()
        sbp_ts = nwb.analysis["SpikingBandPower"]
        tcr_ts = nwb.analysis["ThresholdCrossings"]
        vel_ts = nwb.analysis["index_velocity"]
        sbp = np.asarray(sbp_ts.data[:])
        tcr = np.asarray(tcr_ts.data[:])
        vel = np.asarray(vel_ts.data[:]).ravel()
    return sbp, tcr, vel


def load_link_sessions(
    paths: list,
    include_tcr: bool = False,
) -> tuple:
    """
    Load SBP and index_velocity for each path. Paths are sorted by session date.
    Returns (sessions_sbp, sessions_vel, session_dates) with session_dates as YYYYMMDD strings.
    If include_tcr=True, returns (sessions_sbp, sessions_tcr, sessions_vel, session_dates).
    """
    paths = [Path(p) for p in paths]
    paths_sorted = sorted(paths, key=session_date_from_path)
    sessions_sbp: list[np.ndarray] = []
    sessions_tcr: list[np.ndarray] = []
    sessions_vel: list[np.ndarray] = []
    session_dates: list[str] = []
    for p in paths_sorted:
        if include_tcr:
            sbp, tcr, vel = load_link_sbp_tcr_velocity(p)
            sessions_sbp.append(sbp)
            sessions_tcr.append(tcr)
            sessions_vel.append(vel)
        else:
            sbp, vel = load_link_sbp_velocity(p)
            sessions_sbp.append(sbp)
            sessions_vel.append(vel)
        session_dates.append(session_date_from_path(p))
    if include_tcr:
        return sessions_sbp, sessions_tcr, sessions_vel, session_dates
    return sessions_sbp, sessions_vel, session_dates


def load_nwb_extracellular(path: str, *, channel: Optional[int] = None) -> Any:
    """Stub for loading an extracellular trace from an NWB file."""
    raise NotImplementedError("load_nwb_extracellular is not implemented yet.")


def load_actigraphy_time_series(path: str) -> Any:
    """Stub for loading a simple behavioural / actigraphy time series."""
    raise NotImplementedError("load_actigraphy_time_series is not implemented yet.")


__all__ = [
    "get_repo_root",
    "get_data_dir",
    "list_link_nwb_paths",
    "get_link_nwb_path",
    "open_nwb",
    "session_date_from_path",
    "load_link_sbp_velocity",
    "load_link_sbp_tcr_velocity",
    "load_link_sessions",
    "load_nwb_extracellular",
    "load_actigraphy_time_series",
]

