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


LINK_EXTRACTED_DIR = "link"  # under the data dir: SBP + index_velocity per session, from scripts/fetch_link.py


def session_key_from_path(p: Union[Path, str]) -> str:
    """
    Session key from a LINK file name: the date (20200127), or date + DANDI object id on the 9 dates
    with two sessions (ses-20211015_obj-1wbkdus → 20211015_obj-1wbkdus). key[:8] is always the date.
    """
    m = re.search(r"ses-(\d{8}(?:_obj-[0-9a-z]+)?)_", Path(p).name)
    if not m:
        raise ValueError(f"not a LINK session file name: {Path(p).name}")
    return m.group(1)


def list_link_session_paths(data_dir: Optional[Union[Path, str]] = None) -> dict:
    """
    {session key: path} for every LINK session on disk, sorted by key: NWB files in the data dir and
    extracted .npz files in <data dir>/link/. Stops if one session exists in both forms.
    """
    data_dir = Path(data_dir or get_data_dir())
    paths = list(data_dir.glob("*.nwb")) + list((data_dir / LINK_EXTRACTED_DIR).glob("*.npz"))
    out: dict = {}
    for p in paths:
        k = session_key_from_path(p)
        if k in out:
            raise ValueError(f"session {k} is on disk twice: {out[k].name} and {p.name}")
        out[k] = p
    return dict(sorted(out.items()))


LINK_TARGET_STYLES = ("CO", "RD")


def load_link_target_style(path: Union[Path, str]) -> str:
    """
    Target style of one LINK session, "CO" or "RD", from the NWB identifier (2020-01-27_CO_nwb) or the
    identifier kept in its extracted .npz (FACTS → "LINK target style"; the codes are not explained in the file).
    """
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            identifier = str(z["identifier"])
    else:
        import h5py

        with h5py.File(path, "r") as f:
            identifier = f["identifier"][()].decode()
    style = identifier.split("_")[1] if identifier.count("_") >= 2 else ""
    if style not in LINK_TARGET_STYLES:
        raise ValueError(f"{path.name}: no target style in identifier {identifier!r}")
    return style


def load_link_session(path: Union[Path, str]) -> tuple:
    """(sbp, vel) of one LINK session, from its NWB file or its extracted .npz (same arrays, same dtype)."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            return z["sbp"], z["vel"]
    return load_link_sbp_velocity(path)


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


LINK_KINEMATICS = ("index_position", "mrs_position", "index_velocity", "mrs_velocity")


def load_link_trial_session(nwb_path: Union[Path, str]) -> dict:
    """
    SBP, the four LINK kinematic series and the bin where each trial starts, for the
    trial-based split of the LINK paper (Temmar et al. 2025). Returns
    {"sbp": (T, 96), "kin": (T, 4) in LINK_KINEMATICS order, "trial_start_bins": (n_trials,)}.
    """
    with NWBHDF5IO(str(nwb_path), mode="r") as io:
        nwb = io.read()
        sbp_ts = nwb.analysis["SpikingBandPower"]
        sbp = np.asarray(sbp_ts.data[:])
        kin = np.column_stack([np.asarray(nwb.analysis[k].data[:]).ravel() for k in LINK_KINEMATICS])
        stamps = np.asarray(sbp_ts.timestamps[:])
        starts = np.asarray(nwb.trials["start_time"][:])
    trial_start_bins = np.searchsorted(stamps, starts - 1e-6)
    return {"sbp": sbp, "kin": kin, "trial_start_bins": trial_start_bins}


def load_link_sessions(
    paths: list,
    include_tcr: bool = False,
) -> tuple:
    """
    Load SBP and index_velocity for each path (NWB or extracted .npz). Paths are sorted by session key.
    Returns (sessions_sbp, sessions_vel, session_dates); each label is the session key: the YYYYMMDD
    date, with a DANDI object suffix only on the 9 dates that have two sessions.
    If include_tcr=True, returns (sessions_sbp, sessions_tcr, sessions_vel, session_dates).
    """
    paths = [Path(p) for p in paths]
    paths_sorted = sorted(paths, key=session_key_from_path)
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
            sbp, vel = load_link_session(p)
            sessions_sbp.append(sbp)
            sessions_vel.append(vel)
        session_dates.append(session_key_from_path(p))
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
    "session_key_from_path",
    "list_link_session_paths",
    "load_link_session",
    "load_link_target_style",
    "load_link_sbp_velocity",
    "load_link_sbp_tcr_velocity",
    "load_link_sessions",
    "load_nwb_extracellular",
    "load_actigraphy_time_series",
]

