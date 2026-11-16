"""
Fetch the LINK sessions (DANDI:001201) that are not on disk, keep SBP + index_velocity, delete the NWB (EDA-19).

    python scripts/fetch_link.py --list          # what is on disk, what is missing, how many GB
    python scripts/fetch_link.py                 # fetch every missing session (resumable: just re-run)
    python scripts/fetch_link.py --limit 5       # the next 5 missing sessions
    python scripts/fetch_link.py --only 20211015_obj-1wbkdus --out /tmp/x   # chosen sessions, anywhere

Assets come from the pinned dandiset version (FACTS → "LINK session dates (all 312)"). Per session:
download to <out>/.part/ (a partial file resumes with an HTTP Range request) → check size and the
DANDI sha2-256 → read SBP and index_velocity with the same loader the bench uses on NWB files →
write <out>/sub-Monkey-N_ses-<key>_ecephys.npz (native dtype, float64, so scores match the NWB
exactly) → delete the NWB. A session counts as done once its .npz exists. Sessions already in the
data dir as NWB are skipped unless named with --only. Stops before a download that would leave less
than MIN_FREE_GB free. Standard library + the package loader; no new dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "02_package" / "src"))

import numpy as np  # noqa: E402
from pynwb import NWBHDF5IO  # noqa: E402

from neurosignal.io.loaders import (  # noqa: E402
    LINK_EXTRACTED_DIR, get_data_dir, list_link_session_paths, load_link_sbp_velocity, session_key_from_path,
)

DANDISET, VERSION = "001201", "0.261015.2153"
API = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}/assets/"
MIN_FREE_GB = 2.0
CHUNK = 1 << 20


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def list_assets() -> dict:
    """{session key: {path, size, asset_id}} for every NWB asset of the pinned version."""
    out, url = {}, f"{API}?page_size=1000"
    while url:
        page = get_json(url)
        for a in page["results"]:
            if a["path"].endswith(".nwb"):
                out[session_key_from_path(a["path"])] = {"path": a["path"], "size": a["size"], "asset_id": a["asset_id"]}
        url = page["next"]
    return dict(sorted(out.items()))


def npz_name(key: str) -> str:
    return f"sub-Monkey-N_ses-{key}_ecephys.npz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(asset: dict, part: Path) -> str:
    """Download (or resume) one asset into part; return its sha2-256 from DANDI after checking it."""
    meta = get_json(f"{API}{asset['asset_id']}/")
    want = meta["digest"]["dandi:sha2-256"]
    url = next(u for u in meta["contentUrl"] if "api/assets" in u)
    start = part.stat().st_size if part.exists() and part.stat().st_size < asset["size"] else 0
    if start == 0 and part.exists():
        part.unlink()
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        if start and r.status != 206:
            start = 0  # server ignored the range: start over
        with part.open("ab" if start else "wb") as fh:
            shutil.copyfileobj(r, fh, CHUNK)
    if part.stat().st_size != asset["size"]:
        raise RuntimeError(f"size {part.stat().st_size} != {asset['size']}; re-run to resume")
    if sha256(part) != want:
        part.unlink()
        raise RuntimeError("sha2-256 mismatch; partial file deleted, re-run to fetch again")
    return want


def extract(nwb: Path, out: Path, key: str, asset: dict, digest: str) -> None:
    """Write SBP + index_velocity (and where they came from) to out atomically."""
    sbp, vel = load_link_sbp_velocity(nwb)
    with NWBHDF5IO(str(nwb), mode="r") as io:
        f = io.read()
        identifier, start = f.identifier, str(f.session_start_time)
    tmp = nwb.with_suffix(".tmp.npz")  # next to the NWB in .part/, so a crash never leaves a stray .npz in out
    np.savez(tmp, sbp=sbp, vel=vel, key=key, dandi_path=asset["path"], asset_id=asset["asset_id"],
             sha256=digest, dandi_version=f"{DANDISET}/{VERSION}", identifier=identifier, session_start_time=start)
    os.replace(tmp, out)


def fetch_one(key: str, asset: dict, out_dir: Path) -> None:
    part_dir = out_dir / ".part"
    part_dir.mkdir(parents=True, exist_ok=True)
    part = part_dir / Path(asset["path"]).name
    free_gb = shutil.disk_usage(out_dir).free / 1e9
    if free_gb - asset["size"] / 1e9 < MIN_FREE_GB:
        sys.exit(f"stopping: {free_gb:.1f} GB free, {key} needs {asset['size'] / 1e9:.2f} GB and {MIN_FREE_GB} GB must stay free")
    t0 = time.time()
    digest = download(asset, part)
    extract(part, out_dir / npz_name(key), key, asset, digest)
    part.unlink()
    print(f"ok {key} ({asset['size'] / 1e6:.0f} MB, {time.time() - t0:.0f} s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show what is on disk and what is missing, then exit")
    parser.add_argument("--only", nargs="+", metavar="KEY", help="fetch these session keys (even if on disk as NWB)")
    parser.add_argument("--limit", type=int, help="fetch at most this many sessions")
    parser.add_argument("--out", type=Path, help=f"where the .npz files go (default <data dir>/{LINK_EXTRACTED_DIR})")
    args = parser.parse_args()

    data_dir = get_data_dir(ROOT)
    out_dir = args.out or data_dir / LINK_EXTRACTED_DIR
    assets = list_assets()
    on_disk = list_link_session_paths(data_dir)
    nwb = {k for k, p in on_disk.items() if p.suffix == ".nwb"}
    done = {k for k in assets if (out_dir / npz_name(k)).exists()}

    if args.only:
        unknown = [k for k in args.only if k not in assets]
        if unknown:
            sys.exit(f"not in DANDI {DANDISET} {VERSION}: {unknown}")
        todo = [k for k in args.only if k not in done]
    else:
        todo = [k for k in assets if k not in nwb and k not in done]
    if args.limit is not None:
        todo = todo[: args.limit]

    gb = sum(assets[k]["size"] for k in todo) / 1e9
    print(f"DANDI {DANDISET} {VERSION}: {len(assets)} sessions · {len(nwb)} on disk as NWB · "
          f"{len(done)} extracted in {out_dir} · {len(todo)} to fetch ({gb:.2f} GB download)")
    if args.list or not todo:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, k in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {k}", flush=True)
        try:
            fetch_one(k, assets[k], out_dir)
        except (RuntimeError, OSError) as e:  # network or file trouble: report, keep going, re-run resumes
            print(f"FAILED {k}: {e}", flush=True)
            failed.append(k)
    if failed:
        sys.exit(f"{len(failed)} sessions failed: {failed}; re-run to resume")


if __name__ == "__main__":
    main()
