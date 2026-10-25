"""
Download O'Doherty et al. 2017 broadband sessions and their labels into 01_data/odoherty_broadband/ (D-07).

    python scripts/download_odoherty_broadband.py                      # the 6 sessions in SESSIONS
    python scripts/download_odoherty_broadband.py indy_20161024_03     # chosen sessions
    python scripts/download_odoherty_broadband.py --list               # every broadband session and its size

Files, sizes and md5 checksums come from the Zenodo API: one "broadband for <session>" record per
session (raw NWB 1.0.6, 24,414.0625 Hz) and the main record 583331 (`<session>.mat`: positions and
spike times). Resumes partial files; a file counts as present only if its md5 matches. CC-BY-4.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "01_data" / "odoherty_broadband"
API = "https://zenodo.org/api/records"
MAIN_RECORD = 583331
QUERY = '"Sensorimotor Cortex Electrophysiology: broadband"'
SESSIONS = ("indy_20160915_01", "indy_20160921_01", "indy_20161005_06",
            "indy_20161024_03", "indy_20161207_02", "indy_20170124_01")


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def broadband_files() -> dict:
    """session -> (file key, size, md5, url) for every broadband record."""
    out, page = {}, 1
    while True:
        hits = get_json(f"{API}?q={urllib.parse.quote(QUERY)}&size=25&page={page}")["hits"]["hits"]
        if not hits:
            return out
        for h in hits:
            title = h["metadata"]["title"]
            if "broadband for " in title:
                session = title.rsplit("broadband for ", 1)[1].strip()
                for f in h["files"]:
                    out[session] = (f["key"], f["size"], f["checksum"].split(":")[1], f["links"]["self"])
        page += 1


def label_files() -> dict:
    files = get_json(f"{API}/{MAIN_RECORD}")["files"]
    return {f["key"][:-4]: (f["key"], f["size"], f["checksum"].split(":")[1], f["links"]["self"])
            for f in files if f["key"].endswith(".mat")}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(key: str, size: int, checksum: str, url: str) -> bool:
    path = OUT / key
    if path.exists() and path.stat().st_size == size and md5(path) == checksum:
        print(f"ok (present) {key}")
        return True
    start = path.stat().st_size if path.exists() and path.stat().st_size < size else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req, timeout=120) as r, path.open("ab" if start else "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    ok = md5(path) == checksum
    print(f"{'ok' if ok else 'MD5 MISMATCH'} {key} ({size / 1e9:.2f} GB)")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="*", default=list(SESSIONS))
    parser.add_argument("--list", action="store_true", help="list every broadband session and exit")
    args = parser.parse_args()

    broad = broadband_files()
    if args.list:
        for s, (_, size, _, _) in sorted(broad.items()):
            print(f"{s}  {size / 1e9:.2f} GB")
        print(f"{len(broad)} sessions, {sum(v[1] for v in broad.values()) / 1e9:.1f} GB")
        return
    labels = label_files()
    missing = [s for s in args.sessions if s not in broad or s not in labels]
    if missing:
        sys.exit(f"no broadband record or label file for: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    results = [fetch(*f) for s in args.sessions for f in (broad[s], labels[s])]
    if not all(results):
        sys.exit("some files failed their md5 check; re-run to resume")


if __name__ == "__main__":
    main()
