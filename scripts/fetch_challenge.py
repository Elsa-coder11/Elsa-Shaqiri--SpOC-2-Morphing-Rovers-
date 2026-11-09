"""Download ESA's official SpOC-2 "Morphing Rovers" evaluation code and data.

ESA's repository (https://github.com/esa/SpOC2) has no licence file, so this
project does not redistribute those files. This script fetches them from the
official repository at a pinned commit and lays them out where the official
evaluation code expects them:

    external/morphing_udp.py                 official evaluation code (unmodified)
    data/spoc2/morphing/coordinates.txt      landing sites and sample positions
    data/spoc2/morphing/example_rover.npy    ESA's example chromosome
    data/spoc2/morphing/Maps/MapX.jpg|JPG    the six height maps

Usage:  python scripts/fetch_challenge.py
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

COMMIT = "fac5a6a8b9c1fe8b3105a7c959a76814bbd5e0f1"  # esa/SpOC2, 22 April 2023
ARCHIVE_URL = f"https://github.com/esa/SpOC2/archive/{COMMIT}.zip"
ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
DATA = ROOT / "data" / "spoc2" / "morphing"

# The official code opens Map2..Map5 with an upper-case ".JPG" extension while the
# repository stores ".jpg". Saving under the exact names the code asks for makes it
# work on case-sensitive file systems (Linux, macOS) as well as on Windows.
HEIGHTMAP_NAMES = ["Map1.jpg", "Map2.JPG", "Map3.JPG", "Map4.JPG", "Map5.JPG", "Map6.jpg"]


def main() -> None:
    if (EXTERNAL / "morphing_udp.py").exists() and (DATA / "coordinates.txt").exists():
        print("Challenge files already present - nothing to do.")
        return

    print(f"Downloading {ARCHIVE_URL} ...")
    with urllib.request.urlopen(ARCHIVE_URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    udp_member = next(n for n in archive.namelist() if n.endswith("Morphing Rovers/morphing_udp.py"))
    base = udp_member[: -len("morphing_udp.py")]

    EXTERNAL.mkdir(parents=True, exist_ok=True)
    (DATA / "Maps").mkdir(parents=True, exist_ok=True)

    (EXTERNAL / "morphing_udp.py").write_bytes(archive.read(udp_member))
    for name in ("coordinates.txt", "example_rover.npy"):
        (DATA / name).write_bytes(archive.read(base + "data/" + name))
    for target in HEIGHTMAP_NAMES:
        source = base + "data/Maps/" + target.rsplit(".", 1)[0] + ".jpg"
        (DATA / "Maps" / target).write_bytes(archive.read(source))

    print(f"Saved evaluation code to {EXTERNAL} and data to {DATA}.")


if __name__ == "__main__":
    main()
