"""Fetch NISRA Census 2021 + NIMDM (Multiple Deprivation) tables.

NISRA publishes Census 2021 outputs as CSVs against Data Zones / SOAs / LGDs.
NIMDM 2017 publishes domain & overall ranks against Super Output Areas.

The exact URLs change as NISRA refreshes the catalogue, so this script:
  1. Tries a configurable list of canonical CSV URLs.
  2. Falls back to reading any pre-downloaded copy in outputs/.

If you have the raw files already (e.g. downloaded from
https://www.nisra.gov.uk/statistics/census/census-2021), drop them in
outputs/ with the filenames listed in CENSUS_FILES below.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CENSUS_FILES = {
    # filename in outputs/  ->  source URL (best-effort)
    "nimdm_2017_soa.csv":
        "https://www.nisra.gov.uk/system/files/statistics/NIMDM17_SOA.csv",
    "census_2021_population_density_dz.csv":
        "https://www.nisra.gov.uk/sites/nisra.gov.uk/files/publications/census-2021-population-density.csv",
    "census_2021_tenure_dz.csv":
        "https://www.nisra.gov.uk/sites/nisra.gov.uk/files/publications/census-2021-tenure.csv",
    "census_2021_dwellings_dz.csv":
        "https://www.nisra.gov.uk/sites/nisra.gov.uk/files/publications/census-2021-dwellings.csv",
}


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  already present: {dest.name}")
        return True
    try:
        print(f"  downloading {url}")
        r = requests.get(url, timeout=120, headers={"User-Agent": "belfast-sentinel/0.1"})
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"  wrote {dest.name} ({len(r.content):,} bytes)")
        return True
    except Exception as exc:
        print(f"  WARN: failed to download {url}: {exc}")
        return False


def main() -> None:
    print("[nisra] fetching Census 2021 + NIMDM tables")
    for name, url in CENSUS_FILES.items():
        download(url, OUTPUT_DIR / name)

    print(
        "\nIf any download failed, manually save the file from "
        "https://www.nisra.gov.uk/statistics/census/census-2021 "
        f"into {OUTPUT_DIR}\nwith the filenames above."
    )


if __name__ == "__main__":
    main()
