"""Fetch the Northern Ireland House Price Index (NI HPI) by Local Government District.

NISRA publishes the NI HPI quarterly:
  https://www.nisra.gov.uk/publications/ni-house-price-index-statistical-reports

The CSV download endpoint changes per release. If the URL below fails, manually
download the latest "Standardised price by LGD" CSV into
data_preparation/outputs/ni_hpi_lgd.csv
"""

from __future__ import annotations

from pathlib import Path
import requests

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Best-effort canonical URL — adjust if NISRA refreshes.
NI_HPI_CSV = (
    "https://www.finance-ni.gov.uk/sites/default/files/publications/dfp/"
    "NI-HPI-Q4-2024-Tables.csv"
)


def main() -> None:
    out = OUTPUT_DIR / "ni_hpi_lgd.csv"
    if out.exists():
        print(f"already present: {out}")
        return
    try:
        print(f"downloading {NI_HPI_CSV}")
        r = requests.get(NI_HPI_CSV, timeout=120, headers={"User-Agent": "belfast-sentinel/0.1"})
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"wrote {out} ({len(r.content):,} bytes)")
    except Exception as exc:
        print(f"WARN: failed to download NI HPI: {exc}")
        print(
            "Manually download the latest NI HPI tables CSV from\n"
            "  https://www.finance-ni.gov.uk/topics/dof-statistics-and-research/"
            "northern-ireland-house-price-index\n"
            f"and save as {out}"
        )


if __name__ == "__main__":
    main()
