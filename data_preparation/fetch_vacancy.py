"""Download LPS domestic property vacancy rates (district level) for Belfast.

Outputs:
  outputs/vacancy_grid.csv  — one row per cell_id with vacancy_rate
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VACANCY_URL = (
    "https://admin.opendatani.gov.uk/dataset/b0bd13ad-8224-4f06-8052-9fa7eef57c69"
    "/resource/5c113142-3a52-419b-a80d-b55660abd0d5"
    "/download/domestic-property-vacancy-rates-by-district-council.csv"
)


def main() -> None:
    grid_path = OUTPUT_DIR / "belfast_grid.geojson"
    if not grid_path.exists():
        print("[vacancy] belfast_grid.geojson not found — run build_grid.py first", file=sys.stderr)
        sys.exit(1)

    import geopandas as gpd
    grid = gpd.read_file(grid_path)

    print("[vacancy] downloading domestic vacancy rates...")
    r = requests.get(VACANCY_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    print(f"[vacancy] loaded {len(df)} rows")

    # Filter to Belfast, average across all available dates
    belfast = df[df["District Council"] == "Belfast"].copy()

    rate_col = "Domestic Vacancy Rate %"
    avg_rate = float(belfast[rate_col].mean()) / 100.0  # convert % to [0,1]

    print(f"[vacancy] Belfast average vacancy rate: {avg_rate*100:.2f}%")

    out = pd.DataFrame({
        "cell_id": grid["cell_id"],
        "vacancy_rate": avg_rate,
    })

    out.to_csv(OUTPUT_DIR / "vacancy_grid.csv", index=False)
    print(f"[vacancy] wrote {OUTPUT_DIR / 'vacancy_grid.csv'} ({len(out)} cells)")


if __name__ == "__main__":
    main()
