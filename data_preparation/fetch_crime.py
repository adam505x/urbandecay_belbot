"""Download PSNI recorded crime data and compute Belfast crime rate.

Outputs:
  outputs/crime_grid.csv  — one row per cell_id with crime_rate_per_1k
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CRIME_URL = (
    "https://admin.opendatani.gov.uk/dataset/80dc9542-7b2a-48f5-bbf4-ccc7040d36af"
    "/resource/6fd51851-df78-4469-98c5-4f06953621a0"
    "/download/police-recorded-crime-monthly-data.csv"
)

# Belfast 2021 Census population (LGD level)
BELFAST_POPULATION = 345_418

# Recent years to average over
RECENT_YEARS = [2020, 2021, 2022, 2023, 2024]


def main() -> None:
    grid_path = OUTPUT_DIR / "belfast_grid.geojson"
    if not grid_path.exists():
        print("[crime] belfast_grid.geojson not found — run build_grid.py first", file=sys.stderr)
        sys.exit(1)

    import geopandas as gpd
    grid = gpd.read_file(grid_path)

    print("[crime] downloading PSNI recorded crime data...")
    r = requests.get(CRIME_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    print(f"[crime] loaded {len(df)} rows")

    # Filter to Belfast City, total crime, recent years
    mask = (
        (df["Policing_District"] == "Belfast City")
        & (df["Crime_Type"] == "Total police recorded crime")
        & (df["Calendar_Year"].isin(RECENT_YEARS))
    )
    belfast = df[mask].copy()
    belfast["Count"] = pd.to_numeric(belfast["Count"], errors="coerce")

    if belfast.empty:
        print("[crime] WARN: no Belfast data found, using NI total scaled to Belfast share")
        mask_ni = (
            (df["Policing_District"] == "Northern Ireland")
            & (df["Crime_Type"] == "Total police recorded crime")
            & (df["Calendar_Year"].isin(RECENT_YEARS))
        )
        belfast = df[mask_ni].copy()
        belfast["Count"] = pd.to_numeric(belfast["Count"], errors="coerce")
        # Belfast is ~28% of NI population
        belfast["Count"] = (belfast["Count"] * 0.28).round()

    # Annual total: sum months per year, then average across years
    annual = belfast.groupby("Calendar_Year")["Count"].sum()
    avg_annual_crimes = float(annual.mean())
    crime_rate_per_1k = avg_annual_crimes / BELFAST_POPULATION * 1000

    print(f"[crime] avg annual crimes (Belfast): {avg_annual_crimes:,.0f}")
    print(f"[crime] crime rate: {crime_rate_per_1k:.1f} per 1,000 population")

    # All cells get the LGD-level rate; spatial variation comes from NIMDM crime_domain
    out = pd.DataFrame({
        "cell_id": grid["cell_id"],
        "crime_rate_per_1k": crime_rate_per_1k,
    })

    out.to_csv(OUTPUT_DIR / "crime_grid.csv", index=False)
    print(f"[crime] wrote {OUTPUT_DIR / 'crime_grid.csv'} ({len(out)} cells)")


if __name__ == "__main__":
    main()
