"""Download NI House Price Index by LGD (2005–2025) and compute Belfast trend.

Outputs:
  outputs/hpi_grid.csv  — one row per cell_id (all Belfast cells get same LGD values)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HPI_LGD_URL = (
    "https://admin.opendatani.gov.uk/dataset/0b74f343-051e-4496-a483-301b69d4fae7"
    "/resource/dc7af407-bcb5-4820-81c0-a5e0dd7cbcb9"
    "/download/standardised-price-and-index-by-lgd-q1-2005-q2-2025.csv"
)


def _parse_quarter(q: str) -> float:
    """Convert 'Q1 2005' -> 2005.0, 'Q3 2005' -> 2005.5, etc."""
    parts = q.strip().split()
    year = int(parts[1])
    qnum = int(parts[0][1])
    return year + (qnum - 1) * 0.25


def _trend_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y on x."""
    if len(x) < 2:
        return 0.0
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def main() -> None:
    grid_path = OUTPUT_DIR / "belfast_grid.geojson"
    if not grid_path.exists():
        print("[hpi] belfast_grid.geojson not found — run build_grid.py first", file=sys.stderr)
        sys.exit(1)

    import geopandas as gpd
    grid = gpd.read_file(grid_path)

    print("[hpi] downloading NI House Price Index by LGD...")
    r = requests.get(HPI_LGD_URL, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    print(f"[hpi] loaded {len(df)} quarters, columns: {list(df.columns[:6])}...")

    # Parse time axis
    df["t"] = df["Quarter_Year"].apply(_parse_quarter)
    df = df.sort_values("t")

    # Belfast HPI column
    hpi_col = "Belfast_HPI"
    price_col = "Belfast_Standardised_Price"

    t = df["t"].values
    hpi = df[hpi_col].astype(float).values
    price = df[price_col].astype(float).values

    # Current values (most recent quarter)
    current_hpi = float(hpi[-1])
    current_price = float(price[-1])

    # 5-year trend: slope of HPI over last 20 quarters
    t_5yr = t[-20:]
    hpi_5yr = hpi[-20:]
    slope_5yr = _trend_slope(t_5yr, hpi_5yr)   # HPI points per year

    # 10-year trend: slope over last 40 quarters
    t_10yr = t[-40:] if len(t) >= 40 else t
    hpi_10yr = hpi[-40:] if len(hpi) >= 40 else hpi
    slope_10yr = _trend_slope(t_10yr, hpi_10yr)

    # Normalised price change per year (for decay_index)
    # Positive slope = house prices rising = lower decay risk
    price_slope_5yr = _trend_slope(t_5yr, price[-20:])   # £ per year

    print(f"[hpi] Belfast current HPI={current_hpi:.1f}  price=£{current_price:,.0f}")
    print(f"[hpi] 5yr trend: {slope_5yr:+.2f} HPI pts/yr  (price {price_slope_5yr:+,.0f} £/yr)")
    print(f"[hpi] 10yr trend: {slope_10yr:+.2f} HPI pts/yr")

    # Assign same values to all Belfast grid cells
    out = pd.DataFrame({
        "cell_id": grid["cell_id"],
        "house_price_index": current_hpi,
        "house_price_standardised": current_price,
        "house_price_trend_5yr": slope_5yr,
        "house_price_trend_10yr": slope_10yr,
        "house_price_growth_pct_5yr": (hpi[-1] / hpi[-20] - 1) * 100 if hpi[-20] > 0 else 0.0,
    })

    out.to_csv(OUTPUT_DIR / "hpi_grid.csv", index=False)
    print(f"[hpi] wrote {OUTPUT_DIR / 'hpi_grid.csv'} ({len(out)} cells)")


if __name__ == "__main__":
    main()
