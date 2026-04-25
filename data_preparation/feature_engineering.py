"""Merge all real data sources onto the grid and compute the decay_index target.

Inputs (from outputs/):
  belfast_grid.geojson       — base grid (required)
  nimdm_grid.csv             — NIMDM 2017 deprivation features (required)
  hpi_grid.csv               — House price index + trend (required)
  vacancy_grid.csv           — Domestic vacancy rate (required)
  crime_grid.csv             — PSNI crime rate (optional)
  transport_grid.csv         — Bus stop access score (optional)
  sentinel_features.csv      — NDVI/NDBI/NDWI (optional)

Output:
  outputs/belfast_grid_with_features.geojson

Decay index formula (all inputs normalised to [0,1]):
  decay_index = (
      0.40 * deprivation_score
    + 0.20 * crime_domain           (NIMDM spatial crime)
    + 0.20 * vacancy_rate_norm      (LPS vacancy rate)
    - 0.10 * house_price_growth_norm
    - 0.10 * transport_access_score
  ).clip(0, 1)
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

BELFAST_CENTRE = (-5.9301, 54.5973)  # lon, lat


def _norm(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _load_csv(name: str, required: bool = True) -> pd.DataFrame | None:
    path = OUTPUT_DIR / name
    if path.exists():
        df = pd.read_csv(path)
        print(f"[features] loaded {name} ({len(df)} rows)")
        return df
    if required:
        print(f"[features] ERROR: required file {name} not found", file=sys.stderr)
        sys.exit(1)
    print(f"[features] WARN: optional {name} not found, skipping")
    return None


def main() -> None:
    # --- Base grid ---
    grid_path = OUTPUT_DIR / "belfast_grid.geojson"
    if not grid_path.exists():
        print("[features] belfast_grid.geojson not found", file=sys.stderr)
        sys.exit(1)

    grid = gpd.read_file(grid_path)
    print(f"[features] grid: {len(grid)} cells")

    # --- Required data ---
    nimdm = _load_csv("nimdm_grid.csv", required=True)
    hpi   = _load_csv("hpi_grid.csv",   required=True)
    vac   = _load_csv("vacancy_grid.csv", required=True)

    # --- Optional data ---
    crime     = _load_csv("crime_grid.csv",     required=False)
    transport = _load_csv("transport_grid.csv",  required=False)
    sentinel  = _load_csv("sentinel_features.csv", required=False)

    # --- Merge everything ---
    df = grid[["cell_id"]].copy()
    for src in [nimdm, hpi, vac, crime, transport]:
        if src is not None:
            df = df.merge(src, on="cell_id", how="left")

    if sentinel is not None:
        sentinel_cols = [c for c in sentinel.columns if c != "cell_id"]
        df = df.merge(sentinel[["cell_id"] + sentinel_cols], on="cell_id", how="left")

    # --- Distance to city centre ---
    grid_wgs = grid.to_crs("EPSG:4326")
    centroids = grid_wgs.geometry.centroid
    lon, lat = BELFAST_CENTRE
    # Approximate km distance using equirectangular
    dlat = (centroids.y - lat) * 111.0
    dlon = (centroids.x - lon) * 111.0 * np.cos(np.radians(lat))
    df["dist_to_centre_km"] = np.sqrt(dlat**2 + dlon**2).values

    # --- Impute missing optional columns with median ---
    num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in num_cols if c != "cell_id"]
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # --- Compute decay_index ---
    # Normalise components
    deprivation_norm    = _norm(df["deprivation_score"])
    crime_norm          = _norm(df["crime_domain"])
    vacancy_norm        = _norm(df["vacancy_rate"])

    # House price growth: higher growth → lower decay (invert)
    # Use 5yr trend; if missing, use 0
    if "house_price_trend_5yr" in df.columns:
        hpg_norm = _norm(df["house_price_trend_5yr"])
    else:
        hpg_norm = pd.Series(0.5, index=df.index)

    # Transport access: higher = lower decay (invert)
    if "transport_access_score" in df.columns:
        transport_norm = _norm(df["transport_access_score"])
    else:
        transport_norm = pd.Series(0.5, index=df.index)

    df["decay_index"] = (
        0.40 * deprivation_norm
        + 0.20 * crime_norm
        + 0.20 * vacancy_norm
        - 0.10 * hpg_norm
        - 0.10 * transport_norm
    ).clip(0, 1)

    print(f"[features] decay_index: min={df['decay_index'].min():.3f}  "
          f"mean={df['decay_index'].mean():.3f}  max={df['decay_index'].max():.3f}")

    # Decile (1=least decay, 10=most decay)
    df["decay_decile"] = pd.qcut(df["decay_index"], 10, labels=False) + 1

    # --- Merge back geometry ---
    out_gdf = grid[["cell_id", "geometry"]].merge(df.drop(columns=[], errors="ignore"), on="cell_id")

    out_path = OUTPUT_DIR / "belfast_grid_with_features.geojson"
    out_gdf.to_file(out_path, driver="GeoJSON")
    print(f"[features] wrote {out_path} ({len(out_gdf)} cells, {len(out_gdf.columns)} columns)")


if __name__ == "__main__":
    main()
