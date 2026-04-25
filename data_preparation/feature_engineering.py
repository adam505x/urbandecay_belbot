"""Join real (or synthetic) NI feature tables onto the grid and write the final
feature-engineered GeoJSON the API serves.

Inputs (all optional; missing layers fall back to the synthetic field):
  outputs/belfast_grid.geojson         <- build_grid.py
  outputs/sentinel_features.csv        <- fetch_sentinel.py
  outputs/synthetic_features.csv       <- generate_synthetic.py
  outputs/flood_river.geojson          <- fetch_opendatani.py
  outputs/flood_coastal.geojson
  outputs/flood_surface.geojson
  outputs/flood_climate.geojson
  outputs/nimdm_2017_soa.csv           <- fetch_nisra_census.py
  outputs/ni_hpi_lgd.csv               <- fetch_house_prices.py

Output:
  outputs/belfast_grid_with_features.geojson
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

FLOOD_LAYERS = {
    "flood_river_pct": "flood_river.geojson",
    "flood_coastal_pct": "flood_coastal.geojson",
    "flood_surface_pct": "flood_surface.geojson",
    "flood_climate_pct": "flood_climate.geojson",
}


def overlay_pct(grid: gpd.GeoDataFrame, layer: gpd.GeoDataFrame) -> pd.Series:
    """For each cell return % area intersected by the (dissolved) layer."""
    if layer.empty:
        return pd.Series(0.0, index=grid.index)
    layer = layer.to_crs(grid.crs)
    layer_union = layer.unary_union
    cell_area = grid.geometry.area
    inter_area = grid.geometry.intersection(layer_union).area
    return (inter_area / cell_area).fillna(0.0)


def maybe_load_geojson(path: Path) -> gpd.GeoDataFrame:
    if path.exists():
        try:
            return gpd.read_file(path)
        except Exception as exc:
            print(f"WARN: could not read {path}: {exc}")
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default=str(OUTPUT_DIR / "belfast_grid.geojson"))
    parser.add_argument("--out", default=str(OUTPUT_DIR / "belfast_grid_with_features.geojson"))
    args = parser.parse_args()

    grid_path = Path(args.grid)
    grid = gpd.read_file(grid_path).to_crs("EPSG:4326")
    print(f"[features] grid: {len(grid)} cells")

    # 1. Synthetic baseline (always present, populates every column).
    synth_path = OUTPUT_DIR / "synthetic_features.csv"
    if not synth_path.exists():
        from generate_synthetic import synthesize  # type: ignore

        print("[features] no synthetic baseline found, generating now")
        synthesize(grid_path, synth_path)
    feats = pd.read_csv(synth_path)
    grid = grid.merge(feats, on="cell_id", how="left")

    # 2. Real Sentinel features overwrite synthetic where present.
    sentinel_path = OUTPUT_DIR / "sentinel_features.csv"
    if sentinel_path.exists():
        print("[features] merging real Sentinel features")
        sf = pd.read_csv(sentinel_path)
        for col in [c for c in sf.columns if c != "cell_id"]:
            grid = grid.merge(sf[["cell_id", col]], on="cell_id", how="left", suffixes=("", "_real"))
            real = grid[f"{col}_real"]
            grid[col] = real.where(real.notna(), grid[col])
            grid = grid.drop(columns=[f"{col}_real"])

    # 3. Real DfI flood envelopes overwrite synthetic where present.
    grid_metric = grid.to_crs("EPSG:29903")
    for col, fname in FLOOD_LAYERS.items():
        layer_path = OUTPUT_DIR / fname
        if layer_path.exists():
            print(f"[features] overlaying {fname}")
            layer = maybe_load_geojson(layer_path)
            if not layer.empty:
                grid[col] = overlay_pct(grid_metric, layer).clip(0, 1).values

    # 4. NIMDM deprivation scores by SOA, if present.
    nimdm_path = OUTPUT_DIR / "nimdm_2017_soa.csv"
    if nimdm_path.exists():
        try:
            print("[features] joining NIMDM 2017 by SOA centroid")
            nimdm = pd.read_csv(nimdm_path)
            # Best-effort spatial join: needs SOA boundaries to be useful.
            # If you have them, drop a soa_boundaries.geojson next to this file
            # and we'll do a proper centroid-in-polygon join.
            soa_path = OUTPUT_DIR / "soa_boundaries.geojson"
            if soa_path.exists() and "SOA Code" in nimdm.columns:
                soa = gpd.read_file(soa_path).to_crs("EPSG:4326")
                soa = soa.merge(nimdm, left_on="SOA2011", right_on="SOA Code", how="left")
                joined = gpd.sjoin(
                    grid.set_geometry("geometry"),
                    soa[["geometry", "MDM Decile", "Income Domain Score",
                         "Employment Domain Score", "Health Deprivation and Disability Domain Score",
                         "Crime and Disorder Domain Score", "Living Environment Domain Score"]],
                    how="left", predicate="intersects",
                )
                joined = joined.drop_duplicates("cell_id")
                rename = {
                    "MDM Decile": "deprivation_decile",
                    "Income Domain Score": "income_deprivation",
                    "Employment Domain Score": "employment_deprivation",
                    "Health Deprivation and Disability Domain Score": "health_deprivation",
                    "Crime and Disorder Domain Score": "crime_score",
                    "Living Environment Domain Score": "living_environment",
                }
                for src, dst in rename.items():
                    if src in joined.columns:
                        real = joined[src]
                        grid[dst] = real.where(real.notna(), grid[dst]).astype(float)
        except Exception as exc:
            print(f"WARN: NIMDM join failed: {exc}")

    # 5. NI HPI by LGD overwrites synthetic where present.
    hpi_path = OUTPUT_DIR / "ni_hpi_lgd.csv"
    lgd_path = OUTPUT_DIR / "lgd_boundaries.geojson"
    if hpi_path.exists() and lgd_path.exists():
        try:
            print("[features] joining NI HPI by LGD")
            hpi = pd.read_csv(hpi_path)
            lgd = gpd.read_file(lgd_path).to_crs("EPSG:4326")
            joined = gpd.sjoin(grid, lgd, how="left", predicate="intersects").drop_duplicates("cell_id")
            joined = joined.merge(hpi, left_on="LGDNAME", right_on="LGD", how="left")
            for src, dst in {
                "Standardised_Price": "house_price_index",
                "Annual_Change_Pct": "house_price_trend",
            }.items():
                if src in joined.columns:
                    grid[dst] = joined[src].where(joined[src].notna(), grid[dst]).astype(float)
        except Exception as exc:
            print(f"WARN: NI HPI join failed: {exc}")

    # Final: ensure no NaNs in numeric columns; downstream LightGBM tolerates
    # NaNs but we keep the GeoJSON tidy.
    num_cols = grid.select_dtypes(include="number").columns
    grid[num_cols] = grid[num_cols].fillna(grid[num_cols].median(numeric_only=True))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_file(out, driver="GeoJSON")
    print(f"[features] wrote {out} with {len(grid)} cells, {len(grid.columns)} columns")


if __name__ == "__main__":
    main()
