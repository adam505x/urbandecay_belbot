"""Download NIMDM 2017 (Small Area level) and join deprivation features to the grid.

Outputs:
  outputs/nimdm_grid.csv  — one row per cell_id with deprivation columns
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NIMDM_CSV_URL = (
    "https://admin.opendatani.gov.uk/dataset/e202fde9-7f0b-4d88-8711-e18a8817cff8"
    "/resource/b29fa439-a314-4573-889e-648d3a118691/download/nimdm2017-soa.csv"
)
# SOA 2001 GeoJSON from OpenDataNI (verified working)
SOA2001_GEOJSON_URL = (
    "https://admin.opendatani.gov.uk/dataset/678697e1-ae71-41f3-abba-0ef5f3f352c2"
    "/resource/80392e82-8bee-42de-a1e3-82d1cbaa983f/download/soa2001.json"
)

BELFAST_LGD = "N09000003"

# Max NIMDM rank across NI (4,537 Small Areas)
NIMDM_MAX_RANK = 4537


def _download(url: str, label: str) -> bytes:
    print(f"[nimdm] downloading {label}...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def _extract_shapefile(zip_bytes: bytes, tmp_dir: Path) -> gpd.GeoDataFrame:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(tmp_dir)
    shp_files = list(tmp_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("No .shp file found in zip")
    return gpd.read_file(shp_files[0])


def _norm_rank(series: pd.Series, max_rank: int) -> pd.Series:
    """Convert rank (1=most deprived) to score (1=most deprived) in [0,1]."""
    return 1.0 - (series - 1) / (max_rank - 1)


def main() -> None:
    grid_path = OUTPUT_DIR / "belfast_grid.geojson"
    if not grid_path.exists():
        print("[nimdm] belfast_grid.geojson not found — run build_grid.py first", file=sys.stderr)
        sys.exit(1)

    grid = gpd.read_file(grid_path)

    # --- Load NIMDM CSV ---
    nimdm_raw = _download(NIMDM_CSV_URL, "NIMDM 2017 CSV")
    nimdm = pd.read_csv(io.BytesIO(nimdm_raw))
    print(f"[nimdm] loaded {len(nimdm)} Small Areas from NIMDM")

    # Filter to Belfast LGD
    belfast_nimdm = nimdm[nimdm["LGD2014code"] == BELFAST_LGD].copy()
    print(f"[nimdm] {len(belfast_nimdm)} Small Areas in Belfast LGD")

    # Normalise ranks to [0,1] scores
    belfast_nimdm["deprivation_score"] = _norm_rank(belfast_nimdm["MDM_rank"], NIMDM_MAX_RANK)
    belfast_nimdm["income_deprivation"] = belfast_nimdm["Income_perc"] / 100.0
    belfast_nimdm["employment_deprivation"] = belfast_nimdm["Empl_perc"] / 100.0
    belfast_nimdm["health_deprivation"] = _norm_rank(belfast_nimdm["D3_Health_rank"], NIMDM_MAX_RANK)
    belfast_nimdm["living_environment"] = _norm_rank(belfast_nimdm["D6_LivEnv_rank"], NIMDM_MAX_RANK)
    belfast_nimdm["crime_domain"] = _norm_rank(belfast_nimdm["D7_CD_rank"], NIMDM_MAX_RANK)
    belfast_nimdm["access_to_services"] = _norm_rank(belfast_nimdm["P5_Access_rank"], NIMDM_MAX_RANK)
    # Keep SOA2001name for aggregation to SOA level
    belfast_nimdm["SOA2001name"] = belfast_nimdm["SOA2001name"]

    # Aggregate NIMDM from SA2011 to SOA2001 level (for boundary join)
    soa_cols = ["SOA2001name", "deprivation_score", "income_deprivation", "employment_deprivation",
                "health_deprivation", "living_environment", "crime_domain", "access_to_services"]
    belfast_nimdm = belfast_nimdm[soa_cols]
    soa_nimdm = belfast_nimdm.groupby("SOA2001name").mean(numeric_only=True).reset_index()
    print(f"[nimdm] aggregated to {len(soa_nimdm)} SOA2001 areas")

    # --- Download SOA2001 boundaries (GeoJSON from OpenDataNI) ---
    bounds_gdf = None
    try:
        raw = _download(SOA2001_GEOJSON_URL, "SOA2001 boundaries (GeoJSON)")
        bounds_gdf = gpd.read_file(io.BytesIO(raw))
        bounds_gdf = bounds_gdf.to_crs("EPSG:4326")
        # Find the SOA name column
        name_col = None
        for candidate in ["SOA_LABEL", "SOA2001", "SOA_NAME", "Name", "NAME", "label"]:
            if candidate in bounds_gdf.columns:
                name_col = candidate
                break
        if name_col is None:
            print(f"[nimdm] SOA GeoJSON columns: {list(bounds_gdf.columns)}")
            name_col = bounds_gdf.columns[0]
        print(f"[nimdm] SOA2001 boundaries: {len(bounds_gdf)} areas, join key: '{name_col}'")
    except Exception as exc:
        print(f"[nimdm] WARN: SOA2001 GeoJSON failed: {exc}")

    if bounds_gdf is None or name_col is None:
        # Fallback: Belfast-wide median to all cells
        print("[nimdm] WARN: no boundary file — applying Belfast-median values to all cells")
        feat_cols = [c for c in soa_nimdm.columns if c != "SOA2001name"]
        row = soa_nimdm[feat_cols].median()
        out = pd.DataFrame({"cell_id": grid["cell_id"]})
        for col in feat_cols:
            out[col] = row[col]
        out.to_csv(OUTPUT_DIR / "nimdm_grid.csv", index=False)
        print(f"[nimdm] wrote {OUTPUT_DIR / 'nimdm_grid.csv'} (fallback: Belfast median)")
        return

    # Spatial join: grid centroid → SOA2001 polygon
    grid_wgs = grid.to_crs("EPSG:4326")
    grid_centroids = grid_wgs.copy()
    grid_centroids["geometry"] = grid_wgs.geometry.centroid

    joined = gpd.sjoin(
        grid_centroids[["cell_id", "geometry"]],
        bounds_gdf[[name_col, "geometry"]],
        how="left",
        predicate="within",
    ).drop_duplicates("cell_id")

    joined = joined[["cell_id", name_col]].rename(columns={name_col: "SOA2001name"})

    # Merge NIMDM features
    out = joined.merge(soa_nimdm, on="SOA2001name", how="left")
    out = out.drop(columns=["SOA2001name"])

    # Impute unmatched cells with Belfast median
    feat_cols = [c for c in out.columns if c != "cell_id"]
    medians = soa_nimdm[feat_cols].median()
    for col in feat_cols:
        out[col] = out[col].fillna(medians[col])

    out.to_csv(OUTPUT_DIR / "nimdm_grid.csv", index=False)
    matched = out[feat_cols[0]].notna().sum()
    print(f"[nimdm] wrote {OUTPUT_DIR / 'nimdm_grid.csv'} ({len(out)} cells, {matched} matched spatially)")


if __name__ == "__main__":
    main()
