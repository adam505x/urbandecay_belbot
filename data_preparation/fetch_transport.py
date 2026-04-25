"""Download Translink bus stop locations and compute transport access per grid cell.

Outputs:
  outputs/transport_grid.csv  — one row per cell_id with transport features
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUS_STOPS_URL = (
    "https://admin.opendatani.gov.uk/dataset/495c6964-e8d2-4bf1-9942-8d950b3a0ceb"
    "/resource/2952c083-69da-4524-9621-5aac64a50626/download/export10042024.geojson"
)

# 500m buffer in metres (Irish Grid EPSG:29903)
BUFFER_M = 500


def main() -> None:
    grid_path = OUTPUT_DIR / "belfast_grid.geojson"
    if not grid_path.exists():
        print("[transport] belfast_grid.geojson not found — run build_grid.py first", file=sys.stderr)
        sys.exit(1)

    grid = gpd.read_file(grid_path).to_crs("EPSG:29903")

    print("[transport] downloading Translink bus stops...")
    r = requests.get(BUS_STOPS_URL, timeout=120)
    r.raise_for_status()
    stops = gpd.read_file(io.BytesIO(r.content))
    print(f"[transport] loaded {len(stops)} stops")

    # Filter to active stops only
    if "Status" in stops.columns:
        stops = stops[stops["Status"].str.lower() == "active"]
        print(f"[transport] {len(stops)} active stops after filter")

    stops = stops.to_crs("EPSG:29903")

    # Clip to greater Belfast area (bounding box with margin)
    bbox = grid.total_bounds  # minx, miny, maxx, maxy
    margin = 2000  # 2km margin
    stops = stops.cx[bbox[0]-margin:bbox[2]+margin, bbox[1]-margin:bbox[3]+margin]
    print(f"[transport] {len(stops)} stops within Belfast area")

    stop_coords = np.array([[g.x, g.y] for g in stops.geometry])

    rows = []
    grid_centroids = grid.geometry.centroid

    for idx, (cell_id, centroid) in enumerate(zip(grid["cell_id"], grid_centroids)):
        cx, cy = centroid.x, centroid.y

        if len(stop_coords) == 0:
            n_stops = 0
            dist_nearest = 9999.0
        else:
            dists = np.sqrt((stop_coords[:, 0] - cx)**2 + (stop_coords[:, 1] - cy)**2)
            n_stops = int((dists <= BUFFER_M).sum())
            dist_nearest = float(dists.min())

        rows.append({
            "cell_id": cell_id,
            "n_stops_500m": n_stops,
            "dist_to_nearest_stop_m": dist_nearest,
        })

    out = pd.DataFrame(rows)

    # Normalised transport access score [0,1]:
    # high stops + low distance = high access = high score
    max_stops = float(out["n_stops_500m"].quantile(0.99)) or 1.0
    max_dist = float(out["dist_to_nearest_stop_m"].quantile(0.99)) or 1.0

    stops_norm = (out["n_stops_500m"] / max_stops).clip(0, 1)
    dist_norm = 1.0 - (out["dist_to_nearest_stop_m"] / max_dist).clip(0, 1)
    out["transport_access_score"] = (0.5 * stops_norm + 0.5 * dist_norm)

    print(f"[transport] stop counts: min={out['n_stops_500m'].min()} "
          f"mean={out['n_stops_500m'].mean():.1f} max={out['n_stops_500m'].max()}")
    print(f"[transport] transport_access_score: "
          f"mean={out['transport_access_score'].mean():.3f}")

    out.to_csv(OUTPUT_DIR / "transport_grid.csv", index=False)
    print(f"[transport] wrote {OUTPUT_DIR / 'transport_grid.csv'} ({len(out)} cells)")


if __name__ == "__main__":
    main()
