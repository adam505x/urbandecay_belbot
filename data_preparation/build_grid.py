"""Build a regular grid of cells covering Belfast (or all of Northern Ireland).

Output: data_preparation/outputs/belfast_grid.geojson  (EPSG:4326)
Each cell carries a stable integer cell_id and the LSOA / LGD / Ward it falls in
once the spatial joins have run (see feature_engineering.py).

The grid uses Irish Grid (EPSG:29903) internally so cell sizes are isotropic in
metres, then reprojects to WGS84 for serving.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, box

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Belfast bounding box (rough — covers the metro area + Lagan Valley + Castlereagh)
BELFAST_BBOX_WGS84 = (-6.10, 54.50, -5.75, 54.72)  # (minlon, minlat, maxlon, maxlat)

# Whole-NI bounding box
NI_BBOX_WGS84 = (-8.20, 54.00, -5.40, 55.30)


def make_grid(bbox_wgs84: tuple[float, float, float, float], cell_m: int = 500) -> gpd.GeoDataFrame:
    """Build a regular grid in Irish Grid metres, return as WGS84 GeoDataFrame."""

    bbox_gdf = gpd.GeoDataFrame(
        geometry=[box(*bbox_wgs84)], crs="EPSG:4326"
    ).to_crs("EPSG:29903")
    minx, miny, maxx, maxy = bbox_gdf.total_bounds

    xs = np.arange(np.floor(minx / cell_m) * cell_m, maxx, cell_m)
    ys = np.arange(np.floor(miny / cell_m) * cell_m, maxy, cell_m)

    cells: list[Polygon] = []
    for x in xs:
        for y in ys:
            cells.append(
                Polygon(
                    [
                        (x, y),
                        (x + cell_m, y),
                        (x + cell_m, y + cell_m),
                        (x, y + cell_m),
                    ]
                )
            )

    gdf = gpd.GeoDataFrame({"geometry": cells}, crs="EPSG:29903")
    gdf["cell_id"] = np.arange(len(gdf), dtype=np.int64)
    gdf = gdf.to_crs("EPSG:4326")
    return gdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region", choices=["belfast", "ni"], default="belfast",
        help="belfast = metro bbox; ni = whole of Northern Ireland",
    )
    parser.add_argument("--cell-m", type=int, default=500, help="Cell size in metres")
    parser.add_argument("--out", default=str(OUTPUT_DIR / "belfast_grid.geojson"))
    args = parser.parse_args()

    bbox = BELFAST_BBOX_WGS84 if args.region == "belfast" else NI_BBOX_WGS84
    grid = make_grid(bbox, cell_m=args.cell_m)
    print(f"Built {len(grid)} cells over {args.region} at {args.cell_m}m resolution")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_file(out, driver="GeoJSON")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
