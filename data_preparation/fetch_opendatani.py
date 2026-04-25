"""Fetch OpenDataNI / DfI flood-map layers and clip to the grid bbox.

Uses the DfI Rivers ArcGIS REST endpoints (no auth) to download flood envelopes:
- Flood Map (Rivers)
- Flood Map (Sea / Coastal)
- Flood Map (Surface Water)
- Climate-change projected flood envelope

Outputs: outputs/flood_<layer>.geojson (EPSG:4326)

If a layer is gated behind admin.opendatani.gov.uk and requires an API key,
set the OPENDATANI_API_KEY env var or download the dataset manually into
outputs/ with the same filename.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlencode

import requests

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DfI Strategic Flood Map services (FeatureServer endpoints).
# These slugs may shift over time — check https://admin.opendatani.gov.uk/dataset
# for the latest. The query parameters below are standard ArcGIS REST.
DFI_FLOOD_SERVICES = {
    "river": "https://services1.arcgis.com/4y0u8yRJWxxFQ4iY/arcgis/rest/services/Strategic_Flood_Map_NI_Rivers/FeatureServer/0",
    "coastal": "https://services1.arcgis.com/4y0u8yRJWxxFQ4iY/arcgis/rest/services/Strategic_Flood_Map_NI_Sea/FeatureServer/0",
    "surface": "https://services1.arcgis.com/4y0u8yRJWxxFQ4iY/arcgis/rest/services/Strategic_Flood_Map_NI_Surface_Water/FeatureServer/0",
    "climate": "https://services1.arcgis.com/4y0u8yRJWxxFQ4iY/arcgis/rest/services/Strategic_Flood_Map_NI_Climate_Change/FeatureServer/0",
}


def fetch_arcgis_layer(base_url: str, bbox: tuple[float, float, float, float]) -> dict:
    """Pull all features within bbox as GeoJSON (paginated)."""
    minx, miny, maxx, maxy = bbox
    geometry = json.dumps(
        {"xmin": minx, "ymin": miny, "xmax": maxx, "ymax": maxy, "spatialReference": {"wkid": 4326}}
    )
    params = {
        "where": "1=1",
        "geometry": geometry,
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "f": "geojson",
        "resultRecordCount": 2000,
    }

    features: list[dict] = []
    offset = 0
    while True:
        q = {**params, "resultOffset": offset}
        url = f"{base_url}/query?{urlencode(q)}"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        page = resp.json()
        page_features = page.get("features", [])
        features.extend(page_features)
        if len(page_features) < params["resultRecordCount"] or not page.get("exceededTransferLimit"):
            break
        offset += params["resultRecordCount"]

    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bbox", nargs=4, type=float, default=[-6.10, 54.50, -5.75, 54.72],
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
    )
    parser.add_argument(
        "--layers", nargs="+", default=list(DFI_FLOOD_SERVICES.keys()),
        choices=list(DFI_FLOOD_SERVICES.keys()),
    )
    args = parser.parse_args()

    for layer in args.layers:
        url = DFI_FLOOD_SERVICES[layer]
        out = OUTPUT_DIR / f"flood_{layer}.geojson"
        try:
            print(f"[opendatani] fetching {layer} -> {out}")
            fc = fetch_arcgis_layer(url, tuple(args.bbox))
            with open(out, "w", encoding="utf-8") as f:
                json.dump(fc, f)
            print(f"  wrote {len(fc.get('features', []))} features")
        except Exception as exc:
            print(f"  WARN: failed to fetch {layer}: {exc}")
            print("  -> the synthetic generator will fill this in if no real data is present.")


if __name__ == "__main__":
    main()
