"""Fetch Sentinel-2 (NDVI/NDBI/NDWI) and Sentinel-5P (NO₂) summaries per grid cell.

Two backends are supported, picked via SENTINEL_BACKEND env var:
  - "sh"  : Sentinel Hub Process API (https://sentinel-hub.com)
            requires SH_CLIENT_ID + SH_CLIENT_SECRET
  - "gee" : Google Earth Engine                     requires GEE service-account JSON

Output: outputs/sentinel_features.csv  (one row per cell_id)

If neither backend is configured, this script exits cleanly and
generate_synthetic.py will fabricate plausible values for the demo.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _have_sh() -> bool:
    return bool(os.getenv("SH_CLIENT_ID") and os.getenv("SH_CLIENT_SECRET"))


def _have_gee() -> bool:
    return bool(os.getenv("GEE_SERVICE_ACCOUNT_KEY"))


def fetch_with_sentinel_hub(grid_path: Path, out_path: Path, year: int) -> None:
    import warnings
    try:
        from sentinelhub import (  # type: ignore
            CRS, BBox, DataCollection, MimeType, SHConfig, SentinelHubStatistical,
        )
        from sentinelhub.exceptions import SHRateLimitWarning  # type: ignore
        warnings.filterwarnings("ignore", category=SHRateLimitWarning)
    except ImportError:
        print("sentinelhub-py not installed; pip install sentinelhub", file=sys.stderr)
        sys.exit(2)

    import geopandas as gpd
    import pandas as pd

    cfg = SHConfig()
    cfg.sh_client_id = os.environ["SH_CLIENT_ID"]
    cfg.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
    cfg.sh_base_url = os.environ.get(
        "SH_BASE_URL", "https://sh.dataspace.copernicus.eu"
    )
    cfg.sh_token_url = os.environ.get(
        "SH_TOKEN_URL",
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
    )

    # DataCollection.SENTINEL2_L2A has service_url hardcoded to the legacy
    # services.sentinel-hub.com endpoint. Define CDSE-targeted collections so
    # requests go to the correct CDSE host.
    cdse_url = cfg.sh_base_url
    CDSE_S2 = DataCollection.define_from(
        DataCollection.SENTINEL2_L2A, "CDSE_S2L2A", service_url=cdse_url
    )
    CDSE_S5P = DataCollection.define_from(
        DataCollection.SENTINEL5P, "CDSE_S5P", service_url=cdse_url
    )

    gdf = gpd.read_file(grid_path)

    evalscript_s2 = """
    //VERSION=3
    function setup() {
      return {
        input: [{ bands: ["B03","B04","B08","B11","dataMask"] }],
        output: [
          { id: "default", bands: 3, sampleType: "FLOAT32" },
          { id: "dataMask", bands: 1 }
        ]
      };
    }
    function evaluatePixel(s) {
      let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
      let ndbi = (s.B11 - s.B08) / (s.B11 + s.B08 + 1e-6);
      let ndwi = (s.B03 - s.B08) / (s.B03 + s.B08 + 1e-6);
      return { default: [ndvi, ndbi, ndwi], dataMask: [s.dataMask] };
    }
    """

    evalscript_s5p = """
    //VERSION=3
    function setup() {
      return {
        input: [{ bands: ["NO2","dataMask"] }],
        output: [
          { id: "default", bands: 1, sampleType: "FLOAT32" },
          { id: "dataMask", bands: 1 }
        ]
      };
    }
    function evaluatePixel(s) {
      return { default: [s.NO2], dataMask: [s.dataMask] };
    }
    """

    def _mean_of(stats, band_idx):
        vals = []
        for interval in stats["data"]:
            bands = interval["outputs"]["default"]["bands"]
            m = bands[f"B{band_idx}"]["stats"].get("mean")
            try:
                m = float(m)
            except (TypeError, ValueError):
                continue
            if m == m:  # filter NaN
                vals.append(m)
        return sum(vals) / len(vals) if vals else None

    def _fetch_cell(row):
        bb = row.geometry.bounds
        bbox = BBox(bb, crs=CRS.WGS84)
        s2 = SentinelHubStatistical(
            aggregation=SentinelHubStatistical.aggregation(
                evalscript=evalscript_s2,
                time_interval=(f"{year}-04-01", f"{year}-09-30"),
                aggregation_interval="P1D",
                resolution=(20, 20),
            ),
            input_data=[SentinelHubStatistical.input_data(CDSE_S2)],
            bbox=bbox,
            config=cfg,
        )
        s5p = SentinelHubStatistical(
            aggregation=SentinelHubStatistical.aggregation(
                evalscript=evalscript_s5p,
                time_interval=(f"{year}-01-01", f"{year}-12-31"),
                aggregation_interval="P30D",
                resolution=(7000, 3500),
            ),
            input_data=[SentinelHubStatistical.input_data(CDSE_S5P)],
            bbox=bbox,
            config=cfg,
        )
        s2_data = s2.get_data()[0]
        s5p_data = s5p.get_data()[0]
        return {
            "cell_id": int(row.cell_id),
            "ndvi_mean": _mean_of(s2_data, 0),
            "ndbi_mean": _mean_of(s2_data, 1),
            "ndwi_mean": _mean_of(s2_data, 2),
            "no2_mean": _mean_of(s5p_data, 0),
        }

    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 10 workers saturates the CDSE rate limit without wasting it
    n_workers = 10

    _first_error_logged = [False]

    def _fetch_with_retry(row, max_retries: int = 4):
        for attempt in range(max_retries):
            try:
                return _fetch_cell(row)
            except Exception as exc:
                msg = str(exc).lower()
                if "rate limit" in msg or "429" in msg or "too many" in msg:
                    wait = 2 ** attempt  # 1, 2, 4, 8 s
                    time.sleep(wait)
                    continue
                if not _first_error_logged[0]:
                    print(f"[sh] cell error (cell_id={row.cell_id}): {type(exc).__name__}: {exc}", flush=True)
                    _first_error_logged[0] = True
                return {"cell_id": int(row.cell_id)}
        return {"cell_id": int(row.cell_id)}

    # Resume: skip cells already in the output file
    already_done: set = set()
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path)
            if len(existing.columns) > 1:  # has real data, not just cell_id
                already_done = set(existing["cell_id"].tolist())
                print(f"[sh] resuming: {len(already_done)} cells already done", flush=True)
        except Exception:
            pass

    cell_rows = [r for r in gdf.itertuples() if int(r.cell_id) not in already_done]
    total = len(gdf)
    done = len(already_done)

    print(f"[sh] processing {len(cell_rows)} cells with {n_workers} parallel workers", flush=True)

    batch: list = []
    write_header = not out_path.exists() or len(already_done) == 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_fetch_with_retry, row): row for row in cell_rows}
        for fut in as_completed(futures):
            result = fut.result()
            if len(result) > 1:  # only save rows that have actual feature data
                batch.append(result)
            done += 1
            if done % 100 == 0:
                print(f"[sh] {done}/{total} cells done", flush=True)
            if len(batch) >= 100:
                df_batch = pd.DataFrame(batch)
                df_batch.to_csv(out_path, mode="a", header=write_header, index=False)
                write_header = False
                batch = []

    # flush remaining
    if batch:
        pd.DataFrame(batch).to_csv(out_path, mode="a", header=write_header, index=False)

    print(f"[sh] wrote {out_path}", flush=True)


def fetch_with_gee(grid_path: Path, out_path: Path, year: int) -> None:
    try:
        import ee  # type: ignore
        import geopandas as gpd
        import pandas as pd
    except ImportError:
        print("earthengine-api not installed; pip install earthengine-api", file=sys.stderr)
        sys.exit(2)

    key_path = os.environ["GEE_SERVICE_ACCOUNT_KEY"]
    creds = ee.ServiceAccountCredentials(None, key_path)
    ee.Initialize(creds)

    gdf = gpd.read_file(grid_path)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(f"{year}-04-01", f"{year}-09-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
        .median()
    )
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("ndvi_mean")
    ndbi = s2.normalizedDifference(["B11", "B8"]).rename("ndbi_mean")
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("ndwi_mean")

    s5p = (
        ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .select("tropospheric_NO2_column_number_density")
        .mean()
        .rename("no2_mean")
    )

    img = ndvi.addBands([ndbi, ndwi, s5p])

    rows = []
    print(f"[gee] reducing {len(gdf)} cells")
    for _, row in gdf.iterrows():
        geom = ee.Geometry.Polygon(list(row.geometry.exterior.coords))
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=20, maxPixels=1e9
        ).getInfo()
        rows.append({"cell_id": int(row["cell_id"]), **{k: stats.get(k) for k in img.bandNames().getInfo()}})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[gee] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default=str(OUTPUT_DIR / "belfast_grid.geojson"))
    parser.add_argument("--out", default=str(OUTPUT_DIR / "sentinel_features.csv"))
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    backend = os.getenv("SENTINEL_BACKEND", "auto").lower()
    if backend in ("auto", "sh") and _have_sh():
        fetch_with_sentinel_hub(Path(args.grid), Path(args.out), args.year)
    elif backend in ("auto", "gee") and _have_gee():
        fetch_with_gee(Path(args.grid), Path(args.out), args.year)
    else:
        print(
            "[sentinel] no backend configured.\n"
            "  Set SH_CLIENT_ID + SH_CLIENT_SECRET for Sentinel Hub, or\n"
            "  GEE_SERVICE_ACCOUNT_KEY for Google Earth Engine.\n"
            "  generate_synthetic.py will fabricate values for the demo."
        )


if __name__ == "__main__":
    main()
