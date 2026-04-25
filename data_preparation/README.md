# Data preparation

End-to-end pipeline that turns Open Data NI / NISRA / Sentinel feeds into a
feature-engineered Belfast grid + a trained LightGBM model the API can serve.

## One-shot

```bash
cd data_preparation
pip install -r requirements.txt
python run_pipeline.py
```

That runs every step below, falling back to the synthetic baseline whenever a
real data source is missing credentials. When it finishes, you'll have:

- `backend/belfast_sentinel_model.txt`
- `backend/belfast_sentinel_model_metadata.json`
- `backend/belfast_grid_with_features.geojson`

…which is everything `backend/api.py` needs at startup.

## Steps

| script | what it does | needs |
|---|---|---|
| `build_grid.py` | 500 m grid in Irish Grid metres, reprojected to WGS84. Default bbox covers Belfast metro. | nothing |
| `fetch_opendatani.py` | DfI Strategic Flood Map (rivers / coastal / surface water / climate). Pulls from the public ArcGIS REST endpoints — no API key. | network access |
| `fetch_nisra_census.py` | NIMDM 2017 + Census 2021 tables (population density, tenure, dwellings). Drop a `soa_boundaries.geojson` next to the CSVs to enable the spatial join. | best-effort URLs; or drop CSVs in `outputs/` |
| `fetch_house_prices.py` | NI House Price Index by LGD. Drop `lgd_boundaries.geojson` to enable the join. | best-effort URL |
| `fetch_sentinel.py` | Sentinel-2 NDVI/NDBI/NDWI + Sentinel-5P NO₂ summaries per cell. | one of: `SH_CLIENT_ID`+`SH_CLIENT_SECRET` (Sentinel Hub) or `GEE_SERVICE_ACCOUNT_KEY` (Earth Engine) |
| `generate_synthetic.py` | Fabricates plausible feature values for every cell. Used as the baseline + fills any gaps. | nothing |
| `feature_engineering.py` | Joins everything onto the grid; real data overwrites synthetic where present. | the steps above |
| `train_model.py` | 5-fold stratified LightGBM, refit on full data, saves model + metadata into `backend/`. | nothing |

## Boundary files (optional)

To get full value from the NIMDM and HPI joins, drop these into `outputs/`:

- `soa_boundaries.geojson` — NISRA Super Output Areas (download from NISRA Open Data).
- `lgd_boundaries.geojson` — Local Government District 2014 boundaries.

The pipeline still works without them — it just falls back to synthetic
deprivation/HPI fields.

## Replacing synthetic with real Sentinel data

```bash
# Sentinel Hub
export SH_CLIENT_ID=...
export SH_CLIENT_SECRET=...
pip install sentinelhub
python fetch_sentinel.py

# OR Google Earth Engine
export GEE_SERVICE_ACCOUNT_KEY=/path/to/key.json
pip install earthengine-api
python fetch_sentinel.py

python feature_engineering.py
python train_model.py
```
