# Belfast Sentinel — Urban Decay Predictor for Northern Ireland

A Belfast / Northern Ireland fork of [ManagementMO/Urban-Sentinel](https://github.com/ManagementMO/Urban-Sentinel).
Same architecture (FastAPI + LightGBM + React + Mapbox), but the Toronto 311
service-request feature set has been replaced with Northern Ireland data:

- **Sentinel-2** — NDVI (vegetation), NDBI (built-up), NDWI (water), surface
  temperature trend.
- **Sentinel-5P** — tropospheric NO₂ (a strong proxy for traffic / combustion).
- **DfI Strategic Flood Maps** — river, coastal, surface-water and
  climate-change flood envelopes (OpenDataNI ArcGIS REST).
- **NIMDM 2017** — Multiple Deprivation Measure (decile + income / employment /
  health / crime / living-environment domain scores).
- **NISRA Census 2021** — population density, tenure, dwelling status,
  central-heating coverage.
- **NI House Price Index** — standardised price + 5-year trend by LGD.
- **Optional**: TomTom traffic congestion, NI Footfall index, NI 2021 LiDAR DTM,
  proximity to power transmission / substations.

## Quick start (one command)

```bash
# 1. Build the model + grid (synthetic data is used for any feed missing creds)
cd data_preparation
pip install -r requirements.txt
python run_pipeline.py        # writes belfast_sentinel_model.* into ../backend/

# 2. Run the stack
cd ..
docker compose up --build
```

Open http://localhost:3000.

## Repo layout

```
urbandecay_belbot/
├── backend/                 # FastAPI inference server
│   ├── api.py               # endpoints: /health /api/predict-risk /api/stats
│   │                        #            /api/feature-importance /api/cell-details/{id}
│   │                        #            /api/top-risk-areas
│   ├── Dockerfile           # dev image (uvicorn --reload)
│   ├── Dockerfile.prod      # gunicorn + uvicorn workers
│   └── requirements.txt
│
├── data_preparation/        # turn open data → grid + model
│   ├── build_grid.py            # 500 m grid covering Belfast metro
│   ├── fetch_opendatani.py      # DfI flood map ArcGIS REST
│   ├── fetch_nisra_census.py    # NIMDM + Census 2021
│   ├── fetch_house_prices.py    # NI HPI (LGD)
│   ├── fetch_sentinel.py        # Sentinel Hub OR Earth Engine backend
│   ├── generate_synthetic.py    # plausible synthetic fallback
│   ├── feature_engineering.py   # joins everything onto the grid
│   ├── train_model.py           # 5-fold LightGBM, refit on full data
│   └── run_pipeline.py          # orchestrates all of the above
│
├── frontend/                # React + TypeScript + Mapbox GL
│   ├── src/components/      # LandingPage, Map, Legend, FilterButtonPanel, …
│   ├── src/services/api.ts  # axios client + NI-shaped types
│   └── src/utils/geoHelpers.ts
│
├── docker-compose.yml
├── env.example
└── nginx/nginx.conf
```

## Running without Docker

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
npm install
echo "REACT_APP_MAPBOX_TOKEN=pk.your_token_here" > .env.local
npm start
```

## Replacing the synthetic data

`run_pipeline.py` will *always* produce a working demo by falling back to
synthetic features. To swap in real data:

| Source | What to set / drop |
|---|---|
| Sentinel-2 / 5P (Sentinel Hub) | `SH_CLIENT_ID` + `SH_CLIENT_SECRET` env vars; `pip install sentinelhub` |
| Sentinel-2 / 5P (Earth Engine) | `GEE_SERVICE_ACCOUNT_KEY` pointing to a service-account JSON; `pip install earthengine-api` |
| DfI flood maps | already public — works out of the box if the network is up |
| NIMDM 2017 | drop `outputs/soa_boundaries.geojson` (NISRA SOA boundaries) so the spatial join runs |
| NI HPI | drop `outputs/lgd_boundaries.geojson` (LGD2014) so the LGD join runs |

After dropping new files, re-run:

```bash
python data_preparation/feature_engineering.py
python data_preparation/train_model.py
```

The API hot-reloads new model + grid files on next start.

## API contract

```
GET  /health                          → health + model metadata
GET  /api/predict-risk                → all cells with risk_score
GET  /api/stats                       → numeric column stats
GET  /api/feature-importance          → ranked feature importance
GET  /api/cell-details/{cell_id}      → per-cell breakdown
GET  /api/top-risk-areas?limit=20     → top-N cells by risk_score
```

The frontend already speaks this contract — it's the same shape as the
upstream Urban-Sentinel project.

## Why "belbot"?

Working name from the original brief: *Belfast City Council Decision Model*.
The project is positioned as a decision-support tool for council officers
allocating regeneration funding, with optional outputs for: data-centre siting
(infrastructure proximity layer), and resident-rehoming candidates (top-risk
areas adjacent to lower-risk housing stock).

## Credit

Architecture and frontend skeleton from
[ManagementMO/Urban-Sentinel](https://github.com/ManagementMO/Urban-Sentinel).
All NI-specific data pipeline, feature engineering, and model are this repo's.
