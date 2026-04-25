"""Belfast / Northern Ireland Urban Decay Risk API.

Loads a LightGBM regressor trained on a composite decay_index and exposes
endpoints consumed by the existing React + Mapbox frontend. Includes
decay_2025/2030/2035/2040 forecast fields where available.
"""

import json
import os
from typing import Any, Dict, List, Union

import geopandas as gpd
import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Belfast Sentinel API",
    description="Predictive urban-decay risk for Belfast & Northern Ireland",
    version="0.1.0",
)

cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    os.getenv("FRONTEND_URL", "https://belfast-sentinel.example.com"),
]
if os.getenv("ENVIRONMENT", "development").lower() == "development":
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Columns that are NOT model features (identifiers / labels / display fields).
NON_FEATURE_COLUMNS = {
    "cell_id",
    "decay_index", "decay_decile",
    "decay_2025", "decay_2030", "decay_2035", "decay_2040",
    "is_decayed",
    "target_decay_score",
    "dominant_decay_driver",
    "recent_dominant_decay_driver",
    "lsoa_code", "lsoa_name",
    "ward_code", "ward_name",
    "SOA2001name",
}

# Human-readable descriptions for NI-specific features.
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # NIMDM 2017 (real data, Small Area level)
    "deprivation_score": "NIMDM 2017 overall deprivation score (1=most deprived)",
    "income_deprivation": "NIMDM income-deprivation — % of population income-deprived",
    "employment_deprivation": "NIMDM employment-deprivation domain score",
    "health_deprivation": "NIMDM health & disability domain score",
    "crime_domain": "NIMDM crime & disorder domain score (Small Area level)",
    "living_environment": "NIMDM living-environment domain score",
    "access_to_services": "NIMDM proximity to services domain score",
    # House price (real data, Belfast LGD quarterly)
    "house_price_index": "NI House Price Index — Belfast LGD (latest quarter)",
    "house_price_standardised": "Standardised house price — Belfast LGD",
    "house_price_trend_5yr": "5-year HPI trend slope (HPI points per year)",
    "house_price_trend_10yr": "10-year HPI trend slope (HPI points per year)",
    "house_price_growth_pct_5yr": "5-year house price growth % — Belfast LGD",
    # Vacancy (real data, LPS district level)
    "vacancy_rate": "Domestic property vacancy rate — Belfast district",
    # Crime (real data, PSNI Belfast City)
    "crime_rate_per_1k": "PSNI recorded crimes per 1,000 population",
    # Transport (real data, Translink bus stops)
    "n_stops_500m": "Active bus stops within 500m",
    "dist_to_nearest_stop_m": "Distance to nearest active bus stop (m)",
    "transport_access_score": "Normalised transport accessibility [0-1]",
    # Geometry-derived
    "dist_to_centre_km": "Distance to Belfast city centre (km)",
    # Sentinel-2 (optional — requires Sentinel Hub credentials)
    "ndvi_mean": "Mean NDVI (Sentinel-2) — vegetation cover",
    "ndbi_mean": "Mean NDBI (Sentinel-2) — built-up surface index",
    "ndwi_mean": "Mean NDWI (Sentinel-2) — water/moisture",
    "no2_mean": "Mean tropospheric NO2 (Sentinel-5P)",
}


def get_feature_description(feature: str) -> str:
    return FEATURE_DESCRIPTIONS.get(feature, f"Feature: {feature}")


def get_risk_level(score: float) -> str:
    if score >= 0.8:
        return "Very High"
    if score >= 0.6:
        return "High"
    if score >= 0.4:
        return "Medium"
    if score >= 0.2:
        return "Low"
    return "Very Low"


# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "belfast_sentinel_model.txt")
META_PATH = os.getenv("META_PATH", "belfast_sentinel_model_metadata.json")
GRID_PATH = os.getenv("GRID_PATH", "belfast_grid_with_features.geojson")


def _resolve(path: str) -> str:
    if os.path.exists(path):
        return path
    alt = os.path.join("..", "data_preparation", "outputs", os.path.basename(path))
    if os.path.exists(alt):
        return alt
    return path


model: lgb.Booster
gdf: gpd.GeoDataFrame
model_meta: Dict[str, Any] = {}
feature_importance_df: Union[pd.DataFrame, None] = None


def load_assets() -> None:
    global model, gdf, model_meta, feature_importance_df

    mp = _resolve(MODEL_PATH)
    metp = _resolve(META_PATH)
    gp = _resolve(GRID_PATH)

    print(f"[belfast-sentinel] loading model from {mp}")
    model = lgb.Booster(model_file=mp)

    print(f"[belfast-sentinel] loading metadata from {metp}")
    with open(metp, "r", encoding="utf-8") as f:
        model_meta = json.load(f)

    if "feature_importance" in model_meta:
        feature_importance_df = pd.DataFrame(model_meta["feature_importance"])

    print(f"[belfast-sentinel] loading grid from {gp}")
    gdf = gpd.read_file(gp)
    print(f"[belfast-sentinel] grid loaded: {gdf.shape[0]} cells, {gdf.shape[1]} columns")


load_assets()


def _feature_frame() -> pd.DataFrame:
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    drop_cols = [c for c in df.columns if c in NON_FEATURE_COLUMNS]
    feature_df = df.drop(columns=drop_cols)
    # Use the exact feature order the model was trained on, when available.
    feature_order = model_meta.get("feature_names")
    if feature_order:
        for col in feature_order:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        feature_df = feature_df[feature_order]
    return feature_df


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Belfast Sentinel — predictive urban-decay risk for Northern Ireland"}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": gdf is not None,
        "total_cells": int(len(gdf)) if gdf is not None else 0,
        "model_type": model_meta.get("model_type", "LightGBM"),
        "model_version": model_meta.get("model_version", "0.1.0"),
        "training_date": model_meta.get("training_date"),
        "n_features": model_meta.get("n_features"),
        "performance_metrics": model_meta.get("performance_metrics", {}),
        "region": model_meta.get("region", "Northern Ireland"),
    }


@app.get("/api/predict-risk")
def predict_risk() -> List[Dict[str, Any]]:
    try:
        X = _feature_frame()
        scores = model.predict(X)
        out = gdf.copy()
        out["risk_score"] = scores

        # Frontend expects WKT geometry strings.
        out["geometry"] = out["geometry"].apply(lambda g: g.wkt if g is not None else None)

        # Expose forecast columns if present in the grid GeoJSON.
        for forecast_col in ["decay_2025", "decay_2030", "decay_2035", "decay_2040"]:
            if forecast_col not in out.columns:
                out[forecast_col] = None

        # risk_score == decay_2025 when forecast columns exist; otherwise raw model output.
        if "decay_2025" in gdf.columns:
            out["risk_score"] = gdf["decay_2025"].clip(0, 1)

        # Legacy aliases so the existing popup/legend works without code changes.
        out["is_blighted"] = (out["risk_score"] >= 0.5).astype(bool)
        if "dominant_decay_driver" in out.columns:
            out["overall_most_common_blight"] = out["dominant_decay_driver"]
        else:
            out["overall_most_common_blight"] = "urban_deprivation"
        out["recent_most_common_blight"] = out.get("recent_dominant_decay_driver", "urban_deprivation")
        out["target_blight_count"] = out["risk_score"]

        return out.to_dict(orient="records")
    except Exception as exc:
        print(f"[predict-risk] error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/stats")
def stats() -> Dict[str, Union[int, float, List[str]]]:
    try:
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        out: Dict[str, Union[int, float, List[str]]] = {
            "total_cells": int(len(df)),
            "columns": list(df.columns),
        }
        for col in df.select_dtypes(include="number").columns:
            out[f"{col}_mean"] = float(df[col].mean())
            out[f"{col}_std"] = float(df[col].std())
        return out
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/feature-importance")
def feature_importance() -> Dict[str, Any]:
    try:
        if feature_importance_df is not None:
            rows = feature_importance_df.to_dict(orient="records")
            total = sum(float(r["importance"]) for r in rows) or 1.0
            for r in rows:
                r["description"] = get_feature_description(r["feature"])
                r["contribution_percent"] = float(r["importance"]) / total * 100.0
                r["importance"] = float(r["importance"])
                r["rank"] = int(r["rank"])
            top = rows[:15]
            return {
                "top_features": top,
                "all_features": rows,
                "total_features": len(rows),
                "model_type": model_meta.get("model_type", "LightGBM"),
                "importance_type": "gain",
                "enhanced": True,
                "feature_coverage": {
                    "top_5": sum(r["contribution_percent"] for r in rows[:5]),
                    "top_10": sum(r["contribution_percent"] for r in rows[:10]),
                    "top_15": sum(r["contribution_percent"] for r in rows[:15]),
                },
            }

        feats = list(_feature_frame().columns)
        importances = model.feature_importance(importance_type="gain")
        rows = sorted(
            [
                {
                    "feature": f,
                    "importance": float(i),
                    "rank": idx + 1,
                    "description": get_feature_description(f),
                }
                for idx, (f, i) in enumerate(zip(feats, importances))
            ],
            key=lambda r: r["importance"],
            reverse=True,
        )
        for idx, r in enumerate(rows):
            r["rank"] = idx + 1
        return {
            "top_features": rows[:15],
            "all_features": rows,
            "total_features": len(rows),
            "model_type": "LightGBM",
            "importance_type": "gain",
            "enhanced": False,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/cell-details/{cell_id}")
def cell_details(cell_id: int) -> Dict[str, Any]:
    try:
        match = gdf[gdf["cell_id"] == cell_id]
        if len(match) == 0:
            raise HTTPException(status_code=404, detail=f"Cell {cell_id} not found")
        row = match.iloc[0]

        X = _feature_frame()
        cell_features = X.iloc[[match.index[0] - gdf.index[0]]] if gdf.index[0] != 0 else X.iloc[[match.index[0]]]
        risk = float(model.predict(cell_features)[0])

        feature_values: Dict[str, Dict[str, Any]] = {}
        for col in X.columns:
            try:
                feature_values[col] = {
                    "value": float(row.get(col, 0.0) or 0.0),
                    "description": get_feature_description(col),
                }
            except (TypeError, ValueError):
                feature_values[col] = {"value": 0.0, "description": get_feature_description(col)}

        forecast = {}
        for col in ["decay_2025", "decay_2030", "decay_2035", "decay_2040"]:
            v = row.get(col)
            forecast[col] = float(v) if v is not None and str(v) != "nan" else None

        return {
            "cell_id": int(cell_id),
            "risk_score": risk,
            "risk_level": get_risk_level(risk),
            "coordinates": {"geometry": row.geometry.wkt if row.geometry is not None else None},
            "features": feature_values,
            "forecast_timeline": forecast,
            "historical_data": {
                "is_blighted": risk >= 0.5,
                "target_blight_count": risk,
                "overall_most_common_blight": str(row.get("dominant_decay_driver", "urban_deprivation")),
                "recent_most_common_blight": str(row.get("recent_dominant_decay_driver", "urban_deprivation")),
                "lsoa": str(row.get("lsoa_name", "")),
                "ward": str(row.get("ward_name", "")),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/top-risk-areas")
def top_risk_areas(limit: int = 20) -> List[Dict[str, Any]]:
    try:
        X = _feature_frame()
        scores = model.predict(X)
        df = pd.DataFrame(gdf.drop(columns="geometry"))
        df["risk_score"] = scores
        df = df.sort_values("risk_score", ascending=False).head(limit)

        results: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            score_val = float(r["risk_score"])
            results.append(
                {
                    "cell_id": int(r["cell_id"]),
                    "risk_score": score_val,
                    "risk_level": get_risk_level(score_val),
                    "lsoa_name": str(r.get("lsoa_name", "")),
                    "ward_name": str(r.get("ward_name", "")),
                    "deprivation_decile": float(r.get("deprivation_decile", 0) or 0),
                    "ndvi_mean": float(r.get("ndvi_mean", 0) or 0),
                    "no2_mean": float(r.get("no2_mean", 0) or 0),
                    "flood_river_pct": float(r.get("flood_river_pct", 0) or 0),
                    "is_blighted": bool(r.get("is_decayed", False)),
                    # Legacy fields for the existing TopBlightMenu component
                    "total_complaints_mean": float(r.get("population_density", 0) or 0),
                    "blight_complaints_mean": float(r.get("target_decay_score", 0) or 0),
                }
            )
        return results
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
