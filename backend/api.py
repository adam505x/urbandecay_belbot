"""Belfast / Northern Ireland Urban Decay Risk API.

Loads a LightGBM classifier trained on a feature-engineered Belfast/NI grid and
exposes endpoints that mirror the Urban Sentinel contract so the existing
React + Mapbox frontend can talk to it without changes.
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
    "is_decayed",
    "target_decay_score",
    "dominant_decay_driver",
    "recent_dominant_decay_driver",
    "lsoa_code",
    "lsoa_name",
    "ward_code",
    "ward_name",
}

# Human-readable descriptions for NI-specific features.
FEATURE_DESCRIPTIONS: Dict[str, str] = {
    # Sentinel-2 derived
    "ndvi_mean": "Mean vegetation index (Sentinel-2) — lower values often indicate hard, neglected surfaces",
    "ndvi_std": "Variability of vegetation across the cell",
    "ndvi_trend": "Multi-year NDVI trend (negative = greenery loss)",
    "ndbi_mean": "Mean built-up index (Sentinel-2)",
    "ndbi_trend": "Trend in built-up surface — rising values can indicate sprawl or rising heat",
    "ndwi_mean": "Mean water index (Sentinel-2)",
    "lst_mean": "Mean land-surface temperature (Sentinel-3 / Landsat) — urban heat islands",
    # Sentinel-5P air quality
    "no2_mean": "Mean tropospheric NO₂ (Sentinel-5P) — traffic & combustion proxy",
    "no2_trend": "Trend in NO₂ concentration",
    "aerosol_index": "UV aerosol index (Sentinel-5P)",
    # Flood / climate
    "flood_river_pct": "% of cell within DfI river-flood envelope",
    "flood_coastal_pct": "% of cell within DfI coastal-flood envelope",
    "flood_surface_pct": "% of cell within DfI surface-water flood envelope",
    "flood_climate_pct": "% of cell within DfI climate-change projected flood envelope",
    # NISRA Census 2021 / NIMDM
    "deprivation_decile": "NIMDM 2017 multiple-deprivation decile (1 = most deprived)",
    "income_deprivation": "NIMDM income-deprivation domain score",
    "employment_deprivation": "NIMDM employment-deprivation domain score",
    "health_deprivation": "NIMDM health & disability domain score",
    "crime_score": "NIMDM crime & disorder domain score",
    "living_environment": "NIMDM living-environment domain score",
    "population_density": "Resident population per km² (Census 2021)",
    "pct_rented_social": "% of households in social rented tenure (Census 2021)",
    "pct_no_central_heating": "% of dwellings without central heating",
    "pct_unoccupied_dwellings": "% of dwellings unoccupied at census night",
    # House price / economy
    "house_price_index": "NI House Price Index (latest quarter, LGD-level)",
    "house_price_trend": "5-year house-price growth rate",
    "transactions_per_1k": "Property transactions per 1,000 dwellings",
    # Infrastructure / proximity
    "dist_to_powerline_km": "Distance to nearest 33kV+ transmission line",
    "dist_to_substation_km": "Distance to nearest grid substation",
    "dist_to_water_km": "Distance to nearest river/coast",
    "dist_to_centre_km": "Distance to nearest town/city centre",
    # LiDAR / topography
    "elev_mean": "Mean elevation (NI 2021 LiDAR DTM)",
    "elev_std": "Terrain roughness within the cell",
    "slope_mean": "Mean slope",
    # Traffic
    "traffic_congestion_idx": "TomTom congestion index (0–1)",
    "footfall_score": "Estimated footfall score (NI Footfall index)",
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

        # Frontend expects WKT geometry strings + a few legacy field names.
        out["geometry"] = out["geometry"].apply(lambda g: g.wkt if g is not None else None)
        # Legacy aliases so the existing popup/legend works without code changes.
        if "is_decayed" in out.columns:
            out["is_blighted"] = out["is_decayed"].astype(bool)
        if "dominant_decay_driver" in out.columns:
            out["overall_most_common_blight"] = out["dominant_decay_driver"]
        if "recent_dominant_decay_driver" in out.columns:
            out["recent_most_common_blight"] = out["recent_dominant_decay_driver"]
        if "target_decay_score" in out.columns:
            out["target_blight_count"] = out["target_decay_score"]

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

        return {
            "cell_id": int(cell_id),
            "risk_score": risk,
            "risk_level": get_risk_level(risk),
            "coordinates": {"geometry": row.geometry.wkt if row.geometry is not None else None},
            "features": feature_values,
            "historical_data": {
                "is_blighted": bool(row.get("is_decayed", False)),
                "target_blight_count": float(row.get("target_decay_score", 0) or 0),
                "overall_most_common_blight": str(row.get("dominant_decay_driver", "None")),
                "recent_most_common_blight": str(row.get("recent_dominant_decay_driver", "None")),
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
