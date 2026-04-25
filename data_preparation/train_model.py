"""Train LightGBM regressor on decay_index and produce 2025–2040 forecasts.

Outputs (into backend/):
  belfast_sentinel_model.txt              — LightGBM Booster
  belfast_sentinel_model_metadata.json   — feature names, importance, metrics
  belfast_grid_with_features.geojson     — grid with decay_2025/2030/2035/2040
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"

NON_FEATURES = {
    "cell_id", "geometry",
    "decay_index", "decay_decile",
    # legacy fields
    "is_decayed", "target_decay_score",
    "dominant_decay_driver", "recent_dominant_decay_driver",
    "lsoa_code", "lsoa_name", "ward_code", "ward_name",
    "SOA2001name",
}

LGB_PARAMS = {
    "objective": "regression",
    "metric": ["rmse", "mae"],
    "device": "gpu",
    "num_threads": 12,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 0.1,
    "verbose": -1,
}

# Forecast horizons: years ahead from 2025
FORECAST_HORIZONS = {
    "decay_2025": 0,
    "decay_2030": 5,
    "decay_2035": 10,
    "decay_2040": 15,
}


def _forecast_decay(
    base_decay: np.ndarray,
    deprivation: np.ndarray,
    years_ahead: int,
    hpi_growth_pct_5yr: float,
) -> np.ndarray:
    """
    Project decay_index forward using a polarisation model:
    - Overall improvement tied to Belfast HPI growth (rising market lifts some areas)
    - High-deprivation areas benefit less from house price growth
    - Low-deprivation areas improve faster

    Improvement rate per year: based on HPI 5yr growth % (annualised)
    deprivation_score = 1 → no improvement (area not connected to rising market)
    deprivation_score = 0 → full improvement rate
    """
    if years_ahead == 0:
        return base_decay.copy()

    # Annualised improvement: HPI growing 3-5%/yr historically; cap at 0.5% decay reduction/yr
    hpi_annual_pct = max(0.0, hpi_growth_pct_5yr / 5.0)   # % per year
    base_improvement_rate = min(hpi_annual_pct / 100.0 * 0.3, 0.005)  # max 0.5%/yr

    # deprivation_score ∈ [0,1]; high deprivation = lower market connection
    market_connection = 1.0 - deprivation   # [0,1]

    delta = base_improvement_rate * market_connection * years_ahead
    return (base_decay - delta).clip(0, 1)


def main() -> None:
    grid_path = OUTPUT_DIR / "belfast_grid_with_features.geojson"
    if not grid_path.exists():
        print("[train] belfast_grid_with_features.geojson not found", file=sys.stderr)
        sys.exit(1)

    gdf = gpd.read_file(grid_path)
    print(f"[train] loaded {len(gdf)} cells, {len(gdf.columns)} columns")

    df = pd.DataFrame(gdf.drop(columns="geometry"))
    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])
    ]
    print(f"[train] using {len(feature_cols)} features: {feature_cols}")

    if "decay_index" not in df.columns:
        print("[train] ERROR: decay_index column not found — run feature_engineering.py first", file=sys.stderr)
        sys.exit(1)

    X = df[feature_cols].astype("float32").values
    y = df["decay_index"].astype("float32").values

    # --- Cross-validation ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    oof = np.zeros(len(X))
    boosters: list[lgb.Booster] = []

    for fold, (tr, va) in enumerate(kf.split(X), 1):
        d_tr = lgb.Dataset(X[tr], y[tr], feature_name=feature_cols)
        d_va = lgb.Dataset(X[va], y[va], feature_name=feature_cols, reference=d_tr)
        booster = lgb.train(
            LGB_PARAMS,
            d_tr,
            num_boost_round=1000,
            valid_sets=[d_va],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        preds = booster.predict(X[va], num_iteration=booster.best_iteration)
        oof[va] = preds
        rmse = float(np.sqrt(mean_squared_error(y[va], preds)))
        mae  = float(mean_absolute_error(y[va], preds))
        r2   = float(r2_score(y[va], preds))
        print(f"[train] fold {fold}: rmse={rmse:.4f}  mae={mae:.4f}  r2={r2:.4f}  (iter={booster.best_iteration})")
        fold_metrics.append({"fold": fold, "rmse": rmse, "mae": mae, "r2": r2,
                              "best_iter": booster.best_iteration})
        boosters.append(booster)

    oof_rmse = float(np.sqrt(mean_squared_error(y, oof)))
    oof_mae  = float(mean_absolute_error(y, oof))
    oof_r2   = float(r2_score(y, oof))
    print(f"[train] OOF: rmse={oof_rmse:.4f}  mae={oof_mae:.4f}  r2={oof_r2:.4f}")

    # --- Refit on all data ---
    best_iter = int(np.median([m["best_iter"] for m in fold_metrics]))
    full_ds = lgb.Dataset(X, y, feature_name=feature_cols)
    final_model = lgb.train(LGB_PARAMS, full_ds, num_boost_round=best_iter)

    # --- Forecasts ---
    base_decay = final_model.predict(X, num_iteration=best_iter).clip(0, 1)
    deprivation = df["deprivation_score"].values if "deprivation_score" in df.columns else np.zeros(len(df))
    hpi_growth_pct = float(df["house_price_growth_pct_5yr"].iloc[0]) if "house_price_growth_pct_5yr" in df.columns else 20.0

    forecasts = {}
    for col, years in FORECAST_HORIZONS.items():
        preds = _forecast_decay(base_decay, deprivation, years, hpi_growth_pct)
        forecasts[col] = preds
        print(f"[train] {col}: mean={preds.mean():.3f}  min={preds.min():.3f}  max={preds.max():.3f}")

    # --- Save model ---
    BACKEND_DIR.mkdir(parents=True, exist_ok=True)
    model_path = BACKEND_DIR / "belfast_sentinel_model.txt"
    final_model.save_model(str(model_path))
    print(f"[train] saved model -> {model_path}")

    # --- Feature importance ---
    gains = final_model.feature_importance(importance_type="gain")
    imp = (
        pd.DataFrame({"feature": feature_cols, "importance": gains})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp["rank"] = imp.index + 1

    metadata = {
        "model_type": "LightGBM",
        "objective": "regression",
        "model_version": "1.0.0",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "region": "Northern Ireland (Belfast metro)",
        "n_features": len(feature_cols),
        "n_samples": int(len(X)),
        "feature_names": feature_cols,
        "feature_importance": imp.to_dict(orient="records"),
        "performance_metrics": {
            "oof_rmse": oof_rmse,
            "oof_mae": oof_mae,
            "oof_r2": oof_r2,
            "fold_metrics": fold_metrics,
        },
        "lgbm_params": LGB_PARAMS,
        "best_iteration": best_iter,
    }
    meta_path = BACKEND_DIR / "belfast_sentinel_model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train] saved metadata -> {meta_path}")

    # --- Export GeoJSON with forecasts ---
    for col, preds in forecasts.items():
        gdf[col] = preds

    target_grid = BACKEND_DIR / "belfast_grid_with_features.geojson"
    shutil.copyfile(grid_path, target_grid)

    # Also write the forecast GeoJSON (overwrite with forecast columns added)
    gdf.to_file(target_grid, driver="GeoJSON")
    print(f"[train] saved grid+forecasts -> {target_grid}")


if __name__ == "__main__":
    main()
