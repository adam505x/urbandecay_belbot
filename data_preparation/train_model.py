"""Train a LightGBM classifier on the feature-engineered Belfast grid.

Outputs (into backend/ so the API can find them at startup):
  backend/belfast_sentinel_model.txt              (LightGBM Booster save_model)
  backend/belfast_sentinel_model_metadata.json    (feature names, importance, metrics)
  backend/belfast_grid_with_features.geojson      (copy used by the API)
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, log_loss, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"

NON_FEATURES = {
    "cell_id", "is_decayed", "target_decay_score",
    "dominant_decay_driver", "recent_dominant_decay_driver",
    "lsoa_code", "lsoa_name", "ward_code", "ward_name", "geometry",
}

LGB_PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 25,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 0.1,
    "verbose": -1,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(OUTPUT_DIR / "belfast_grid_with_features.geojson"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gdf = gpd.read_file(args.input)
    print(f"[train] loaded {len(gdf)} cells, {len(gdf.columns)} columns")

    df = pd.DataFrame(gdf.drop(columns="geometry"))
    feature_cols = [c for c in df.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]
    print(f"[train] using {len(feature_cols)} numeric features")

    X = df[feature_cols].astype("float32").values
    y = df["is_decayed"].astype(int).values

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_metrics = []
    oof = np.zeros(len(X))
    boosters: list[lgb.Booster] = []

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        d_tr = lgb.Dataset(X[tr], y[tr], feature_name=feature_cols)
        d_va = lgb.Dataset(X[va], y[va], feature_name=feature_cols, reference=d_tr)
        booster = lgb.train(
            LGB_PARAMS,
            d_tr,
            num_boost_round=args.num_boost_round,
            valid_sets=[d_va],
            callbacks=[
                lgb.early_stopping(args.early_stopping, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        preds = booster.predict(X[va], num_iteration=booster.best_iteration)
        oof[va] = preds
        auc = roc_auc_score(y[va], preds)
        ap = average_precision_score(y[va], preds)
        ll = log_loss(y[va], preds, labels=[0, 1])
        print(f"[train] fold {fold}: auc={auc:.4f} ap={ap:.4f} logloss={ll:.4f} (best_iter={booster.best_iteration})")
        fold_metrics.append({"fold": fold, "auc": auc, "ap": ap, "logloss": ll, "best_iter": booster.best_iteration})
        boosters.append(booster)

    overall_auc = roc_auc_score(y, oof)
    overall_ap = average_precision_score(y, oof)
    overall_ll = log_loss(y, oof, labels=[0, 1])
    print(f"[train] OOF: auc={overall_auc:.4f} ap={overall_ap:.4f} logloss={overall_ll:.4f}")

    # Refit on all data using the median best_iteration from the folds.
    best_iter = int(np.median([m["best_iter"] for m in fold_metrics]))
    full_dataset = lgb.Dataset(X, y, feature_name=feature_cols)
    final_model = lgb.train(LGB_PARAMS, full_dataset, num_boost_round=best_iter)

    BACKEND_DIR.mkdir(parents=True, exist_ok=True)
    model_path = BACKEND_DIR / "belfast_sentinel_model.txt"
    final_model.save_model(str(model_path))
    print(f"[train] saved model -> {model_path}")

    # Importance.
    gains = final_model.feature_importance(importance_type="gain")
    imp = (
        pd.DataFrame({"feature": feature_cols, "importance": gains})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp["rank"] = imp.index + 1

    metadata = {
        "model_type": "LightGBM",
        "model_version": "0.1.0",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "region": "Northern Ireland (Belfast metro)",
        "n_features": len(feature_cols),
        "n_samples": int(len(X)),
        "feature_names": feature_cols,
        "feature_importance": imp.to_dict(orient="records"),
        "performance_metrics": {
            "oof_auc": float(overall_auc),
            "oof_ap": float(overall_ap),
            "oof_logloss": float(overall_ll),
            "fold_metrics": fold_metrics,
        },
        "lgbm_params": LGB_PARAMS,
        "best_iteration": best_iter,
    }
    meta_path = BACKEND_DIR / "belfast_sentinel_model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train] saved metadata -> {meta_path}")

    # Copy the feature GeoJSON next to the model so the API can find it.
    target_grid = BACKEND_DIR / "belfast_grid_with_features.geojson"
    shutil.copyfile(args.input, target_grid)
    print(f"[train] copied grid -> {target_grid}")


if __name__ == "__main__":
    main()
