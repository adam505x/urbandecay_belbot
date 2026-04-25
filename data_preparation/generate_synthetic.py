"""Generate plausible synthetic feature values for every grid cell.

This is the fallback used when the real OpenDataNI / Sentinel / NISRA fetchers
have not been run (e.g. no API keys yet). It produces a feature CSV with the
same schema feature_engineering.py expects, so the rest of the pipeline runs
end-to-end and the demo can be brought up immediately.

The synthetic field generator is biased toward Belfast — high-density
deprivation around the inner east, north Belfast and Falls/Shankill corridor —
so the resulting risk map is at least *visually* plausible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def _gauss_field(centroids: np.ndarray, hot_spots: list[tuple[float, float, float]]) -> np.ndarray:
    """Sum of Gaussian bumps around given (lon, lat, sigma) hotspots."""
    out = np.zeros(len(centroids))
    for lon, lat, sigma in hot_spots:
        d2 = ((centroids[:, 0] - lon) ** 2 + (centroids[:, 1] - lat) ** 2) / (sigma**2)
        out += np.exp(-d2)
    return out


def synthesize(grid_path: Path, out_path: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    gdf = gpd.read_file(grid_path)
    cents = np.array([[g.centroid.x, g.centroid.y] for g in gdf.geometry])

    # Belfast deprivation hotspots (rough, illustrative only).
    deprivation_hotspots = [
        (-5.945, 54.610, 0.012),  # Falls / Lower Shankill
        (-5.913, 54.620, 0.012),  # New Lodge / North Belfast
        (-5.898, 54.585, 0.012),  # Inner East / Short Strand
        (-5.870, 54.561, 0.012),  # Castlereagh fringe
    ]
    affluence_hotspots = [
        (-5.840, 54.580, 0.020),  # Stormont / Cherryvalley
        (-5.985, 54.572, 0.020),  # Malone / Lisburn Rd
        (-5.870, 54.660, 0.020),  # Glengormley
    ]

    deprivation_field = _gauss_field(cents, deprivation_hotspots)
    affluence_field = _gauss_field(cents, affluence_hotspots)
    decay_field = deprivation_field - 0.7 * affluence_field
    decay_field = (decay_field - decay_field.min()) / (np.ptp(decay_field) + 1e-9)

    n = len(gdf)
    df = pd.DataFrame({"cell_id": gdf["cell_id"].values})

    # Sentinel-2 vegetation/built-up: lower NDVI in deprived built-up areas.
    df["ndvi_mean"] = np.clip(0.55 - 0.45 * decay_field + rng.normal(0, 0.05, n), -0.1, 0.9)
    df["ndvi_std"] = np.clip(0.05 + 0.08 * rng.random(n), 0.02, 0.25)
    df["ndvi_trend"] = np.clip(-0.02 * decay_field + rng.normal(0, 0.01, n), -0.08, 0.05)
    df["ndbi_mean"] = np.clip(-0.1 + 0.4 * decay_field + rng.normal(0, 0.05, n), -0.4, 0.5)
    df["ndbi_trend"] = np.clip(0.01 * decay_field + rng.normal(0, 0.005, n), -0.03, 0.04)
    df["ndwi_mean"] = np.clip(-0.1 + rng.normal(0, 0.05, n), -0.4, 0.4)
    df["lst_mean"] = 14.0 + 4.5 * decay_field + rng.normal(0, 0.6, n)

    # Sentinel-5P NO2 — higher near deprived and arterial corridors.
    df["no2_mean"] = np.clip(2e-5 + 6e-5 * decay_field + rng.normal(0, 5e-6, n), 0, None)
    df["no2_trend"] = rng.normal(0, 1e-6, n)
    df["aerosol_index"] = rng.normal(0, 0.3, n)

    # Flood envelopes — coastal (Belfast Lough) bias by latitude/longitude.
    coastal_proximity = np.exp(-((cents[:, 0] + 5.93) ** 2 + (cents[:, 1] - 54.66) ** 2) / 0.005)
    df["flood_river_pct"] = np.clip(rng.beta(0.5, 6, n) + 0.1 * decay_field, 0, 1)
    df["flood_coastal_pct"] = np.clip(0.3 * coastal_proximity + rng.beta(0.3, 8, n), 0, 1)
    df["flood_surface_pct"] = np.clip(rng.beta(0.5, 5, n) + 0.05 * decay_field, 0, 1)
    df["flood_climate_pct"] = np.clip(df["flood_river_pct"] + 0.05 * rng.random(n), 0, 1)

    # NISRA Census 2021 / NIMDM-style.
    df["deprivation_decile"] = np.clip(np.round(10 - 9 * decay_field + rng.normal(0, 0.7, n)), 1, 10).astype(int)
    df["income_deprivation"] = np.clip(0.05 + 0.55 * decay_field + rng.normal(0, 0.05, n), 0, 1)
    df["employment_deprivation"] = np.clip(0.05 + 0.50 * decay_field + rng.normal(0, 0.05, n), 0, 1)
    df["health_deprivation"] = np.clip(0.05 + 0.45 * decay_field + rng.normal(0, 0.06, n), 0, 1)
    df["crime_score"] = np.clip(0.10 + 0.65 * decay_field + rng.normal(0, 0.07, n), 0, 1)
    df["living_environment"] = np.clip(0.10 + 0.50 * decay_field + rng.normal(0, 0.06, n), 0, 1)
    df["population_density"] = np.clip(800 + 6500 * decay_field + rng.normal(0, 400, n), 0, None)
    df["pct_rented_social"] = np.clip(0.05 + 0.55 * decay_field + rng.normal(0, 0.06, n), 0, 0.95)
    df["pct_no_central_heating"] = np.clip(0.005 + 0.05 * decay_field + rng.normal(0, 0.01, n), 0, 0.3)
    df["pct_unoccupied_dwellings"] = np.clip(0.04 + 0.10 * decay_field + rng.normal(0, 0.02, n), 0, 0.5)

    # House prices — affluence-driven.
    affluence_norm = (affluence_field - affluence_field.min()) / (np.ptp(affluence_field) + 1e-9)
    df["house_price_index"] = 110 + 80 * affluence_norm - 30 * decay_field + rng.normal(0, 8, n)
    df["house_price_trend"] = 0.05 + 0.15 * affluence_norm - 0.10 * decay_field + rng.normal(0, 0.02, n)
    df["transactions_per_1k"] = np.clip(8 + 12 * affluence_norm + rng.normal(0, 2, n), 0, None)

    # Infrastructure proximity.
    df["dist_to_powerline_km"] = np.clip(rng.exponential(1.5, n), 0, 8)
    df["dist_to_substation_km"] = np.clip(rng.exponential(2.0, n), 0, 12)
    df["dist_to_water_km"] = np.clip(np.abs(cents[:, 1] - 54.66) * 60 + rng.normal(0, 0.4, n), 0, None)
    df["dist_to_centre_km"] = np.clip(
        np.sqrt((cents[:, 0] + 5.93) ** 2 + (cents[:, 1] - 54.597) ** 2) * 70 + rng.normal(0, 0.3, n),
        0, None,
    )

    # LiDAR / topo.
    df["elev_mean"] = np.clip(20 + 80 * (1 - decay_field) + rng.normal(0, 15, n), 0, None)
    df["elev_std"] = np.clip(rng.gamma(2, 1.5, n), 0, None)
    df["slope_mean"] = np.clip(rng.gamma(2, 0.8, n), 0, None)

    # Traffic / footfall.
    df["traffic_congestion_idx"] = np.clip(0.1 + 0.6 * decay_field + rng.normal(0, 0.08, n), 0, 1)
    df["footfall_score"] = np.clip(0.2 + 0.5 * affluence_norm + 0.3 * decay_field + rng.normal(0, 0.1, n), 0, 1)

    # Synthetic label — noisy combination of decay drivers.
    label_logit = (
        2.5 * decay_field
        - 1.5 * df["ndvi_mean"]
        + 0.8 * df["ndbi_mean"]
        + 1.2 * df["crime_score"]
        + 1.0 * df["pct_unoccupied_dwellings"]
        + 0.6 * df["flood_surface_pct"]
        - 0.5 * (df["house_price_index"] - 150) / 50
        + rng.normal(0, 0.4, n)
    )
    prob = 1 / (1 + np.exp(-label_logit + 1.5))
    df["target_decay_score"] = prob
    df["is_decayed"] = (rng.random(n) < prob).astype(int)

    drivers = ["unoccupied_dwellings", "surface_flooding", "low_vegetation",
               "high_no2", "social_housing_concentration", "crime_pressure"]
    df["dominant_decay_driver"] = rng.choice(drivers, n, p=[0.25, 0.10, 0.20, 0.15, 0.20, 0.10])
    df["recent_dominant_decay_driver"] = rng.choice(drivers, n, p=[0.30, 0.10, 0.20, 0.15, 0.15, 0.10])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[synthetic] wrote {out_path} ({len(df)} rows, {len(df.columns)} cols)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default=str(OUTPUT_DIR / "belfast_grid.geojson"))
    parser.add_argument("--out", default=str(OUTPUT_DIR / "synthetic_features.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    synthesize(Path(args.grid), Path(args.out), args.seed)


if __name__ == "__main__":
    main()
