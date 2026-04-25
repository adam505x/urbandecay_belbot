"""End-to-end pipeline: build grid -> fetch real data (best effort) -> synth ->
feature-engineer -> train.

This is the one-shot entrypoint. If the real data fetchers are missing
credentials or hit network errors, the synthetic baseline is used so the demo
still ends with a working model + GeoJSON in backend/.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader — no python-dotenv dependency required."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")

STEPS = [
    [sys.executable, str(HERE / "build_grid.py"), "--region", "belfast", "--cell-m", "500"],
    [sys.executable, str(HERE / "fetch_opendatani.py")],
    [sys.executable, str(HERE / "fetch_nisra_census.py")],
    [sys.executable, str(HERE / "fetch_house_prices.py")],
    [sys.executable, str(HERE / "fetch_sentinel.py")],
    [sys.executable, str(HERE / "generate_synthetic.py")],
    [sys.executable, str(HERE / "feature_engineering.py")],
    [sys.executable, str(HERE / "train_model.py")],
]


def main() -> None:
    for cmd in STEPS:
        print(f"\n>>> {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            label = Path(cmd[1]).stem
            if label.startswith("fetch_") or label == "generate_synthetic":
                print(f"  [skip] {label} returned {rc}; pipeline continues with whatever data is available")
                continue
            print(f"  [fail] {label} returned {rc}; aborting")
            sys.exit(rc)


if __name__ == "__main__":
    main()
