"""End-to-end pipeline: build grid → fetch real data → engineer features → train.

Required steps (abort on failure):
  build_grid, fetch_nimdm, fetch_hpi, fetch_vacancy, feature_engineering, train_model

Optional steps (skip on failure, pipeline continues):
  fetch_crime, fetch_transport, fetch_sentinel
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load_dotenv(path: Path) -> None:
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

# Use all 12 CPU threads for numpy/scipy/geopandas operations
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

REQUIRED = [
    [sys.executable, str(HERE / "build_grid.py"), "--region", "belfast", "--cell-m", "500"],
    [sys.executable, str(HERE / "fetch_nimdm.py")],
    [sys.executable, str(HERE / "fetch_hpi.py")],
    [sys.executable, str(HERE / "fetch_vacancy.py")],
]

OPTIONAL = [
    [sys.executable, str(HERE / "fetch_crime.py")],
    [sys.executable, str(HERE / "fetch_transport.py")],
    [sys.executable, str(HERE / "fetch_sentinel.py")],
]

REQUIRED_FINAL = [
    [sys.executable, str(HERE / "feature_engineering.py")],
    [sys.executable, str(HERE / "train_model.py")],
]


def main() -> None:
    for cmd in REQUIRED:
        print(f"\n>>> {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  [ABORT] {Path(cmd[1]).stem} failed (rc={rc}) — required step")
            sys.exit(rc)

    for cmd in OPTIONAL:
        print(f"\n>>> {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  [skip] {Path(cmd[1]).stem} failed (rc={rc}) — optional, continuing")

    for cmd in REQUIRED_FINAL:
        print(f"\n>>> {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  [ABORT] {Path(cmd[1]).stem} failed (rc={rc})")
            sys.exit(rc)

    print("\n[pipeline] complete.")


if __name__ == "__main__":
    main()
