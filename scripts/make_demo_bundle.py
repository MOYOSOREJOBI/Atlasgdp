from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from atlas_gdp.config import load_settings


def main() -> None:
    settings = load_settings()
    payload = {"offline_mode": True, "note": "Use data/samples/offline_macro.csv for bundled demo runs."}
    (settings.paths.samples / "demo_bundle.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {settings.paths.samples / 'demo_bundle.json'}")


if __name__ == "__main__":
    main()
