from __future__ import annotations

import json
from pathlib import Path

from atlas_gdp.pipeline.run import load_forecast_payloads_from_manifest, manifest_country_list


def test_manifest_loader_only_returns_manifest_countries(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "runs" / "20260303T180900Z_7K2P"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "forecast_BRA.json").write_text(json.dumps({"country": "BRA", "forecast": []}), encoding="utf-8")
    (run_dir / "forecast_OLD.json").write_text(json.dumps({"country": "OLD", "forecast": []}), encoding="utf-8")
    manifest = {
        "run_id": "20260303T180900Z_7K2P",
        "created_at_utc": "2026-03-03T18:09:00+00:00",
        "as_of": "2026-03-03",
        "countries": ["BRA"],
        "horizon_q": 7,
        "scenario": {
            "financial_tightening": 0.0,
            "commodity_shock": 0.0,
            "demand_boost": 0.0,
        },
        "paths": {
            "forecast": {"BRA": "forecast_BRA.json"},
            "drivers": {"BRA": "drivers_BRA.json"},
            "world_snapshot": "world_snapshot.csv",
            "train_report": "train_report.json",
            "backtest_report": None,
        },
        "manifest_path": str(run_dir / "manifest.json"),
        "artifacts_root": str(run_dir),
    }

    assert manifest_country_list(manifest) == ["BRA"]
    payloads = load_forecast_payloads_from_manifest(manifest)
    assert sorted(payloads.keys()) == ["BRA"]
