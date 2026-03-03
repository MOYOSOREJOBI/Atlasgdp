from __future__ import annotations

from datetime import date
from pathlib import Path

from atlas_gdp.pipeline.run import load_manifest, run_pipeline


def _configure_demo_env(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "atlas_gdp.test.yml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  countries: [BRA, USA]",
                "  start_year: 2023",
                "  end_year: 2026",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("ATLAS_GDP_ROOT", str(tmp_path))
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ATLAS_GDP_CONFIG", str(config_path))


def test_manifest_has_required_keys(monkeypatch, tmp_path: Path) -> None:
    _configure_demo_env(monkeypatch, tmp_path)
    result = run_pipeline(
        as_of=date(2026, 3, 3),
        countries=["BRA"],
        horizon_q=3,
        scenario={"financial_tightening": 0.0, "commodity_shock": 0.0, "demand_boost": 0.0},
        offline=True,
    )
    manifest = load_manifest(result.manifest_path)

    required = {"run_id", "created_at_utc", "as_of", "countries", "horizon_q", "scenario", "paths", "connector_status"}
    assert required.issubset(manifest.keys())
    assert {"forecast", "drivers", "world_snapshot", "train_report", "backtest_report"}.issubset(manifest["paths"].keys())
    assert manifest["countries"] == ["BRA"]
