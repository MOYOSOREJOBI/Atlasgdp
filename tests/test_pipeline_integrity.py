from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from atlas_gdp.pipeline.service import run_pipeline


def _configure_demo_env(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "atlas_gdp.test.yml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  countries: [USA, CAN, GBR]",
                "  start_year: 2023",
                "  end_year: 2026",
                "run:",
                "  default_horizons: [0, 1, 2]",
                "  backtest_horizon: 1",
                "  backtest_min_train_rows: 4",
                "  backtest_max_origins: 2",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("ATLAS_GDP_ROOT", str(tmp_path))
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ATLAS_GDP_CONFIG", str(config_path))


def test_run_pipeline_writes_manifest_and_removes_stale_latest_files(monkeypatch, tmp_path: Path) -> None:
    _configure_demo_env(monkeypatch, tmp_path)
    latest_dir = tmp_path / "artifacts" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    stale = latest_dir / "forecast_OLD.json"
    stale.write_text("{}", encoding="utf-8")

    manifest = run_pipeline(
        as_of="2026-03-03",
        horizons=[0, 1, 2],
        scenario={"tighten": 0.0, "commodity": 0.0, "demand": 0.0},
        countries=["USA", "CAN"],
        include_backtest=False,
    )

    assert manifest["countries"] == ["USA", "CAN"]
    assert not stale.exists()
    assert Path(manifest["manifest_path"]).exists()
    latest_manifest = latest_dir / "manifest.json"
    assert latest_manifest.exists()
    latest_payload = json.loads(latest_manifest.read_text(encoding="utf-8"))
    assert latest_payload["run_id"] == manifest["run_id"]
    latest_files = sorted(path.name for path in latest_dir.glob("forecast_*.json"))
    assert latest_files == ["forecast_CAN.json", "forecast_USA.json"]


def test_forecast_files_and_backtest_outputs_follow_manifest(monkeypatch, tmp_path: Path) -> None:
    _configure_demo_env(monkeypatch, tmp_path)
    manifest = run_pipeline(
        as_of="2026-03-03",
        horizons=[0, 1, 2],
        scenario={"tighten": 0.0, "commodity": 0.0, "demand": 0.0},
        countries=["USA"],
        include_backtest=True,
    )

    forecast_path = Path(manifest["country_artifacts"]["USA"]["forecast"])
    drivers_path = Path(manifest["country_artifacts"]["USA"]["drivers"])
    metrics_path = Path(manifest["backtest"]["files"]["metrics"])
    predictions_path = Path(manifest["backtest"]["files"]["predictions"])
    coverage_plot_path = Path(manifest["backtest"]["files"]["coverage_plot"])
    assert forecast_path.exists()
    assert drivers_path.exists()
    assert metrics_path.exists()
    assert predictions_path.exists()
    assert coverage_plot_path.exists()

    forecast_payload = json.loads(forecast_path.read_text(encoding="utf-8"))
    assert {"country", "as_of", "forecast", "shock_prob", "trace"}.issubset(forecast_payload.keys())
    first_row = forecast_payload["forecast"][0]
    assert {"horizon_q", "period", "mean", "p10", "p50", "p90"}.issubset(first_row.keys())

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert {"overall", "by_country", "origins", "horizon_q", "calibration"}.issubset(metrics_payload.keys())
    assert metrics_payload["horizon_q"] == 1

    prediction_df = pd.read_csv(predictions_path)
    assert {"country", "origin", "point", "actual", "horizon_q"}.issubset(prediction_df.columns)
