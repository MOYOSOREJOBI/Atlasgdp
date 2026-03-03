from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from atlas_gdp.config import load_settings
from atlas_gdp.pipeline.service import load_manifest as load_manifest_file
from atlas_gdp.pipeline.service import run_pipeline as run_service_pipeline
from atlas_gdp.pipeline.storage import read_json, read_latest_run_id, write_json


@dataclass(frozen=True)
class RunResult:
    run_id: str
    manifest_path: Path
    created_at_utc: str


def _external_scenario(scenario: dict[str, Any] | None) -> dict[str, float]:
    scenario = scenario or {}
    return {
        "financial_tightening": float(scenario.get("financial_tightening", scenario.get("tighten", 0.0))),
        "commodity_shock": float(scenario.get("commodity_shock", scenario.get("commodity", 0.0))),
        "demand_boost": float(scenario.get("demand_boost", scenario.get("demand", 0.0))),
    }


def _internal_scenario(scenario: dict[str, Any] | None) -> dict[str, float]:
    external = _external_scenario(scenario)
    return {
        "tighten": external["financial_tightening"],
        "commodity": external["commodity_shock"],
        "demand": external["demand_boost"],
    }


def _normalize_manifest(inner_manifest: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(inner_manifest["artifacts_root"])
    train_report_path = run_dir / "train_report.json"
    if not train_report_path.exists():
        write_json(
            train_report_path,
            {
                "run_id": inner_manifest["run_id"],
                "created_at_utc": inner_manifest["created_at_utc"],
                "as_of": inner_manifest["as_of"],
                "countries": inner_manifest["countries"],
            },
        )

    normalized = {
        "run_id": inner_manifest["run_id"],
        "created_at_utc": inner_manifest["created_at_utc"],
        "as_of": inner_manifest["as_of"],
        "countries": list(inner_manifest.get("countries", [])),
        "horizon_q": int(max(inner_manifest.get("horizons", [0])) if inner_manifest.get("horizons") else 0),
        "scenario": _external_scenario(inner_manifest.get("scenario", {})),
        "connector_status": inner_manifest.get("connector_status", {}),
        "method_notes": inner_manifest.get("method_notes"),
        "config_snapshot": inner_manifest.get("config_snapshot", {}),
        "settings_snapshot": inner_manifest.get("settings_snapshot", {}),
        "paths": {
            "forecast": {country: Path(files["forecast"]).name for country, files in inner_manifest.get("country_artifacts", {}).items()},
            "drivers": {country: Path(files["drivers"]).name for country, files in inner_manifest.get("country_artifacts", {}).items()},
            "world_snapshot": Path(inner_manifest["world_snapshot_path"]).name,
            "train_report": train_report_path.name,
            "backtest_report": Path(
                inner_manifest.get("backtest", {}).get("files", {}).get("summary")
                or inner_manifest.get("backtest", {}).get("files", {}).get("metrics")
            ).name
            if (
                inner_manifest.get("backtest", {}).get("files", {}).get("summary")
                or inner_manifest.get("backtest", {}).get("files", {}).get("metrics")
            )
            else None,
            "backtest_plot": Path(inner_manifest["backtest"]["files"]["coverage_plot"]).name
            if inner_manifest.get("backtest", {}).get("files", {}).get("coverage_plot")
            else None,
        },
        "manifest_path": inner_manifest["manifest_path"],
        "artifacts_root": inner_manifest["artifacts_root"],
    }
    return normalized


def run_pipeline(
    as_of: date,
    countries: list[str],
    horizon_q: int,
    scenario: dict[str, Any] | None,
    offline: bool,
) -> RunResult:
    settings = load_settings()
    service_manifest = run_service_pipeline(
        as_of=as_of.isoformat(),
        horizons=list(range(max(0, int(horizon_q)) + 1)),
        scenario=_internal_scenario(scenario),
        countries=countries or None,
        include_backtest=True,
        settings=settings,
    )
    normalized = _normalize_manifest(service_manifest)
    manifest_path = Path(service_manifest["manifest_path"])
    write_json(manifest_path, normalized)
    latest_manifest_path = settings.paths.artifacts / "latest" / "manifest.json"
    write_json(latest_manifest_path, normalized)
    return RunResult(
        run_id=normalized["run_id"],
        manifest_path=manifest_path,
        created_at_utc=normalized["created_at_utc"],
    )


def load_manifest(path: str | Path) -> dict[str, Any]:
    return load_manifest_file(path)


def load_latest_manifest() -> dict[str, Any] | None:
    settings = load_settings()
    run_id = read_latest_run_id(settings)
    if not run_id:
        return None
    manifest_path = settings.paths.artifacts / "runs" / run_id / "manifest.json"
    if not manifest_path.exists():
        return None
    return load_manifest(manifest_path)


def manifest_country_list(manifest: dict[str, Any]) -> list[str]:
    return sorted(str(country) for country in manifest.get("countries", []))


def load_forecast_payloads_from_manifest(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    run_root = Path(manifest["artifacts_root"])
    payloads: dict[str, dict[str, Any]] = {}
    for country, filename in manifest.get("paths", {}).get("forecast", {}).items():
        path = run_root / filename
        if path.exists():
            payloads[str(country)] = read_json(path)
    return payloads
