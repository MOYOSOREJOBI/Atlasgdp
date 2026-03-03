from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_PYDANTIC_SETTINGS = True
except ImportError:  # pragma: no cover - exercised in this environment
    BaseSettings = object  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    SettingsConfigDict = dict  # type: ignore[assignment]
    _HAS_PYDANTIC_SETTINGS = False


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    processed: Path
    samples: Path
    artifacts: Path
    reports: Path

    def ensure(self) -> None:
        self.raw.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)
        self.samples.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    offline_mode: bool
    seed: int
    paths: Paths
    db_url: str
    fred_api_key: str | None
    bea_api_key: str | None
    ask_atlas_enabled: bool
    ask_atlas_api_key: str | None
    config_path: Path | None
    raw_config: dict[str, Any]
    default_countries: list[str]
    default_release_lag_days: int
    settings_snapshot: dict[str, Any]


def default_config() -> dict[str, Any]:
    return {
        "seed": 42,
        "data": {
            "countries": ["USA", "CAN", "GBR", "BRA", "DEU", "FRA", "JPN", "IND", "CHN", "MEX"],
            "start_year": 1995,
            "end_year": 2026,
            "release_lag_days": 45,
        },
        "run": {
            "default_horizons": [0, 1, 2, 4, 8],
            "backtest_horizon": 1,
            "backtest_min_train_rows": 8,
            "backtest_max_origins": 12,
        },
        "world_bank": {
            "target": "NY.GDP.MKTP.KD.ZG",
            "indicators": [
                "NY.GDP.MKTP.KD.ZG",
                "FP.CPI.TOTL.ZG",
                "SP.POP.GROW",
                "NE.GDI.TOTL.ZS",
                "NE.TRD.GNFS.ZS",
                "SL.UEM.TOTL.ZS",
            ],
        },
        "ensemble": {"quantiles": [0.1, 0.5, 0.9]},
        "fred": {"series": ["VIXCLS", "DFF", "DGS10"]},
    }


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _config_path_from_env() -> Path | None:
    raw = os.getenv("ATLAS_GDP_CONFIG")
    return Path(raw).resolve() if raw else None


def _load_yaml_payload(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("ATLAS_GDP_CONFIG must point to a mapping-style YAML file")
    return payload


def _yaml_runtime_values() -> dict[str, Any]:
    path = _config_path_from_env()
    payload = _load_yaml_payload(path)
    if not payload:
        return {}
    values: dict[str, Any] = {}
    mapping = {
        "offline_mode": "offline_mode",
        "artifact_root": "artifact_root",
        "db_url": "db_url",
        "fred_api_key": "fred_api_key",
        "bea_api_key": "bea_api_key",
        "ask_atlas_enabled": "ask_atlas_enabled",
        "ask_atlas_api_key": "ask_atlas_api_key",
        "seed": "seed",
    }
    for key, field_name in mapping.items():
        if key in payload:
            values[field_name] = payload[key]
        elif key.upper() in payload:
            values[field_name] = payload[key.upper()]
    data_cfg = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
    if "countries" in data_cfg:
        values["default_countries"] = data_cfg["countries"]
    if "release_lag_days" in data_cfg:
        values["default_release_lag_days"] = data_cfg["release_lag_days"]
    return values


if _HAS_PYDANTIC_SETTINGS:
    class _EnvSettings(BaseSettings):
        model_config = SettingsConfigDict(extra="ignore", populate_by_name=True)

        root: Path = Field(default=Path("."), alias="ATLAS_GDP_ROOT")
        artifact_root: Path | None = Field(default=None, alias="ARTIFACT_ROOT")
        offline_mode: bool = Field(default=False, alias="OFFLINE_MODE")
        db_url: str | None = Field(default=None, alias="DB_URL")
        fred_api_key: str | None = Field(default=None, alias="FRED_API_KEY")
        bea_api_key: str | None = Field(default=None, alias="BEA_API_KEY")
        ask_atlas_enabled: bool = Field(default=False, alias="ASK_ATLAS_ENABLED")
        ask_atlas_api_key: str | None = Field(default=None, alias="ASK_ATLAS_API_KEY")
        seed: int = Field(default=int(default_config()["seed"]), alias="ATLAS_GDP_SEED")
        default_countries: list[str] = Field(default_factory=lambda: list(default_config()["data"]["countries"]), alias="DEFAULT_COUNTRIES")
        default_release_lag_days: int = Field(default=int(default_config()["data"]["release_lag_days"]), alias="DEFAULT_RELEASE_LAG_DAYS")
else:
    class _EnvSettings:  # pragma: no cover - exercised here without dependency
        def __init__(self) -> None:
            yaml_values = _yaml_runtime_values()
            defaults = default_config()
            self.root = Path(os.getenv("ATLAS_GDP_ROOT", str(yaml_values.get("root", ".")))).resolve()
            raw_artifact_root = os.getenv("ARTIFACT_ROOT")
            if raw_artifact_root is None:
                raw_artifact_root = yaml_values.get("artifact_root")
            self.artifact_root = Path(raw_artifact_root).resolve() if raw_artifact_root else None
            self.offline_mode = _as_bool(os.getenv("OFFLINE_MODE"), bool(yaml_values.get("offline_mode", False)))
            self.db_url = os.getenv("DB_URL", str(yaml_values.get("db_url") or ""))
            self.fred_api_key = os.getenv("FRED_API_KEY", yaml_values.get("fred_api_key"))
            self.bea_api_key = os.getenv("BEA_API_KEY", yaml_values.get("bea_api_key"))
            self.ask_atlas_enabled = _as_bool(os.getenv("ASK_ATLAS_ENABLED"), bool(yaml_values.get("ask_atlas_enabled", False)))
            self.ask_atlas_api_key = os.getenv("ASK_ATLAS_API_KEY", yaml_values.get("ask_atlas_api_key"))
            self.seed = int(os.getenv("ATLAS_GDP_SEED", str(yaml_values.get("seed", defaults["seed"]))))
            self.default_countries = _parse_country_list(
                os.getenv("DEFAULT_COUNTRIES"),
                yaml_values.get("default_countries", defaults["data"]["countries"]),
            )
            self.default_release_lag_days = int(
                os.getenv(
                    "DEFAULT_RELEASE_LAG_DAYS",
                    str(yaml_values.get("default_release_lag_days", defaults["data"]["release_lag_days"])),
                )
            )


def _parse_country_list(raw: str | None, fallback: Any) -> list[str]:
    if raw is None:
        return list(fallback) if isinstance(fallback, list) else [str(fallback)]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_paths(root: Path, artifact_root: Path) -> Paths:
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    samples = root / "data" / "samples"
    reports = artifact_root / "reports"
    return Paths(root=root, raw=raw, processed=processed, samples=samples, artifacts=artifact_root, reports=reports)


def load_settings() -> Settings:
    config_path = _config_path_from_env()
    yaml_payload = _load_yaml_payload(config_path)
    runtime_yaml = _yaml_runtime_values()
    raw_config = _merge_dict(default_config(), yaml_payload)
    env_settings = _EnvSettings()
    resolved_countries = _parse_country_list(
        os.getenv("DEFAULT_COUNTRIES"),
        runtime_yaml.get("default_countries", getattr(env_settings, "default_countries", [])),
    )
    resolved_release_lag = int(
        os.getenv(
            "DEFAULT_RELEASE_LAG_DAYS",
            str(runtime_yaml.get("default_release_lag_days", getattr(env_settings, "default_release_lag_days", raw_config["data"]["release_lag_days"]))),
        )
    )

    seed_value = int(os.getenv("ATLAS_GDP_SEED", str(runtime_yaml.get("seed", env_settings.seed))))
    raw_config["seed"] = seed_value
    raw_config.setdefault("data", {})
    raw_config["data"]["countries"] = list(resolved_countries)
    raw_config["data"]["release_lag_days"] = int(resolved_release_lag)

    root = Path(os.getenv("ATLAS_GDP_ROOT", str(runtime_yaml.get("root", env_settings.root)))).resolve()
    artifact_root = Path(
        os.getenv(
            "ARTIFACT_ROOT",
            str(runtime_yaml.get("artifact_root", env_settings.artifact_root or (root / "artifacts"))),
        )
    ).resolve()
    paths = _build_paths(root, artifact_root)
    paths.ensure()

    db_url = os.getenv("DB_URL", str(runtime_yaml.get("db_url", env_settings.db_url or f"sqlite:///{artifact_root / 'atlas_gdp.db'}")))
    offline_mode = _as_bool(os.getenv("OFFLINE_MODE"), bool(runtime_yaml.get("offline_mode", env_settings.offline_mode)))
    ask_atlas_enabled = _as_bool(
        os.getenv("ASK_ATLAS_ENABLED"),
        bool(runtime_yaml.get("ask_atlas_enabled", env_settings.ask_atlas_enabled)),
    )
    fred_api_key = os.getenv("FRED_API_KEY", runtime_yaml.get("fred_api_key", env_settings.fred_api_key))
    bea_api_key = os.getenv("BEA_API_KEY", runtime_yaml.get("bea_api_key", env_settings.bea_api_key))
    ask_atlas_api_key = os.getenv("ASK_ATLAS_API_KEY", runtime_yaml.get("ask_atlas_api_key", env_settings.ask_atlas_api_key))
    snapshot = {
        "offline_mode": bool(offline_mode),
        "root": str(root),
        "artifact_root": str(artifact_root),
        "db_url": str(db_url),
        "default_countries": list(resolved_countries),
        "default_release_lag_days": int(resolved_release_lag),
        "config_path": str(config_path) if config_path else None,
        "pydantic_settings_available": bool(_HAS_PYDANTIC_SETTINGS),
    }
    return Settings(
        offline_mode=bool(offline_mode),
        seed=int(seed_value),
        paths=paths,
        db_url=str(db_url),
        fred_api_key=fred_api_key,
        bea_api_key=bea_api_key,
        ask_atlas_enabled=bool(ask_atlas_enabled),
        ask_atlas_api_key=ask_atlas_api_key,
        config_path=config_path,
        raw_config=raw_config,
        default_countries=list(resolved_countries),
        default_release_lag_days=int(resolved_release_lag),
        settings_snapshot=snapshot,
    )
