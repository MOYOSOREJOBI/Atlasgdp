from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from atlas_gdp.data.sdmx_connector import (
    ConnectorLoadResult,
    ConnectorUnavailableError,
    cache_paths,
    load_cached_connector,
    merge_series_payload,
)


OECD_DATASET = "quarterly_macro"
OECD_SERIES_URLS = {
    "industrial_production": "ATLAS_OECD_INDUSTRIAL_PRODUCTION_URL",
    "retail_sales": "ATLAS_OECD_RETAIL_SALES_URL",
    "pmi": "ATLAS_OECD_PMI_URL",
}
OECD_EMPTY_COLUMNS = ["country", "date", "industrial_production", "retail_sales", "pmi"]


def _message_to_records(message: Any) -> list[dict[str, object]]:
    frame = message.to_pandas()
    if isinstance(frame, dict):
        parts = []
        for key, value in frame.items():
            chunk = value.reset_index(name="value") if isinstance(value, pd.Series) else value.reset_index()
            chunk["series_key"] = str(key)
            parts.append(chunk)
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    elif isinstance(frame, pd.Series):
        frame = frame.reset_index(name="value")
    else:
        frame = frame.reset_index()
    if frame.empty:
        return []
    frame.columns = [str(col).lower() for col in frame.columns]
    value_col = "value" if "value" in frame.columns else str(frame.columns[-1])
    date_col = next((col for col in frame.columns if "time" in col or col in {"period", "date"}), None)
    country_col = next((col for col in frame.columns if any(token in col for token in ["country", "ref_area", "geo", "location", "area"])), None)
    if date_col is None:
        raise ValueError("Unable to locate a time dimension in the OECD SDMX response")
    normalized = pd.DataFrame(
        {
            "country": frame[country_col].astype(str).str.upper().str[:3] if country_col else "ALL",
            "date": pd.to_datetime(frame[date_col], errors="coerce"),
            "value": pd.to_numeric(frame[value_col], errors="coerce"),
        }
    ).dropna(subset=["date", "value"])
    normalized["date"] = normalized["date"] + pd.offsets.QuarterEnd(0)
    return normalized.to_dict(orient="records")


def _fetch_oecd_series(url: str) -> list[dict[str, object]]:
    try:
        import pandasdmx as sdmx
    except ImportError as exc:
        raise ConnectorUnavailableError(
            "missing_dependency",
            "pandaSDMX is required for live OECD SDMX pulls. Install project dependencies.",
        ) from exc
    if hasattr(sdmx, "api") and hasattr(sdmx.api, "read_url"):
        message = sdmx.api.read_url(url)
    elif hasattr(sdmx, "read_url"):
        message = sdmx.read_url(url)
    else:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        raise ConnectorUnavailableError(
            "missing_dependency",
            "Installed pandaSDMX build does not expose a URL reader for live OECD pulls.",
        )
    return _message_to_records(message)


def _fetch_oecd_payload() -> dict[str, Any]:
    series: dict[str, list[dict[str, object]]] = {}
    configured = False
    for name, env_var in OECD_SERIES_URLS.items():
        url = os.getenv(env_var)
        if not url:
            continue
        configured = True
        series[name] = _fetch_oecd_series(url)
    if not configured:
        raise ConnectorUnavailableError(
            "not_configured",
            "OECD SDMX URLs are not configured. Set ATLAS_OECD_INDUSTRIAL_PRODUCTION_URL, "
            "ATLAS_OECD_RETAIL_SALES_URL, and/or ATLAS_OECD_PMI_URL."
        )
    return {"source": "oecd", "dataset": OECD_DATASET, "series": series}


def _parse_oecd_payload(payload: dict[str, Any]) -> pd.DataFrame:
    return merge_series_payload(
        payload,
        {
            "industrial_production": "industrial_production",
            "retail_sales": "retail_sales",
            "pmi": "pmi",
        },
        empty_columns=OECD_EMPTY_COLUMNS,
    )


def load_oecd_quarterly(raw_dir: str | os.PathLike[str], offline_mode: bool = False) -> ConnectorLoadResult:
    legacy_file = Path(raw_dir) / "oecd_quarterly.csv"
    if legacy_file.exists():
        frame = pd.read_csv(legacy_file, parse_dates=["date"])
        return ConnectorLoadResult(frame=frame, status="offline_legacy_csv" if offline_mode else "online_legacy_csv")
    cache_key = "|".join(f"{name}={os.getenv(env_var, '')}" for name, env_var in sorted(OECD_SERIES_URLS.items()))
    return load_cached_connector(
        raw_dir=raw_dir,
        source="oecd",
        dataset=OECD_DATASET,
        cache_key=cache_key,
        offline_mode=offline_mode,
        parser=_parse_oecd_payload,
        fetcher=_fetch_oecd_payload,
    )


__all__ = ["OECD_DATASET", "OECD_EMPTY_COLUMNS", "OECD_SERIES_URLS", "load_oecd_quarterly", "cache_paths"]
