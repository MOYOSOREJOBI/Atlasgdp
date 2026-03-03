from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from atlas_gdp.data.cache import cache_path, read_cache, write_cache
from atlas_gdp.data.schemas import PANEL_COLUMNS


WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"


def _offline_panel(countries: list[str], start: int, end: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for country in countries:
        base = (abs(hash(country)) % 300) / 100.0 - 1.5
        for year in range(start, end + 1):
            cycle = ((year - start) % 7) - 3
            rows.append(
                {
                    "country": country,
                    "date": pd.Timestamp(f"{year}-12-31"),
                    "frequency": "A",
                    "gdp_growth": base + cycle * 0.4,
                    "inflation": 1.5 + (cycle * 0.3),
                    "population_growth": 0.4 + ((year + len(country)) % 5) * 0.15,
                    "investment_share": 20 + ((year + 2) % 6) * 1.2,
                    "trade_share": 45 + ((year + 3) % 9) * 1.8,
                    "unemployment": 4.5 + ((year + 1) % 8) * 0.5,
                }
            )
    return pd.DataFrame(rows, columns=PANEL_COLUMNS)


def _fetch_indicator(raw_dir: Path, country: str, indicator: str) -> list[dict[str, Any]]:
    url = WB_URL.format(country=country, indicator=indicator)
    key = f"{url}|per_page=20000|format=json"
    path = cache_path(raw_dir, "world_bank", key)
    cached = read_cache(path)
    if cached is not None:
        return cached
    response = requests.get(url, params={"per_page": 20000, "format": "json"}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    write_cache(path, payload)
    return payload


def build_world_bank_panel(
    raw_dir: str | Path,
    countries: list[str],
    start: int,
    end: int,
    indicators: list[str],
    offline_mode: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_base = Path(raw_dir)
    if offline_mode:
        sample_path = raw_base.parent / "samples" / "offline_macro.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path, parse_dates=["date"])
            df = df[df["country"].isin(countries)].copy()
            df = df[(df["date"].dt.year >= start) & (df["date"].dt.year <= end)].copy()
            present = set(df["country"].unique())
            missing = [country for country in countries if country not in present]
            if missing:
                df = pd.concat([df, _offline_panel(missing, start, end)], ignore_index=True)
            if df.empty:
                df = _offline_panel(countries, start, end)
        else:
            df = _offline_panel(countries, start, end)
        lineage = {
            "mode": "offline",
            "countries": countries,
            "start": start,
            "end": end,
            "row_count": int(len(df)),
            "generated_at": datetime.now(UTC).isoformat(),
        }
        return df, lineage

    frames: dict[str, pd.DataFrame] = {}
    lineage_calls: list[dict[str, Any]] = []
    for indicator in indicators:
        indicator_rows: list[dict[str, Any]] = []
        for country in countries:
            payload = _fetch_indicator(raw_base, country, indicator)
            observations = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
            for item in observations:
                year_str = str(item.get("date", ""))
                try:
                    year = int(year_str)
                except ValueError:
                    continue
                if year < start or year > end:
                    continue
                value = item.get("value")
                if value is None:
                    continue
                indicator_rows.append(
                    {
                        "country": country,
                        "date": pd.Timestamp(f"{year}-12-31"),
                        indicator: float(value),
                    }
                )
            lineage_calls.append(
                {
                    "country": country,
                    "indicator": indicator,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "row_count": len(observations),
                }
            )
        frames[indicator] = pd.DataFrame(indicator_rows)

    target = indicators[0]
    merged = frames[target].rename(columns={target: "gdp_growth"})
    name_map = {
        "FP.CPI.TOTL.ZG": "inflation",
        "SP.POP.GROW": "population_growth",
        "NE.GDI.TOTL.ZS": "investment_share",
        "NE.TRD.GNFS.ZS": "trade_share",
        "SL.UEM.TOTL.ZS": "unemployment",
    }
    for indicator in indicators[1:]:
        alias = name_map.get(indicator, indicator.lower())
        frame = frames[indicator].rename(columns={indicator: alias})
        merged = merged.merge(frame, on=["country", "date"], how="left")

    merged["frequency"] = "A"
    for col in ["inflation", "population_growth", "investment_share", "trade_share", "unemployment"]:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[PANEL_COLUMNS].sort_values(["country", "date"]).reset_index(drop=True)
    data_hash = hashlib.sha256(pd.util.hash_pandas_object(merged.fillna(0), index=True).values.tobytes()).hexdigest()
    lineage = {
        "mode": "online",
        "countries": countries,
        "indicators": indicators,
        "row_count": int(len(merged)),
        "missingness": {col: float(merged[col].isna().mean()) for col in merged.columns if col not in {"country", "date", "frequency"}},
        "calls": lineage_calls,
        "data_hash": data_hash,
    }
    return merged, lineage
