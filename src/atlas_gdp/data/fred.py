from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from atlas_gdp.data.cache import cache_path, read_cache, write_cache


FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_series(raw_dir: str | Path, series_id: str, api_key: str | None) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame(columns=["date", series_id])
    key = f"{series_id}|{api_key}"
    path = cache_path(Path(raw_dir), "fred", key)
    cached = read_cache(path)
    if cached is None:
        response = requests.get(
            FRED_URL,
            params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
            timeout=60,
        )
        response.raise_for_status()
        cached = response.json()
        write_cache(path, cached)
    rows = []
    for item in cached.get("observations", []):
        value = item.get("value")
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        rows.append({"date": pd.Timestamp(item["date"]), series_id: val})
    return pd.DataFrame(rows)
