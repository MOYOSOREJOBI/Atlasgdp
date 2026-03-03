from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


class ConnectorUnavailableError(RuntimeError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ConnectorLoadResult:
    frame: pd.DataFrame
    status: str
    note: str | None = None


def cache_paths(raw_dir: str | Path, source: str, dataset: str, cache_key: str) -> tuple[Path, Path]:
    root = Path(raw_dir) / source / dataset
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    return root / f"{digest}.json", root / f"{digest}.parquet"


def load_cached_connector(
    *,
    raw_dir: str | Path,
    source: str,
    dataset: str,
    cache_key: str,
    offline_mode: bool,
    parser: Callable[[dict[str, Any]], pd.DataFrame],
    fetcher: Callable[[], dict[str, Any]],
) -> ConnectorLoadResult:
    raw_path, parquet_path = cache_paths(raw_dir, source, dataset, cache_key)
    if parquet_path.exists():
        return ConnectorLoadResult(
            frame=pd.read_parquet(parquet_path),
            status="offline_cache_hit" if offline_mode else "online_cache_hit",
        )
    if raw_path.exists():
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        frame = parser(payload)
        frame.to_parquet(parquet_path, index=False)
        return ConnectorLoadResult(
            frame=frame,
            status="offline_cache_hit" if offline_mode else "online_cache_hit",
        )
    if offline_mode:
        raise ConnectorUnavailableError(
            "offline_cache_missing",
            f"{source.upper()} cache is missing in offline mode. Run once online to populate cache.",
        )
    payload = fetcher()
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    frame = parser(payload)
    frame.to_parquet(parquet_path, index=False)
    return ConnectorLoadResult(frame=frame, status="online_cached")


def merge_series_payload(
    payload: dict[str, Any],
    rename_map: dict[str, str],
    *,
    empty_columns: list[str],
) -> pd.DataFrame:
    series_payload = payload.get("series", {})
    if not series_payload:
        return pd.DataFrame(columns=empty_columns)
    merged: pd.DataFrame | None = None
    for raw_name, output_name in rename_map.items():
        records = series_payload.get(raw_name, [])
        frame = pd.DataFrame(records)
        if frame.empty:
            continue
        frame["date"] = pd.to_datetime(frame["date"]) + pd.offsets.QuarterEnd(0)
        frame = frame.rename(columns={"value": output_name})
        keep_cols = ["country", "date", output_name]
        frame = frame[keep_cols]
        merged = frame if merged is None else merged.merge(frame, on=["country", "date"], how="outer")
    if merged is None:
        return pd.DataFrame(columns=empty_columns)
    for column in empty_columns:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged[empty_columns].sort_values(["country", "date"]).reset_index(drop=True)
