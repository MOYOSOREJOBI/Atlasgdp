from __future__ import annotations

import pandas as pd


def asof_filter(df: pd.DataFrame, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of_date)
    return df[df["date"] <= as_of].copy()


def release_calendar_simulator(df: pd.DataFrame, lag_days: int = 45) -> pd.DataFrame:
    out = df.copy()
    out["available_date"] = out["date"] + pd.to_timedelta(lag_days, unit="D")
    return out


def asof_release_filter(df: pd.DataFrame, as_of_date: str | pd.Timestamp) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of_date)
    if "available_date" not in df.columns:
        return asof_filter(df, as_of)
    return df[df["available_date"] <= as_of].copy()
