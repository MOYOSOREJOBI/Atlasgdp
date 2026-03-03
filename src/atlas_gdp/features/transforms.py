from __future__ import annotations

import numpy as np
import pandas as pd


def annual_log_growth(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return 100.0 * (np.log(s) - np.log(s.shift(1)))


def quarterly_qoq_saar(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return 400.0 * (np.log(s) - np.log(s.shift(1)))


def winsorize_series(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def add_lags(df: pd.DataFrame, columns: list[str], lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby("country")[col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, columns: list[str], windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        for window in windows:
            grouped = out.groupby("country")[col]
            out[f"{col}_rollmean_{window}"] = grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
            out[f"{col}_rollstd_{window}"] = grouped.transform(lambda s: s.rolling(window, min_periods=1).std()).fillna(0.0)
            out[f"{col}_momentum_{window}"] = grouped.transform(lambda s: s - s.shift(window))
    return out
