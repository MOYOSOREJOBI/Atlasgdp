from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA


def _annual_to_quarterly(annual_df: pd.DataFrame) -> pd.DataFrame:
    if annual_df.empty:
        return annual_df.copy()
    quarter_offsets = {
        1: {"month": 3, "gdp_growth": -0.12},
        2: {"month": 6, "gdp_growth": 0.00},
        3: {"month": 9, "gdp_growth": 0.08},
        4: {"month": 12, "gdp_growth": 0.04},
    }
    rows: list[dict[str, object]] = []
    for _, row in annual_df.iterrows():
        base = row.to_dict()
        year = pd.Timestamp(base["date"]).year
        for quarter, meta in quarter_offsets.items():
            payload = dict(base)
            payload["date"] = pd.Timestamp(year=year, month=int(meta["month"]), day=1) + pd.offsets.QuarterEnd(0)
            payload["frequency"] = "Q"
            if "gdp_growth" in payload and pd.notna(payload["gdp_growth"]):
                payload["gdp_growth"] = float(payload["gdp_growth"]) + float(meta["gdp_growth"])
            rows.append(payload)
    out = pd.DataFrame(rows)
    return out.sort_values(["country", "date"]).reset_index(drop=True)


def _prepare_quarterly_frame(quarterly_df: pd.DataFrame) -> pd.DataFrame:
    q = quarterly_df.copy()
    q["date"] = pd.to_datetime(q["date"]) + pd.offsets.QuarterEnd(0)
    q = q.sort_values(["country", "date"]).reset_index(drop=True)
    return q


def _prepare_monthly_frame(monthly_df: pd.DataFrame, as_of: str | pd.Timestamp | None = None) -> tuple[pd.DataFrame, list[str]]:
    monthly = monthly_df.copy()
    if monthly.empty:
        return monthly, []
    monthly["date"] = pd.to_datetime(monthly["date"])
    if "available_date" in monthly.columns:
        monthly["available_date"] = pd.to_datetime(monthly["available_date"])
    monthly["month_end"] = monthly["date"].dt.to_period("M").dt.end_time.dt.normalize()
    metadata = {"country", "date", "available_date", "frequency", "month_end"}
    indicator_cols = [
        col
        for col in monthly.columns
        if col not in metadata and pd.api.types.is_numeric_dtype(monthly[col])
    ]
    if not indicator_cols:
        return pd.DataFrame(columns=["country", "month_end", "available_date"]), []
    aggregations: dict[str, str] = {col: "mean" for col in indicator_cols}
    if "available_date" in monthly.columns:
        aggregations["available_date"] = "max"
    monthly = monthly.groupby(["country", "month_end"], as_index=False).agg(aggregations)
    if "available_date" not in monthly.columns:
        monthly["available_date"] = monthly["month_end"] + pd.Timedelta(days=15)
    monthly = monthly.sort_values(["country", "month_end"]).reset_index(drop=True)
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of)
        monthly = monthly.loc[monthly["available_date"] <= as_of_ts].copy()
    return monthly.reset_index(drop=True), indicator_cols


def _quarterly_monthly_aggregates(monthly: pd.DataFrame, indicator_cols: list[str]) -> pd.DataFrame:
    if monthly.empty or not indicator_cols:
        return pd.DataFrame(columns=["country", "date"])
    working = monthly.copy()
    working["date"] = working["month_end"].dt.to_period("Q").dt.end_time.dt.normalize()
    rename_map = {col: f"monthly_agg_{col}" for col in indicator_cols}
    aggregated = working.groupby(["country", "date"], as_index=False)[indicator_cols].mean(numeric_only=True)
    return aggregated.rename(columns=rename_map)


def _quarterly_midas_lags(monthly: pd.DataFrame, quarter_index: pd.DataFrame, indicator_cols: list[str], lag_count: int = 2) -> pd.DataFrame:
    if monthly.empty or quarter_index.empty or not indicator_cols:
        return pd.DataFrame(columns=["country", "date"])
    rows: list[dict[str, object]] = []
    for country, q_group in quarter_index.groupby("country", sort=False):
        country_months = monthly.loc[monthly["country"] == country].sort_values("month_end").reset_index(drop=True)
        if country_months.empty:
            continue
        for quarter_end in q_group["date"].tolist():
            eligible = country_months.loc[country_months["month_end"] <= quarter_end].tail(lag_count + 1)
            payload: dict[str, object] = {"country": country, "date": quarter_end}
            for indicator in indicator_cols:
                values = eligible[indicator].tolist()
                for lag in range(lag_count + 1):
                    idx = len(values) - 1 - lag
                    payload[f"{indicator}_m{lag}"] = values[idx] if idx >= 0 else pd.NA
            rows.append(payload)
    return pd.DataFrame(rows)


def _quarterly_dfm_factors(monthly: pd.DataFrame, indicator_cols: list[str], n_factors: int = 2) -> pd.DataFrame:
    if monthly.empty or not indicator_cols:
        return pd.DataFrame(columns=["country", "date"])
    filled = monthly[indicator_cols].copy()
    filled = filled.fillna(filled.mean(numeric_only=True)).fillna(0.0)
    centered = filled - filled.mean()
    scale = filled.std(ddof=0).replace(0.0, 1.0)
    standardized = centered.divide(scale, axis="columns")
    n_components = min(n_factors, standardized.shape[0], standardized.shape[1])
    if n_components <= 0:
        return pd.DataFrame(columns=["country", "date"])
    pca = PCA(n_components=n_components)
    factors = pca.fit_transform(standardized)
    factor_cols = [f"dfm_factor_{idx + 1}" for idx in range(n_components)]
    monthly_factors = monthly[["country", "month_end"]].copy()
    for idx, col in enumerate(factor_cols):
        monthly_factors[col] = factors[:, idx]
    monthly_factors["date"] = monthly_factors["month_end"].dt.to_period("Q").dt.end_time.dt.normalize()
    return monthly_factors.groupby(["country", "date"], as_index=False)[factor_cols].mean(numeric_only=True)


def align_mixed_frequency(
    annual_df: pd.DataFrame,
    quarterly_df: pd.DataFrame | None = None,
    monthly_df: pd.DataFrame | None = None,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    base = _annual_to_quarterly(annual_df)
    if quarterly_df is not None and not quarterly_df.empty:
        q = _prepare_quarterly_frame(quarterly_df)
        base = base.merge(q, on=["country", "date"], how="left")
    if monthly_df is not None and not monthly_df.empty:
        monthly, indicator_cols = _prepare_monthly_frame(monthly_df, as_of=as_of)
        monthly_agg = _quarterly_monthly_aggregates(monthly, indicator_cols)
        monthly_lags = _quarterly_midas_lags(monthly, base[["country", "date"]], indicator_cols)
        dfm_factors = _quarterly_dfm_factors(monthly, indicator_cols)
        month_counts = monthly.assign(date=monthly["month_end"].dt.to_period("Q").dt.end_time.dt.normalize())
        month_counts = month_counts.groupby(["country", "date"]).size().reset_index(name="month_count")
        for frame in [monthly_agg, monthly_lags, dfm_factors, month_counts]:
            if not frame.empty:
                base = base.merge(frame, on=["country", "date"], how="left")
    return base.sort_values(["country", "date"]).reset_index(drop=True)
