from __future__ import annotations

import math
import re
import subprocess
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from atlas_gdp.config import Settings, load_settings
from atlas_gdp.data.bea_demo import load_bea_demo_components
from atlas_gdp.data.fred import fetch_fred_series
from atlas_gdp.data.imf_sdmx import load_imf_quarterly
from atlas_gdp.data.oecd_sdmx import load_oecd_quarterly
from atlas_gdp.data.sdmx_connector import ConnectorUnavailableError
from atlas_gdp.data.wb import build_world_bank_panel
from atlas_gdp.evaluation.backtest import summarize_prediction_frame
from atlas_gdp.features.asof import asof_release_filter, release_calendar_simulator
from atlas_gdp.features.mixed_frequency import align_mixed_frequency
from atlas_gdp.features.transforms import add_lags, add_rolling_features
from atlas_gdp.models.baseline_panel import predict_baseline, train_baseline_panel
from atlas_gdp.models.bridge_bottomup import predict_bridge, train_bridge
from atlas_gdp.models.bvar import predict_bvar, train_bvar
from atlas_gdp.models.dfm_kalman import predict_dfm, train_dfm
from atlas_gdp.models.ensemble import predict_ensemble, train_ensemble
from atlas_gdp.models.midas import predict_midas, train_midas
from atlas_gdp.logging_utils import get_logger, log_event
from atlas_gdp.pipeline.storage import (
    copy_tree_contents,
    ensure_run_dirs,
    make_run_id,
    read_json,
    read_latest_pointer,
    record_run,
    utc_now,
    write_json,
    write_latest_pointer,
)
from atlas_gdp.reporting.drivers_report import current_drivers, delta_decomposition


BASE_FEATURE_COLUMNS = [
    "country",
    "inflation",
    "population_growth",
    "investment_share",
    "trade_share",
    "unemployment",
    "gdp_growth_lag1",
    "gdp_growth_lag2",
    "gdp_growth_lag3",
    "inflation_lag1",
    "trade_share_lag1",
    "gdp_growth_rollmean_2",
    "gdp_growth_rollstd_2",
    "inflation_rollmean_2",
    "year_trend",
    "regime_flag",
]

METHOD_NOTES = (
    "Quarterly GDP rows are synthesized at quarter-end from annual World Bank series. "
    "Monthly inputs are aligned at quarter frequency using quarter means from months available as-of. "
    "MIDAS is approximated with explicit monthly lag features plus exponential Almon weights. "
    "DFM is approximated with PCA factors fit on standardized monthly indicators at monthly frequency, then averaged to quarter."
)


def _load_fred_panel(settings: Settings) -> pd.DataFrame:
    series_ids = settings.raw_config.get("fred", {}).get("series", [])
    merged: pd.DataFrame | None = None
    for series_id in series_ids:
        frame = fetch_fred_series(settings.paths.raw, series_id, settings.fred_api_key)
        if frame.empty:
            continue
        merged = frame if merged is None else merged.merge(frame, on="date", how="outer")
    if merged is None or merged.empty:
        return pd.DataFrame()
    merged = merged.sort_values("date").reset_index(drop=True)
    if {"DGS10", "DFF"}.issubset(merged.columns):
        merged["term_spread"] = merged["DGS10"] - merged["DFF"]
    merged["country"] = "USA"
    return merged


def _attach_source_lineage(lineage: dict[str, Any], name: str, row_count: int, enabled: bool, note: str | None = None) -> None:
    sources = lineage.setdefault("sources", [])
    payload = {"name": name, "row_count": int(row_count), "enabled": bool(enabled)}
    if note:
        payload["note"] = note
    sources.append(payload)


def build_dataset(
    settings: Settings | None = None,
    countries: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    settings = settings or load_settings()
    data_cfg = settings.raw_config.get("data", {})
    use_countries = countries or list(data_cfg.get("countries", []))
    start = int(start_year or data_cfg.get("start_year", 1995))
    end = int(end_year or data_cfg.get("end_year", utc_now().year))

    wb_df, lineage = build_world_bank_panel(
        raw_dir=settings.paths.raw,
        countries=use_countries,
        start=start,
        end=end,
        indicators=settings.raw_config["world_bank"]["indicators"],
        offline_mode=settings.offline_mode,
    )
    _attach_source_lineage(lineage, "world_bank", len(wb_df), True, "offline sample" if settings.offline_mode else "live API")
    connector_status: dict[str, str] = {
        "world_bank": "offline_sample" if settings.offline_mode else "online_api",
        "fred": "disabled",
    }

    try:
        oecd_result = load_oecd_quarterly(settings.paths.raw, offline_mode=settings.offline_mode)
        oecd = oecd_result.frame
        connector_status["oecd"] = oecd_result.status
        oecd_note = oecd_result.note or f"real SDMX connector ({oecd_result.status})"
    except ConnectorUnavailableError as exc:
        oecd = pd.DataFrame(columns=["country", "date", "industrial_production", "retail_sales", "pmi"])
        connector_status["oecd"] = exc.status
        oecd_note = str(exc)
    except Exception as exc:
        oecd = pd.DataFrame(columns=["country", "date", "industrial_production", "retail_sales", "pmi"])
        connector_status["oecd"] = "error"
        oecd_note = f"OECD connector failed: {exc}"

    try:
        imf_result = load_imf_quarterly(settings.paths.raw, offline_mode=settings.offline_mode)
        imf = imf_result.frame
        connector_status["imf"] = imf_result.status
        imf_note = imf_result.note or f"real SDMX connector ({imf_result.status})"
    except ConnectorUnavailableError as exc:
        imf = pd.DataFrame(columns=["country", "date", "credit_gap", "current_account"])
        connector_status["imf"] = exc.status
        imf_note = str(exc)
    except Exception as exc:
        imf = pd.DataFrame(columns=["country", "date", "credit_gap", "current_account"])
        connector_status["imf"] = "error"
        imf_note = f"IMF connector failed: {exc}"

    bea_result = load_bea_demo_components(settings.paths.raw)
    bea = bea_result.frame
    connector_status["bea_demo"] = bea_result.status

    fred = _load_fred_panel(settings)
    if not fred.empty:
        connector_status["fred"] = "online_cached" if not settings.offline_mode else "offline_cache_hit"

    _attach_source_lineage(lineage, "oecd", len(oecd), not oecd.empty, oecd_note)
    _attach_source_lineage(lineage, "imf", len(imf), not imf.empty, imf_note)
    _attach_source_lineage(lineage, "fred", len(fred), not fred.empty, "optional monthly enrichment; requires FRED_API_KEY")
    _attach_source_lineage(lineage, "bea_demo", len(bea), not bea.empty, bea_result.note or "demo-only local component bridge")
    lineage["connector_status"] = connector_status

    if not bea.empty:
        bea["country"] = "USA"
    quarterly = None
    if not oecd.empty or not imf.empty:
        quarterly = oecd.merge(imf, on=["country", "date"], how="outer")

    panel = align_mixed_frequency(wb_df, quarterly_df=quarterly, monthly_df=fred)
    wb_df.to_parquet(settings.paths.processed / "annual_panel.parquet", index=False)
    (quarterly if quarterly is not None else pd.DataFrame()).to_parquet(settings.paths.processed / "quarterly_features.parquet", index=False)
    panel.to_parquet(settings.paths.processed / "panel.parquet", index=False)
    bea.to_parquet(settings.paths.processed / "bea_components.parquet", index=False)
    fred.to_parquet(settings.paths.processed / "fred_features.parquet", index=False)
    write_json(settings.paths.artifacts / "data_lineage.json", lineage)
    return panel, bea, fred, lineage


def _load_processed_inputs(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel_path = settings.paths.processed / "panel.parquet"
    if not panel_path.exists():
        build_dataset(settings=settings)
    panel = pd.read_parquet(settings.paths.processed / "panel.parquet")
    bea_path = settings.paths.processed / "bea_components.parquet"
    bea = pd.read_parquet(bea_path) if bea_path.exists() else pd.DataFrame()
    lineage_path = settings.paths.artifacts / "data_lineage.json"
    lineage = read_json(lineage_path) if lineage_path.exists() else {"sources": []}
    return panel, bea, lineage


def _engineer_panel(panel: pd.DataFrame, bea: pd.DataFrame, settings: Settings | None = None) -> pd.DataFrame:
    settings = settings or load_settings()
    engineered = release_calendar_simulator(panel, lag_days=settings.default_release_lag_days)
    engineered = engineered.sort_values(["country", "date"]).reset_index(drop=True)
    engineered = add_lags(engineered, ["gdp_growth", "inflation", "trade_share"], [1, 2, 3])
    engineered = add_rolling_features(engineered, ["gdp_growth", "inflation"], [2, 3])
    engineered["year_trend"] = engineered["date"].dt.year - engineered["date"].dt.year.min()
    unemployment_roll = engineered.groupby("country")["unemployment"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    engineered["regime_flag"] = (engineered["unemployment"].fillna(0.0) > unemployment_roll.fillna(0.0)).astype(int)
    grouped = engineered.groupby("country", sort=False)
    engineered["target"] = grouped["gdp_growth"].shift(-1)
    engineered["target_date"] = grouped["date"].shift(-1)
    engineered["target_available_date"] = grouped["available_date"].shift(-1)
    engineered["target_source_country"] = grouped["country"].shift(-1)
    target_mask = engineered["target"].notna()
    if not engineered.loc[target_mask, "target_source_country"].eq(engineered.loc[target_mask, "country"]).all():
        raise ValueError("Target leakage detected: target source country differs from current row country")
    engineered = engineered.dropna(subset=["target", "target_date", "target_available_date"]).reset_index(drop=True)
    engineered = engineered.drop(columns=["target_source_country"])
    if not bea.empty:
        merge_keys = ["country", "date"] if "country" in bea.columns else ["date"]
        engineered = engineered.merge(bea, on=merge_keys, how="left")
    return engineered


def load_engineered_panel(settings: Settings | None = None, as_of: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    settings = settings or load_settings()
    panel, bea, lineage = _load_processed_inputs(settings)
    if as_of is not None:
        annual_path = settings.paths.processed / "annual_panel.parquet"
        quarterly_path = settings.paths.processed / "quarterly_features.parquet"
        monthly_path = settings.paths.processed / "fred_features.parquet"
        if annual_path.exists():
            annual = pd.read_parquet(annual_path)
            quarterly = pd.read_parquet(quarterly_path) if quarterly_path.exists() else pd.DataFrame()
            monthly = pd.read_parquet(monthly_path) if monthly_path.exists() else pd.DataFrame()
            panel = align_mixed_frequency(
                annual,
                quarterly_df=quarterly if not quarterly.empty else None,
                monthly_df=monthly if not monthly.empty else None,
                as_of=as_of,
            )
    full_engineered = _engineer_panel(panel, bea, settings=settings)
    if as_of is None:
        return panel, full_engineered, lineage
    filtered = asof_release_filter(full_engineered, as_of)
    return panel, filtered, lineage


def choose_split_available_date(
    usable: pd.DataFrame,
    as_of: str | pd.Timestamp,
    min_valid_rows: int = 4,
) -> tuple[pd.Timestamp, dict[str, Any]]:
    as_of_ts = pd.Timestamp(as_of)
    eligible = usable.loc[usable["available_date"] <= as_of_ts].copy()
    if eligible.empty:
        raise ValueError("No rows are available on or before the requested as-of date")

    split_candidates = sorted(pd.Timestamp(value) for value in eligible["available_date"].dropna().unique().tolist())
    best_choice: pd.Timestamp | None = None
    best_valid = -1
    for boundary in reversed(split_candidates):
        train_rows = int((eligible["available_date"] <= boundary).sum())
        valid_rows = int(((eligible["available_date"] > boundary) & (eligible["available_date"] <= as_of_ts)).sum())
        if train_rows == 0 or valid_rows == 0:
            continue
        if valid_rows >= min_valid_rows:
            return boundary, {
                "split_available_date": boundary.isoformat(),
                "requested_min_valid_rows": int(min_valid_rows),
                "actual_valid_rows": valid_rows,
                "fallback_used": False,
            }
        if valid_rows > best_valid:
            best_choice = boundary
            best_valid = valid_rows

    if best_choice is None:
        raise ValueError("Unable to create a non-empty forward holdout for the requested as-of date")
    return best_choice, {
        "split_available_date": best_choice.isoformat(),
        "requested_min_valid_rows": int(min_valid_rows),
        "actual_valid_rows": int(best_valid),
        "fallback_used": True,
    }


def split_train_valid(
    usable: pd.DataFrame,
    as_of: str | pd.Timestamp,
    min_valid_rows: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    as_of_ts = pd.Timestamp(as_of)
    eligible = usable.loc[usable["available_date"] <= as_of_ts].copy()
    split_date, _ = choose_split_available_date(eligible, as_of_ts, min_valid_rows=min_valid_rows)
    train_mask = eligible["available_date"] <= split_date
    valid_mask = (eligible["available_date"] > split_date) & (eligible["available_date"] <= as_of_ts)
    train_df = eligible.loc[train_mask].copy()
    valid_df = eligible.loc[valid_mask].copy()
    if train_df.empty:
        raise ValueError("No training rows available before the chosen split boundary")
    if valid_df.empty:
        raise ValueError("No forward holdout rows available for the chosen split boundary")
    return train_df, valid_df


def _feature_columns(usable: pd.DataFrame) -> list[str]:
    feature_columns = list(BASE_FEATURE_COLUMNS)
    dynamic = []
    for col in usable.columns:
        if col.startswith("monthly_agg_") or col.startswith("dfm_factor_"):
            dynamic.append(col)
            continue
        if re.search(r"_m\d+$", col):
            dynamic.append(col)
    feature_columns.extend([col for col in ["VIXCLS", "DFF", "DGS10", "term_spread"] if col in usable.columns])
    feature_columns.extend(sorted(set(dynamic)))
    return feature_columns


def _fit_models(train_df: pd.DataFrame, valid_df: pd.DataFrame, settings: Settings) -> tuple[dict[str, Any], pd.DataFrame]:
    feature_columns = _feature_columns(pd.concat([train_df, valid_df], ignore_index=True))
    baseline = train_baseline_panel(train_df, feature_columns)
    baseline_pred = predict_baseline(baseline, valid_df)

    bridge = train_bridge(train_df)
    bridge_pred = predict_bridge(bridge, valid_df)

    numeric_features = [
        c
        for c in train_df.columns
        if c not in {"country", "date", "frequency", "available_date", "target", "target_date", "target_available_date"}
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    dfm_features = [c for c in numeric_features if c.startswith("dfm_factor_")]
    if not dfm_features:
        dfm_features = numeric_features[: min(6, len(numeric_features))]
    dfm = train_dfm(train_df, dfm_features)
    dfm_pred = predict_dfm(dfm, valid_df)

    midas_cols = sorted(
        [c for c in numeric_features if re.search(r"_m\d+$", c)],
        key=lambda value: (re.sub(r"_m\d+$", "", value), int(value.rsplit("_m", 1)[1])),
    )
    if not midas_cols:
        midas_cols = numeric_features[: min(4, len(numeric_features))]
    midas = train_midas(train_df, midas_cols)
    midas_pred = predict_midas(midas, valid_df)

    bvar_cols = numeric_features[: min(5, len(numeric_features))]
    bvar = train_bvar(train_df, bvar_cols)
    bvar_pred = predict_bvar(bvar, valid_df)

    pred_df = pd.DataFrame(index=valid_df.index)
    pred_df["ridge"] = baseline_pred["ridge"].to_numpy()
    pred_df["hgb"] = baseline_pred["hgb"].to_numpy()
    pred_df["bridge"] = bridge_pred.to_numpy()
    pred_df["dfm"] = dfm_pred["dfm_mean"].to_numpy()
    pred_df["midas"] = midas_pred.to_numpy()
    pred_df["bvar"] = bvar_pred["bvar_mean"].to_numpy()

    ensemble = train_ensemble(pred_df, valid_df["target"], settings.raw_config["ensemble"]["quantiles"])
    ensemble_pred = predict_ensemble(ensemble, pred_df)
    scored = valid_df[["country", "date", "target", "target_date", "available_date", "target_available_date"]].copy()
    scored = scored.rename(columns={"target": "actual"})
    for col in pred_df.columns:
        scored[col] = pred_df[col].to_numpy()
    for col in ensemble_pred.columns:
        scored[col] = ensemble_pred[col].to_numpy()

    bundle = {
        "baseline": baseline,
        "bridge": bridge,
        "dfm": dfm,
        "midas": midas,
        "bvar": bvar,
        "ensemble": ensemble,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "created_utc": utc_now().isoformat(),
        "config": settings.raw_config,
    }
    return bundle, scored


def _train_bundle_from_train_only(train_df: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    feature_columns = _feature_columns(train_df)
    baseline = train_baseline_panel(train_df, feature_columns)
    bridge = train_bridge(train_df)

    numeric_features = [
        c
        for c in train_df.columns
        if c not in {"country", "date", "frequency", "available_date", "target", "target_date", "target_available_date"}
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]
    dfm_features = [c for c in numeric_features if c.startswith("dfm_factor_")]
    if not dfm_features:
        dfm_features = numeric_features[: min(6, len(numeric_features))]
    dfm = train_dfm(train_df, dfm_features)

    midas_cols = sorted(
        [c for c in numeric_features if re.search(r"_m\d+$", c)],
        key=lambda value: (re.sub(r"_m\d+$", "", value), int(value.rsplit("_m", 1)[1])),
    )
    if not midas_cols:
        midas_cols = numeric_features[: min(4, len(numeric_features))]
    midas = train_midas(train_df, midas_cols)

    bvar_cols = numeric_features[: min(5, len(numeric_features))]
    bvar = train_bvar(train_df, bvar_cols)

    pred_df = pd.DataFrame(index=train_df.index)
    pred_df["ridge"] = predict_baseline(baseline, train_df)["ridge"].to_numpy()
    pred_df["hgb"] = predict_baseline(baseline, train_df)["hgb"].to_numpy()
    pred_df["bridge"] = predict_bridge(bridge, train_df).to_numpy()
    pred_df["dfm"] = predict_dfm(dfm, train_df)["dfm_mean"].to_numpy()
    pred_df["midas"] = predict_midas(midas, train_df).to_numpy()
    pred_df["bvar"] = predict_bvar(bvar, train_df)["bvar_mean"].to_numpy()

    ensemble = train_ensemble(pred_df, train_df["target"], settings.raw_config["ensemble"]["quantiles"])
    return {
        "baseline": baseline,
        "bridge": bridge,
        "dfm": dfm,
        "midas": midas,
        "bvar": bvar,
        "ensemble": ensemble,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "created_utc": utc_now().isoformat(),
        "config": settings.raw_config,
    }


def train_for_as_of(settings: Settings | None = None, as_of: str | None = None) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = settings or load_settings()
    if as_of is None:
        raise ValueError("as_of is required")
    panel, usable, _ = load_engineered_panel(settings=settings, as_of=as_of)
    train_df, valid_df = split_train_valid(usable, as_of)
    bundle, scored = _fit_models(train_df, valid_df, settings)
    return bundle, panel, usable, scored


def train_for_as_of_with_report(
    settings: Settings | None = None,
    as_of: str | None = None,
    min_valid_rows: int = 4,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    settings = settings or load_settings()
    if as_of is None:
        raise ValueError("as_of is required")
    panel, usable, _ = load_engineered_panel(settings=settings, as_of=as_of)
    split_date, split_meta = choose_split_available_date(usable, as_of, min_valid_rows=min_valid_rows)
    train_df, valid_df = split_train_valid(usable, as_of, min_valid_rows=min_valid_rows)
    bundle, scored = _fit_models(train_df, valid_df, settings)

    combined = pd.concat(
        [
            train_df.assign(_split="train")[["country", "_split"]],
            valid_df.assign(_split="valid")[["country", "_split"]],
        ],
        ignore_index=True,
    )
    counts_by_country = []
    for country, group in combined.groupby("country", sort=True):
        counts_by_country.append(
            {
                "country": str(country),
                "train_rows": int((group["_split"] == "train").sum()),
                "valid_rows": int((group["_split"] == "valid").sum()),
                "total_rows": int(len(group)),
            }
        )

    report = {
        "as_of": pd.Timestamp(as_of).isoformat(),
        "split_available_date": split_date.isoformat(),
        "requested_min_valid_rows": int(min_valid_rows),
        "fallback_used": bool(split_meta["fallback_used"]),
        "counts": {
            "overall": {
                "total_rows": int(len(train_df) + len(valid_df)),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
            },
            "by_country": counts_by_country,
        },
        "paths": {
            "bundle_path": str(settings.paths.artifacts / "atlas_gdp_bundle.joblib"),
            "forecast_frame_path": str(settings.paths.artifacts / "latest_forecast_frame.parquet"),
            "train_report_path": str(settings.paths.artifacts / "train_report.json"),
        },
        "parameters": {
            "as_of": pd.Timestamp(as_of).isoformat(),
            "split_available_date": split_date.isoformat(),
            "min_valid_rows": int(min_valid_rows),
        },
    }
    return bundle, panel, usable, scored, report


def _quarter_label(value: pd.Timestamp) -> str:
    period = value.to_period("Q")
    return f"{period.year}-Q{period.quarter}"


def build_history(panel: pd.DataFrame, country: str, as_of: str, limit: int = 8) -> list[dict[str, Any]]:
    ordered = panel[(panel["country"] == country) & (panel["date"] <= pd.Timestamp(as_of))].sort_values("date").tail(limit)
    return [{"period": _quarter_label(pd.Timestamp(row["date"])), "value": float(row["gdp_growth"])} for _, row in ordered.iterrows()]


def _make_state_frame(state: dict[str, Any], bundle: dict[str, Any]) -> pd.DataFrame:
    required = set(BASE_FEATURE_COLUMNS)
    required.update(bundle["numeric_features"])
    required.update(bundle["bridge"].features)
    required.update(bundle["midas"].feature_columns)
    required.update(bundle["bvar"].features)
    required.update({"country", "date"})
    row = {col: state.get(col, 0.0) for col in required}
    row["country"] = state.get("country", "UNK")
    row["date"] = pd.Timestamp(state.get("date", pd.Timestamp("today")))
    return pd.DataFrame([row])


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _predict_one_step(bundle: dict[str, Any], state: dict[str, Any]) -> dict[str, float]:
    state_df = _make_state_frame(state, bundle)
    baseline = predict_baseline(bundle["baseline"], state_df)
    bridge = predict_bridge(bundle["bridge"], state_df)
    dfm = predict_dfm(bundle["dfm"], state_df)
    midas = predict_midas(bundle["midas"], state_df)
    bvar = predict_bvar(bundle["bvar"], state_df)

    pred_df = pd.DataFrame(index=state_df.index)
    pred_df["ridge"] = baseline["ridge"].to_numpy()
    pred_df["hgb"] = baseline["hgb"].to_numpy()
    pred_df["bridge"] = bridge.to_numpy()
    pred_df["dfm"] = dfm["dfm_mean"].to_numpy()
    pred_df["midas"] = midas.to_numpy()
    pred_df["bvar"] = bvar["bvar_mean"].to_numpy()
    ensemble = predict_ensemble(bundle["ensemble"], pred_df).iloc[0]

    mean = float(ensemble["point"])
    p10 = float(ensemble.get("q10", mean - 0.6))
    p50 = float(ensemble.get("q50", mean))
    p90 = float(ensemble.get("q90", mean + 0.6))
    p25 = (p10 + p50) / 2.0
    p75 = (p50 + p90) / 2.0
    std = max((p90 - p10) / 2.563, 1e-6)
    shock_prob = _normal_cdf((0.0 - mean) / std)
    return {
        "ridge": float(pred_df.iloc[0]["ridge"]),
        "hgb": float(pred_df.iloc[0]["hgb"]),
        "bridge": float(pred_df.iloc[0]["bridge"]),
        "dfm": float(pred_df.iloc[0]["dfm"]),
        "midas": float(pred_df.iloc[0]["midas"]),
        "bvar": float(pred_df.iloc[0]["bvar"]),
        "mean": mean,
        "p10": p10,
        "p25": float(p25),
        "p50": p50,
        "p75": float(p75),
        "p90": p90,
        "shock_prob": max(0.0, min(1.0, float(shock_prob))),
    }


def _compute_anchors(country_hist: pd.DataFrame) -> dict[str, float]:
    cols = [
        "inflation",
        "population_growth",
        "investment_share",
        "trade_share",
        "unemployment",
        "gdp_growth",
        "VIXCLS",
        "DFF",
        "DGS10",
        "term_spread",
    ]
    anchors: dict[str, float] = {}
    for col in cols:
        if col in country_hist.columns:
            series = country_hist[col].dropna()
            if not series.empty:
                anchors[col] = float(series.tail(4).mean())
    if "term_spread" not in anchors and {"DGS10", "DFF"}.issubset(anchors):
        anchors["term_spread"] = anchors["DGS10"] - anchors["DFF"]
    return anchors


def _apply_scenario_to_state(state: dict[str, Any], scenario: dict[str, float] | None, anchors: dict[str, float], horizon: int) -> dict[str, Any]:
    if not scenario:
        return dict(state)
    scale = 0.85**horizon
    tighten = float(scenario.get("tighten", 0.0)) * scale
    commodity = float(scenario.get("commodity", 0.0)) * scale
    demand = float(scenario.get("demand", 0.0)) * scale
    stressed = dict(state)
    stressed["inflation"] = float(state.get("inflation", anchors.get("inflation", 0.0))) + 0.18 * commodity - 0.06 * tighten + 0.08 * demand
    stressed["investment_share"] = float(state.get("investment_share", anchors.get("investment_share", 0.0))) - 0.60 * tighten - 0.20 * commodity + 0.45 * demand
    stressed["trade_share"] = float(state.get("trade_share", anchors.get("trade_share", 0.0))) - 0.35 * commodity - 0.10 * tighten + 0.20 * demand
    stressed["unemployment"] = max(0.0, float(state.get("unemployment", anchors.get("unemployment", 0.0))) + 0.25 * tighten + 0.10 * commodity - 0.12 * demand)
    if "VIXCLS" in state or "VIXCLS" in anchors:
        stressed["VIXCLS"] = max(0.0, float(state.get("VIXCLS", anchors.get("VIXCLS", 18.0))) + 4.0 * tighten + 2.5 * commodity - 1.5 * demand)
    if "DFF" in state or "DFF" in anchors:
        stressed["DFF"] = max(0.0, float(state.get("DFF", anchors.get("DFF", 2.0))) + 0.35 * tighten - 0.05 * demand)
    base_term = float(state.get("term_spread", anchors.get("term_spread", 1.2)))
    stressed["term_spread"] = base_term - 0.25 * tighten - 0.15 * commodity + 0.20 * demand
    if "DGS10" in state or "DGS10" in anchors:
        base_dgs10 = float(state.get("DGS10", anchors.get("DGS10", stressed.get("DFF", 2.0) + base_term)))
        stressed["DGS10"] = base_dgs10 - 0.05 * tighten - 0.05 * commodity + 0.05 * demand
    return stressed


def _advance_state(state: dict[str, Any], point: float, anchors: dict[str, float]) -> dict[str, Any]:
    next_state = dict(state)
    prev_lag1 = float(state.get("gdp_growth_lag1", state.get("gdp_growth", point)))
    prev_lag2 = float(state.get("gdp_growth_lag2", prev_lag1))
    prev_inflation = float(state.get("inflation", anchors.get("inflation", 0.0)))
    prev_trade = float(state.get("trade_share", anchors.get("trade_share", 0.0)))
    prev_investment = float(state.get("investment_share", anchors.get("investment_share", 0.0)))
    prev_population = float(state.get("population_growth", anchors.get("population_growth", 0.0)))
    prev_unemployment = float(state.get("unemployment", anchors.get("unemployment", 0.0)))
    prev_vix = float(state.get("VIXCLS", anchors.get("VIXCLS", 18.0)))
    prev_dff = float(state.get("DFF", anchors.get("DFF", 2.0)))
    prev_dgs10 = float(state.get("DGS10", anchors.get("DGS10", 3.0)))

    next_state["gdp_growth"] = point
    next_state["gdp_growth_lag3"] = prev_lag2
    next_state["gdp_growth_lag2"] = prev_lag1
    next_state["gdp_growth_lag1"] = point
    next_state["gdp_growth_rollmean_2"] = float(pd.Series([point, prev_lag1]).mean())
    next_state["gdp_growth_rollstd_2"] = float(pd.Series([point, prev_lag1]).std(ddof=0))
    next_state["inflation_lag1"] = prev_inflation
    next_state["trade_share_lag1"] = prev_trade
    next_state["inflation"] = 0.82 * prev_inflation + 0.18 * anchors.get("inflation", prev_inflation)
    next_state["trade_share"] = 0.90 * prev_trade + 0.10 * anchors.get("trade_share", prev_trade)
    next_state["investment_share"] = 0.90 * prev_investment + 0.10 * anchors.get("investment_share", prev_investment)
    next_state["population_growth"] = 0.95 * prev_population + 0.05 * anchors.get("population_growth", prev_population)
    next_state["unemployment"] = 0.88 * prev_unemployment + 0.12 * anchors.get("unemployment", prev_unemployment)
    next_state["inflation_rollmean_2"] = 0.5 * (next_state["inflation"] + prev_inflation)
    next_state["year_trend"] = float(state.get("year_trend", 0.0)) + 0.25
    next_state["regime_flag"] = int(next_state["unemployment"] > anchors.get("unemployment", next_state["unemployment"]))
    next_state["VIXCLS"] = 0.80 * prev_vix + 0.20 * anchors.get("VIXCLS", prev_vix)
    next_state["DFF"] = 0.85 * prev_dff + 0.15 * anchors.get("DFF", prev_dff)
    next_state["DGS10"] = 0.88 * prev_dgs10 + 0.12 * anchors.get("DGS10", prev_dgs10)
    next_state["term_spread"] = float(next_state.get("DGS10", prev_dgs10) - next_state.get("DFF", prev_dff))
    next_period = pd.Timestamp(state["date"]).to_period("Q") + 1
    next_state["date"] = next_period.end_time.normalize()
    return next_state


def _build_country_drivers(country_hist: pd.DataFrame, current_state: dict[str, Any], as_of: str, top_n: int = 8) -> dict[str, Any]:
    signs = {
        "inflation": -1.0,
        "inflation_lag1": -0.7,
        "unemployment": -1.0,
        "trade_share": 0.6,
        "trade_share_lag1": 0.4,
        "investment_share": 0.8,
        "population_growth": 0.3,
        "gdp_growth_lag1": 1.0,
        "gdp_growth_lag2": 0.7,
        "gdp_growth_rollmean_2": 0.9,
        "regime_flag": -0.5,
        "VIXCLS": -0.5,
        "DFF": -0.3,
        "term_spread": 0.4,
    }

    def sign_for(feature: str) -> float | None:
        if feature in signs:
            return signs[feature]
        if feature.startswith("monthly_agg_"):
            return signs.get(feature.removeprefix("monthly_agg_"))
        if feature.startswith("dfm_factor_"):
            return 0.25
        match = re.search(r"^(?P<base>.+)_m\d+$", feature)
        if match:
            return signs.get(match.group("base"))
        return None

    rows: list[dict[str, float | str]] = []
    candidate_features = set(signs)
    candidate_features.update(col for col in country_hist.columns if col.startswith("monthly_agg_") or col.startswith("dfm_factor_"))
    candidate_features.update(col for col in country_hist.columns if re.search(r"_m\d+$", col))
    for feature in sorted(candidate_features):
        sign = sign_for(feature)
        if sign is None:
            continue
        if feature not in current_state or feature not in country_hist.columns:
            continue
        hist = country_hist[feature].dropna()
        current = float(current_state.get(feature, 0.0) or 0.0)
        if hist.empty:
            contribution = current * sign
        else:
            center = float(hist.mean())
            scale = float(hist.std(ddof=0)) or 1.0
            contribution = ((current - center) / scale) * sign
        rows.append({"feature": feature, "contribution": float(contribution)})
    drivers = sorted(rows, key=lambda item: abs(float(item["contribution"])), reverse=True)[:top_n]
    return {"as_of": as_of, "drivers": drivers, "method": "country_relative_zscore"}


def recursive_forecast(bundle: dict[str, Any], country_hist: pd.DataFrame, as_of: str, horizons: list[int], scenario: dict[str, float] | None = None) -> tuple[list[dict[str, Any]], float, dict[str, Any], list[dict[str, Any]]]:
    ordered = country_hist.sort_values("date")
    state = ordered.iloc[-1].to_dict()
    anchors = _compute_anchors(ordered)
    requested = sorted(set(horizons))
    max_h = max(requested) if requested else 0
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    last_shock = 0.0
    start_period = pd.Timestamp(as_of).to_period("Q")

    for horizon in range(max_h + 1):
        working_state = _apply_scenario_to_state(state, scenario, anchors, horizon)
        pred = _predict_one_step(bundle, working_state)
        traces.append({"horizon_q": horizon, **pred})
        if horizon in requested:
            period = start_period + horizon
            rows.append(
                {
                    "horizon_q": horizon,
                    "period": f"{period.year}-Q{period.quarter}",
                    "mean": pred["mean"],
                    "p10": pred["p10"],
                    "p25": pred["p25"],
                    "p50": pred["p50"],
                    "p75": pred["p75"],
                    "p90": pred["p90"],
                }
            )
        last_shock = pred["shock_prob"]
        state = _advance_state(working_state, pred["mean"], anchors)

    drivers = _build_country_drivers(ordered, _apply_scenario_to_state(ordered.iloc[-1].to_dict(), scenario, anchors, 0), as_of)
    return rows, float(last_shock), drivers, traces


def _serialize_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in bundle.items():
        if is_dataclass(value):
            summary: dict[str, Any] = {}
            for field in fields(value):
                field_value = getattr(value, field.name)
                if isinstance(field_value, (str, int, float, bool)) or field_value is None:
                    summary[field.name] = field_value
                elif isinstance(field_value, list):
                    summary[field.name] = field_value
                elif isinstance(field_value, dict):
                    summary[field.name] = field_value
                else:
                    summary[field.name] = type(field_value).__name__
            out[key] = summary
        elif isinstance(value, dict):
            out[key] = value
        elif isinstance(value, list):
            out[key] = value
        else:
            out[key] = str(type(value).__name__)
    return out


def _git_hash(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None


def run_rolling_origin_backtest(
    settings: Settings,
    full_engineered: pd.DataFrame,
    horizon: int = 1,
    *,
    expanding: bool = True,
    rolling_window: int | None = None,
    max_origins: int | None = None,
    logger: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if full_engineered.empty:
        empty = pd.DataFrame(
            columns=[
                "country",
                "origin",
                "origin_label",
                "target_date",
                "target_period",
                "horizon_q",
                "point",
                "q10",
                "q25",
                "q50",
                "q75",
                "q90",
                "actual",
                "train_rows",
                "train_max_available_date",
            ]
        )
        return empty, summarize_prediction_frame(empty, horizon)

    min_train_rows = int(settings.raw_config.get("run", {}).get("backtest_min_train_rows", 8))
    configured_max_origins = int(settings.raw_config.get("run", {}).get("backtest_max_origins", 12))
    origin_dates = sorted(pd.Timestamp(value) for value in full_engineered["available_date"].dropna().unique().tolist())
    if max_origins is None:
        max_origins = configured_max_origins
    prediction_rows: list[dict[str, Any]] = []
    countries = sorted(full_engineered["country"].dropna().unique().tolist())
    max_horizon = max(1, int(horizon))
    forecast_horizons = list(range(max_horizon))
    valid_origins: list[pd.Timestamp] = []
    available_target_dates = set(pd.Timestamp(value) for value in full_engineered["date"].dropna().tolist())
    for origin in origin_dates:
        furthest_target = (origin.to_period("Q") + (max_horizon - 1)).end_time.normalize()
        if furthest_target in available_target_dates:
            valid_origins.append(origin)
    origin_dates = valid_origins
    if max_origins and max_origins > 0:
        origin_dates = origin_dates[-max_origins:]

    for origin in origin_dates:
        origin_label = _quarter_label(pd.Timestamp(origin))
        trained_countries = 0
        for country in countries:
            country_hist = full_engineered.loc[full_engineered["country"] == country].sort_values("date").reset_index(drop=True)
            if country_hist.empty:
                continue
            train_country = country_hist.loc[country_hist["available_date"] <= origin].copy()
            if train_country.empty:
                continue
            if not expanding and rolling_window:
                train_country = train_country.tail(int(rolling_window)).copy()
            if len(train_country) < min_train_rows:
                continue

            last_available = pd.Timestamp(train_country["available_date"].max())
            if last_available > origin:
                raise ValueError("Backtest leakage detected: training data includes rows unavailable at the origin")

            bundle = _train_bundle_from_train_only(train_country, settings)
            forecast_rows, _, _, _ = recursive_forecast(
                bundle,
                train_country,
                origin.isoformat(),
                forecast_horizons,
                scenario={"tighten": 0.0, "commodity": 0.0, "demand": 0.0},
            )
            if not forecast_rows:
                continue

            actual_map = country_hist.set_index("date")["gdp_growth"]
            for row in forecast_rows:
                eval_horizon = int(row["horizon_q"]) + 1
                target_period = pd.Period(str(row["period"]), freq="Q")
                target_date = target_period.end_time.normalize()
                actual = actual_map.get(target_date)
                if pd.isna(actual):
                    continue
                prediction_rows.append(
                    {
                        "country": country,
                        "origin": origin,
                        "origin_label": origin_label,
                        "target_date": target_date,
                        "target_period": str(row["period"]),
                        "horizon_q": eval_horizon,
                        "point": float(row["mean"]),
                        "q10": float(row["p10"]),
                        "q25": float(row["p25"]),
                        "q50": float(row["p50"]),
                        "q75": float(row["p75"]),
                        "q90": float(row["p90"]),
                        "actual": float(actual),
                        "forecast": float(row["mean"]),
                        "train_rows": int(len(train_country)),
                        "train_max_available_date": last_available,
                    }
                )
            trained_countries += 1
        if logger is not None:
            logger(f"Backtest origin {origin_label}: retrained {trained_countries} country model sets.")

    if prediction_rows:
        combined = pd.DataFrame(prediction_rows).sort_values(["origin", "country", "horizon_q"]).reset_index(drop=True)
    else:
        combined = pd.DataFrame(
            columns=[
                "country",
                "origin",
                "origin_label",
                "target_date",
                "target_period",
                "horizon_q",
                "point",
                "q10",
                "q25",
                "q50",
                "q75",
                "q90",
                "actual",
                "forecast",
                "train_rows",
                "train_max_available_date",
            ]
        )
    summary = summarize_prediction_frame(combined, max_horizon)
    return combined, summary


def _write_country_outputs(
    run_dir: Path,
    bundle: dict[str, Any],
    panel: pd.DataFrame,
    usable: pd.DataFrame,
    as_of: str,
    horizons: list[int],
    scenario: dict[str, float],
    countries: list[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    exported_rows: list[dict[str, Any]] = []
    country_files: dict[str, dict[str, str]] = {}
    for country in countries:
        country_hist = usable[usable["country"] == country].sort_values("date")
        if country_hist.empty:
            continue
        history = build_history(panel, country, as_of)
        forecast_rows, shock_prob, drivers_payload, traces = recursive_forecast(bundle, country_hist, as_of, horizons, scenario=scenario)
        payload = {
            "country": country,
            "as_of": as_of,
            "target": "qoq_saar",
            "history": history,
            "forecast": forecast_rows,
            "shock_prob": shock_prob,
            "model_weights": getattr(bundle["ensemble"], "weight_stability", {}),
            "trace": traces,
            "scenario": scenario,
        }
        forecast_path = run_dir / f"forecast_{country}.json"
        drivers_path = run_dir / f"drivers_{country}.json"
        write_json(forecast_path, payload)
        write_json(drivers_path, drivers_payload)
        first = forecast_rows[0] if forecast_rows else {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
        exported_rows.append(
            {
                "country": country,
                "as_of": as_of,
                "mean": float(first["mean"]),
                "p10": float(first["p10"]),
                "p50": float(first["p50"]),
                "p90": float(first["p90"]),
                "shock_prob": float(shock_prob),
            }
        )
        country_files[country] = {
            "forecast": str(forecast_path),
            "drivers": str(drivers_path),
        }
    return exported_rows, country_files


def run_pipeline(
    as_of: str,
    horizons: list[int],
    scenario: dict[str, float] | None = None,
    countries: list[str] | None = None,
    include_backtest: bool = True,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    scenario = scenario or {"tighten": 0.0, "commodity": 0.0, "demand": 0.0}
    run_id = make_run_id("atlas")
    logger = get_logger("atlas_gdp.run")
    run_started = time.perf_counter()
    log_event(
        logger,
        "run_started",
        run_id=run_id,
        as_of=as_of,
        horizons=horizons,
        requested_countries=countries or [],
    )

    step_started = time.perf_counter()
    built_panel, _, _, _ = build_dataset(settings=settings)
    log_event(
        logger,
        "build_dataset_completed",
        run_id=run_id,
        duration_ms=int((time.perf_counter() - step_started) * 1000),
        rows=int(len(built_panel)),
    )

    step_started = time.perf_counter()
    bundle, panel, usable, scored = train_for_as_of(settings=settings, as_of=as_of)
    log_event(
        logger,
        "train_completed",
        run_id=run_id,
        duration_ms=int((time.perf_counter() - step_started) * 1000),
        training_rows=int(len(usable)),
        scored_rows=int(len(scored)),
    )

    _, full_engineered, lineage = load_engineered_panel(settings=settings, as_of=None)
    available_countries = sorted(usable["country"].dropna().unique().tolist())
    target_countries = [country for country in (countries or available_countries) if country in available_countries]
    if not target_countries:
        target_countries = available_countries

    run_dir, latest_dir = ensure_run_dirs(settings, run_id)

    bundle_path = run_dir / "atlas_gdp_bundle.joblib"
    scored_path = run_dir / "latest_forecast_frame.parquet"
    joblib.dump(bundle, bundle_path)
    scored.to_parquet(scored_path, index=False)

    driver_cols = [c for c in bundle["numeric_features"][:10] if c in usable.columns]
    drivers = current_drivers(usable, driver_cols)
    deltas = delta_decomposition(usable, driver_cols)
    drivers.to_csv(run_dir / "drivers_current.csv", index=False)
    deltas.to_csv(run_dir / "drivers_delta.csv", index=False)

    world_rows, country_files = _write_country_outputs(
        run_dir=run_dir,
        bundle=bundle,
        panel=panel,
        usable=usable,
        as_of=as_of,
        horizons=horizons,
        scenario=scenario,
        countries=target_countries,
    )
    snapshot = pd.DataFrame(world_rows).sort_values(["shock_prob", "mean"], ascending=[False, False])
    snapshot_path = run_dir / "world_snapshot.csv"
    snapshot.to_csv(snapshot_path, index=False)

    backtest_files: dict[str, str] = {}
    backtest_metrics: dict[str, Any] = {}
    if include_backtest:
        from atlas_gdp.evaluation.plots import plot_coverage_calibration, plot_fan_chart, plot_forecast_vs_actual
        from atlas_gdp.reporting.model_card import write_model_card

        bt_horizon = int(settings.raw_config.get("run", {}).get("backtest_horizon", 1))
        bt_started = time.perf_counter()
        bt_predictions, backtest_metrics = run_rolling_origin_backtest(settings, full_engineered, horizon=bt_horizon)
        log_event(
            logger,
            "backtest_completed",
            run_id=run_id,
            duration_ms=int((time.perf_counter() - bt_started) * 1000),
            rows=int(len(bt_predictions)),
            origins=len(backtest_metrics.get("origins", [])),
        )
        backtest_pred_path = run_dir / "backtest_predictions.csv"
        backtest_summary_path = run_dir / "backtest_summary.json"
        calibration_path = run_dir / "backtest_calibration.csv"
        plots_dir = run_dir / "plots"
        bt_predictions.to_csv(backtest_pred_path, index=False)
        write_json(backtest_summary_path, backtest_metrics)
        pd.DataFrame(backtest_metrics.get("calibration", [])).to_csv(calibration_path, index=False)
        if not bt_predictions.empty:
            bt_plot = bt_predictions.copy()
            bt_plot["date"] = pd.to_datetime(bt_plot["target_date"])
            plot_forecast_vs_actual(bt_plot, plots_dir / "forecast_vs_actual.png")
            plot_fan_chart(bt_plot, plots_dir / "fan_chart.png")
        else:
            fallback_plot = scored.copy()
            fallback_plot["forecast"] = fallback_plot["point"]
            plot_forecast_vs_actual(fallback_plot, plots_dir / "forecast_vs_actual.png")
            plot_fan_chart(fallback_plot, plots_dir / "fan_chart.png")
        calibration_df = pd.DataFrame(backtest_metrics.get("calibration", []))
        if not calibration_df.empty:
            plot_coverage_calibration(calibration_df, plots_dir / "coverage_calibration.png")
        write_model_card(backtest_metrics.get("overall", backtest_metrics), run_dir / "MODEL_CARD.md")
        backtest_files = {
            "predictions": str(backtest_pred_path),
            "metrics": str(backtest_summary_path),
            "summary": str(backtest_summary_path),
            "calibration": str(calibration_path),
            "forecast_vs_actual": str(plots_dir / "forecast_vs_actual.png"),
            "fan_chart": str(plots_dir / "fan_chart.png"),
            "coverage_plot": str(plots_dir / "coverage_calibration.png"),
            "model_card": str(run_dir / "MODEL_CARD.md"),
        }

    lineage_path = run_dir / "data_lineage.json"
    write_json(lineage_path, lineage)
    manifest = {
        "run_id": run_id,
        "created_at_utc": utc_now().isoformat(),
        "as_of": as_of,
        "horizons": horizons,
        "countries": target_countries,
        "scenario": scenario,
        "manifest_path": str(run_dir / "manifest.json"),
        "artifacts_root": str(run_dir),
        "latest_path": str(latest_dir),
        "bundle_path": str(bundle_path),
        "scored_frame_path": str(scored_path),
        "world_snapshot_path": str(snapshot_path),
        "lineage_path": str(lineage_path),
        "country_artifacts": country_files,
        "model_weights": getattr(bundle["ensemble"], "weight_stability", {}),
        "bundle_summary": _serialize_bundle(bundle),
        "config_snapshot": settings.raw_config,
        "settings_snapshot": settings.settings_snapshot,
        "connectors_used": lineage.get("sources", []),
        "connector_status": lineage.get("connector_status", {}),
        "method_notes": METHOD_NOTES,
        "backtest": {
            "metrics": backtest_metrics,
            "files": backtest_files,
        },
        "git_hash": _git_hash(settings.paths.root),
    }
    write_json(run_dir / "manifest.json", manifest)
    copy_tree_contents(run_dir, latest_dir)
    write_latest_pointer(settings, manifest)
    record_run(settings, manifest)
    latest_bundle_path = settings.paths.artifacts / "atlas_gdp_bundle.joblib"
    latest_scored_path = settings.paths.artifacts / "latest_forecast_frame.parquet"
    joblib.dump(bundle, latest_bundle_path)
    scored.to_parquet(latest_scored_path, index=False)
    write_json(settings.paths.artifacts / "data_lineage.json", lineage)
    log_event(
        logger,
        "run_completed",
        run_id=run_id,
        duration_ms=int((time.perf_counter() - run_started) * 1000),
        exported_countries=target_countries,
        forecast_rows=int(len(snapshot)),
    )
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    return read_json(Path(path))


def load_latest_manifest(settings: Settings | None = None) -> dict[str, Any] | None:
    settings = settings or load_settings()
    pointer = read_latest_pointer(settings)
    if pointer is None:
        return None
    manifest_path = Path(pointer["manifest_path"])
    if not manifest_path.exists():
        return None
    return load_manifest(manifest_path)
