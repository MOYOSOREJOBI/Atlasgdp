from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from atlas_gdp.evaluation.dm_test import diebold_mariano
from atlas_gdp.evaluation.metrics import crps_normal_approx, interval_coverage, point_metrics


@dataclass
class BacktestResult:
    metrics: dict[str, float]
    forecast_frame: pd.DataFrame
    summary: dict[str, Any]


def _period_label(value: str | pd.Timestamp) -> str:
    period = pd.Timestamp(value).to_period("Q")
    return f"{period.year}-Q{period.quarter}"


def _actual_col(df: pd.DataFrame) -> str:
    if "actual" in df.columns:
        return "actual"
    if "target" in df.columns:
        return "target"
    raise KeyError("backtest frame requires either 'actual' or 'target'")


def _base_metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "r2": 0.0,
            "cov50": 0.0,
            "cov90": 0.0,
            "crps": 0.0,
            "rows": 0.0,
        }

    actual = df[_actual_col(df)].to_numpy(dtype=float)
    point = df["point"].to_numpy(dtype=float)
    lower50 = df["q25"].to_numpy(dtype=float) if "q25" in df.columns else point - 0.35
    upper50 = df["q75"].to_numpy(dtype=float) if "q75" in df.columns else point + 0.35
    lower90 = df["q10"].to_numpy(dtype=float) if "q10" in df.columns else point - 1.0
    upper90 = df["q90"].to_numpy(dtype=float) if "q90" in df.columns else point + 1.0
    std = np.maximum((upper90 - lower90) / 2.563, 1e-6)

    if len(df) < 2:
        abs_err = np.abs(actual - point)
        sq_err = (actual - point) ** 2
        denom = np.maximum(np.abs(actual), 1e-6)
        metrics = {
            "mae": float(np.mean(abs_err)),
            "rmse": float(np.sqrt(np.mean(sq_err))),
            "mape": float(np.mean(abs_err / denom)),
            "r2": 0.0,
        }
    else:
        metrics = point_metrics(actual, point)
    metrics["cov50"] = interval_coverage(actual, lower50, upper50)
    metrics["cov90"] = interval_coverage(actual, lower90, upper90)
    metrics["coverage_80"] = metrics["cov90"]
    metrics["crps"] = crps_normal_approx(actual, point, std)
    metrics["rows"] = float(len(df))
    if "ridge" in df.columns and "hgb" in df.columns:
        loss_a = np.abs(df["ridge"].to_numpy(dtype=float) - actual)
        loss_b = np.abs(df["hgb"].to_numpy(dtype=float) - actual)
        metrics.update(diebold_mariano(loss_a, loss_b))
    return metrics


def score_prediction_frame(df: pd.DataFrame) -> dict[str, float]:
    return _base_metrics(df)


def summarize_prediction_frame(df: pd.DataFrame, horizon_q: int) -> dict[str, Any]:
    forecast_frame = df.copy()
    if not forecast_frame.empty and "actual" not in forecast_frame.columns:
        forecast_frame["actual"] = forecast_frame[_actual_col(forecast_frame)]

    by_country: dict[str, dict[str, float]] = {}
    if "country" in forecast_frame.columns:
        for country, group in forecast_frame.groupby("country", sort=True):
            metrics = _base_metrics(group)
            by_country[str(country)] = {
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "cov50": float(metrics["cov50"]),
                "cov90": float(metrics["cov90"]),
                "rows": int(len(group)),
            }

    by_horizon: dict[str, dict[str, float]] = {}
    if "horizon_q" in forecast_frame.columns:
        for horizon, group in forecast_frame.groupby("horizon_q", sort=True):
            metrics = _base_metrics(group)
            by_horizon[str(int(horizon))] = {
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "cov50": float(metrics["cov50"]),
                "cov90": float(metrics["cov90"]),
                "rows": int(len(group)),
            }

    overall_metrics = _base_metrics(forecast_frame)
    calibration = [
        {"nominal": 0.5, "observed": float(overall_metrics["cov50"])},
        {"nominal": 0.9, "observed": float(overall_metrics["cov90"])},
    ]
    if "origin_label" in forecast_frame.columns:
        origins = sorted(str(value) for value in forecast_frame["origin_label"].dropna().unique().tolist())
    elif "origin" in forecast_frame.columns:
        origins = sorted(_period_label(value) for value in forecast_frame["origin"].dropna().unique().tolist())
    else:
        origins = []

    summary = {
        "by_country": by_country,
        "overall": {
            "mae": float(overall_metrics["mae"]),
            "rmse": float(overall_metrics["rmse"]),
            "mape": float(overall_metrics["mape"]),
            "r2": float(overall_metrics["r2"]),
            "cov50": float(overall_metrics["cov50"]),
            "cov90": float(overall_metrics["cov90"]),
            "crps": float(overall_metrics["crps"]),
            "rows": int(len(forecast_frame)),
        },
        "by_horizon": by_horizon,
        "calibration": calibration,
        "origins": origins,
        "horizon_q": int(horizon_q),
        "interval_method": "training-residual ensemble quantiles (model-based, not fully probabilistic)",
    }
    return summary


def rolling_origin_backtest(df: pd.DataFrame, horizon_q: int | None = None) -> BacktestResult:
    forecast_frame = df.copy()
    actual_col = _actual_col(forecast_frame) if not forecast_frame.empty or {"actual", "target"}.intersection(forecast_frame.columns) else "actual"
    if "actual" not in forecast_frame.columns and actual_col in forecast_frame.columns:
        forecast_frame["actual"] = forecast_frame[actual_col]
    if "forecast" not in forecast_frame.columns and "point" in forecast_frame.columns:
        forecast_frame["forecast"] = forecast_frame["point"]
    resolved_horizon = int(horizon_q if horizon_q is not None else forecast_frame["horizon_q"].max() if "horizon_q" in forecast_frame.columns and not forecast_frame.empty else 1)
    summary = summarize_prediction_frame(forecast_frame, resolved_horizon)
    return BacktestResult(metrics=summary["overall"], forecast_frame=forecast_frame, summary=summary)
