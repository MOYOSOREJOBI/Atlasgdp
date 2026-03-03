from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def guarded_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": guarded_mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def crps_normal_approx(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    std = np.maximum(std, 1e-6)
    z = (y_true - mean) / std
    return float(np.mean(std * (z * (2 * _norm_cdf(z) - 1) + 2 * _norm_pdf(z) - 1 / np.sqrt(np.pi))))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))
