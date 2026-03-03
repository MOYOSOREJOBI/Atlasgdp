from __future__ import annotations

import numpy as np


def conformal_interval(point_forecast: np.ndarray, residuals: np.ndarray, alpha: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    radius = np.quantile(np.abs(residuals), 1.0 - alpha)
    lower = point_forecast - radius
    upper = point_forecast + radius
    return lower, upper


def linear_pool_quantiles(forecasts: list[np.ndarray], quantiles: list[float]) -> dict[float, np.ndarray]:
    stacked = np.stack(forecasts, axis=0)
    return {q: np.quantile(stacked, q, axis=0) for q in quantiles}
