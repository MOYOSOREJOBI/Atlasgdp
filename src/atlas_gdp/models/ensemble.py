from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class EnsembleBundle:
    point_model: Ridge
    base_columns: list[str]
    quantiles: list[float]
    weight_stability: dict[str, float]
    residual_quantiles: dict[float, float]


def train_ensemble(train_pred_df: pd.DataFrame, target: pd.Series, quantiles: list[float]) -> EnsembleBundle:
    base_columns = list(train_pred_df.columns)
    x = train_pred_df[base_columns].fillna(0.0)
    model = Ridge(alpha=1.0)
    model.fit(x, target)
    coef = model.coef_
    stability = {col: float(abs(w)) for col, w in zip(base_columns, coef)}
    point = model.predict(x)
    residuals = target.to_numpy() - point
    residual_quantiles = {float(q): float(np.quantile(residuals, q)) for q in quantiles if float(q) != 0.5}
    return EnsembleBundle(
        point_model=model,
        base_columns=base_columns,
        quantiles=quantiles,
        weight_stability=stability,
        residual_quantiles=residual_quantiles,
    )


def predict_ensemble(bundle: EnsembleBundle, pred_df: pd.DataFrame) -> pd.DataFrame:
    x = pred_df[bundle.base_columns].fillna(0.0)
    point = bundle.point_model.predict(x)
    out = {"point": point}
    for q in bundle.quantiles:
        key = f"q{int(float(q) * 100):02d}"
        if float(q) == 0.5:
            out[key] = point
        else:
            offset = bundle.residual_quantiles.get(float(q), 0.0)
            out[key] = point + offset
    return pd.DataFrame(out, index=pred_df.index)
