from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class BVARBundle:
    model: Ridge
    features: list[str]
    residual_std: float


def train_bvar(train_df: pd.DataFrame, features: list[str]) -> BVARBundle:
    model = Ridge(alpha=2.0)
    x = train_df[features].fillna(0.0)
    y = train_df["target"]
    model.fit(x, y)
    residuals = y - model.predict(x)
    return BVARBundle(model=model, features=features, residual_std=float(np.std(residuals)))


def predict_bvar(bundle: BVARBundle, df: pd.DataFrame) -> pd.DataFrame:
    mean = bundle.model.predict(df[bundle.features].fillna(0.0))
    return pd.DataFrame({"bvar_mean": mean, "bvar_std": bundle.residual_std}, index=df.index)
