from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


@dataclass
class DFMBundle:
    pca: PCA | None
    model: Ridge
    feature_columns: list[str]
    residual_std: float


def train_dfm(train_df: pd.DataFrame, feature_columns: list[str], n_factors: int = 2) -> DFMBundle:
    x = train_df[feature_columns].fillna(0.0)
    if all(col.startswith("dfm_factor_") for col in feature_columns):
        pca = None
        factors = x.to_numpy()
    else:
        pca = PCA(n_components=min(n_factors, x.shape[1]))
        factors = pca.fit_transform(x)
    model = Ridge(alpha=1.0)
    model.fit(factors, train_df["target"])
    residuals = train_df["target"] - model.predict(factors)
    return DFMBundle(pca=pca, model=model, feature_columns=feature_columns, residual_std=float(np.std(residuals)))


def predict_dfm(bundle: DFMBundle, df: pd.DataFrame) -> pd.DataFrame:
    x = df[bundle.feature_columns].fillna(0.0)
    factors = x.to_numpy() if bundle.pca is None else bundle.pca.transform(x)
    mean = bundle.model.predict(factors)
    var = np.full(len(df), bundle.residual_std**2)
    return pd.DataFrame({"dfm_mean": mean, "dfm_var": var}, index=df.index)
