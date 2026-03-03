from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def exponential_almon_weights(length: int, theta1: float = -0.1, theta2: float = -0.01) -> np.ndarray:
    j = np.arange(length, dtype=float)
    raw = np.exp(theta1 * j + theta2 * j * j)
    return raw / raw.sum()


@dataclass
class MIDASBundle:
    model: Ridge
    feature_columns: list[str]
    weights: np.ndarray


def train_midas(train_df: pd.DataFrame, feature_columns: list[str]) -> MIDASBundle:
    weights = exponential_almon_weights(len(feature_columns))
    weighted = train_df[feature_columns].fillna(0.0).to_numpy() * weights
    model = Ridge(alpha=1.0)
    model.fit(weighted, train_df["target"])
    return MIDASBundle(model=model, feature_columns=feature_columns, weights=weights)


def predict_midas(bundle: MIDASBundle, df: pd.DataFrame) -> pd.Series:
    weighted = df[bundle.feature_columns].fillna(0.0).to_numpy() * bundle.weights
    return pd.Series(bundle.model.predict(weighted), index=df.index, name="midas")
