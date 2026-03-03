from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class BridgeBundle:
    model: Ridge
    features: list[str]


def train_bridge(train_df: pd.DataFrame) -> BridgeBundle:
    features = [c for c in ["consumption", "investment", "government", "net_exports"] if c in train_df.columns]
    if not features:
        features = [c for c in ["inflation", "investment_share", "trade_share"] if c in train_df.columns]
    model = Ridge(alpha=1.0)
    model.fit(train_df[features].fillna(0.0), train_df["target"])
    return BridgeBundle(model=model, features=features)


def predict_bridge(bundle: BridgeBundle, df: pd.DataFrame) -> pd.Series:
    return pd.Series(bundle.model.predict(df[bundle.features].fillna(0.0)), index=df.index, name="bridge")
