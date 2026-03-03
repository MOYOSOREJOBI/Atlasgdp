from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class BaselinePanelBundle:
    ridge: Pipeline
    hgb: Pipeline
    feature_columns: list[str]
    hgb_columns: list[str]


def train_baseline_panel(train_df: pd.DataFrame, feature_columns: list[str]) -> BaselinePanelBundle:
    numeric = [c for c in feature_columns if c not in {"country"}]
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["country"],
            ),
        ]
    )
    ridge = Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])
    ridge.fit(train_df[feature_columns], train_df["target"])
    x_hgb = pd.get_dummies(train_df[feature_columns], columns=["country"], dtype=float)
    hgb = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(max_depth=3, learning_rate=0.05, n_estimators=200, random_state=42)),
        ]
    )
    hgb.fit(x_hgb, train_df["target"])
    return BaselinePanelBundle(ridge=ridge, hgb=hgb, feature_columns=feature_columns, hgb_columns=list(x_hgb.columns))


def predict_baseline(bundle: BaselinePanelBundle, df: pd.DataFrame) -> pd.DataFrame:
    x = df[bundle.feature_columns]
    x_hgb = pd.get_dummies(x, columns=["country"], dtype=float)
    return pd.DataFrame(
        {
            "ridge": bundle.ridge.predict(x),
            "hgb": bundle.hgb.predict(x_hgb.reindex(columns=bundle.hgb_columns, fill_value=0.0)),
        },
        index=df.index,
    )
