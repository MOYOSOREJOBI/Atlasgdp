from __future__ import annotations

import pandas as pd


def current_drivers(df: pd.DataFrame, feature_columns: list[str], top_n: int = 5) -> pd.DataFrame:
    latest = df.iloc[-1]
    rows = []
    for col in feature_columns:
        rows.append({"feature": col, "value": float(latest[col]) if pd.notna(latest[col]) else 0.0})
    out = pd.DataFrame(rows)
    out["abs_value"] = out["value"].abs()
    return out.sort_values("abs_value", ascending=False).head(top_n).drop(columns=["abs_value"])


def delta_decomposition(df: pd.DataFrame, feature_columns: list[str], top_n: int = 5) -> pd.DataFrame:
    if len(df) < 2:
        return current_drivers(df, feature_columns, top_n=top_n)
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    rows = []
    for col in feature_columns:
        a = float(curr[col]) if pd.notna(curr[col]) else 0.0
        b = float(prev[col]) if pd.notna(prev[col]) else 0.0
        rows.append({"feature": col, "delta": a - b})
    out = pd.DataFrame(rows)
    out["abs_delta"] = out["delta"].abs()
    return out.sort_values("abs_delta", ascending=False).head(top_n).drop(columns=["abs_delta"])
