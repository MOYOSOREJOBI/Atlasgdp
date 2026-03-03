from __future__ import annotations

from pathlib import Path

import pandas as pd

from atlas_gdp.config import load_settings
from atlas_gdp.pipeline.service import run_rolling_origin_backtest


def _configure_backtest_env(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "atlas_gdp.test.yml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  backtest_horizon: 1",
                "  backtest_min_train_rows: 4",
                "  backtest_max_origins: 2",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("ATLAS_GDP_ROOT", str(tmp_path))
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("ATLAS_GDP_CONFIG", str(config_path))


def _synthetic_engineered_panel() -> pd.DataFrame:
    dates = pd.date_range("2021-03-31", periods=9, freq="QE")
    rows: list[dict[str, object]] = []
    for country, base in [("BRA", 0.8), ("USA", 1.6)]:
        for idx, value_date in enumerate(dates):
            rows.append(
                {
                    "country": country,
                    "date": value_date.normalize(),
                    "frequency": "Q",
                    "available_date": (value_date + pd.Timedelta(days=45)).normalize(),
                    "gdp_growth": base + (0.2 * idx),
                    "inflation": 2.5 + (0.03 * idx),
                    "population_growth": 0.7,
                    "investment_share": 21.0 + (0.1 * idx),
                    "trade_share": 40.0 + idx,
                    "unemployment": 5.0 + (0.04 * idx),
                }
            )
    df = pd.DataFrame(rows).sort_values(["country", "date"]).reset_index(drop=True)
    grouped = df.groupby("country", sort=False)
    df["gdp_growth_lag1"] = grouped["gdp_growth"].shift(1)
    df["gdp_growth_lag2"] = grouped["gdp_growth"].shift(2)
    df["gdp_growth_lag3"] = grouped["gdp_growth"].shift(3)
    df["inflation_lag1"] = grouped["inflation"].shift(1)
    df["trade_share_lag1"] = grouped["trade_share"].shift(1)
    df["gdp_growth_rollmean_2"] = grouped["gdp_growth"].transform(lambda s: s.rolling(2, min_periods=2).mean())
    df["gdp_growth_rollstd_2"] = grouped["gdp_growth"].transform(lambda s: s.rolling(2, min_periods=2).std(ddof=0))
    df["inflation_rollmean_2"] = grouped["inflation"].transform(lambda s: s.rolling(2, min_periods=2).mean())
    df["year_trend"] = df["date"].dt.year - df["date"].dt.year.min()
    df["regime_flag"] = (df["unemployment"] > grouped["unemployment"].transform(lambda s: s.rolling(3, min_periods=1).mean())).astype(int)
    df["target"] = grouped["gdp_growth"].shift(-1)
    df["target_date"] = grouped["date"].shift(-1)
    df["target_available_date"] = grouped["available_date"].shift(-1)
    return df.dropna().reset_index(drop=True)


def test_backtest_respects_origin_cutoff(monkeypatch, tmp_path: Path) -> None:
    _configure_backtest_env(monkeypatch, tmp_path)
    settings = load_settings()
    engineered = _synthetic_engineered_panel()

    predictions, _ = run_rolling_origin_backtest(settings, engineered, horizon=1)

    assert not predictions.empty
    assert (pd.to_datetime(predictions["train_max_available_date"]) <= pd.to_datetime(predictions["origin"])).all()

