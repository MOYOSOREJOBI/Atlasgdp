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
                "  backtest_horizon: 2",
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
    dates = pd.date_range("2021-03-31", periods=12, freq="QE")
    rows: list[dict[str, object]] = []
    for country, base in [("BRA", 1.0), ("USA", 2.0)]:
        for idx, value_date in enumerate(dates):
            gdp = base + (0.15 * idx)
            inflation = 3.0 - (0.05 * idx)
            trade_share = 45.0 + idx
            rows.append(
                {
                    "country": country,
                    "date": value_date.normalize(),
                    "frequency": "Q",
                    "available_date": (value_date + pd.Timedelta(days=45)).normalize(),
                    "gdp_growth": gdp,
                    "inflation": inflation,
                    "population_growth": 0.8 + (0.01 * idx),
                    "investment_share": 20.0 + (0.2 * idx),
                    "trade_share": trade_share,
                    "unemployment": 5.0 + (0.05 * idx),
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


def test_walk_forward_backtest_emits_expected_rows(monkeypatch, tmp_path: Path) -> None:
    _configure_backtest_env(monkeypatch, tmp_path)
    settings = load_settings()
    engineered = _synthetic_engineered_panel()

    predictions, summary = run_rolling_origin_backtest(settings, engineered, horizon=2)

    assert len(summary["origins"]) == 2
    assert len(predictions) == 8
    assert set(predictions["horizon_q"].tolist()) == {1, 2}
    assert {"point", "q25", "q75", "q10", "q90", "actual"}.issubset(predictions.columns)
