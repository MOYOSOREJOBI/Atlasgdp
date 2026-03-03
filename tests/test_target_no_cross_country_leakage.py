from __future__ import annotations

import pandas as pd


def test_target_shift_does_not_cross_country_when_rows_are_interleaved() -> None:
    frame = pd.DataFrame(
        {
            "country": ["USA", "CAN", "USA", "CAN"],
            "date": pd.to_datetime(["2024-03-31", "2024-03-31", "2024-06-30", "2024-06-30"]),
            "gdp_growth": [1.0, 10.0, 2.0, 20.0],
        }
    )
    ordered = frame.sort_values(["country", "date"]).reset_index(drop=True)
    targets = ordered.groupby("country", sort=False)["gdp_growth"].shift(-1)

    usa_rows = ordered.loc[ordered["country"] == "USA"].copy()
    can_rows = ordered.loc[ordered["country"] == "CAN"].copy()
    usa_targets = targets.loc[usa_rows.index]
    can_targets = targets.loc[can_rows.index]

    assert usa_targets.iloc[0] == 2.0
    assert can_targets.iloc[0] == 20.0
    assert pd.isna(usa_targets.iloc[-1])
    assert pd.isna(can_targets.iloc[-1])
