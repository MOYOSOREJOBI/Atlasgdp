from __future__ import annotations

import pandas as pd

from atlas_gdp.features.mixed_frequency import align_mixed_frequency


def test_asof_excludes_future_monthly_releases() -> None:
    annual = pd.DataFrame(
        [
            {
                "country": "BRA",
                "date": pd.Timestamp("2024-12-31"),
                "gdp_growth": 2.0,
            }
        ]
    )
    monthly = pd.DataFrame(
        [
            {
                "country": "BRA",
                "date": pd.Timestamp("2024-01-15"),
                "available_date": pd.Timestamp("2024-02-05"),
                "PMI": 10.0,
            },
            {
                "country": "BRA",
                "date": pd.Timestamp("2024-02-15"),
                "available_date": pd.Timestamp("2024-03-05"),
                "PMI": 20.0,
            },
            {
                "country": "BRA",
                "date": pd.Timestamp("2024-03-15"),
                "available_date": pd.Timestamp("2024-05-10"),
                "PMI": 30.0,
            },
        ]
    )

    aligned = align_mixed_frequency(annual, monthly_df=monthly, as_of="2024-04-01")
    q1 = aligned.loc[aligned["date"] == pd.Timestamp("2024-03-31")].iloc[0]

    assert float(q1["monthly_agg_PMI"]) == 15.0
    assert float(q1["PMI_m0"]) == 20.0
    assert float(q1["PMI_m1"]) == 10.0
    assert pd.isna(q1["PMI_m2"])

