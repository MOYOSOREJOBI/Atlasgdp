from __future__ import annotations

import pandas as pd

from atlas_gdp.features.mixed_frequency import align_mixed_frequency


def test_monthly_series_aligns_to_quarterly_rows() -> None:
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
            {"country": "BRA", "date": pd.Timestamp("2024-01-15"), "PMI": 1.0},
            {"country": "BRA", "date": pd.Timestamp("2024-02-15"), "PMI": 2.0},
            {"country": "BRA", "date": pd.Timestamp("2024-03-15"), "PMI": 3.0},
            {"country": "BRA", "date": pd.Timestamp("2024-04-15"), "PMI": 4.0},
            {"country": "BRA", "date": pd.Timestamp("2024-05-15"), "PMI": 5.0},
            {"country": "BRA", "date": pd.Timestamp("2024-06-15"), "PMI": 6.0},
        ]
    )

    aligned = align_mixed_frequency(annual, monthly_df=monthly)

    q1 = aligned.loc[aligned["date"] == pd.Timestamp("2024-03-31")].iloc[0]
    q2 = aligned.loc[aligned["date"] == pd.Timestamp("2024-06-30")].iloc[0]

    assert float(q1["monthly_agg_PMI"]) == 2.0
    assert float(q2["monthly_agg_PMI"]) == 5.0
    assert float(q1["PMI_m0"]) == 3.0
    assert float(q1["PMI_m1"]) == 2.0
    assert float(q1["PMI_m2"]) == 1.0
    assert float(q2["PMI_m0"]) == 6.0

