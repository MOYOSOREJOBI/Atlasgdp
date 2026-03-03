from __future__ import annotations

import pandas as pd

from atlas_gdp.features.asof import asof_filter, asof_release_filter, release_calendar_simulator


def test_asof_release_filter_avoids_future_rows() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "value": [1.0, 2.0],
        }
    )
    delayed = release_calendar_simulator(df, lag_days=30)
    out = asof_release_filter(delayed, "2024-02-10")
    assert len(out) == 1
    assert out.iloc[0]["value"] == 1.0
