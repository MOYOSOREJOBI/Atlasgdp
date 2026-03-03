from __future__ import annotations

import pandas as pd

from atlas_gdp.pipeline.service import choose_split_available_date, split_train_valid


def test_asof_split_moves_boundary_backward_to_create_nonempty_valid_set() -> None:
    usable = pd.DataFrame(
        {
            "country": ["USA", "CAN", "USA", "CAN", "USA", "CAN"],
            "date": pd.to_datetime(
                [
                    "2023-03-31",
                    "2023-03-31",
                    "2023-06-30",
                    "2023-06-30",
                    "2023-09-30",
                    "2023-09-30",
                ]
            ),
            "available_date": pd.to_datetime(
                [
                    "2023-05-15",
                    "2023-05-15",
                    "2023-08-15",
                    "2023-08-15",
                    "2023-11-15",
                    "2023-11-15",
                ]
            ),
            "target": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            "target_date": pd.to_datetime(
                [
                    "2023-06-30",
                    "2023-06-30",
                    "2023-09-30",
                    "2023-09-30",
                    "2023-12-31",
                    "2023-12-31",
                ]
            ),
            "target_available_date": pd.to_datetime(
                [
                    "2023-08-15",
                    "2023-08-15",
                    "2023-11-15",
                    "2023-11-15",
                    "2024-02-15",
                    "2024-02-15",
                ]
            ),
            "inflation": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
        }
    )
    split_date, meta = choose_split_available_date(usable, "2023-09-01", min_valid_rows=2)
    train_df, valid_df = split_train_valid(usable, "2023-09-01", min_valid_rows=2)

    assert split_date == pd.Timestamp("2023-05-15")
    assert meta["actual_valid_rows"] == 2
    assert len(train_df) == 2
    assert len(valid_df) == 2
    assert valid_df["available_date"].max() <= pd.Timestamp("2023-09-01")
    assert valid_df["available_date"].min() > split_date
