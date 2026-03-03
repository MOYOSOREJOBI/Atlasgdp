from __future__ import annotations

import json

from atlas_gdp.data.oecd_sdmx import OECD_DATASET, OECD_SERIES_URLS, load_oecd_quarterly
from atlas_gdp.data.sdmx_connector import cache_paths


def test_cached_raw_payload_round_trips_to_dataframe(tmp_path) -> None:
    cache_key = "|".join(f"{name}=" for name in sorted(OECD_SERIES_URLS))
    raw_path, parquet_path = cache_paths(tmp_path, "oecd", OECD_DATASET, cache_key)
    raw_path.write_text(
        json.dumps(
            {
                "source": "oecd",
                "dataset": OECD_DATASET,
                "series": {
                    "industrial_production": [
                        {"country": "BRA", "date": "2024-03-31", "value": 1.2},
                    ],
                    "retail_sales": [
                        {"country": "BRA", "date": "2024-03-31", "value": 2.3},
                    ],
                    "pmi": [
                        {"country": "BRA", "date": "2024-03-31", "value": 53.0},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    result = load_oecd_quarterly(tmp_path, offline_mode=True)

    assert result.status == "offline_cache_hit"
    assert parquet_path.exists()
    assert result.frame.loc[0, "country"] == "BRA"
    assert float(result.frame.loc[0, "industrial_production"]) == 1.2
    assert float(result.frame.loc[0, "retail_sales"]) == 2.3
    assert float(result.frame.loc[0, "pmi"]) == 53.0
