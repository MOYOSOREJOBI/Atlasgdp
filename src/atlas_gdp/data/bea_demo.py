from __future__ import annotations

from pathlib import Path

import pandas as pd

from atlas_gdp.data.sdmx_connector import ConnectorLoadResult


def load_bea_demo_components(raw_dir: str | Path) -> ConnectorLoadResult:
    path = Path(raw_dir) / "bea_demo_components.csv"
    if path.exists():
        frame = pd.read_csv(path, parse_dates=["date"])
        return ConnectorLoadResult(frame=frame, status="demo_local_file", note="BEA connector is demo-only in this repo.")
    rows = []
    for i, q in enumerate(pd.period_range("2018Q1", "2025Q4", freq="Q")):
        rows.append(
            {
                "date": q.end_time.normalize(),
                "consumption": 1.5 + (i % 4) * 0.2,
                "investment": 0.6 + (i % 5) * 0.15,
                "government": 0.4 + (i % 3) * 0.05,
                "net_exports": -0.2 + (i % 4) * 0.03,
            }
        )
    return ConnectorLoadResult(
        frame=pd.DataFrame(rows),
        status="demo_embedded",
        note="BEA connector is demo-only in this repo. Provide bea_demo_components.csv to override.",
    )
