from __future__ import annotations

from atlas_gdp.data.wb import build_world_bank_panel


def test_world_bank_offline_panel_builds() -> None:
    df, lineage = build_world_bank_panel(
        raw_dir="data/raw",
        countries=["USA", "CAN"],
        start=2018,
        end=2020,
        indicators=[
            "NY.GDP.MKTP.KD.ZG",
            "FP.CPI.TOTL.ZG",
            "SP.POP.GROW",
            "NE.GDI.TOTL.ZS",
            "NE.TRD.GNFS.ZS",
            "SL.UEM.TOTL.ZS",
        ],
        offline_mode=True,
    )
    assert not df.empty
    assert lineage["mode"] == "offline"
