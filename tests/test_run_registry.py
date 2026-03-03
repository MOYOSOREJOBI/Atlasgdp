from __future__ import annotations

import sqlite3
from pathlib import Path

from atlas_gdp.pipeline.storage import record_run
from atlas_gdp.settings import load_settings


def test_record_run_inserts_registry_row(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ATLAS_GDP_ROOT", str(tmp_path))
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    settings = load_settings()
    record_run(
        settings,
        run_id="atlas_test_001",
        manifest_path=str(tmp_path / "artifacts" / "runs" / "atlas_test_001" / "manifest.json"),
        created_at="2026-03-03T20:00:00+00:00",
        as_of="2026-03-03",
        horizon_q=4,
        scenario={"tighten": 0.25},
    )

    db_path = Path(settings.db_url.removeprefix("sqlite:///"))
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT run_id, created_at, as_of, horizon_q, scenario_json, manifest_path FROM runs WHERE run_id = ?",
            ("atlas_test_001",),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == "atlas_test_001"
    assert row[2] == "2026-03-03"
    assert row[3] == 4
    assert '"tighten": 0.25' in row[4]
