from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_train_report_json_contains_required_keys(tmp_path) -> None:
    config_path = tmp_path / "atlas_gdp.test.yml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  countries: [USA, CAN, GBR]",
                "  start_year: 2023",
                "  end_year: 2026",
            ]
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["OFFLINE_MODE"] = "1"
    env["ATLAS_GDP_ROOT"] = str(tmp_path)
    env["ARTIFACT_ROOT"] = str(tmp_path / "artifacts")
    env["ATLAS_GDP_CONFIG"] = str(config_path)
    env["PYTHONPATH"] = str((Path.cwd() / "src").resolve())

    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--as_of", "2026-03-03", "--min_valid_rows", "2"],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0

    report_path = tmp_path / "artifacts" / "train_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    required_keys = {"as_of", "split_available_date", "counts", "paths", "parameters"}
    assert required_keys.issubset(payload.keys())
    assert {"overall", "by_country"}.issubset(payload["counts"].keys())
    assert {"train_report_path", "bundle_path", "forecast_frame_path"}.issubset(payload["paths"].keys())
