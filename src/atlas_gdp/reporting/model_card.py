from __future__ import annotations

from pathlib import Path

import json


def write_model_card(metrics: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ATLAS-GDP Model Card",
        "",
        "## Summary",
        "ATLAS-GDP is a mixed-frequency GDP nowcasting and forecasting ensemble.",
        "",
        "## Latest metrics",
        "```json",
        json.dumps(metrics, indent=2),
        "```",
        "",
        "## Limitations",
        "- Data revisions matter.",
        "- Coverage varies by country.",
        "- Structural breaks degrade all models.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
