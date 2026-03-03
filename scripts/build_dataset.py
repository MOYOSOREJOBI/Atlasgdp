from __future__ import annotations

import argparse
import json

from atlas_gdp.config import load_settings
from atlas_gdp.pipeline import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", required=False)
    parser.add_argument("--start", type=int, required=False)
    parser.add_argument("--end", type=int, required=False)
    args = parser.parse_args()

    settings = load_settings()
    panel, _, _, lineage = build_dataset(
        settings=settings,
        countries=args.countries,
        start_year=args.start,
        end_year=args.end,
    )
    print(
        json.dumps(
            {
                "rows": int(len(panel)),
                "countries": sorted(panel["country"].dropna().unique().tolist()),
                "lineage_sources": len(lineage.get("sources", [])),
                "panel_path": str(settings.paths.processed / "panel.parquet"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
