from __future__ import annotations

import argparse
import json

import joblib

from atlas_gdp.config import load_settings
from atlas_gdp.pipeline.storage import write_json
from atlas_gdp.pipeline.service import train_for_as_of_with_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="nowcast", choices=["nowcast", "forecast"])
    parser.add_argument("--as_of", required=True, type=str)
    parser.add_argument("--min_valid_rows", default=4, type=int)
    args = parser.parse_args()

    settings = load_settings()
    bundle, _, _, scored, report = train_for_as_of_with_report(
        settings=settings,
        as_of=args.as_of,
        min_valid_rows=args.min_valid_rows,
    )
    bundle_path = settings.paths.artifacts / "atlas_gdp_bundle.joblib"
    forecast_frame_path = settings.paths.artifacts / "latest_forecast_frame.parquet"
    train_report_path = settings.paths.artifacts / "train_report.json"

    joblib.dump(bundle, bundle_path)
    scored.to_parquet(forecast_frame_path, index=False)
    write_json(train_report_path, report)
    print(
        json.dumps(
            {
                "task": args.task,
                "as_of": args.as_of,
                "split_available_date": report["split_available_date"],
                "counts": report["counts"],
                "bundle_path": str(bundle_path),
                "forecast_frame_path": str(forecast_frame_path),
                "train_report_path": str(train_report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
