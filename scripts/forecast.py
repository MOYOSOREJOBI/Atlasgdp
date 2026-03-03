from __future__ import annotations

import argparse
import json

import pandas as pd

from atlas_gdp.pipeline.run import load_manifest, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default=None, type=str)
    parser.add_argument("--countries", nargs="*", default=None)
    parser.add_argument("--all_countries", action="store_true")
    parser.add_argument("--horizons", nargs="+", required=True, type=int)
    parser.add_argument("--as_of", default=None, type=str)
    parser.add_argument("--tighten", type=float, default=0.0)
    parser.add_argument("--commodity", type=float, default=0.0)
    parser.add_argument("--demand", type=float, default=0.0)
    args = parser.parse_args()

    if args.all_countries or (args.country is None and not args.countries):
        countries: list[str] = []
    elif args.countries:
        countries = args.countries
    else:
        countries = [args.country] if args.country else []

    run_result = run_pipeline(
        as_of=pd.Timestamp(args.as_of or pd.Timestamp("today").strftime("%Y-%m-%d")).date(),
        countries=countries,
        horizon_q=max(args.horizons) if args.horizons else 0,
        scenario={
            "financial_tightening": args.tighten,
            "commodity_shock": args.commodity,
            "demand_boost": args.demand,
        },
        offline=True,
    )
    manifest = load_manifest(run_result.manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
