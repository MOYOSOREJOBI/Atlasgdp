from __future__ import annotations

import argparse
import json

import pandas as pd

from atlas_gdp.config import load_settings
from atlas_gdp.evaluation.backtest import summarize_prediction_frame
from atlas_gdp.pipeline import build_dataset
from atlas_gdp.pipeline.service import load_engineered_panel, run_pipeline as run_service_pipeline, run_rolling_origin_backtest, train_for_as_of


def _cmd_build_dataset(args: argparse.Namespace) -> None:
    settings = load_settings()
    panel, _, _, lineage = build_dataset(settings=settings, countries=args.countries, start_year=args.start, end_year=args.end)
    print(
        json.dumps(
            {
                "rows": int(len(panel)),
                "countries": sorted(panel["country"].dropna().unique().tolist()),
                "sources": lineage.get("sources", []),
            },
            indent=2,
        )
    )


def _cmd_train(args: argparse.Namespace) -> None:
    settings = load_settings()
    bundle, _, _, scored = train_for_as_of(settings=settings, as_of=args.as_of)
    import joblib

    joblib.dump(bundle, settings.paths.artifacts / "atlas_gdp_bundle.joblib")
    scored.to_parquet(settings.paths.artifacts / "latest_forecast_frame.parquet", index=False)
    print(json.dumps({"as_of": args.as_of, "rows": int(len(scored))}, indent=2))


def _cmd_run(args: argparse.Namespace) -> None:
    settings = load_settings()
    manifest = run_service_pipeline(
        as_of=args.as_of or pd.Timestamp("today").strftime("%Y-%m-%d"),
        horizons=args.horizons or list(settings.raw_config.get("run", {}).get("default_horizons", [0, 1, 2, 4, 8])),
        scenario={"tighten": args.tighten, "commodity": args.commodity, "demand": args.demand},
        countries=args.countries,
        include_backtest=not args.skip_backtest,
        settings=settings,
    )
    print(json.dumps(manifest, indent=2))


def _cmd_backtest(args: argparse.Namespace) -> None:
    from atlas_gdp.evaluation.plots import plot_fan_chart, plot_forecast_vs_actual
    from atlas_gdp.reporting.model_card import write_model_card

    settings = load_settings()
    build_dataset(settings=settings)
    _, full_engineered, _ = load_engineered_panel(settings=settings, as_of=None)
    predictions, summary = run_rolling_origin_backtest(settings, full_engineered, horizon=args.horizon)
    if not predictions.empty:
        plot_frame = predictions.copy()
        plot_frame["forecast"] = plot_frame["point"]
        plot_forecast_vs_actual(plot_frame, settings.paths.reports / "forecast_vs_actual.png")
        plot_fan_chart(plot_frame, settings.paths.reports / "fan_chart.png")
    filtered = predictions.copy()
    if args.date_from:
        filtered = filtered[filtered["origin"] >= pd.Timestamp(args.date_from)].copy()
    if args.date_to:
        filtered = filtered[filtered["origin"] <= pd.Timestamp(args.date_to)].copy()
    summary = summarize_prediction_frame(filtered, args.horizon) if not filtered.empty else summary
    write_model_card(summary.get("overall", {}), "MODEL_CARD.md")
    print(json.dumps({"summary": summary, "rows": int(len(filtered))}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlas-gdp")
    sub = parser.add_subparsers(dest="command", required=True)

    ds = sub.add_parser("build-dataset")
    ds.add_argument("--countries", nargs="+")
    ds.add_argument("--start", type=int)
    ds.add_argument("--end", type=int)
    ds.set_defaults(func=_cmd_build_dataset)

    train = sub.add_parser("train")
    train.add_argument("--as-of", "--as_of", dest="as_of", required=True)
    train.set_defaults(func=_cmd_train)

    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("--as-of", "--as_of", dest="as_of")
    run_cmd.add_argument("--horizons", nargs="*", type=int)
    run_cmd.add_argument("--countries", nargs="*")
    run_cmd.add_argument("--tighten", type=float, default=0.0)
    run_cmd.add_argument("--commodity", type=float, default=0.0)
    run_cmd.add_argument("--demand", type=float, default=0.0)
    run_cmd.add_argument("--skip-backtest", "--skip_backtest", dest="skip_backtest", action="store_true")
    run_cmd.set_defaults(func=_cmd_run)

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--horizon", type=int, default=1)
    backtest.add_argument("--from", "--date-from", dest="date_from")
    backtest.add_argument("--to", "--date-to", dest="date_to")
    backtest.set_defaults(func=_cmd_backtest)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def app() -> None:
    main()


if __name__ == "__main__":
    main()
