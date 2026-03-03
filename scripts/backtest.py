from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from atlas_gdp.config import load_settings
from atlas_gdp.evaluation.backtest import summarize_prediction_frame
from atlas_gdp.evaluation.plots import plot_coverage_calibration, plot_fan_chart, plot_forecast_vs_actual
from atlas_gdp.pipeline.service import build_dataset, load_engineered_panel, load_latest_manifest, run_rolling_origin_backtest
from atlas_gdp.pipeline.storage import write_json
from atlas_gdp.reporting.model_card import write_model_card


def _select_output_root(settings, latest_manifest: dict[str, object] | None) -> Path:
    if latest_manifest and latest_manifest.get("artifacts_root"):
        return Path(str(latest_manifest["artifacts_root"]))
    fallback = settings.paths.artifacts / "latest"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _update_manifest(manifest: dict[str, object], output_root: Path, summary_path: Path, prediction_path: Path, calibration_path: Path, plots_dir: Path) -> None:
    files = {
        "predictions": str(prediction_path),
        "metrics": str(summary_path),
        "summary": str(summary_path),
        "calibration": str(calibration_path),
        "forecast_vs_actual": str(plots_dir / "forecast_vs_actual.png"),
        "fan_chart": str(plots_dir / "fan_chart.png"),
        "coverage_plot": str(plots_dir / "coverage_calibration.png"),
        "model_card": str(output_root / "MODEL_CARD.md"),
    }
    manifest["backtest"] = {
        "metrics": json.loads(summary_path.read_text(encoding="utf-8")),
        "files": files,
    }
    if isinstance(manifest.get("paths"), dict):
        manifest["paths"]["backtest_report"] = summary_path.name
        manifest["paths"]["backtest_plot"] = (plots_dir / "coverage_calibration.png").name
    write_json(output_root / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="date_from", required=False, type=str)
    parser.add_argument("--to", dest="date_to", required=False, type=str)
    parser.add_argument("--horizon", dest="horizon", required=False, type=int)
    parser.add_argument("--origins", dest="origins", required=False, type=int)
    parser.add_argument("--rolling-window", dest="rolling_window", required=False, type=int)
    args = parser.parse_args()

    settings = load_settings()
    build_dataset(settings=settings)
    _, full_engineered, lineage = load_engineered_panel(settings=settings, as_of=None)
    latest_manifest = load_latest_manifest(settings=settings)
    output_root = _select_output_root(settings, latest_manifest)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    prediction_df, summary = run_rolling_origin_backtest(
        settings,
        full_engineered,
        horizon=int(args.horizon or settings.raw_config.get("run", {}).get("backtest_horizon", 1)),
        expanding=args.rolling_window is None,
        rolling_window=args.rolling_window,
        max_origins=args.origins,
        logger=print,
    )
    if args.date_from or args.date_to:
        mask = pd.Series(True, index=prediction_df.index)
        if args.date_from:
            mask &= prediction_df["origin"] >= pd.Timestamp(args.date_from)
        if args.date_to:
            mask &= prediction_df["origin"] <= pd.Timestamp(args.date_to)
        prediction_df = prediction_df.loc[mask].copy()
        summary = summarize_prediction_frame(prediction_df, int(args.horizon or summary.get("horizon_q", 1)))
        summary["filtered_from"] = args.date_from
        summary["filtered_to"] = args.date_to

    prediction_path = output_root / "backtest_predictions.csv"
    summary_path = output_root / "backtest_summary.json"
    calibration_path = output_root / "backtest_calibration.csv"
    prediction_df.to_csv(prediction_path, index=False)
    write_json(summary_path, summary)
    pd.DataFrame(summary.get("calibration", [])).to_csv(calibration_path, index=False)

    plot_frame = prediction_df.copy()
    if not plot_frame.empty:
        plot_frame["date"] = pd.to_datetime(plot_frame["target_date"])
        plot_forecast_vs_actual(plot_frame, plots_dir / "forecast_vs_actual.png")
        plot_fan_chart(plot_frame, plots_dir / "fan_chart.png")
    calibration_df = pd.DataFrame(summary.get("calibration", []))
    if not calibration_df.empty:
        plot_coverage_calibration(calibration_df, plots_dir / "coverage_calibration.png")
    write_model_card(summary.get("overall", {}), output_root / "MODEL_CARD.md")

    if latest_manifest:
        latest_manifest["connectors_used"] = latest_manifest.get("connectors_used", lineage.get("sources", []))
        _update_manifest(latest_manifest, output_root, summary_path, prediction_path, calibration_path, plots_dir)

    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "prediction_path": str(prediction_path),
                "overall": summary.get("overall", {}),
                "origins": summary.get("origins", []),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
