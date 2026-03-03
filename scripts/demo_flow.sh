#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export OFFLINE_MODE="${OFFLINE_MODE:-1}"
export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR/src}"

python -m atlas_gdp.cli run --as_of 2026-03-03 --horizons 0 1 2 3 4 5 6 7 --countries BRA --skip_backtest
python -m atlas_gdp.cli run --as_of 2026-03-03 --horizons 0 1 2 3 4 5 6 7 --countries BRA --tighten 1.0 --skip_backtest
python -m atlas_gdp.cli run --as_of 2026-03-03 --horizons 0 1 2 3 4 5 6 7 --countries USA --skip_backtest
python -m atlas_gdp.cli backtest --horizon 1 --from 2020-01-01 --to 2026-03-03

printf '%s\n' "Demo flow complete."
printf '%s\n' "Start the UI with:"
printf '%s\n' "  streamlit run app/streamlit_app.py --server.port 8501 --server.headless true"
printf '%s\n' "Then open Forecast, Simulator, Backtest, and Audit tabs to inspect the latest run manifest."
