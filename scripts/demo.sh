#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export OFFLINE_MODE="${OFFLINE_MODE:-1}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_CMD="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_CMD="$(command -v python3)"
fi

if ! "$PYTHON_CMD" -c "import streamlit" >/dev/null 2>&1; then
  echo "Missing runtime dependencies for the selected Python interpreter: $PYTHON_CMD"
  echo "Run this first:"
  echo "  $PYTHON_CMD -m pip install --upgrade pip setuptools wheel"
  echo "  $PYTHON_CMD -m pip install -e \".[dev]\""
  exit 1
fi

if [[ -x "$ROOT_DIR/.venv/bin/atlas-gdp" ]]; then
  ATLAS_CMD=("$ROOT_DIR/.venv/bin/atlas-gdp")
elif command -v atlas-gdp >/dev/null 2>&1; then
  ATLAS_CMD=("$(command -v atlas-gdp)")
else
  ATLAS_CMD=("$PYTHON_CMD" -m atlas_gdp.cli)
fi

echo "==> Building dataset (OFFLINE_MODE=${OFFLINE_MODE})"
"${ATLAS_CMD[@]}" build-dataset

echo "==> Running BRA demo pipeline for 2026-03-03 horizon 7"
"${ATLAS_CMD[@]}" run --as-of 2026-03-03 --horizons 0 1 2 3 4 5 6 7 --countries BRA

echo "==> Launching Streamlit on http://localhost:8501"
echo "Demo checks:"
echo "  1. Select BRA"
echo "  2. Set As-of date to 2026-03-03"
echo "  3. Click Run / Refresh (recompute)"
echo "  4. Confirm KPIs populate and Last run updates"
echo "  5. Move Financial tightening to +1.0 and rerun"
echo "  6. Open Backtest and Audit tabs"

if [[ "${ATLAS_DEMO_SKIP_STREAMLIT:-0}" == "1" ]]; then
  echo "==> ATLAS_DEMO_SKIP_STREAMLIT=1, skipping Streamlit launch"
  exit 0
fi

exec "$PYTHON_CMD" -m streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true
