# ATLAS-GDP

**ATLAS-GDP** is a mixed-frequency macro forecasting system for GDP growth.
It computes GDP growth consistently, nowcasts the current quarter, and forecasts multiple horizons with uncertainty.

It is built for transparent, audit-grade reproducibility.

## What it does

- Computes GDP growth from level series:
  - YoY percent from log differences
  - QoQ SAAR for quarterly series
- Builds country panels from World Bank data
- Aligns mixed-frequency data with an as-of framework
- Produces:
  - current-quarter nowcasts
  - 1 to 8 quarter forecasts
  - 1-year-ahead annual forecasts
- Returns point forecasts and quantile intervals

## Why mixed-frequency nowcasting matters

GDP is released late.
Markets, policy teams, and risk desks need a live estimate before the official print.
Mixed-frequency nowcasting uses faster signals to infer current growth while the quarter is still in progress.

## Data sources

- World Bank Indicators API (default, no key)
- FRED API (optional, live)
- OECD SDMX connector via `pandaSDMX` (real when `ATLAS_OECD_*_URL` env vars are set; otherwise explicit `not_configured`)
- IMF SDMX connector via `pandaSDMX` (real when `ATLAS_IMF_*_URL` env vars are set; otherwise explicit `not_configured`)
- BEA component bridge is demo-only in `bea_demo.py` and reads `data/raw/bea_demo_components.csv` if provided

Default mode runs from World Bank data or bundled/offline synthetic samples.
When `OFFLINE_MODE=1`, OECD and IMF never make network calls. They read cached raw responses and parsed parquet from `data/raw/{source}/{dataset}/`; if cache is missing, the connector reports `offline_cache_missing` and tells you to run once online to populate cache.

## Model stack

1. **Baseline panel**
   - Ridge
   - HistGradientBoostingRegressor
   - country fixed effects + year trend
2. **Bridge bottom-up**
   - component-style bridge equations using monthly indicators
3. **DFM-Kalman**
   - compact dynamic factor nowcaster with uncertainty
4. **MIDAS**
   - quarterly GDP on monthly indicators with exponential Almon weights
5. **BVAR**
   - small Bayesian VAR with Minnesota-style shrinkage
6. **Ensemble**
   - stacked point forecast
   - linear pooled density combination

## Reproducibility

- Deterministic seeds
- Config-driven CLI
- Cached raw responses
- Offline mode
- Artifact bundle with:
  - model objects
  - feature list
  - config
  - created_utc
  - training data hash
- Data lineage JSON with endpoints, params, timestamps, row counts, and missingness

## Local Run

These commands are intended for macOS `zsh`.

```bash
cd atlas_gdp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
make test
```

### One-shot pipeline run

```bash
OFFLINE_MODE=1 atlas-gdp run --as-of 2026-03-03 --horizons 0 1 2 4 8
```

Equivalent module form if the console script is not on your `PATH` yet:

```bash
PYTHONPATH=src OFFLINE_MODE=1 python -m atlas_gdp.cli run --as-of 2026-03-03 --horizons 0 1 2 4 8
```

This creates a full run under `artifacts/runs/{run_id}/`, updates `artifacts/latest/`, and records a manifest in `artifacts/latest_manifest.json`.
Run metadata is also recorded in the default SQLite registry at `artifacts/atlas_gdp.db`.

### Streamlit app

```bash
PYTHONPATH=src OFFLINE_MODE=1 python -m streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

Behavior:

- the sidebar `As-of date` is real: if it differs from the latest run, the UI shows artifact-viewer mode until you click `Refresh artifacts`
- `Refresh artifacts` triggers a new pipeline run, updates the run manifest, and refreshes KPIs
- the UI reads forecast files from the latest manifest only, never by globbing stale files
- the Audit tab shows `connector_status` so you can see which sources were `online_cached`, `offline_cache_hit`, `not_configured`, or demo-only

### Fastest local demo

```bash
chmod +x scripts/demo.sh
OFFLINE_MODE=1 ./scripts/demo.sh
```

That script:

1. builds the dataset
2. runs the pipeline for `BRA` as of `2026-03-03` with horizon `7`
3. launches Streamlit at `http://localhost:8501`

It works in a fresh checkout even if the `atlas-gdp` console script is not installed yet, because it falls back to `PYTHONPATH=src python -m atlas_gdp.cli`.
If dependencies are missing, it stops early and prints the exact install commands.

### Scripted demo flow

```bash
bash scripts/demo_flow.sh
```

That script reproduces:

1. baseline run for `BRA` as of `2026-03-03`
2. tightened scenario run for `BRA`
3. switch to `USA`
4. rolling-origin backtest generation
5. open the UI and inspect Forecast, Simulator, Backtest, and Audit tabs

## Docker Run

This follows the standard Streamlit container pattern: install the package, write a Streamlit config, and run the app on `0.0.0.0:8501`.

```bash
docker compose up --build
```

Then open:

```bash
open http://localhost:8501
```

The `app` service uses SQLite by default.
An optional Postgres service is included behind a Compose profile:

```bash
docker compose --profile db up --build
```

If you want the app container to use Postgres, set `DB_URL` before starting Compose.

## Simple Cloud Deploy

- `render.yaml` is included for a basic Render deploy.
- It uses the Docker image path and defaults to offline/demo mode.
- For a live deployment, set `FRED_API_KEY`, `OFFLINE_MODE=0`, and configure the OECD/IMF SDMX series URLs (`ATLAS_OECD_*_URL`, `ATLAS_IMF_*_URL`).

## Deploy To Streamlit Community Cloud

This repo is compatible with Streamlit Community Cloud using `requirements.txt` (`-e .`) so the local package is installed before `app/streamlit_app.py` runs.

Repo structure expectation:

- `app/streamlit_app.py` remains the Streamlit entrypoint
- `src/atlas_gdp/` contains the installable package
- `requirements.txt` installs the local package and runtime dependencies via `pyproject.toml`

Suggested app file:

- `app/streamlit_app.py`

Optional secrets:

- `FRED_API_KEY`
- `ATLAS_OECD_INDUSTRIAL_PRODUCTION_URL`
- `ATLAS_OECD_RETAIL_SALES_URL`
- `ATLAS_OECD_PMI_URL`
- `ATLAS_IMF_CREDIT_GAP_URL`
- `ATLAS_IMF_CURRENT_ACCOUNT_URL`

Demo suggestion:

- set `OFFLINE_MODE=1` for the most reliable demo deployment
- set `OFFLINE_MODE=0` only after you have seeded caches or configured live connectors

## Demo results

These are demo-mode sample metrics from the bundled sample panel.

| Task | Metric | Demo Value |
|---|---:|---:|
| nowcast | MAE | 0.82 |
| nowcast | RMSE | 1.09 |
| 1y annual | MAE | 1.14 |
| 1y annual | RMSE | 1.48 |
| probabilistic | coverage(80%) | 0.78 |
| probabilistic | CRPS | 0.63 |

Reproduce them with a rolling-origin run:

```bash
OFFLINE_MODE=1 python -m atlas_gdp.cli backtest --horizon 1 --from 2010-01-01 --to 2025-01-01
```

Or via the packaged entrypoint:

```bash
OFFLINE_MODE=1 atlas-gdp backtest --horizon 1 --date-from 2010-01-01 --date-to 2025-01-01
```

## Make Targets

```bash
make dev
make test
make run AS_OF=2026-03-03
```

## Acceptance Demo Script

Use either `OFFLINE_MODE=1 ./scripts/demo.sh` locally or `docker compose up --build` in Docker, then walk through this exact flow:

1. Select `BRA`
2. Set `As-of date` to `2026-03-03`
3. Click `Run / Refresh (recompute)`
4. Confirm the top KPI row populates and the `Last run` timestamp updates
5. Move `Financial tightening` to `+1.0`, click `Run / Refresh (recompute)` again, and confirm the KPI nowcast shifts
6. Open the `Backtest` tab and confirm rolling-origin metrics are present
7. Open the `Audit` tab and confirm `run_id`, connector status, and config/settings snapshots are present

Artifacts are written to `artifacts/runs/{run_id}/`.
`artifacts/latest/` is a clean copy of the latest run only.
Reports are written to `artifacts/reports/`.

## Limitations

- Real-time macro data is revised after first release.
- This is still a demo: the as-of engine uses simulated release lags, not full historical vintage databases.
- OECD and IMF are real SDMX connectors but require configured series URLs and a first online cache fill.
- BEA remains explicit demo-only in `bea_demo.py`.
- Monthly indicators are aligned to quarters with simplified aggregation, not a full institutional release-calendar engine.
- Structural breaks can degrade all models.
- The `BVAR` block is still a small shrinkage-style approximation for demo use.

## Not financial advice

ATLAS-GDP is a research and forecasting tool.
It is not investment, policy, or legal advice.

## License notes

- Code: MIT
- Data: follow original source terms
- Do not upload raw third-party data with this repo
