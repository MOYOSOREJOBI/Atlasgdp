# Overall Completion: 60%

## Executive Summary

This repo is still not a production-grade demo.

The core model pipeline is materially stronger than the original prototype:

- target leakage across countries is fixed
- the train/validation split is now based on `available_date`
- rolling-origin backtesting is real and retrains per origin
- run folders, manifests, latest pointers, and a run registry exist
- connector behavior is more explicit

But the product layer is still inconsistent. The biggest remaining problem is contract integrity:

- the CLI and the Streamlit app still do not use one canonical manifest schema
- the easiest demo path (`scripts/demo.sh`) still runs the CLI path first, which writes the incompatible manifest shape
- the UI can therefore launch after the demo script and silently show sample fallback data until the user clicks `Run / Refresh`
- scenario effects are still applied twice in the UI path: once in backend recomputation and again client-side

That means the codebase is now a serious prototype with better controls and better traceability, but it still fails the “production-grade demo” bar.

## Verification Run

### Commands Run

- `pytest -q`
- `.venv/bin/pytest -q`
- `ATLAS_DEMO_SKIP_STREAMLIT=1 OFFLINE_MODE=1 ./scripts/demo.sh`
- `PYTHONPATH=src OFFLINE_MODE=1 .venv/bin/python - <<...>>` using `atlas_gdp.pipeline.run.run_pipeline(...)` with `financial_tightening=1.0`
- `curl -sS http://localhost:8503/healthz`
- `curl -sS http://localhost:8503 | head -n 5`

### Results

- `pytest -q` failed in the default shell environment with a hard abort inside system Anaconda/scikit-learn.
- `.venv/bin/pytest -q` passed: `17 passed in 46.03s`.
- `ATLAS_DEMO_SKIP_STREAMLIT=1 OFFLINE_MODE=1 ./scripts/demo.sh` passed end-to-end:
  - built dataset
  - ran the BRA pipeline for `2026-03-03`, horizon `7`
  - produced run `20260303T211529Z_LX2M`
  - skipped the final Streamlit launch by explicit flag
- A fresh normalized wrapper run passed:
  - run `20260303T212148Z_X55C`
  - `financial_tightening=+1.0`
  - first BRA mean forecast: `-1.040550784623791`
- Local Streamlit server started successfully on `http://localhost:8503` when run outside the sandbox.
  - `GET /healthz` returned `ok`
  - `GET /` returned Streamlit HTML

### Important Runtime Evidence

- Current normalized run loader works:
  - run `20260303T212148Z_X55C`
  - manifest countries: `['BRA']`
  - forecast payload countries loaded from manifest: `['BRA']`
- Current CLI/demo-script run loader fails:
  - run `20260303T211529Z_LX2M`
  - manifest countries: `['BRA']`
  - forecast payload countries loaded from manifest: `[]`
- This means the UI loader still breaks on the very manifest shape produced by the CLI and the demo script.

## Scores

| Category | Score | Verdict |
|---|---:|---|
| Correctness | 59 | Core leakage fixes are real, but the UI path still misrepresents scenario effects, `OFFLINE_MODE` is incomplete, and the target is still synthetic quarterly GDP. |
| Evaluation Rigor | 74 | Rolling-origin retraining and prediction-based metrics are real, but intervals are still residual-based approximations and the target process is still synthetic. |
| UX Truthfulness | 42 | The app can still launch on sample fallback after the demo script, and scenario effects are double-applied in the UI path. |
| Deployability | 61 | Docker, Compose, README, and a now-usable demo script exist, but the CLI/demo/UI split still undermines the easiest path and coverage tooling is still absent. |
| Test Coverage | 68 | The repo has useful targeted tests, but no coverage report, no UI integration tests, and no CLI-to-UI contract tests. |
| Maintainability | 57 | `sys.path` hacks are gone, but there are still duplicate run paths, duplicate manifest shapes, and stubbed artifact normalization that discards detail. |
| Overall | 60 | Better engineering, still not a coherent demo product. |

## Checklist Findings

### 1) Correctness

#### Works

- Target shift is grouped by country in [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py#L203) through [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py#L212).
- The code explicitly checks for cross-country leakage and raises if it occurs in [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py#L207) through [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py#L210).
- Training uses `available_date`-based forward holdout selection in [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py#L243) through [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py#L298).
- Backtesting enforces `train_country["available_date"] <= origin` and hard-fails on leakage in [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py#L840) through [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py#L850).

#### Fails / Risks

- The quarterly target is still fabricated from annual World Bank data in [mixed_frequency.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/features/mixed_frequency.py#L7) through [mixed_frequency.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/features/mixed_frequency.py#L28). This is not a real quarterly GDP target.
- The UI scenario path is still not trustworthy:
  - backend rerun receives the scenario in [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py#L687) through [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py#L697)
  - then the frontend applies another deterministic scenario shift in [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py#L727)
- The actual scenario semantics are questionable:
  - baseline normalized BRA run (`2026-03-03`): `-1.465089652793843`
  - `financial_tightening=+1.0`: `-1.040550784623791`
  - the forecast changed, but it moved in the opposite direction of the plain-English label

#### Validation Split

- The split logic does produce non-empty validation or an explicit fallback metadata record.
- That part is structurally correct now.
- But the normalized UI wrapper does not preserve the real train report. It writes a stub `train_report.json` with only run metadata in [run.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/run.py#L41) through [run.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/run.py#L52), so the actual split counts and fallback status are lost on the UI-facing path.

### 2) Artifacts

#### Works

- Run folders exist under `artifacts/runs/{run_id}`.
- Latest pointer works:
  - `artifacts/latest_run_id.txt` matched the most recent normalized run: `20260303T212148Z_X55C`
  - `artifacts/latest/manifest.json` matched the same run id
- Normalized manifests include `paths.forecast`, `paths.drivers`, `paths.world_snapshot`, `paths.train_report`, and `paths.backtest_report`.
- For normalized runs, the loader only returns countries listed in the manifest.

#### Fails / Risks

- Manifest schema is still not stable across entrypoints:
  - CLI/service path writes the full schema with `country_artifacts`
  - UI wrapper rewrites the same file into a normalized schema with `paths`
- Manifest schema is not versioned:
  - `schema_version` is absent from the latest normalized manifest
- The demo script still uses the CLI path, so it seeds the incompatible full manifest before launching the app.
- On that full manifest shape, `load_forecast_payloads_from_manifest()` returns no countries.
- Since [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py#L459) through [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py#L484) falls back to sample forecast data when no payloads are loaded, stale/non-manifest countries can effectively reappear in the UI through fallback data, even if the latest run only contains `BRA`.

### 3) UI Behavior

#### Verified

- `As-of date` is not discarded anymore. The UI reads it from [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py#L650).
- `Run / Refresh (recompute)` triggers a real rerun via [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py#L663) through [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py#L705).
- The title area shows `Last run: {run_id} at {created_at}` in [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py#L737) through [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py#L741).
- The UI shows an explicit artifact-viewer banner if the current `as_of` differs from the latest run in [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py#L740) through [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py#L741).

#### Verified By Runtime Output (Backend Contract Used By UI)

- As-of changes results:
  - normalized run `20260303T192026Z_UKUC` (`2025-03-03`): `-1.8612753759773213`
  - normalized run `20260303T192033Z_FZCX` (`2026-03-03`): `-1.465089652793843`
- Scenario changes results:
  - normalized run `20260303T192033Z_FZCX` (`financial_tightening=0.0`): `-1.465089652793843`
  - normalized run `20260303T212148Z_X55C` (`financial_tightening=1.0`): `-1.040550784623791`
- Refresh implies recomputation:
  - run ids changed across reruns (`...UKUC`, `...FZCX`, `...X55C`)

#### Fails / Risks

- The easiest demo flow is still misleading:
  - [demo.sh](/Users/mac/Desktop/atlas_gdp/scripts/demo.sh#L29) launches Streamlit immediately after a CLI run
  - that CLI run writes the non-normalized manifest
  - the app can therefore start in sample-fallback mode until the user manually clicks `Run / Refresh`
- Browser-level interaction was not fully automated in this audit because the Playwright browser tool could not keep a session open in this environment. UI behavior above is verified by code inspection, HTTP smoke checks, and the exact backend rerun contract the UI calls.

### 4) Backtest

#### Works

- Rolling-origin retraining exists in [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py#L782) through [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py#L919).
- The runtime logs confirmed retraining and completion:
  - `backtest_completed` with `origins: 12`
  - prediction rows: `96`
- Metrics are computed from prediction rows in [backtest.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/evaluation/backtest.py#L33) through [backtest.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/evaluation/backtest.py#L141), not from a static pre-saved score file.
- Coverage is computed from `q25/q75` and `q10/q90` using `interval_coverage()` in [backtest.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/evaluation/backtest.py#L48) through [backtest.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/evaluation/backtest.py#L69).

#### Risks

- Intervals are still residual-based approximations, not true predictive distributions.
- `interval_method` is honest about this, which is good, but it is still not production-grade uncertainty modeling.

### 5) Connectors

#### Real vs Demo

- Real online/cached when configured:
  - OECD SDMX: [oecd_sdmx.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/data/oecd_sdmx.py)
  - IMF SDMX: [imf_sdmx.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/data/imf_sdmx.py)
  - World Bank: live or offline sample depending on mode
  - FRED: live when API key is present
- Explicit demo-only:
  - BEA demo bridge: [bea_demo.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/data/bea_demo.py)

#### Verified

- OECD/IMF offline behavior and cache roundtrips are covered by tests in [test_offline_mode_blocks_network.py](/Users/mac/Desktop/atlas_gdp/tests/test_offline_mode_blocks_network.py) and [test_cache_roundtrip.py](/Users/mac/Desktop/atlas_gdp/tests/test_cache_roundtrip.py).
- Current demo-script run showed connector statuses:
  - `world_bank: offline_sample`
  - `oecd: offline_cache_missing`
  - `imf: offline_cache_missing`
  - `bea_demo: demo_embedded`

#### Fails / Risks

- `OFFLINE_MODE` still does not block FRED network calls if `FRED_API_KEY` is set:
  - [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py#L73) through [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py#L87)
  - [fred.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/data/fred.py#L14) through [fred.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/data/fred.py#L28)

### 6) Packaging + Config

#### Works

- `sys.path` hacks are gone from Python source files. A repo-wide search found none in `app/`, `scripts/`, or `src/`.
- Config loading is centralized in [settings.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/settings.py#L205) through [settings.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/settings.py#L271).
- Env and YAML resolution both exist in [settings.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/settings.py#L103) through [settings.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/settings.py#L145).
- The CLI works via module invocation:
  - `PYTHONPATH=src .venv/bin/python -m atlas_gdp.cli ...`
- Docker, Compose, and README demo instructions exist.

#### Fails / Risks

- Editable package installation was not revalidated in a fresh offline environment during this audit.
- The installed console script path still was not the path used for reliable verification; the working path in this environment is still `PYTHONPATH=src python -m atlas_gdp.cli`.
- `pytest-cov` / coverage tooling is not configured in [pyproject.toml](/Users/mac/Desktop/atlas_gdp/pyproject.toml) or [ci.yml](/Users/mac/Desktop/atlas_gdp/.github/workflows/ci.yml).

### 7) Tests

#### Verified

- `.venv/bin/pytest -q` passed with `17` tests.
- The current suite does cover:
  - target leakage
  - as-of split behavior
  - mixed-frequency quarterly alignment
  - offline connector blocking and cache roundtrip
  - manifest helper behavior
  - backtest walk-forward shapes and no-future-leakage
  - settings loading
  - run registry

#### Missing / Under-Tested

- No coverage report exists, so coverage percentage is unknown.
- No UI integration tests exist for [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py).
- No test proves the CLI-written manifest can be rendered by the Streamlit loaders.
- No test proves the demo script starts the app into a truthful initial state.
- No test covers the stub `train_report.json` behavior in normalized runs.

## What Works

- Country-grouped target shifting and leakage guard
- `available_date`-based train/validation split
- Rolling-origin backtest with retraining and prediction-based metrics
- Run directories and latest pointers
- Structured run logging
- Docker and Compose scaffolding
- Mac-friendly local demo script that now actually runs in the repo venv
- Streamlit server boots and responds over HTTP

## What Doesn’t

- The CLI/demo path still writes the wrong manifest shape for the UI
- The UI can still silently show sample fallback data after the advertised demo flow
- Scenario effects are still double-applied in the UI path
- `OFFLINE_MODE` is still not enforced consistently across all connectors
- Manifest schema is still unversioned
- The normalized run path fabricates a minimal `train_report.json` and discards the real train report detail

## What Is Still Risky

1. The easiest demo path is still not truthful on first load because it seeds the incompatible manifest before launching the app.
2. CLI and UI still have two manifest schemas and two runtime contracts.
3. Scenario labeling and scenario mechanics are still inconsistent.
4. FRED can still hit the network in offline mode.
5. The target is still synthetic quarterized annual GDP.
6. Sample fallback behavior still hides real loader failures.
7. Manifest schema changes can still break loaders silently because there is no `schema_version`.
8. The UI-facing `train_report.json` is currently a stub, not the real training report.
9. Browser-level UI behavior is still not covered by tests.
10. Installed-package validation is still weaker than module-path validation in the current local environment.

## Top 10 Remaining Risks (Ranked)

1. **CLI/UI manifest contract is still broken.**
   - Targets: [cli.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/cli.py), [run.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/run.py), [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py)
2. **The demo script still launches the app on a CLI-generated manifest that the UI loaders cannot read.**
   - Targets: [demo.sh](/Users/mac/Desktop/atlas_gdp/scripts/demo.sh), [cli.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/cli.py), [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py)
3. **Scenario effects are double-applied in the UI path.**
   - Targets: [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py), [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py)
4. **Scenario semantics are not trustworthy.**
   - Target: [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py#L595)
5. **`OFFLINE_MODE` is still incomplete because FRED bypasses it.**
   - Targets: [fred.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/data/fred.py), [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py)
6. **Manifest schema is still unversioned.**
   - Targets: [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py), [run.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/run.py)
7. **The UI-facing train report is fake/minimal.**
   - Target: [run.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/run.py#L41)
8. **The target is still synthetic, not real quarterly GDP.**
   - Target: [mixed_frequency.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/features/mixed_frequency.py)
9. **Sample fallback still masks hard data-path failures.**
   - Target: [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py#L459)
10. **No UI end-to-end tests exist.**
   - Targets: [tests](/Users/mac/Desktop/atlas_gdp/tests), [app/streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py)

## Next 5 PRs

1. **Unify manifest writing and reading**
   - Targets: [cli.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/cli.py), [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py), [run.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/run.py), [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py)
   - Make one canonical manifest schema.
   - Add `schema_version`.
   - Make CLI and UI both use the same manifest writer.

2. **Fix the demo path so first load is truthful**
   - Targets: [demo.sh](/Users/mac/Desktop/atlas_gdp/scripts/demo.sh), [cli.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/cli.py), [run.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/run.py)
   - Make `scripts/demo.sh` use the normalized UI-facing runner, not the raw CLI/service manifest path.

3. **Remove double scenario application**
   - Targets: [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py), [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py)
   - Choose one scenario source of truth.
   - Delete the other path.

4. **Make offline mode real for FRED**
   - Targets: [fred.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/data/fred.py), [service.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/service.py), [test_offline_mode_blocks_network.py](/Users/mac/Desktop/atlas_gdp/tests/test_offline_mode_blocks_network.py)
   - Add cache-only behavior.
   - Add a regression test.

5. **Preserve the real training report in normalized runs**
   - Targets: [run.py](/Users/mac/Desktop/atlas_gdp/src/atlas_gdp/pipeline/run.py), [train.py](/Users/mac/Desktop/atlas_gdp/scripts/train.py), [streamlit_app.py](/Users/mac/Desktop/atlas_gdp/app/streamlit_app.py)
   - Stop fabricating the minimal `train_report.json`.
   - Reuse the real split/count artifact and surface it in Audit.

## Bottom Line

This is a materially better codebase than the original prototype, but it is still not a production-grade demo.

The hardest remaining problem is no longer the modeling core. It is the mismatch between the product entrypoints:

- CLI
- demo script
- UI loader
- manifest schemas

Until those behave as one coherent system, the demo is still too easy to misread.
