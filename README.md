# ML-Ops-Live

A production-style MLOps platform built around a real, continuously published public dataset:
**NYC TLC Green Taxi trip records**, used to predict **trip duration (minutes)**.

> Status: **Phase 1–3 complete (Data Pipeline, Baseline & Candidate Models, SQLite MLflow Tracking & Model Registry).**

## Planned capabilities

- Reproducible data ingestion from the official TLC monthly Parquet files
- Leakage-safe, time-based train/validation/test splits
- Baseline and candidate models with MLflow experiment tracking
- FastAPI model serving with persistent prediction records
- Data/prediction drift monitoring
- Automated retraining with champion/challenger evaluation
- Human approval gate and rollback
- Automated tests and GitHub Actions CI
- Monitoring dashboard and load testing

## Environment

- Windows 11, PowerShell, Python 3.11
- Native local processes (no Docker)

## Project layout

```
configs/          configuration files
data/raw/         downloaded TLC files (git-ignored)
data/interim/     intermediate data (git-ignored)
data/processed/   model-ready data (git-ignored)
models/           local model artifacts (git-ignored)
scripts/          command-line entry points
src/mlops_live/   application package
tests/            automated tests
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

For an exact, fully pinned environment use `requirements-lock.txt` instead of `requirements-dev.txt`.

## Checks

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## Usage

### Ingest Data
Download monthly Green Taxi parquet data directly from NYC TLC:

```powershell
python scripts/download_data.py --year 2024 --month 1
```

### Prepare Data
Clean raw records, extract leakage-safe features, and write chronological train/validation/test splits (`data/processed/train.parquet`, `val.parquet`, `test.parquet`):

```powershell
python scripts/prepare_data.py --input-file data/raw/green_tripdata_2024-01.parquet --output-dir data/processed
```
(Pass `--force` to overwrite existing split files.)

### Train & Evaluate Baseline vs. Candidate Models (with MLflow Tracking & Registry)
Train the median dummy reference, one-hot `Ridge` linear baseline, and target-encoded `HistGradientBoostingRegressor` candidate on `train.parquet`, evaluate on `val.parquet` (keeping `test.parquet` isolated), log runs to SQLite-backed MLflow (`mlflow.db` + `mlruns/`), register `@champion` and `@challenger` model versions, and save `models/model.joblib` and `models/metrics.json`:

```powershell
python scripts/train.py --train-file data/processed/train.parquet --val-file data/processed/val.parquet --output-dir models
```
(Pass `--force` to overwrite existing model artifacts; pass `--promote-winner` to promote a newly trained winner over an existing `@champion`.)

## Data source

NYC Taxi & Limousine Commission (TLC) Trip Record Data:
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
