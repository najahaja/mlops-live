"""Tests for holdout evaluation, monitoring dashboard, and concurrent load testing."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from scripts.evaluate_holdout import main as evaluate_holdout_main
from scripts.load_test import main as load_test_main
from scripts.train import main as train_main

from mlops_live.data import FEATURE_COLUMNS, TARGET_COLUMN
from mlops_live.models import build_ridge_baseline
from mlops_live.serving import create_app


def _make_split(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dist = rng.uniform(1.0, 10.0, size=n)
    return pd.DataFrame(
        {
            "PULocationID": rng.choice([74, 75, 41, 42], size=n).astype("int64"),
            "DOLocationID": rng.choice([42, 74, 238, 239], size=n).astype("int64"),
            "trip_distance": dist.astype("float64"),
            "passenger_count": rng.choice([1.0, 2.0], size=n).astype("float64"),
            "pickup_hour": rng.integers(0, 24, size=n).astype("int64"),
            "pickup_day_of_week": rng.integers(0, 7, size=n).astype("int64"),
            TARGET_COLUMN: (3.0 + dist * 2.8 + rng.normal(0.0, 0.2, size=n)).astype("float64"),
        }
    )


def test_dashboard_and_concurrent_load_test(tmp_path: Path) -> None:
    """Verify /dashboard, /api/dashboard-summary, and concurrent load testing."""
    train_df = _make_split(40, seed=1)
    pipe = build_ridge_baseline()
    pipe.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipe, model_path)
    pred_db = tmp_path / "predictions.db"
    no_mlflow_db = tmp_path / "no_mlflow.db"
    load_report_path = tmp_path / "load_test_report.json"

    # 1. Run concurrent load test (20 requests across 4 worker threads)
    load_report = load_test_main(
        [
            "--num-requests",
            "20",
            "--concurrency",
            "4",
            "--tracking-db",
            str(no_mlflow_db),
            "--fallback-model",
            str(model_path),
            "--predictions-db",
            str(pred_db),
            "--output-json",
            str(load_report_path),
            "--max-p95-latency-ms",
            "2000.0",
        ]
    )
    assert load_report["num_requests"] == 20
    assert load_report["success_count"] == 20
    assert load_report["error_rate"] == 0.0
    assert load_report["sla"]["sla_passed"] is True
    assert load_report_path.exists()

    # 2. Verify /api/dashboard-summary and /dashboard HTML reflect the 20 logged requests
    app = create_app(
        tracking_db=no_mlflow_db,
        fallback_model_path=model_path,
        predictions_db=pred_db,
    )
    with TestClient(app) as client:
        summary_res = client.get("/api/dashboard-summary")
        assert summary_res.status_code == 200
        summary_json = summary_res.json()
        assert summary_json["total_predictions_logged"] == 20
        assert summary_json["latency_stats"]["mean_ms"] >= 0.0
        assert len(summary_json["recent_predictions"]) == 15

        html_res = client.get("/dashboard")
        assert html_res.status_code == 200
        assert "ML-Ops-Live — Production MLOps Dashboard" in html_res.text


def test_evaluate_holdout_cli(tmp_path: Path) -> None:
    """Verify holdout evaluation CLI scores @champion and @challenger on isolated test.parquet."""
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"

    _make_split(60, seed=10).to_parquet(train_path, index=False)
    _make_split(20, seed=11).to_parquet(val_path, index=False)
    _make_split(25, seed=12).to_parquet(test_path, index=False)

    tracking_db = tmp_path / "mlflow.db"
    artifact_dir = tmp_path / "mlruns"
    models_dir = tmp_path / "models"
    holdout_json = tmp_path / "holdout_test_metrics.json"

    train_main(
        [
            "--train-file",
            str(train_path),
            "--val-file",
            str(val_path),
            "--output-dir",
            str(models_dir),
            "--tracking-db",
            str(tracking_db),
            "--artifact-dir",
            str(artifact_dir),
        ]
    )

    payload = evaluate_holdout_main(
        [
            "--test-file",
            str(test_path),
            "--tracking-db",
            str(tracking_db),
            "--output-json",
            str(holdout_json),
        ]
    )
    assert payload["test_rows"] == 25
    assert payload["champion"]["test_metrics"]["num_samples"] == 25
    assert payload["challenger"]["test_metrics"]["num_samples"] == 25
    assert holdout_json.exists()
