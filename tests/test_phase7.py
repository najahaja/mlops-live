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
from mlops_live.lifecycle import record_lifecycle_audit
from mlops_live.models import build_ridge_baseline
from mlops_live.monitoring.drift import DriftReportRecord
from mlops_live.serving import create_app, create_prediction_engine, log_prediction_record


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


def test_dashboard_escapes_hostile_html_and_xss_payloads(tmp_path: Path) -> None:
    """Verify /dashboard escapes <script> and attribute-like HTML strings while JSON stays raw."""
    import json

    from sqlalchemy.orm import Session

    train_df = _make_split(30, seed=5)
    pipe = build_ridge_baseline()
    pipe.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    joblib.dump(pipe, model_path)
    hostile_model_name = 'ridge<script>alert(1)</script>" onmouseover="evil()'
    metrics_path.write_text(json.dumps({"selected_model": hostile_model_name}), encoding="utf-8")

    pred_db = tmp_path / "predictions.db"
    engine = create_prediction_engine(pred_db)
    try:
        log_prediction_record(
            engine,
            {
                "request_id": "<img/src=x>",
                "created_at_utc": "2026-10-29T12:00:00+00:00",
                "model_name": hostile_model_name,
                "model_version": '1" onfocus="bad()',
                "model_alias": "fallback",
                "model_source": "local_joblib",
                "PULocationID": 74,
                "DOLocationID": 42,
                "trip_distance": 2.5,
                "passenger_count": 1.0,
                "pickup_hour": 10,
                "pickup_day_of_week": 2,
                "predicted_duration_minutes": 11.2,
                "latency_ms": 3.4,
            },
        )
        record_lifecycle_audit(
            engine=engine,
            event_type='eval"<script>',
            actor="<script>alert('actor')</script>",
            previous_champion_version="1<x>",
            new_champion_version="2&y",
            challenger_version="3'z",
            gate_passed=True,
            notes='Note with <script>alert("xss")</script> & "quotes"',
            details={},
        )
        with Session(engine) as session, session.begin():
            session.add(
                DriftReportRecord(
                    created_at_utc="2026-10-29T12:05:00+00:00",
                    reference_source="train.parquet",
                    current_source='<svg onload="alert(2)">',
                    reference_rows=100,
                    current_rows=50,
                    max_psi=0.31,
                    max_ks_stat=0.14,
                    overall_drift_detected=True,
                    retraining_recommended=True,
                    drifted_columns_csv='trip_distance,<b onmouseover="bad()">col</b>',
                    report_json="{}",
                )
            )
    finally:
        engine.dispose()

    app = create_app(
        tracking_db=tmp_path / "no_mlflow.db",
        fallback_model_path=model_path,
        fallback_metrics_path=metrics_path,
        predictions_db=pred_db,
    )
    with TestClient(app) as client:
        # JSON endpoint preserves raw values without corrupting strings
        summary = client.get("/api/dashboard-summary").json()
        assert summary["active_model"]["model_name"] == hostile_model_name
        assert summary["lifecycle_audit_logs"][0]["actor"] == "<script>alert('actor')</script>"

        # HTML endpoint escapes all hostile tags and quotes
        html_body = client.get("/dashboard").text
        assert "<script>alert(1)</script>" not in html_body
        assert "<script>alert('actor')</script>" not in html_body
        assert '<svg onload="alert(2)">' not in html_body
        assert "<img/src=x>" not in html_body
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_body
        assert "&lt;script&gt;alert(&#x27;actor&#x27;)&lt;/script&gt;" in html_body
        assert "&lt;svg onload=&quot;alert(2)&quot;&gt;" in html_body
        assert "&amp; &quot;quotes&quot;" in html_body
