"""Integration tests for FastAPI model serving and SQLite prediction logging."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops_live.data import FEATURE_COLUMNS, TARGET_COLUMN
from mlops_live.models import build_ridge_baseline, run_baseline_vs_candidate_training
from mlops_live.serving import (
    count_logged_predictions,
    create_app,
    create_prediction_engine,
    fetch_recent_predictions,
    load_serving_model,
)
from mlops_live.tracking import build_sqlite_tracking_uri


@pytest.fixture
def synthetic_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create small synthetic train and validation DataFrames."""
    rng = np.random.default_rng(123)

    def make_df(n: int) -> pd.DataFrame:
        dist = rng.uniform(1.0, 10.0, size=n)
        return pd.DataFrame(
            {
                "PULocationID": rng.choice([74, 75, 41, 42], size=n).astype("int64"),
                "DOLocationID": rng.choice([42, 74, 238, 239], size=n).astype("int64"),
                "trip_distance": dist.astype("float64"),
                "passenger_count": rng.choice([1.0, 2.0], size=n).astype("float64"),
                "pickup_hour": rng.integers(0, 24, size=n).astype("int64"),
                "pickup_day_of_week": rng.integers(0, 7, size=n).astype("int64"),
                TARGET_COLUMN: (3.0 + dist * 3.0 + rng.normal(0.0, 0.2, size=n)).astype("float64"),
            }
        )

    return make_df(40), make_df(15)


def test_serving_fallback_to_local_joblib_and_sqlite_logging(
    synthetic_splits: tuple[pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """Verify FastAPI falls back to models/model.joblib and logs predictions to SQLite."""
    train_df, _ = synthetic_splits
    pipeline = build_ridge_baseline()
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    model_path = models_dir / "model.joblib"
    metrics_path = models_dir / "metrics.json"
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps({"selected_model": "ridge_baseline"}), encoding="utf-8")

    predictions_db = tmp_path / "predictions.db"
    non_existent_tracking_db = tmp_path / "missing_mlflow.db"

    app = create_app(
        tracking_db=non_existent_tracking_db,
        fallback_model_path=model_path,
        fallback_metrics_path=metrics_path,
        predictions_db=predictions_db,
    )

    with TestClient(app) as client:
        # 1. Check /health before any predictions
        health_res = client.get("/health")
        assert health_res.status_code == 200
        health_data = health_res.json()
        assert health_data["status"] == "ok"
        assert health_data["model_loaded"] is True
        assert health_data["model_name"] == "ridge_baseline"
        assert health_data["model_source"] == "local_joblib"
        assert health_data["predictions_logged"] == 0

        # 2. Check /model-info
        info_res = client.get("/model-info")
        assert info_res.status_code == 200
        info_data = info_res.json()
        assert info_data["feature_columns"] == FEATURE_COLUMNS
        assert info_data["target_column"] == TARGET_COLUMN
        assert info_data["model_source"] == "local_joblib"

        # 3. Call POST /predict with valid payload
        payload = {
            "PULocationID": 74,
            "DOLocationID": 42,
            "trip_distance": 3.5,
            "passenger_count": 2.0,
            "pickup_hour": 9,
            "pickup_day_of_week": 1,
        }
        pred_res = client.post("/predict", json=payload)
        assert pred_res.status_code == 200
        pred_data = pred_res.json()
        assert pred_data["predicted_duration_minutes"] >= 1.0
        assert pred_data["model_name"] == "ridge_baseline"
        assert pred_data["model_source"] == "local_joblib"
        assert pred_data["latency_ms"] >= 0.0
        assert len(pred_data["request_id"]) > 10

        # 4. Verify invalid payload returns 422 and does not increment prediction log count
        invalid_payload = {**payload, "trip_distance": -5.0}
        bad_res = client.post("/predict", json=invalid_payload)
        assert bad_res.status_code == 422

        # 5. Check /health after prediction
        health_after = client.get("/health").json()
        assert health_after["predictions_logged"] == 1

    # Verify SQLite predictions.db persisted exact features and metadata
    engine = create_prediction_engine(predictions_db)
    try:
        assert count_logged_predictions(engine) == 1
        rows = fetch_recent_predictions(engine, limit=10)
        assert len(rows) == 1
        row = rows[0]
        assert row["request_id"] == pred_data["request_id"]
        assert row["PULocationID"] == 74
        assert row["DOLocationID"] == 42
        assert row["trip_distance"] == pytest.approx(3.5)
        assert row["passenger_count"] == pytest.approx(2.0)
        assert row["pickup_hour"] == 9
        assert row["pickup_day_of_week"] == 1
        assert row["predicted_duration_minutes"] == pytest.approx(
            pred_data["predicted_duration_minutes"]
        )
    finally:
        engine.dispose()


def test_serving_loads_registered_champion_from_mlflow_and_reloads(
    synthetic_splits: tuple[pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """Verify FastAPI loads @champion from SQLite MLflow Registry and supports /reload-model."""
    train_df, val_df = synthetic_splits
    models_dir = tmp_path / "models"
    tracking_db = tmp_path / "mlflow.db"
    artifact_dir = tmp_path / "mlruns"
    predictions_db = tmp_path / "predictions.db"

    tracking_uri = build_sqlite_tracking_uri(tracking_db)
    run_baseline_vs_candidate_training(
        train_df=train_df,
        val_df=val_df,
        output_dir=models_dir,
        mlflow_tracking_uri=tracking_uri,
        mlflow_artifact_root=artifact_dir,
    )

    app = create_app(
        tracking_db=tracking_db,
        fallback_model_path=models_dir / "model.joblib",
        fallback_metrics_path=models_dir / "metrics.json",
        predictions_db=predictions_db,
        alias="champion",
    )

    with TestClient(app) as client:
        info = client.get("/model-info").json()
        assert info["model_source"] == "mlflow_registry"
        assert info["model_alias"] == "champion"
        assert info["model_version"] in {"1", "2"}

        pred_res = client.post(
            "/predict",
            json={
                "PULocationID": 75,
                "DOLocationID": 238,
                "trip_distance": 4.2,
                "passenger_count": 1.0,
                "pickup_hour": 17,
                "pickup_day_of_week": 3,
            },
        )
        assert pred_res.status_code == 200
        assert pred_res.json()["model_source"] == "mlflow_registry"

        reload_res = client.post("/reload-model")
        assert reload_res.status_code == 200
        assert reload_res.json()["model_source"] == "mlflow_registry"


def test_load_serving_model_raises_when_neither_source_exists(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No registered model"):
        load_serving_model(
            tracking_db=tmp_path / "no_mlflow.db",
            fallback_model_path=tmp_path / "no_model.joblib",
        )


def test_delayed_ground_truth_feedback_and_realized_online_metrics(
    synthetic_splits: tuple[pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """Verify POST /feedback records ground-truth duration and computes realized MAE/RMSE."""
    train_df, _ = synthetic_splits
    pipeline = build_ridge_baseline()
    pipeline.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    predictions_db = tmp_path / "predictions.db"

    app = create_app(
        tracking_db=tmp_path / "no_mlflow.db",
        fallback_model_path=model_path,
        predictions_db=predictions_db,
    )

    with TestClient(app) as client:
        # Before any feedback, realized_performance has feedback_count=0 and None metrics
        init_summary = client.get("/api/dashboard-summary").json()
        assert init_summary["realized_performance"] == {
            "feedback_count": 0,
            "realized_mae": None,
            "realized_rmse": None,
        }

        # Log a prediction
        pred_res = client.post(
            "/predict",
            json={
                "PULocationID": 74,
                "DOLocationID": 42,
                "trip_distance": 3.0,
                "passenger_count": 1.0,
                "pickup_hour": 10,
                "pickup_day_of_week": 2,
            },
        )
        assert pred_res.status_code == 200
        pred_body = pred_res.json()
        req_id = pred_body["request_id"]
        pred_min = float(pred_body["predicted_duration_minutes"])

        # Submit delayed ground-truth feedback (+2.0 min above predicted duration)
        fb_res = client.post(
            "/feedback",
            json={
                "request_id": req_id,
                "actual_duration_minutes": round(min(60.0, pred_min + 2.0), 4),
            },
        )
        assert fb_res.status_code == 200
        fb_body = fb_res.json()
        assert fb_body["request_id"] == req_id
        assert fb_body["abs_error_minutes"] == pytest.approx(2.0, abs=1e-2)

        # Unknown request_id returns 404
        not_found = client.post(
            "/feedback",
            json={"request_id": "non-existent-id", "actual_duration_minutes": 12.0},
        )
        assert not_found.status_code == 404

        # Verify dashboard summary and HTML reflect realized MAE/RMSE
        after_summary = client.get("/api/dashboard-summary").json()
        assert after_summary["realized_performance"]["feedback_count"] == 1
        assert after_summary["realized_performance"]["realized_mae"] == pytest.approx(2.0, abs=1e-2)
        assert after_summary["realized_performance"]["realized_rmse"] == pytest.approx(
            2.0, abs=1e-2
        )

        html_text = client.get("/dashboard").text
        assert "Realized Online MAE (Feedback)" in html_text
        assert "(1 labels)" in html_text
