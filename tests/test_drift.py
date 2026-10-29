"""Unit and CLI tests for PSI and Kolmogorov-Smirnov drift monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.check_drift import main as check_drift_main

from mlops_live.data import TARGET_COLUMN
from mlops_live.monitoring import (
    compute_ks_test,
    compute_psi,
    evaluate_dataset_drift,
    fetch_recent_drift_reports,
)
from mlops_live.serving import create_prediction_engine, log_prediction_record


def _make_taxi_df(
    n: int,
    seed: int = 42,
    distance_shift: float = 0.0,
    duration_shift: float = 0.0,
    hour_override: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dist = rng.uniform(1.0, 8.0, size=n) + distance_shift
    hours = (
        np.full(n, hour_override, dtype="int64")
        if hour_override is not None
        else rng.integers(0, 24, size=n, dtype="int64")
    )
    return pd.DataFrame(
        {
            "PULocationID": rng.choice([74, 75, 41, 42, 166], size=n).astype("int64"),
            "DOLocationID": rng.choice([42, 74, 238, 239, 151], size=n).astype("int64"),
            "trip_distance": dist.astype("float64"),
            "passenger_count": rng.choice([1.0, 2.0, 3.0], size=n).astype("float64"),
            "pickup_hour": hours,
            "pickup_day_of_week": rng.integers(0, 7, size=n).astype("int64"),
            TARGET_COLUMN: (4.0 + dist * 2.5 + duration_shift).astype("float64"),
        }
    )


def test_compute_psi_and_ks_identical_vs_shifted() -> None:
    rng = np.random.default_rng(10)
    ref = rng.normal(15.0, 3.0, size=500)
    cur_same = rng.normal(15.0, 3.0, size=500)
    cur_shifted = rng.normal(25.0, 3.0, size=500)

    psi_same = compute_psi(ref, cur_same)
    psi_shifted = compute_psi(ref, cur_shifted)
    assert psi_same < 0.10
    assert psi_shifted > 0.50

    ks_same, p_same = compute_ks_test(ref, cur_same)
    ks_shifted, p_shifted = compute_ks_test(ref, cur_shifted)
    assert ks_same < 0.10
    assert p_same > 0.01
    assert ks_shifted > 0.70
    assert p_shifted < 1e-10

    with pytest.raises(ValueError, match="non-empty"):
        compute_psi([], [1.0, 2.0])


def test_evaluate_dataset_drift_no_drift_vs_significant_drift() -> None:
    ref_df = _make_taxi_df(400, seed=1)
    stable_cur_df = _make_taxi_df(400, seed=2)
    drifted_cur_df = _make_taxi_df(
        400, seed=3, distance_shift=12.0, duration_shift=25.0, hour_override=3
    )

    stable_report = evaluate_dataset_drift(ref_df, stable_cur_df)
    assert stable_report.overall_drift_detected is False
    assert stable_report.retraining_recommended is False
    assert stable_report.drifted_columns == []

    drifted_report = evaluate_dataset_drift(ref_df, drifted_cur_df)
    assert drifted_report.overall_drift_detected is True
    assert drifted_report.retraining_recommended is True
    assert "trip_distance" in drifted_report.drifted_columns
    assert "pickup_hour" in drifted_report.drifted_columns
    assert "predicted_duration_minutes" in drifted_report.drifted_columns


def test_check_drift_cli_with_parquet_and_sqlite_predictions_db(tmp_path: Path) -> None:
    ref_df = _make_taxi_df(200, seed=10)
    ref_path = tmp_path / "train.parquet"
    ref_df.to_parquet(ref_path, index=False)

    # 1. Populate synthetic drifted production logs in SQLite predictions.db
    pred_db = tmp_path / "predictions.db"
    engine = create_prediction_engine(pred_db)
    try:
        drifted_df = _make_taxi_df(100, seed=20, distance_shift=15.0, duration_shift=30.0)
        for idx, row in drifted_df.iterrows():
            log_prediction_record(
                engine,
                {
                    "request_id": f"req-{idx}",
                    "created_at_utc": "2026-10-29T12:00:00+00:00",
                    "model_name": "hist_gbdt_candidate",
                    "model_version": "2",
                    "model_alias": "champion",
                    "model_source": "mlflow_registry",
                    "PULocationID": int(row["PULocationID"]),
                    "DOLocationID": int(row["DOLocationID"]),
                    "trip_distance": float(row["trip_distance"]),
                    "passenger_count": float(row["passenger_count"]),
                    "pickup_hour": int(row["pickup_hour"]),
                    "pickup_day_of_week": int(row["pickup_day_of_week"]),
                    "predicted_duration_minutes": float(row[TARGET_COLUMN]),
                    "latency_ms": 5.0,
                },
            )
    finally:
        engine.dispose()

    report_json = tmp_path / "drift_report.json"
    report = check_drift_main(
        [
            "--reference-file",
            str(ref_path),
            "--use-predictions-db",
            "--predictions-db",
            str(pred_db),
            "--tracking-db",
            str(tmp_path / "no_mlflow.db"),
            "--fallback-model",
            str(tmp_path / "no_model.joblib"),
            "--output-json",
            str(report_json),
        ]
    )

    assert report.overall_drift_detected is True
    assert report.retraining_recommended is True
    assert report_json.exists()

    saved_payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert saved_payload["overall_drift_detected"] is True
    assert "trip_distance" in saved_payload["drifted_columns"]

    # Verify drift report was also persisted to SQLite drift_reports table
    engine = create_prediction_engine(pred_db)
    try:
        history = fetch_recent_drift_reports(engine)
        assert len(history) == 1
        assert history[0]["overall_drift_detected"] is True
        assert "trip_distance" in history[0]["drifted_columns"]
    finally:
        engine.dispose()
