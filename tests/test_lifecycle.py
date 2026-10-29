"""End-to-end tests for retraining, champion/challenger gate, human approval, and rollback."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scripts.manage_model import main as manage_model_main
from scripts.retrain_and_evaluate import main as retrain_main
from scripts.train import main as train_main

from mlops_live.data import TARGET_COLUMN
from mlops_live.lifecycle import fetch_lifecycle_audit_logs
from mlops_live.serving import create_prediction_engine


@pytest.fixture
def lifecycle_splits(tmp_path: Path) -> tuple[Path, Path]:
    """Create synthetic train and validation Parquet splits with strong non-linear signal."""
    rng = np.random.default_rng(77)

    def build_df(n: int) -> pd.DataFrame:
        pu = rng.choice([74, 75, 41, 42], size=n)
        do = rng.choice([42, 74, 238, 239], size=n)
        dist = rng.uniform(0.8, 12.0, size=n)
        pax = rng.choice([1.0, 2.0], size=n)
        hour = rng.integers(0, 24, size=n)
        dow = rng.integers(0, 7, size=n)

        # Non-linear interaction so HistGBDT beats linear Ridge on validation set
        rush_factor = np.where((hour >= 7) & (hour <= 19) & (dow < 5), 1.6, 0.9)
        non_linear_dist = np.power(dist, 1.25) * 2.4 * rush_factor
        duration = np.clip(2.5 + non_linear_dist + rng.normal(0.0, 0.15, size=n), 1.0, 60.0)
        return pd.DataFrame(
            {
                "PULocationID": pu.astype("int64"),
                "DOLocationID": do.astype("int64"),
                "trip_distance": dist.astype("float64"),
                "passenger_count": pax.astype("float64"),
                "pickup_hour": hour.astype("int64"),
                "pickup_day_of_week": dow.astype("int64"),
                TARGET_COLUMN: duration.astype("float64"),
            }
        )

    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    build_df(120).to_parquet(train_path, index=False)
    build_df(40).to_parquet(val_path, index=False)
    return train_path, val_path


def test_complete_retrain_evaluate_approve_and_rollback_lifecycle(
    lifecycle_splits: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    train_path, val_path = lifecycle_splits
    models_dir = tmp_path / "models"
    tracking_db = tmp_path / "mlflow.db"
    artifact_dir = tmp_path / "mlruns"
    predictions_db = tmp_path / "predictions.db"
    eval_report = tmp_path / "champion_vs_challenger.json"

    # 1. Bootstrap initial models: v2 (hist_gbdt_candidate) is @champion, v1 is @challenger
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

    status_initial = manage_model_main(
        [
            "--status",
            "--tracking-db",
            str(tracking_db),
            "--predictions-db",
            str(predictions_db),
        ]
    )
    assert status_initial["champion"]["version"] == "2"
    assert status_initial["challenger"]["version"] == "1"

    prod_model_path = models_dir / "model.joblib"
    prod_metrics_path = models_dir / "metrics.json"
    initial_model_bytes = prod_model_path.read_bytes()
    initial_metrics_text = prod_metrics_path.read_text(encoding="utf-8")

    # 2. Gate blocks promoting inferior @challenger v1 over @champion v2 without --force-promote
    with pytest.raises(ValueError, match="did not pass promotion gate"):
        manage_model_main(
            [
                "--approve",
                "--approved-by",
                "najahaja",
                "--eval-file",
                str(val_path),
                "--tracking-db",
                str(tracking_db),
                "--predictions-db",
                str(predictions_db),
                "--fallback-model",
                str(prod_model_path),
            ]
        )
    assert prod_model_path.read_bytes() == initial_model_bytes
    assert prod_metrics_path.read_text(encoding="utf-8") == initial_metrics_text

    # 3. Run automated retraining workflow: stages new @challenger v4 while keeping @champion at v2
    # and writes candidate artifacts to models/challenger/ without overwriting models/model.joblib
    retrain_res = retrain_main(
        [
            "--train-file",
            str(train_path),
            "--val-file",
            str(val_path),
            "--models-dir",
            str(models_dir),
            "--tracking-db",
            str(tracking_db),
            "--artifact-dir",
            str(artifact_dir),
            "--predictions-db",
            str(predictions_db),
            "--eval-report",
            str(eval_report),
        ]
    )
    assert retrain_res.retrained is True
    assert retrain_res.active_champion_version == "2"
    assert retrain_res.staged_challenger_version == "4"
    assert eval_report.exists()
    assert (models_dir / "challenger" / "model.joblib").exists()
    assert (models_dir / "challenger" / "metrics.json").exists()
    assert prod_model_path.read_bytes() == initial_model_bytes
    assert prod_metrics_path.read_text(encoding="utf-8") == initial_metrics_text

    report_payload = json.loads(eval_report.read_text(encoding="utf-8"))
    assert report_payload["champion"]["version"] == "2"
    assert report_payload["challenger"]["version"] == "4"
    assert report_payload["gate_passed"] is True

    # 4. Human approval requires non-empty --approved-by
    with pytest.raises(ValueError, match="requires a non-empty --approved-by"):
        manage_model_main(
            [
                "--approve",
                "--tracking-db",
                str(tracking_db),
                "--predictions-db",
                str(predictions_db),
            ]
        )
    assert prod_model_path.read_bytes() == initial_model_bytes

    # 4b. Simulate a failure writing the production fallback artifact during approval:
    # verify MLflow @champion stays at v2 and existing production artifacts remain intact.
    with pytest.MonkeyPatch.context() as mp:
        import joblib

        mp.setattr(
            joblib,
            "dump",
            lambda *_a, **_kw: (_ for _ in ()).throw(OSError("Simulated disk write error")),
        )
        with pytest.raises(OSError, match="Simulated disk write error"):
            manage_model_main(
                [
                    "--approve",
                    "--approved-by",
                    "najahaja",
                    "--eval-file",
                    str(val_path),
                    "--tracking-db",
                    str(tracking_db),
                    "--predictions-db",
                    str(predictions_db),
                    "--fallback-model",
                    str(prod_model_path),
                ]
            )
    assert prod_model_path.read_bytes() == initial_model_bytes
    assert prod_metrics_path.read_text(encoding="utf-8") == initial_metrics_text
    status_after_failed_promo = manage_model_main(
        [
            "--status",
            "--tracking-db",
            str(tracking_db),
            "--predictions-db",
            str(predictions_db),
        ]
    )
    assert status_after_failed_promo["champion"]["version"] == "2"

    # 5. Human approval promotes @challenger v4 -> @champion and sets @previous_champion -> v2
    approval_res = manage_model_main(
        [
            "--approve",
            "--approved-by",
            "najahaja",
            "--eval-file",
            str(val_path),
            "--tracking-db",
            str(tracking_db),
            "--predictions-db",
            str(predictions_db),
            "--fallback-model",
            str(prod_model_path),
        ]
    )
    assert approval_res["event_type"] == "human_approval_promotion"
    assert approval_res["previous_champion_version"] == "2"
    assert approval_res["new_champion_version"] == "4"
    promoted_metrics = json.loads(prod_metrics_path.read_text(encoding="utf-8"))
    assert promoted_metrics["mlflow"]["champion"]["version"] == "4"

    status_after_approval = manage_model_main(
        [
            "--status",
            "--tracking-db",
            str(tracking_db),
            "--predictions-db",
            str(predictions_db),
        ]
    )
    assert status_after_approval["champion"]["version"] == "4"
    assert status_after_approval["previous_champion"]["version"] == "2"

    # 6. Rollback to @previous_champion restores v2 as @champion
    rollback_prev_res = manage_model_main(
        [
            "--rollback-previous",
            "--approved-by",
            "najahaja",
            "--reason",
            "latency_spike_test",
            "--tracking-db",
            str(tracking_db),
            "--predictions-db",
            str(predictions_db),
            "--fallback-model",
            str(prod_model_path),
        ]
    )
    assert rollback_prev_res["event_type"] == "model_rollback"
    assert rollback_prev_res["previous_champion_version"] == "4"
    assert rollback_prev_res["new_champion_version"] == "2"
    restored_v2_metrics = json.loads(prod_metrics_path.read_text(encoding="utf-8"))
    assert restored_v2_metrics["mlflow"]["champion"]["version"] == "2"

    # 7. Explicit version rollback (--rollback-to-version 1) switches @champion to v1
    rollback_v1_res = manage_model_main(
        [
            "--rollback-to-version",
            "1",
            "--approved-by",
            "najahaja",
            "--tracking-db",
            str(tracking_db),
            "--predictions-db",
            str(predictions_db),
            "--fallback-model",
            str(prod_model_path),
        ]
    )
    assert rollback_v1_res["new_champion_version"] == "1"
    restored_v1_metrics = json.loads(prod_metrics_path.read_text(encoding="utf-8"))
    assert restored_v1_metrics["selected_model"] == "ridge_baseline"
    assert restored_v1_metrics["mlflow"]["champion"]["version"] == "1"

    # 8. Verify complete SQLite governance audit trail in predictions.db
    engine = create_prediction_engine(predictions_db)
    try:
        audits = fetch_lifecycle_audit_logs(engine, limit=20)
        event_types = [a["event_type"] for a in audits]
        assert "champion_vs_challenger_eval" in event_types
        assert "retrain_and_stage_challenger" in event_types
        assert "human_approval_promotion" in event_types
        assert "model_rollback" in event_types
    finally:
        engine.dispose()
