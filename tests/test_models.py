"""Unit and CLI tests for baseline and candidate model training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scripts.train import main as train_cli_main

from mlops_live.data import FEATURE_COLUMNS, TARGET_COLUMN
from mlops_live.models import (
    RegressionMetrics,
    build_ridge_baseline,
    compare_models,
    evaluate_predictions,
    train_and_evaluate_model,
)


@pytest.fixture
def synthetic_train_val_splits(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    """Create synthetic non-linear train and validation Parquet splits in tmp_path."""
    rng = np.random.default_rng(42)

    def make_split(n: int) -> pd.DataFrame:
        pu = rng.choice([74, 75, 41, 42, 166], size=n)
        do = rng.choice([42, 74, 238, 239, 151], size=n)
        dist = rng.uniform(0.5, 12.0, size=n)
        pax = rng.choice([1.0, 2.0, 3.0], size=n)
        hour = rng.integers(0, 24, size=n)
        dow = rng.integers(0, 7, size=n)

        # True signal with rush-hour interaction so tree/linear models beat dummy median
        rush_multiplier = np.where((hour >= 7) & (hour <= 19) & (dow < 5), 1.4, 1.0)
        route_bonus = np.where(pu == 74, 3.0, 0.0) + np.where(do == 238, 2.5, 0.0)
        duration = np.clip(
            3.0 + (dist * 3.2 * rush_multiplier) + route_bonus + rng.normal(0.0, 0.3, size=n),
            1.0,
            60.0,
        )
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

    train_df = make_split(80)
    val_df = make_split(30)

    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    return train_path, val_path, train_df, val_df


def test_evaluate_predictions_and_compare_models() -> None:
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_base = np.array([12.0, 18.0, 34.0, 36.0])  # errors: 2, 2, 4, 4 -> MAE = 3.0
    y_cand = np.array([11.0, 19.0, 31.0, 39.0])  # errors: 1, 1, 1, 1 -> MAE = 1.0

    base_metrics = evaluate_predictions(y_true, y_base)
    cand_metrics = evaluate_predictions(y_true, y_cand)

    assert base_metrics.mae == pytest.approx(3.0)
    assert cand_metrics.mae == pytest.approx(1.0)
    assert cand_metrics.rmse == pytest.approx(1.0)
    assert cand_metrics.num_samples == 4

    comp = compare_models(baseline=base_metrics, candidate=cand_metrics)
    assert comp["candidate_is_better"] is True
    assert comp["mae_reduction_minutes"] == pytest.approx(2.0)

    # Validation checks on invalid inputs
    with pytest.raises(ValueError, match="empty"):
        evaluate_predictions([], [])
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_predictions([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        evaluate_predictions([1.0, np.nan], [1.0, 2.0])


def test_train_and_evaluate_validates_feature_columns(
    synthetic_train_val_splits: tuple[Path, Path, pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, train_df, val_df = synthetic_train_val_splits
    bad_X = train_df[FEATURE_COLUMNS].drop(columns=["trip_distance"])

    with pytest.raises(ValueError, match="missing required columns"):
        train_and_evaluate_model(
            model_name="ridge",
            pipeline=build_ridge_baseline(),
            X_train=bad_X,
            y_train=train_df[TARGET_COLUMN],
            X_val=val_df[FEATURE_COLUMNS],
            y_val=val_df[TARGET_COLUMN],
        )


def test_train_cli_end_to_end_and_overwrite_protection(
    synthetic_train_val_splits: tuple[Path, Path, pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    train_path, val_path, _, val_df = synthetic_train_val_splits
    models_dir = tmp_path / "models_out"

    summary = train_cli_main(
        [
            "--train-file",
            str(train_path),
            "--val-file",
            str(val_path),
            "--output-dir",
            str(models_dir),
        ]
    )

    assert summary.model_path == models_dir / "model.joblib"
    assert summary.metrics_path == models_dir / "metrics.json"
    assert summary.model_path.exists()
    assert summary.metrics_path.exists()

    # Both learned models should beat the dummy median baseline on validation MAE
    by_name: dict[str, RegressionMetrics] = {
        r.model_name: r.val_metrics for r in summary.all_results
    }
    assert by_name["ridge_baseline"].mae < by_name["dummy_median"].mae
    assert by_name["hist_gbdt_candidate"].mae < by_name["dummy_median"].mae

    # Verify saved model artifact reloads and predicts cleanly on validation features
    loaded_pipeline = joblib.load(summary.model_path)
    preds = loaded_pipeline.predict(val_df[FEATURE_COLUMNS])
    assert len(preds) == len(val_df)
    assert np.isfinite(preds).all()

    # Verify metrics.json schema and contents
    metrics_payload = json.loads(summary.metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["selected_model"] == summary.selected_model
    assert metrics_payload["feature_columns"] == FEATURE_COLUMNS
    assert metrics_payload["target_column"] == TARGET_COLUMN
    assert set(metrics_payload["models"].keys()) == {
        "dummy_median",
        "ridge_baseline",
        "hist_gbdt_candidate",
    }

    # Overwrite protection blocks a second run without --force
    with pytest.raises(FileExistsError, match="Pass --force to overwrite"):
        train_cli_main(
            [
                "--train-file",
                str(train_path),
                "--val-file",
                str(val_path),
                "--output-dir",
                str(models_dir),
            ]
        )

    # Succeeds when --force is explicitly passed
    summary_forced = train_cli_main(
        [
            "--train-file",
            str(train_path),
            "--val-file",
            str(val_path),
            "--output-dir",
            str(models_dir),
            "--force",
        ]
    )
    assert summary_forced.model_path.exists()
