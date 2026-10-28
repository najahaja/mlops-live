"""Baseline and candidate regression model pipelines and training workflow."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from mlops_live.data.clean import FEATURE_COLUMNS, TARGET_COLUMN
from mlops_live.models.evaluate import RegressionMetrics, compare_models, evaluate_predictions

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = ["PULocationID", "DOLocationID", "pickup_hour", "pickup_day_of_week"]
NUMERIC_FEATURES = ["trip_distance", "passenger_count"]
LOCATION_FEATURES = ["PULocationID", "DOLocationID"]
PASSTHROUGH_TREE_FEATURES = [
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_day_of_week",
]


class TrainedModelResult(NamedTuple):
    """Result container for a fitted model pipeline and its train/val metrics."""

    model_name: str
    pipeline: Pipeline
    train_metrics: RegressionMetrics
    val_metrics: RegressionMetrics
    fit_seconds: float
    hyperparameters: dict[str, Any]


def build_dummy_baseline() -> Pipeline:
    """Create a trivial median-prediction baseline for sanity-checking."""
    return Pipeline([("model", DummyRegressor(strategy="median"))])


def build_ridge_baseline(alpha: float = 1.0, random_state: int = 42) -> Pipeline:
    """Create a linear baseline pipeline with one-hot categorical encoding and scaled numerics."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=alpha, random_state=random_state)),
        ]
    )


def build_hist_gbdt_candidate(
    max_iter: int = 200,
    learning_rate: float = 0.08,
    max_leaf_nodes: int = 63,
    random_state: int = 42,
    cv_splits: int = 5,
) -> Pipeline:
    """Create a gradient-boosted tree candidate pipeline with cross-fitted target encoding."""
    cv_splitter = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "loc_target_enc",
                TargetEncoder(cv=cv_splitter, smooth="auto"),
                LOCATION_FEATURES,
            ),
            ("passthrough", "passthrough", PASSTHROUGH_TREE_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=max_iter,
                    learning_rate=learning_rate,
                    max_leaf_nodes=max_leaf_nodes,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _validate_feature_matrix(X: pd.DataFrame, split_name: str) -> None:
    """Ensure feature DataFrame is non-empty, has exact FEATURE_COLUMNS, and no NaNs."""
    if X.empty:
        raise ValueError(f"Feature matrix '{split_name}' is empty.")
    missing_cols = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing_cols:
        raise ValueError(
            f"Feature matrix '{split_name}' is missing required columns: {missing_cols}"
        )
    if list(X.columns) != FEATURE_COLUMNS:
        raise ValueError(
            f"Feature matrix '{split_name}' columns {list(X.columns)} do not match "
            f"expected order {FEATURE_COLUMNS}."
        )
    if X.isna().any().any():
        raise ValueError(f"Feature matrix '{split_name}' contains NaN values.")


def train_and_evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    hyperparameters: dict[str, Any] | None = None,
) -> TrainedModelResult:
    """Fit a model pipeline on training data and evaluate on both train and validation splits.

    Args:
        model_name: Identifier for the model candidate.
        pipeline: Unfitted scikit-learn Pipeline.
        X_train: Training feature DataFrame matching FEATURE_COLUMNS.
        y_train: Training target Series.
        X_val: Validation feature DataFrame matching FEATURE_COLUMNS.
        y_val: Validation target Series.
        hyperparameters: Optional dictionary of model hyperparameters for metadata logging.

    Returns:
        TrainedModelResult with fitted pipeline, metrics, and fit duration.
    """
    _validate_feature_matrix(X_train, "X_train")
    _validate_feature_matrix(X_val, "X_val")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train row counts do not match.")
    if len(X_val) != len(y_val):
        raise ValueError("X_val and y_val row counts do not match.")

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    train_preds = pipeline.predict(X_train)
    val_preds = pipeline.predict(X_val)

    train_metrics = evaluate_predictions(y_train, train_preds)
    val_metrics = evaluate_predictions(y_val, val_preds)

    logger.info(
        "[%s] fit=%.3fs | Train MAE=%.4f, RMSE=%.4f, R2=%.4f | Val MAE=%.4f, RMSE=%.4f, R2=%.4f",
        model_name,
        fit_seconds,
        train_metrics.mae,
        train_metrics.rmse,
        train_metrics.r2,
        val_metrics.mae,
        val_metrics.rmse,
        val_metrics.r2,
    )

    return TrainedModelResult(
        model_name=model_name,
        pipeline=pipeline,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        fit_seconds=round(fit_seconds, 4),
        hyperparameters=hyperparameters or {},
    )


def save_model_artifacts(
    selected: TrainedModelResult,
    all_results: list[TrainedModelResult],
    baseline_vs_candidate: dict[str, float | bool],
    output_dir: Path,
    force: bool = False,
) -> tuple[Path, Path]:
    """Safely persist the selected model pipeline (.joblib) and evaluation report (.json).

    Uses two-phase temporary staging and `.bak` backup rollback when `force=True`.
    """
    model_path = output_dir / "model.joblib"
    metrics_path = output_dir / "metrics.json"
    targets = [model_path, metrics_path]

    if not force:
        existing = [str(p) for p in targets if p.exists()]
        if existing:
            joined = ", ".join(existing)
            raise FileExistsError(
                f"Model artifact(s) already exist: {joined}. Pass --force to overwrite."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_model": selected.model_name,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "selected_val_metrics": selected.val_metrics.to_dict(),
        "selected_train_metrics": selected.train_metrics.to_dict(),
        "baseline_vs_candidate": baseline_vs_candidate,
        "models": {
            res.model_name: {
                "hyperparameters": res.hyperparameters,
                "fit_seconds": res.fit_seconds,
                "train_metrics": res.train_metrics.to_dict(),
                "val_metrics": res.val_metrics.to_dict(),
            }
            for res in all_results
        },
    }

    temp_paths = {t: t.with_name(f"{t.name}.tmp") for t in targets}
    backup_paths = {t: t.with_name(f"{t.name}.bak") for t in targets}
    existed_before = {t: t.exists() for t in targets}

    try:
        # Phase 1: Write and validate staged files
        joblib.dump(selected.pipeline, temp_paths[model_path])
        loaded_check = joblib.load(temp_paths[model_path])
        if not hasattr(loaded_check, "predict"):
            raise ValueError("Staged model artifact failed reload verification.")

        temp_paths[metrics_path].write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        json.loads(temp_paths[metrics_path].read_text(encoding="utf-8"))

        # Phase 2a: Back up existing targets if overwriting
        for target in targets:
            if existed_before[target] and target.exists():
                target.replace(backup_paths[target])

        # Phase 2b: Promote staged files to target paths
        for target in targets:
            temp_paths[target].replace(target)
    except Exception:
        for target in targets:
            if existed_before[target]:
                bak = backup_paths[target]
                if bak.exists():
                    try:
                        bak.replace(target)
                    except OSError as err:
                        logger.error("Failed to restore %s from %s: %s", target, bak, err)
            elif target.exists():
                try:
                    target.unlink()
                except OSError as err:
                    logger.error("Failed to remove partial artifact %s: %s", target, err)
        raise
    else:
        for bak in backup_paths.values():
            if bak.exists():
                bak.unlink()
    finally:
        for tmp in temp_paths.values():
            if tmp.exists():
                tmp.unlink()

    return model_path, metrics_path


def run_baseline_vs_candidate_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    force: bool = False,
    random_state: int = 42,
) -> tuple[TrainedModelResult, list[TrainedModelResult], dict[str, float | bool], Path, Path]:
    """Train dummy, linear baseline, and HistGBDT candidate models, select champion, and save."""
    required_cols = [*FEATURE_COLUMNS, TARGET_COLUMN]
    for name, df_part in [("train_df", train_df), ("val_df", val_df)]:
        missing = [c for c in required_cols if c not in df_part.columns]
        if missing:
            raise ValueError(f"Split '{name}' is missing required columns: {missing}")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]

    cv_splits = min(5, max(2, len(X_train) // 2))

    dummy_res = train_and_evaluate_model(
        model_name="dummy_median",
        pipeline=build_dummy_baseline(),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        hyperparameters={"strategy": "median"},
    )

    ridge_params = {"alpha": 1.0, "random_state": random_state}
    ridge_res = train_and_evaluate_model(
        model_name="ridge_baseline",
        pipeline=build_ridge_baseline(**ridge_params),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        hyperparameters=ridge_params,
    )

    hgb_params = {
        "max_iter": 200,
        "learning_rate": 0.08,
        "max_leaf_nodes": 63,
        "random_state": random_state,
        "cv_splits": cv_splits,
    }
    hgb_res = train_and_evaluate_model(
        model_name="hist_gbdt_candidate",
        pipeline=build_hist_gbdt_candidate(**hgb_params),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        hyperparameters=hgb_params,
    )

    comparison = compare_models(baseline=ridge_res.val_metrics, candidate=hgb_res.val_metrics)
    selected = hgb_res if comparison["candidate_is_better"] else ridge_res
    all_results = [dummy_res, ridge_res, hgb_res]

    model_path, metrics_path = save_model_artifacts(
        selected=selected,
        all_results=all_results,
        baseline_vs_candidate=comparison,
        output_dir=output_dir,
        force=force,
    )

    return selected, all_results, comparison, model_path, metrics_path
