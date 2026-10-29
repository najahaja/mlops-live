"""SQLite-backed MLflow experiment tracking and model registry management."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from mlops_live.data.clean import FEATURE_COLUMNS, TARGET_COLUMN
from mlops_live.models.train import TrainedModelResult

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "nyc-green-taxi-duration"
DEFAULT_REGISTERED_MODEL_NAME = "nyc-green-taxi-duration-regressor"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

PINNED_MODEL_PIP_REQUIREMENTS = [
    "scikit-learn==1.9.1",
    "pandas==3.1.0",
    "numpy==2.4.6",
]

SKOPS_TRUSTED_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.model_selection._split.KFold",
]


class LoggedRunInfo(NamedTuple):
    """Metadata for a single model logged to MLflow."""

    model_name: str
    run_id: str
    registered_version: str | None
    alias: str | None


class TrackingSummary(NamedTuple):
    """Summary of an MLflow experiment tracking and model registry session."""

    tracking_uri: str
    experiment_id: str
    experiment_name: str
    registered_model_name: str
    runs: list[LoggedRunInfo]
    champion_version: str
    champion_model_name: str
    challenger_version: str
    challenger_model_name: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracking summary for inclusion in metrics.json."""
        return {
            "tracking_uri": self.tracking_uri,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "registered_model_name": self.registered_model_name,
            "champion": {
                "version": self.champion_version,
                "model_name": self.champion_model_name,
                "alias": CHAMPION_ALIAS,
            },
            "challenger": {
                "version": self.challenger_version,
                "model_name": self.challenger_model_name,
                "alias": CHALLENGER_ALIAS,
            },
            "runs": [
                {
                    "model_name": r.model_name,
                    "run_id": r.run_id,
                    "registered_version": r.registered_version,
                    "alias": r.alias,
                }
                for r in self.runs
            ],
        }


def build_sqlite_tracking_uri(db_path: Path = Path("mlflow.db")) -> str:
    """Build a SQLAlchemy-compatible SQLite tracking URI from a local file path."""
    resolved = db_path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{resolved.as_posix()}"


def configure_mlflow(
    tracking_uri: str,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    artifact_root: Path | None = None,
) -> str:
    """Configure MLflow tracking URI and ensure the target experiment exists.

    Args:
        tracking_uri: SQLite or file tracking URI.
        experiment_name: Name of the MLflow experiment.
        artifact_root: Directory for MLflow run artifacts (defaults to ./mlruns).

    Returns:
        The MLflow experiment_id string.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        resolved_artifacts = (artifact_root or Path("mlruns")).resolve()
        resolved_artifacts.mkdir(parents=True, exist_ok=True)
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=resolved_artifacts.as_uri(),
        )
    else:
        experiment_id = exp.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def get_alias_version_info(
    alias: str,
    tracking_uri: str,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
) -> tuple[str, str] | None:
    """Return (version, model_name_tag) for a given model registry alias, or None if unset."""
    client = MlflowClient(tracking_uri=tracking_uri)
    try:
        mv = client.get_model_version_by_alias(name=registered_model_name, alias=alias)
        model_name_tag = mv.tags.get("model_name", "unknown") if mv.tags else "unknown"
        return str(mv.version), model_name_tag
    except MlflowException:
        return None


def log_and_register_training_runs(
    all_results: list[TrainedModelResult],
    selected: TrainedModelResult,
    tracking_uri: str,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    artifact_root: Path | None = None,
    promote_winner: bool = False,
) -> TrackingSummary:
    """Log all trained models to MLflow and register @champion and @challenger versions.

    Behavior for @champion and @challenger aliases:
    - If no @champion alias exists yet in the registry (initial bootstrap) or `promote_winner=True`,
      the winning model (`selected`) is assigned `@champion` and the runner-up registered model
      is assigned `@challenger`.
    - If an incumbent `@champion` already exists and `promote_winner=False`, the incumbent
      `@champion` is preserved and the newly trained winning model is assigned `@challenger`.
    """
    experiment_id = configure_mlflow(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_root=artifact_root,
    )
    client = MlflowClient(tracking_uri=tracking_uri)

    existing_champion = get_alias_version_info(
        alias=CHAMPION_ALIAS,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )

    logged_by_name: dict[str, tuple[str, str | None]] = {}

    for res in all_results:
        is_selected = res.model_name == selected.model_name
        with mlflow.start_run(experiment_id=experiment_id, run_name=res.model_name) as run:
            run_id = run.info.run_id

            params_to_log: dict[str, Any] = {
                **res.hyperparameters,
                "num_features": len(FEATURE_COLUMNS),
                "train_samples": res.train_metrics.num_samples,
                "val_samples": res.val_metrics.num_samples,
            }
            mlflow.log_params(params_to_log)

            mlflow.log_metrics(
                {
                    "train_mae": res.train_metrics.mae,
                    "train_rmse": res.train_metrics.rmse,
                    "train_r2": res.train_metrics.r2,
                    "val_mae": res.val_metrics.mae,
                    "val_rmse": res.val_metrics.rmse,
                    "val_r2": res.val_metrics.r2,
                    "fit_seconds": res.fit_seconds,
                }
            )

            mlflow.set_tags(
                {
                    "model_name": res.model_name,
                    "target_column": TARGET_COLUMN,
                    "feature_columns": ",".join(FEATURE_COLUMNS),
                    "selected_in_run": str(is_selected).lower(),
                }
            )

            registered_version: str | None = None
            if res.model_name != "dummy_median":
                model_info = mlflow.sklearn.log_model(
                    sk_model=res.pipeline,
                    name="model",
                    pip_requirements=PINNED_MODEL_PIP_REQUIREMENTS,
                    skops_trusted_types=SKOPS_TRUSTED_TYPES,
                )
                mv = mlflow.register_model(
                    model_uri=model_info.model_uri,
                    name=registered_model_name,
                )
                registered_version = str(mv.version)
                client.set_model_version_tag(
                    name=registered_model_name,
                    version=registered_version,
                    key="model_name",
                    value=res.model_name,
                )
                client.set_model_version_tag(
                    name=registered_model_name,
                    version=registered_version,
                    key="val_mae",
                    value=f"{res.val_metrics.mae:.6f}",
                )
                client.set_model_version_tag(
                    name=registered_model_name,
                    version=registered_version,
                    key="val_rmse",
                    value=f"{res.val_metrics.rmse:.6f}",
                )

            logged_by_name[res.model_name] = (run_id, registered_version)

    # Determine runner-up registered model (between ridge_baseline and hist_gbdt_candidate)
    registered_candidates = [
        r.model_name for r in all_results if logged_by_name[r.model_name][1] is not None
    ]
    winner_name = selected.model_name
    runner_up_names = [name for name in registered_candidates if name != winner_name]
    runner_up_name = runner_up_names[0] if runner_up_names else winner_name

    winner_version = logged_by_name[winner_name][1]
    runner_up_version = logged_by_name[runner_up_name][1]
    if winner_version is None or runner_up_version is None:
        raise ValueError("Expected registered versions for winner and runner-up models.")

    alias_by_model_name: dict[str, str | None] = {r.model_name: None for r in all_results}

    if existing_champion is None or promote_winner:
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=CHAMPION_ALIAS,
            version=winner_version,
        )
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=CHALLENGER_ALIAS,
            version=runner_up_version,
        )
        champion_version = winner_version
        champion_model_name = winner_name
        challenger_version = runner_up_version
        challenger_model_name = runner_up_name
        alias_by_model_name[winner_name] = CHAMPION_ALIAS
        if runner_up_name != winner_name:
            alias_by_model_name[runner_up_name] = CHALLENGER_ALIAS
    else:
        # Preserve incumbent @champion and stage newly trained winner as @challenger
        champion_version, champion_model_name = existing_champion
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=CHALLENGER_ALIAS,
            version=winner_version,
        )
        challenger_version = winner_version
        challenger_model_name = winner_name
        alias_by_model_name[winner_name] = CHALLENGER_ALIAS

    runs_info = [
        LoggedRunInfo(
            model_name=res.model_name,
            run_id=logged_by_name[res.model_name][0],
            registered_version=logged_by_name[res.model_name][1],
            alias=alias_by_model_name[res.model_name],
        )
        for res in all_results
    ]

    logger.info(
        "MLflow registry updated (%s): @%s -> v%s (%s), @%s -> v%s (%s)",
        registered_model_name,
        CHAMPION_ALIAS,
        champion_version,
        champion_model_name,
        CHALLENGER_ALIAS,
        challenger_version,
        challenger_model_name,
    )

    return TrackingSummary(
        tracking_uri=tracking_uri,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        registered_model_name=registered_model_name,
        runs=runs_info,
        champion_version=champion_version,
        champion_model_name=champion_model_name,
        challenger_version=challenger_version,
        challenger_model_name=challenger_model_name,
    )


def promote_challenger_to_champion(
    tracking_uri: str,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
) -> tuple[str, str | None]:
    """Promote the current @challenger model version to @champion in the MLflow Registry.

    Returns:
        Tuple of (new_champion_version, previous_champion_version).
    """
    challenger_info = get_alias_version_info(
        alias=CHALLENGER_ALIAS,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    if challenger_info is None:
        raise ValueError(f"No @{CHALLENGER_ALIAS} alias found for model '{registered_model_name}'.")

    previous_champion = get_alias_version_info(
        alias=CHAMPION_ALIAS,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    prev_version = previous_champion[0] if previous_champion is not None else None
    new_champion_version = challenger_info[0]

    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=CHAMPION_ALIAS,
        version=new_champion_version,
    )
    return new_champion_version, prev_version


def load_registered_sklearn_model(
    tracking_uri: str,
    alias: str = CHAMPION_ALIAS,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
) -> Pipeline:
    """Load a registered scikit-learn pipeline by alias (e.g. 'champion' or 'challenger')."""
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{registered_model_name}@{alias}"
    loaded = mlflow.sklearn.load_model(model_uri)
    return loaded
