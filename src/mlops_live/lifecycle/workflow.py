"""Automated retraining, champion-vs-challenger evaluation, human approval, and rollback."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sqlalchemy import Boolean, Engine, Integer, String, Text, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from mlops_live.data.clean import FEATURE_COLUMNS, TARGET_COLUMN
from mlops_live.models.evaluate import RegressionMetrics, compare_models, evaluate_predictions
from mlops_live.models.train import run_baseline_vs_candidate_training
from mlops_live.monitoring.drift import DriftReport, evaluate_dataset_drift, save_drift_report
from mlops_live.serving.database import Base
from mlops_live.tracking.mlflow_tracker import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL_NAME,
    get_alias_version_info,
    load_registered_sklearn_model,
)

logger = logging.getLogger(__name__)

PREVIOUS_CHAMPION_ALIAS = "previous_champion"


class ModelLifecycleAudit(Base):
    """SQLite audit table tracking retraining, evaluation, human approval, and rollbacks."""

    __tablename__ = "model_lifecycle_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at_utc: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    previous_champion_version: Mapped[str] = mapped_column(String(64), nullable=False)
    new_champion_version: Mapped[str] = mapped_column(String(64), nullable=False)
    challenger_version: Mapped[str] = mapped_column(String(64), nullable=False)
    gate_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    notes: Mapped[str] = mapped_column(String(512), nullable=False)
    details_json: Mapped[str] = mapped_column(Text, nullable=False)


class ChampionChallengerEvaluation(NamedTuple):
    """Side-by-side evaluation result comparing @champion and @challenger."""

    created_at_utc: str
    eval_source: str
    eval_rows: int
    champion_version: str
    champion_model_name: str
    champion_metrics: RegressionMetrics
    challenger_version: str
    challenger_model_name: str
    challenger_metrics: RegressionMetrics
    comparison: dict[str, float | bool]
    min_mae_improvement: float
    gate_passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluation report to a JSON-compatible dictionary."""
        return {
            "created_at_utc": self.created_at_utc,
            "eval_source": self.eval_source,
            "eval_rows": self.eval_rows,
            "min_mae_improvement": self.min_mae_improvement,
            "gate_passed": self.gate_passed,
            "champion": {
                "alias": CHAMPION_ALIAS,
                "version": self.champion_version,
                "model_name": self.champion_model_name,
                "metrics": self.champion_metrics.to_dict(),
            },
            "challenger": {
                "alias": CHALLENGER_ALIAS,
                "version": self.challenger_version,
                "model_name": self.challenger_model_name,
                "metrics": self.challenger_metrics.to_dict(),
            },
            "comparison": self.comparison,
        }


class LifecycleActionSummary(NamedTuple):
    """Summary returned after a promotion or rollback action."""

    event_type: str
    actor: str
    previous_champion_version: str
    new_champion_version: str
    new_champion_model_name: str
    challenger_version: str
    gate_passed: bool
    notes: str


class RetrainingWorkflowResult(NamedTuple):
    """Summary returned by the automated retraining and challenger evaluation workflow."""

    retrained: bool
    drift_report: DriftReport | None
    evaluation: ChampionChallengerEvaluation | None
    selected_candidate_name: str | None
    staged_challenger_version: str | None
    active_champion_version: str | None


def record_lifecycle_audit(
    engine: Engine,
    event_type: str,
    actor: str,
    previous_champion_version: str,
    new_champion_version: str,
    challenger_version: str,
    gate_passed: bool,
    notes: str,
    details: dict[str, Any],
) -> None:
    """Persist an immutable lifecycle governance audit entry to SQLite."""
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        row = ModelLifecycleAudit(
            created_at_utc=datetime.now(UTC).isoformat(),
            event_type=event_type,
            actor=actor,
            previous_champion_version=previous_champion_version,
            new_champion_version=new_champion_version,
            challenger_version=challenger_version,
            gate_passed=gate_passed,
            notes=notes,
            details_json=json.dumps(details, indent=2),
        )
        session.add(row)


def fetch_lifecycle_audit_logs(engine: Engine, limit: int = 50) -> list[dict[str, Any]]:
    """Fetch recent model lifecycle governance events from SQLite."""
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        stmt = select(ModelLifecycleAudit).order_by(ModelLifecycleAudit.id.desc()).limit(limit)
        rows = session.scalars(stmt).all()
        return [
            {
                "id": r.id,
                "created_at_utc": r.created_at_utc,
                "event_type": r.event_type,
                "actor": r.actor,
                "previous_champion_version": r.previous_champion_version,
                "new_champion_version": r.new_champion_version,
                "challenger_version": r.challenger_version,
                "gate_passed": r.gate_passed,
                "notes": r.notes,
            }
            for r in rows
        ]


def evaluate_champion_vs_challenger(
    eval_df: pd.DataFrame,
    tracking_uri: str,
    eval_source: str = "val.parquet",
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    min_mae_improvement: float = 0.0,
    output_json_path: Path | None = Path("reports/generated/champion_vs_challenger.json"),
    db_engine: Engine | None = None,
) -> ChampionChallengerEvaluation:
    """Evaluate registered @champion and @challenger models side-by-side on eval_df."""
    required = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [c for c in required if c not in eval_df.columns]
    if missing:
        raise ValueError(f"Evaluation DataFrame is missing required columns: {missing}")

    champ_info = get_alias_version_info(
        alias=CHAMPION_ALIAS,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    chall_info = get_alias_version_info(
        alias=CHALLENGER_ALIAS,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    if champ_info is None:
        raise ValueError(f"No @{CHAMPION_ALIAS} version registered for '{registered_model_name}'.")
    if chall_info is None:
        raise ValueError(
            f"No @{CHALLENGER_ALIAS} version registered for '{registered_model_name}'."
        )

    champ_ver, champ_name = champ_info
    chall_ver, chall_name = chall_info

    champ_model = load_registered_sklearn_model(
        tracking_uri=tracking_uri,
        alias=CHAMPION_ALIAS,
        registered_model_name=registered_model_name,
    )
    chall_model = load_registered_sklearn_model(
        tracking_uri=tracking_uri,
        alias=CHALLENGER_ALIAS,
        registered_model_name=registered_model_name,
    )

    X_eval = eval_df[FEATURE_COLUMNS]
    y_eval = eval_df[TARGET_COLUMN]

    champ_preds = champ_model.predict(X_eval)
    chall_preds = chall_model.predict(X_eval)

    champ_metrics = evaluate_predictions(y_eval, champ_preds)
    chall_metrics = evaluate_predictions(y_eval, chall_preds)
    comparison = compare_models(baseline=champ_metrics, candidate=chall_metrics)

    # Gate check: challenger must achieve at least min_mae_improvement and not degrade RMSE
    mae_gain = champ_metrics.mae - chall_metrics.mae
    rmse_non_worse = chall_metrics.rmse <= champ_metrics.rmse + 1e-6
    gate_passed = bool(mae_gain >= min_mae_improvement - 1e-6 and rmse_non_worse)

    evaluation = ChampionChallengerEvaluation(
        created_at_utc=datetime.now(UTC).isoformat(),
        eval_source=eval_source,
        eval_rows=len(eval_df),
        champion_version=champ_ver,
        champion_model_name=champ_name,
        champion_metrics=champ_metrics,
        challenger_version=chall_ver,
        challenger_model_name=chall_name,
        challenger_metrics=chall_metrics,
        comparison=comparison,
        min_mae_improvement=min_mae_improvement,
        gate_passed=gate_passed,
    )

    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_json_path.with_name(f"{output_json_path.name}.tmp")
        try:
            tmp_path.write_text(json.dumps(evaluation.to_dict(), indent=2), encoding="utf-8")
            tmp_path.replace(output_json_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    if db_engine is not None:
        record_lifecycle_audit(
            engine=db_engine,
            event_type="champion_vs_challenger_eval",
            actor="evaluator",
            previous_champion_version=champ_ver,
            new_champion_version=champ_ver,
            challenger_version=chall_ver,
            gate_passed=gate_passed,
            notes=(
                f"Evaluated @champion v{champ_ver} (MAE={champ_metrics.mae:.4f}) vs "
                f"@challenger v{chall_ver} (MAE={chall_metrics.mae:.4f}); gate_passed={gate_passed}"
            ),
            details=evaluation.to_dict(),
        )

    return evaluation


def run_retraining_workflow(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tracking_uri: str,
    models_dir: Path = Path("models"),
    artifact_root: Path = Path("mlruns"),
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    only_if_drifted: bool = False,
    current_drift_df: pd.DataFrame | None = None,
    drift_report_path: Path = Path("reports/generated/drift_report.json"),
    eval_report_path: Path = Path("reports/generated/champion_vs_challenger.json"),
    min_mae_improvement: float = 0.0,
    db_engine: Engine | None = None,
) -> RetrainingWorkflowResult:
    """Run drift check (optional), retrain candidate models, stage @challenger, and evaluate.

    Note: This workflow never promotes @challenger to @champion automatically when an
    incumbent @champion exists. Promotion requires explicit human approval.
    """
    drift_report: DriftReport | None = None
    if current_drift_df is not None or only_if_drifted:
        window_df = current_drift_df if current_drift_df is not None else val_df
        drift_report = evaluate_dataset_drift(
            reference_df=train_df,
            current_df=window_df,
            reference_source="train_split",
            current_source="retrain_trigger_window",
        )
        save_drift_report(drift_report, output_json_path=drift_report_path, db_engine=db_engine)
        if only_if_drifted and not drift_report.retraining_recommended:
            champ_info = get_alias_version_info(
                alias=CHAMPION_ALIAS,
                tracking_uri=tracking_uri,
                registered_model_name=registered_model_name,
            )
            champ_ver = champ_info[0] if champ_info is not None else None
            logger.info("No actionable drift detected; skipping retraining.")
            return RetrainingWorkflowResult(
                retrained=False,
                drift_report=drift_report,
                evaluation=None,
                selected_candidate_name=None,
                staged_challenger_version=None,
                active_champion_version=champ_ver,
            )

    selected, _, _, _, _ = run_baseline_vs_candidate_training(
        train_df=train_df,
        val_df=val_df,
        output_dir=models_dir,
        force=True,
        mlflow_tracking_uri=tracking_uri,
        mlflow_artifact_root=artifact_root,
        mlflow_experiment_name=experiment_name,
        mlflow_registered_model_name=registered_model_name,
        promote_winner=False,
    )

    evaluation = evaluate_champion_vs_challenger(
        eval_df=val_df,
        tracking_uri=tracking_uri,
        eval_source="val_split",
        registered_model_name=registered_model_name,
        min_mae_improvement=min_mae_improvement,
        output_json_path=eval_report_path,
        db_engine=db_engine,
    )

    if db_engine is not None:
        record_lifecycle_audit(
            engine=db_engine,
            event_type="retrain_and_stage_challenger",
            actor="automated_retrainer",
            previous_champion_version=evaluation.champion_version,
            new_champion_version=evaluation.champion_version,
            challenger_version=evaluation.challenger_version,
            gate_passed=evaluation.gate_passed,
            notes=(
                f"Retrained and staged @{CHALLENGER_ALIAS} v{evaluation.challenger_version} "
                f"({selected.model_name}); awaiting human approval."
            ),
            details=evaluation.to_dict(),
        )

    return RetrainingWorkflowResult(
        retrained=True,
        drift_report=drift_report,
        evaluation=evaluation,
        selected_candidate_name=selected.model_name,
        staged_challenger_version=evaluation.challenger_version,
        active_champion_version=evaluation.champion_version,
    )


def _sync_production_fallback_artifacts(
    pipeline: Any,
    fallback_model_path: Path,
    champion_version: str,
    champion_model_name: str,
    prefer_challenger_metrics: bool = False,
) -> None:
    """Safely update fallback model.joblib and companion metrics.json with .tmp/.bak rollback."""
    fallback_model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = fallback_model_path.with_name("metrics.json")
    challenger_metrics_path = fallback_model_path.parent / "challenger" / "metrics.json"

    source_metrics_path: Path | None = None
    if prefer_challenger_metrics and challenger_metrics_path.exists():
        source_metrics_path = challenger_metrics_path
    elif metrics_path.exists():
        source_metrics_path = metrics_path

    targets = [fallback_model_path]
    if source_metrics_path is not None:
        targets.append(metrics_path)

    temp_paths = {t: t.with_name(f"{t.name}.tmp") for t in targets}
    backup_paths = {t: t.with_name(f"{t.name}.bak") for t in targets}
    existed_before = {t: t.exists() for t in targets}

    try:
        joblib.dump(pipeline, temp_paths[fallback_model_path])
        loaded_check = joblib.load(temp_paths[fallback_model_path])
        if not hasattr(loaded_check, "predict"):
            raise ValueError("Staged fallback model artifact failed reload verification.")

        if source_metrics_path is not None:
            payload = json.loads(source_metrics_path.read_text(encoding="utf-8"))
            payload["selected_model"] = champion_model_name
            if isinstance(payload.get("mlflow"), dict):
                payload["mlflow"]["champion"] = {
                    "version": champion_version,
                    "model_name": champion_model_name,
                    "alias": CHAMPION_ALIAS,
                }
            temp_paths[metrics_path].write_text(json.dumps(payload, indent=2), encoding="utf-8")
            json.loads(temp_paths[metrics_path].read_text(encoding="utf-8"))

        for target in targets:
            if existed_before[target] and target.exists():
                target.replace(backup_paths[target])

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


def approve_and_promote_challenger(
    tracking_uri: str,
    approved_by: str,
    eval_df: pd.DataFrame | None = None,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    min_mae_improvement: float = 0.0,
    force_promote: bool = False,
    fallback_model_path: Path | None = Path("models/model.joblib"),
    db_engine: Engine | None = None,
) -> LifecycleActionSummary:
    """Promote @challenger to @champion after human approval and quantitative gate check."""
    approver = approved_by.strip()
    if not approver:
        raise ValueError("Human approval requires a non-empty 'approved_by' identifier.")

    champ_info = get_alias_version_info(
        alias=CHAMPION_ALIAS,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    chall_info = get_alias_version_info(
        alias=CHALLENGER_ALIAS,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    if chall_info is None:
        raise ValueError(f"No @{CHALLENGER_ALIAS} found for '{registered_model_name}'.")

    prev_champ_ver = champ_info[0] if champ_info is not None else "none"
    chall_ver, chall_name = chall_info

    gate_passed = True
    eval_details: dict[str, Any] = {}
    if eval_df is not None:
        evaluation = evaluate_champion_vs_challenger(
            eval_df=eval_df,
            tracking_uri=tracking_uri,
            eval_source="approval_gate_eval",
            registered_model_name=registered_model_name,
            min_mae_improvement=min_mae_improvement,
            output_json_path=None,
            db_engine=None,
        )
        gate_passed = evaluation.gate_passed
        eval_details = evaluation.to_dict()
        if not gate_passed and not force_promote:
            raise ValueError(
                f"Challenger v{chall_ver} (MAE={evaluation.challenger_metrics.mae:.4f}, "
                f"RMSE={evaluation.challenger_metrics.rmse:.4f}) did not pass promotion gate "
                f"against Champion v{prev_champ_ver} (MAE={evaluation.champion_metrics.mae:.4f}, "
                f"RMSE={evaluation.champion_metrics.rmse:.4f}). Pass --force-promote to override."
            )

    # Load and verify the @challenger model artifact before mutating aliases or fallback files
    promoted_pipeline = load_registered_sklearn_model(
        tracking_uri=tracking_uri,
        alias=CHALLENGER_ALIAS,
        registered_model_name=registered_model_name,
    )

    if fallback_model_path is not None:
        _sync_production_fallback_artifacts(
            pipeline=promoted_pipeline,
            fallback_model_path=fallback_model_path,
            champion_version=chall_ver,
            champion_model_name=chall_name,
            prefer_challenger_metrics=True,
        )

    client = MlflowClient(tracking_uri=tracking_uri)
    if champ_info is not None and prev_champ_ver != chall_ver:
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=PREVIOUS_CHAMPION_ALIAS,
            version=prev_champ_ver,
        )

    client.set_registered_model_alias(
        name=registered_model_name,
        alias=CHAMPION_ALIAS,
        version=chall_ver,
    )

    notes = (
        f"Approved by '{approver}': promoted @{CHALLENGER_ALIAS} v{chall_ver} ({chall_name}) "
        f"to @{CHAMPION_ALIAS} (previous @{CHAMPION_ALIAS}: v{prev_champ_ver})."
    )
    if db_engine is not None:
        record_lifecycle_audit(
            engine=db_engine,
            event_type="human_approval_promotion",
            actor=approver,
            previous_champion_version=prev_champ_ver,
            new_champion_version=chall_ver,
            challenger_version=chall_ver,
            gate_passed=gate_passed,
            notes=notes,
            details=eval_details,
        )

    logger.info(notes)
    return LifecycleActionSummary(
        event_type="human_approval_promotion",
        actor=approver,
        previous_champion_version=prev_champ_ver,
        new_champion_version=chall_ver,
        new_champion_model_name=chall_name,
        challenger_version=chall_ver,
        gate_passed=gate_passed,
        notes=notes,
    )


def rollback_champion_model(
    tracking_uri: str,
    target_version: str | None = None,
    actor: str = "operator",
    reason: str = "manual_rollback",
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    fallback_model_path: Path | None = Path("models/model.joblib"),
    db_engine: Engine | None = None,
) -> LifecycleActionSummary:
    """Roll back @champion to an explicit model version or to @previous_champion."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    current_champ = get_alias_version_info(
        alias=CHAMPION_ALIAS,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    current_champ_ver = current_champ[0] if current_champ is not None else "none"

    if target_version is None:
        prev_info = get_alias_version_info(
            alias=PREVIOUS_CHAMPION_ALIAS,
            tracking_uri=tracking_uri,
            registered_model_name=registered_model_name,
        )
        if prev_info is None:
            raise ValueError(
                f"No @{PREVIOUS_CHAMPION_ALIAS} alias exists and no target version was given."
            )
        resolved_target_ver = prev_info[0]
    else:
        resolved_target_ver = str(target_version).strip()

    try:
        mv = client.get_model_version(name=registered_model_name, version=resolved_target_ver)
    except MlflowException as exc:
        raise ValueError(
            f"Target version '{resolved_target_ver}' does not exist in '{registered_model_name}'."
        ) from exc

    target_model_name = mv.tags.get("model_name", "unknown") if mv.tags else "unknown"

    # Verify the target version artifact loads cleanly before switching @champion
    version_uri = f"models:/{registered_model_name}/{resolved_target_ver}"
    restored_pipeline = mlflow.sklearn.load_model(version_uri)

    if fallback_model_path is not None:
        _sync_production_fallback_artifacts(
            pipeline=restored_pipeline,
            fallback_model_path=fallback_model_path,
            champion_version=resolved_target_ver,
            champion_model_name=target_model_name,
            prefer_challenger_metrics=False,
        )

    if current_champ is not None and current_champ_ver != resolved_target_ver:
        client.set_registered_model_alias(
            name=registered_model_name,
            alias=PREVIOUS_CHAMPION_ALIAS,
            version=current_champ_ver,
        )

    client.set_registered_model_alias(
        name=registered_model_name,
        alias=CHAMPION_ALIAS,
        version=resolved_target_ver,
    )

    chall_info = get_alias_version_info(
        alias=CHALLENGER_ALIAS,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    chall_ver = chall_info[0] if chall_info is not None else "none"

    notes = (
        f"Rollback by '{actor}' ({reason}): restored @{CHAMPION_ALIAS} from "
        f"v{current_champ_ver} to v{resolved_target_ver} ({target_model_name})."
    )
    if db_engine is not None:
        record_lifecycle_audit(
            engine=db_engine,
            event_type="model_rollback",
            actor=actor,
            previous_champion_version=current_champ_ver,
            new_champion_version=resolved_target_ver,
            challenger_version=chall_ver,
            gate_passed=True,
            notes=notes,
            details={"reason": reason, "restored_version": resolved_target_ver},
        )

    logger.info(notes)
    return LifecycleActionSummary(
        event_type="model_rollback",
        actor=actor,
        previous_champion_version=current_champ_ver,
        new_champion_version=resolved_target_ver,
        new_champion_model_name=target_model_name,
        challenger_version=chall_ver,
        gate_passed=True,
        notes=notes,
    )
