"""CLI script for automated drift-aware retraining and champion-vs-challenger evaluation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from mlops_live.data import validate_parquet_file
from mlops_live.lifecycle import RetrainingWorkflowResult, run_retraining_workflow
from mlops_live.serving import create_prediction_engine
from mlops_live.tracking import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGISTERED_MODEL_NAME,
    build_sqlite_tracking_uri,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrain_and_evaluate")


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for automated retraining and challenger evaluation."""
    parser = argparse.ArgumentParser(
        description="Retrain models, stage @challenger, and evaluate against @champion."
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/train.parquet"),
        help="Training Parquet file (default: data/processed/train.parquet)",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("data/processed/val.parquet"),
        help="Validation Parquet file for evaluation (default: data/processed/val.parquet)",
    )
    parser.add_argument(
        "--only-if-drifted",
        action="store_true",
        help="Only run retraining if PSI/KS drift is detected on --val-file",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory for local model artifacts (default: models)",
    )
    parser.add_argument(
        "--tracking-db",
        type=Path,
        default=Path("mlflow.db"),
        help="MLflow SQLite database (default: mlflow.db)",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("mlruns"),
        help="MLflow artifact directory (default: mlruns)",
    )
    parser.add_argument(
        "--predictions-db",
        type=Path,
        default=Path("predictions.db"),
        help="SQLite database for drift and lifecycle audit logs (default: predictions.db)",
    )
    parser.add_argument(
        "--eval-report",
        type=Path,
        default=Path("reports/generated/champion_vs_challenger.json"),
        help="Output JSON report path (default: reports/generated/champion_vs_challenger.json)",
    )
    parser.add_argument(
        "--min-mae-improvement",
        type=float,
        default=0.0,
        help="Minimum validation MAE reduction (minutes) required to pass gate (default: 0.0)",
    )
    return parser


def main(argv: list[str] | None = None) -> RetrainingWorkflowResult:
    """Execute retraining and challenger evaluation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    validate_parquet_file(args.train_file)
    validate_parquet_file(args.val_file)

    train_df = pd.read_parquet(args.train_file)
    val_df = pd.read_parquet(args.val_file)

    tracking_uri = build_sqlite_tracking_uri(args.tracking_db)
    engine = create_prediction_engine(args.predictions_db)
    try:
        result = run_retraining_workflow(
            train_df=train_df,
            val_df=val_df,
            tracking_uri=tracking_uri,
            models_dir=args.models_dir,
            artifact_root=args.artifact_dir,
            experiment_name=DEFAULT_EXPERIMENT_NAME,
            registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
            only_if_drifted=args.only_if_drifted,
            eval_report_path=args.eval_report,
            min_mae_improvement=args.min_mae_improvement,
            db_engine=engine,
        )
    finally:
        engine.dispose()

    if not result.retrained:
        logger.info(
            "Retraining skipped (no drift). Active @champion remains v%s.",
            result.active_champion_version,
        )
        return result

    ev = result.evaluation
    assert ev is not None
    logger.info("Champion vs. Challenger Evaluation Summary:")
    logger.info(
        "  @champion   v%s (%s) -> MAE=%.4f, RMSE=%.4f, R2=%.4f",
        ev.champion_version,
        ev.champion_model_name,
        ev.champion_metrics.mae,
        ev.champion_metrics.rmse,
        ev.champion_metrics.r2,
    )
    logger.info(
        "  @challenger v%s (%s) -> MAE=%.4f, RMSE=%.4f, R2=%.4f",
        ev.challenger_version,
        ev.challenger_model_name,
        ev.challenger_metrics.mae,
        ev.challenger_metrics.rmse,
        ev.challenger_metrics.r2,
    )
    logger.info(
        "  Gate passed: %s | Saved report: %s",
        ev.gate_passed,
        args.eval_report,
    )
    logger.info(
        "  Human approval required to promote @challenger -> @champion: "
        "python scripts/manage_model.py --approve --approved-by <your_name>"
    )
    return result


if __name__ == "__main__":
    main()
