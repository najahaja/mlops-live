"""CLI script for human model approval, champion-vs-challenger evaluation, and version rollback."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_live.data import validate_parquet_file
from mlops_live.lifecycle import (
    PREVIOUS_CHAMPION_ALIAS,
    approve_and_promote_challenger,
    evaluate_champion_vs_challenger,
    fetch_lifecycle_audit_logs,
    rollback_champion_model,
)
from mlops_live.serving import create_prediction_engine
from mlops_live.tracking import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    DEFAULT_REGISTERED_MODEL_NAME,
    build_sqlite_tracking_uri,
    get_alias_version_info,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("manage_model")


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for model governance, approval, and rollback."""
    parser = argparse.ArgumentParser(
        description="Manage registered models: inspect status, evaluate, approve, or roll back."
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display @champion, @challenger, @previous_champion aliases and recent audit logs",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate @champion vs @challenger on --eval-file without promoting",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve and promote @challenger to @champion (requires --approved-by)",
    )
    parser.add_argument(
        "--approved-by",
        type=str,
        default="",
        help="Human operator name approving the promotion or executing rollback",
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Allow promoting @challenger even if it did not beat @champion on validation metrics",
    )
    parser.add_argument(
        "--rollback-to-version",
        type=str,
        default=None,
        help="Roll back @champion to an explicit model version number (e.g., '1' or '2')",
    )
    parser.add_argument(
        "--rollback-previous",
        action="store_true",
        help="Roll back @champion to the @previous_champion alias",
    )
    parser.add_argument(
        "--reason",
        type=str,
        default="manual_operator_rollback",
        help="Reason recorded in the audit log for rollback",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("data/processed/val.parquet"),
        help="Evaluation Parquet file for promotion gate (default: data/processed/val.parquet)",
    )
    parser.add_argument(
        "--eval-report",
        type=Path,
        default=Path("reports/generated/champion_vs_challenger.json"),
        help=(
            "Output JSON report path for --evaluate "
            "(default: reports/generated/champion_vs_challenger.json)"
        ),
    )
    parser.add_argument(
        "--tracking-db",
        type=Path,
        default=Path("mlflow.db"),
        help="MLflow SQLite database (default: mlflow.db)",
    )
    parser.add_argument(
        "--predictions-db",
        type=Path,
        default=Path("predictions.db"),
        help="SQLite database for governance audit logs (default: predictions.db)",
    )
    parser.add_argument(
        "--fallback-model",
        type=Path,
        default=Path("models/model.joblib"),
        help="Local joblib fallback artifact updated on change (default: models/model.joblib)",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Execute model lifecycle governance action."""
    parser = build_parser()
    args = parser.parse_args(argv)

    tracking_uri = build_sqlite_tracking_uri(args.tracking_db)
    engine = create_prediction_engine(args.predictions_db)

    try:
        if args.approve:
            if not args.approved_by.strip():
                raise ValueError("--approve requires a non-empty --approved-by <name>.")
            eval_df = None
            if args.eval_file.exists():
                validate_parquet_file(args.eval_file)
                eval_df = pd.read_parquet(args.eval_file)
            summary = approve_and_promote_challenger(
                tracking_uri=tracking_uri,
                approved_by=args.approved_by,
                eval_df=eval_df,
                registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
                force_promote=args.force_promote,
                fallback_model_path=args.fallback_model,
                db_engine=engine,
            )
            return summary._asdict()

        if args.rollback_to_version is not None or args.rollback_previous:
            actor = args.approved_by.strip() or "operator"
            summary = rollback_champion_model(
                tracking_uri=tracking_uri,
                target_version=args.rollback_to_version,
                actor=actor,
                reason=args.reason,
                registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
                fallback_model_path=args.fallback_model,
                db_engine=engine,
            )
            return summary._asdict()

        if args.evaluate:
            validate_parquet_file(args.eval_file)
            eval_df = pd.read_parquet(args.eval_file)
            evaluation = evaluate_champion_vs_challenger(
                eval_df=eval_df,
                tracking_uri=tracking_uri,
                eval_source=str(args.eval_file),
                registered_model_name=DEFAULT_REGISTERED_MODEL_NAME,
                output_json_path=args.eval_report,
                db_engine=engine,
            )
            logger.info(
                "Evaluation: @champion v%s (MAE=%.4f) vs @challenger v%s (MAE=%.4f) | gate=%s",
                evaluation.champion_version,
                evaluation.champion_metrics.mae,
                evaluation.challenger_version,
                evaluation.challenger_metrics.mae,
                evaluation.gate_passed,
            )
            return evaluation.to_dict()

        # Default or --status: inspect registry aliases and audit logs
        champ = get_alias_version_info(CHAMPION_ALIAS, tracking_uri=tracking_uri)
        chall = get_alias_version_info(CHALLENGER_ALIAS, tracking_uri=tracking_uri)
        prev = get_alias_version_info(PREVIOUS_CHAMPION_ALIAS, tracking_uri=tracking_uri)
        audits = fetch_lifecycle_audit_logs(engine, limit=10)

        status_payload = {
            "champion": {"version": champ[0], "model_name": champ[1]} if champ else None,
            "challenger": {"version": chall[0], "model_name": chall[1]} if chall else None,
            "previous_champion": {"version": prev[0], "model_name": prev[1]} if prev else None,
            "recent_audit_logs": audits,
        }
        logger.info("Model Registry Status: %s", status_payload)
        return status_payload
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
