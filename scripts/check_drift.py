"""CLI script to evaluate feature and prediction drift using PSI and two-sample KS tests."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from mlops_live.data import FEATURE_COLUMNS, validate_parquet_file
from mlops_live.monitoring import (
    DriftReport,
    evaluate_dataset_drift,
    load_current_window_from_sqlite,
    save_drift_report,
)
from mlops_live.serving import create_prediction_engine, load_serving_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_drift")


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for drift evaluation."""
    parser = argparse.ArgumentParser(
        description="Run PSI and Kolmogorov-Smirnov drift detection on features and predictions."
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=Path("data/processed/train.parquet"),
        help="Reference baseline Parquet file (default: data/processed/train.parquet)",
    )
    parser.add_argument(
        "--current-file",
        type=Path,
        default=Path("data/processed/val.parquet"),
        help="Current batch Parquet file (default: data/processed/val.parquet)",
    )
    parser.add_argument(
        "--use-predictions-db",
        action="store_true",
        help="Evaluate drift against logged production requests in --predictions-db",
    )
    parser.add_argument(
        "--predictions-db",
        type=Path,
        default=Path("predictions.db"),
        help="SQLite database for prediction logs and drift reports (default: predictions.db)",
    )
    parser.add_argument(
        "--tracking-db",
        type=Path,
        default=Path("mlflow.db"),
        help="MLflow SQLite database for loading active @champion model (default: mlflow.db)",
    )
    parser.add_argument(
        "--fallback-model",
        type=Path,
        default=Path("models/model.joblib"),
        help="Fallback joblib model path (default: models/model.joblib)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/generated/drift_report.json"),
        help="Path to write JSON drift report (default: reports/generated/drift_report.json)",
    )
    parser.add_argument(
        "--psi-threshold",
        type=float,
        default=0.25,
        help="PSI threshold for significant drift (default: 0.25)",
    )
    parser.add_argument(
        "--ks-threshold",
        type=float,
        default=0.10,
        help="KS statistic threshold for continuous drift (default: 0.10)",
    )
    return parser


def main(argv: list[str] | None = None) -> DriftReport:
    """Run drift detection CLI and persist report to JSON and SQLite."""
    parser = build_parser()
    args = parser.parse_args(argv)

    validate_parquet_file(args.reference_file)
    reference_df = pd.read_parquet(args.reference_file)

    if args.use_predictions_db:
        current_df = load_current_window_from_sqlite(args.predictions_db)
        current_source = f"sqlite:{args.predictions_db}"
    else:
        validate_parquet_file(args.current_file)
        current_df = pd.read_parquet(args.current_file)
        current_source = str(args.current_file)

    ref_preds = None
    cur_preds = None
    try:
        bundle = load_serving_model(
            tracking_db=args.tracking_db,
            fallback_model_path=args.fallback_model,
        )
        ref_preds = bundle.pipeline.predict(reference_df[FEATURE_COLUMNS])
        if "predicted_duration_minutes" in current_df.columns:
            cur_preds = current_df["predicted_duration_minutes"]
        else:
            cur_preds = bundle.pipeline.predict(current_df[FEATURE_COLUMNS])
    except FileNotFoundError:
        logger.info(
            "No active model found for model-prediction scoring; using target column if present."
        )

    report = evaluate_dataset_drift(
        reference_df=reference_df,
        current_df=current_df,
        reference_source=str(args.reference_file),
        current_source=current_source,
        reference_predictions=ref_preds,
        current_predictions=cur_preds,
        psi_threshold=args.psi_threshold,
        ks_threshold=args.ks_threshold,
    )

    engine = create_prediction_engine(args.predictions_db)
    try:
        saved_path = save_drift_report(
            report=report,
            output_json_path=args.output_json,
            db_engine=engine,
        )
    finally:
        engine.dispose()

    logger.info("Drift evaluation completed:")
    logger.info("  Reference: %s (%d rows)", report.reference_source, report.reference_rows)
    logger.info("  Current:   %s (%d rows)", report.current_source, report.current_rows)
    for col, m in report.feature_drift.items():
        logger.info(
            "  Feature %-20s | status=%-17s | PSI=%.4f (%s) | KS=%.4f (p=%.4g) | drifted=%s",
            col,
            m.evaluation_status,
            m.psi,
            m.psi_severity,
            m.ks_statistic,
            m.ks_p_value,
            m.drift_detected,
        )
        if m.status_reason:
            logger.info("    -> %s: %s", col, m.status_reason)
    if report.prediction_drift is not None:
        pm = report.prediction_drift
        logger.info(
            "  Prediction %-17s | status=%-17s | PSI=%.4f (%s) | KS=%.4f (p=%.4g) | drifted=%s",
            pm.column,
            pm.evaluation_status,
            pm.psi,
            pm.psi_severity,
            pm.ks_statistic,
            pm.ks_p_value,
            pm.drift_detected,
        )
    logger.info(
        "  Overall drift: %s | Retraining recommended: %s | Drifted: %s | Insufficient data: %s",
        report.overall_drift_detected,
        report.retraining_recommended,
        report.drifted_columns,
        report.insufficient_data_columns,
    )
    logger.info("  Saved JSON report: %s", saved_path)

    return report


if __name__ == "__main__":
    main()
