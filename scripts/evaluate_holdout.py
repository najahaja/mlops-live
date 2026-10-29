"""CLI script to evaluate @champion and @challenger models on the isolated holdout test.parquet."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlops_live.data import FEATURE_COLUMNS, TARGET_COLUMN, validate_parquet_file
from mlops_live.models import compare_models, evaluate_predictions
from mlops_live.tracking import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    DEFAULT_REGISTERED_MODEL_NAME,
    build_sqlite_tracking_uri,
    get_alias_version_info,
    load_registered_sklearn_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_holdout")


def _compute_error_quantiles(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute p50, p90, and p95 absolute error in minutes."""
    abs_err = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))
    return {
        "p50_abs_error_min": round(float(np.percentile(abs_err, 50)), 4),
        "p90_abs_error_min": round(float(np.percentile(abs_err, 90)), 4),
        "p95_abs_error_min": round(float(np.percentile(abs_err, 95)), 4),
    }


def evaluate_holdout_split(
    test_file: Path = Path("data/processed/test.parquet"),
    tracking_db: Path = Path("mlflow.db"),
    output_json: Path = Path("reports/generated/holdout_test_metrics.json"),
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
) -> dict[str, Any]:
    """Evaluate registered @champion (and @challenger if present) on isolated test.parquet."""
    validate_parquet_file(test_file)
    test_df = pd.read_parquet(test_file)

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    tracking_uri = build_sqlite_tracking_uri(tracking_db)
    champ_info = get_alias_version_info(
        alias=CHAMPION_ALIAS,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    if champ_info is None:
        raise ValueError(f"No @{CHAMPION_ALIAS} model registered in {tracking_db}.")

    champ_ver, champ_name = champ_info
    champ_model = load_registered_sklearn_model(
        tracking_uri=tracking_uri,
        alias=CHAMPION_ALIAS,
        registered_model_name=registered_model_name,
    )
    champ_preds = champ_model.predict(X_test)
    champ_metrics = evaluate_predictions(y_test, champ_preds)
    champ_quantiles = _compute_error_quantiles(y_test, champ_preds)

    payload: dict[str, Any] = {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "test_file": str(test_file),
        "test_rows": len(test_df),
        "champion": {
            "alias": CHAMPION_ALIAS,
            "version": champ_ver,
            "model_name": champ_name,
            "test_metrics": champ_metrics.to_dict(),
            "error_quantiles": champ_quantiles,
        },
        "challenger": None,
        "comparison": None,
    }

    chall_info = get_alias_version_info(
        alias=CHALLENGER_ALIAS,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    if chall_info is not None:
        chall_ver, chall_name = chall_info
        chall_model = load_registered_sklearn_model(
            tracking_uri=tracking_uri,
            alias=CHALLENGER_ALIAS,
            registered_model_name=registered_model_name,
        )
        chall_preds = chall_model.predict(X_test)
        chall_metrics = evaluate_predictions(y_test, chall_preds)
        chall_quantiles = _compute_error_quantiles(y_test, chall_preds)
        payload["challenger"] = {
            "alias": CHALLENGER_ALIAS,
            "version": chall_ver,
            "model_name": chall_name,
            "test_metrics": chall_metrics.to_dict(),
            "error_quantiles": chall_quantiles,
        }
        payload["comparison"] = compare_models(baseline=chall_metrics, candidate=champ_metrics)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_json.with_name(f"{output_json.name}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(output_json)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    logger.info("Holdout Test Set Evaluation (%s, %d rows):", test_file, len(test_df))
    logger.info(
        "  @champion   v%s (%s) -> Test MAE=%.4f, RMSE=%.4f, R2=%.4f "
        "(p50=%.2fm, p90=%.2fm, p95=%.2fm)",
        champ_ver,
        champ_name,
        champ_metrics.mae,
        champ_metrics.rmse,
        champ_metrics.r2,
        champ_quantiles["p50_abs_error_min"],
        champ_quantiles["p90_abs_error_min"],
        champ_quantiles["p95_abs_error_min"],
    )
    if payload["challenger"] is not None:
        c = payload["challenger"]
        cm = c["test_metrics"]
        logger.info(
            "  @challenger v%s (%s) -> Test MAE=%.4f, RMSE=%.4f, R2=%.4f",
            c["version"],
            c["model_name"],
            cm["mae"],
            cm["rmse"],
            cm["r2"],
        )
    logger.info("  Saved holdout report: %s", output_json)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for holdout test evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate @champion and @challenger models on isolated test.parquet."
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=Path("data/processed/test.parquet"),
        help="Holdout test Parquet file (default: data/processed/test.parquet)",
    )
    parser.add_argument(
        "--tracking-db",
        type=Path,
        default=Path("mlflow.db"),
        help="MLflow SQLite database (default: mlflow.db)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/generated/holdout_test_metrics.json"),
        help="Path to write holdout metrics JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run holdout test evaluation CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return evaluate_holdout_split(
        test_file=args.test_file,
        tracking_db=args.tracking_db,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
