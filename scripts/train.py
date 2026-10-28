"""CLI entry point to train and evaluate baseline vs. candidate models on train/val splits."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from mlops_live.data import validate_parquet_file
from mlops_live.models import TrainedModelResult, run_baseline_vs_candidate_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_models")


class TrainingRunSummary(NamedTuple):
    """Summary returned by the training CLI."""

    selected_model: str
    selected_result: TrainedModelResult
    all_results: list[TrainedModelResult]
    comparison: dict[str, float | bool]
    model_path: Path
    metrics_path: Path


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for model training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Train baseline and candidate models and evaluate on val.parquet."
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/processed/train.parquet"),
        help="Path to prepared training Parquet file (default: data/processed/train.parquet)",
    )
    parser.add_argument(
        "--val-file",
        type=Path,
        default=Path("data/processed/val.parquet"),
        help="Path to prepared validation Parquet file (default: data/processed/val.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to write model.joblib and metrics.json (default: models)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing model.joblib and metrics.json if present",
    )
    return parser


def main(argv: list[str] | None = None) -> TrainingRunSummary:
    """Run training and validation evaluation workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Check overwrite protection before spending time fitting models
    if not args.force:
        existing = [
            str(p)
            for p in (args.output_dir / "model.joblib", args.output_dir / "metrics.json")
            if p.exists()
        ]
        if existing:
            joined = ", ".join(existing)
            raise FileExistsError(
                f"Model artifact(s) already exist: {joined}. Pass --force to overwrite."
            )

    validate_parquet_file(args.train_file)
    validate_parquet_file(args.val_file)

    train_df = pd.read_parquet(args.train_file)
    val_df = pd.read_parquet(args.val_file)

    logger.info(
        "Loaded splits: train=%s (%d rows), val=%s (%d rows). Test split remains isolated.",
        args.train_file,
        len(train_df),
        args.val_file,
        len(val_df),
    )

    selected, all_results, comparison, model_path, metrics_path = (
        run_baseline_vs_candidate_training(
            train_df=train_df,
            val_df=val_df,
            output_dir=args.output_dir,
            force=args.force,
        )
    )

    logger.info("Model comparison summary (ridge_baseline -> hist_gbdt_candidate):")
    logger.info(
        "  Val MAE reduction:  %.4f min (%.2f%%)",
        comparison["mae_reduction_minutes"],
        comparison["mae_reduction_pct"],
    )
    logger.info(
        "  Val RMSE reduction: %.4f min (%.2f%%)",
        comparison["rmse_reduction_minutes"],
        comparison["rmse_reduction_pct"],
    )
    logger.info("  Val R2 gain:        %+.4f", comparison["r2_gain"])
    logger.info("  Selected model:     %s", selected.model_name)
    logger.info("  Saved model:        %s", model_path)
    logger.info("  Saved metrics JSON: %s", metrics_path)

    return TrainingRunSummary(
        selected_model=selected.model_name,
        selected_result=selected,
        all_results=all_results,
        comparison=comparison,
        model_path=model_path,
        metrics_path=metrics_path,
    )


if __name__ == "__main__":
    main()
