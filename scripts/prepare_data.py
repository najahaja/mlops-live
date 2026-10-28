"""CLI script to clean raw NYC Green Taxi data and save chronological train/val/test splits."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from mlops_live.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TimedDatasetSplits,
    split_with_timestamps,
    validate_parquet_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_data")

EXPECTED_SPLIT_COLUMNS = [*FEATURE_COLUMNS, TARGET_COLUMN]


class PreparationSummary(NamedTuple):
    """Summary metadata for a completed data-preparation run."""

    input_file: Path
    total_raw_rows: int
    cleaned_rows: int
    excluded_rows: int
    train_path: Path
    val_path: Path
    test_path: Path
    train_rows: int
    val_rows: int
    test_rows: int
    train_time_range: tuple[pd.Timestamp, pd.Timestamp]
    val_time_range: tuple[pd.Timestamp, pd.Timestamp]
    test_time_range: tuple[pd.Timestamp, pd.Timestamp]


def _assemble_split_df(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Combine feature DataFrame and target Series into a single split DataFrame."""
    split_df = X.copy()
    split_df[TARGET_COLUMN] = y
    return split_df[EXPECTED_SPLIT_COLUMNS]


def validate_prepared_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_ts: pd.Series,
    val_ts: pd.Series,
    test_ts: pd.Series,
    expected_total_rows: int,
) -> None:
    """Validate split schemas, non-emptiness, row counts, and chronological boundaries.

    Args:
        train_df: Prepared training DataFrame (features + target).
        val_df: Prepared validation DataFrame (features + target).
        test_df: Prepared test DataFrame (features + target).
        train_ts: Chronological pickup timestamps for the training split.
        val_ts: Chronological pickup timestamps for the validation split.
        test_ts: Chronological pickup timestamps for the test split.
        expected_total_rows: Total cleaned rows expected across all three splits.

    Raises:
        ValueError: If any validation check fails.
    """
    splits = {"train": (train_df, train_ts), "val": (val_df, val_ts), "test": (test_df, test_ts)}

    for name, (df_part, ts_part) in splits.items():
        if df_part.empty:
            raise ValueError(f"Prepared '{name}' split is empty.")
        if len(df_part) != len(ts_part):
            raise ValueError(
                f"Row count mismatch between '{name}' data ({len(df_part)}) "
                f"and timestamps ({len(ts_part)})."
            )
        if list(df_part.columns) != EXPECTED_SPLIT_COLUMNS:
            raise ValueError(
                f"Split '{name}' has unexpected columns {list(df_part.columns)}; "
                f"expected {EXPECTED_SPLIT_COLUMNS}."
            )
        if df_part.isna().any().any():
            raise ValueError(f"Split '{name}' contains unexpected NaN values.")
        if not ts_part.is_monotonic_increasing:
            raise ValueError(f"Split '{name}' timestamps are not sorted chronologically.")

    total_split_rows = len(train_df) + len(val_df) + len(test_df)
    if total_split_rows != expected_total_rows:
        raise ValueError(
            f"Split row sum ({total_split_rows}) does not match cleaned rows "
            f"({expected_total_rows})."
        )

    train_max = train_ts.iloc[-1]
    val_min = val_ts.iloc[0]
    val_max = val_ts.iloc[-1]
    test_min = test_ts.iloc[0]

    if train_max > val_min:
        raise ValueError(
            f"Chronological boundary violation between train and val: "
            f"train max ({train_max}) > val min ({val_min})."
        )
    if val_max > test_min:
        raise ValueError(
            f"Chronological boundary violation between val and test: "
            f"val max ({val_max}) > test min ({test_min})."
        )


def _check_overwrite_paths(target_paths: list[Path], force: bool) -> None:
    """Ensure no target output file exists unless force=True."""
    if force:
        return
    existing = [str(p) for p in target_paths if p.exists()]
    if existing:
        joined = ", ".join(existing)
        raise FileExistsError(f"Output file(s) already exist: {joined}. Pass --force to overwrite.")


def _write_splits_safely(split_frames: dict[Path, pd.DataFrame], force: bool) -> None:
    """Stage split Parquet files, back up existing targets on overwrite, and roll back on failure.

    Note on multi-file atomicity:
        Standard Windows/POSIX filesystems provide atomic rename (`Path.replace`) per file,
        not a single transactional rename across three separate file paths. To prevent a
        failed overwrite from leaving a mixture of old and new split files:
        1. All three splits are written to `.tmp` files and validated before any existing
           target file is modified.
        2. When `force=True`, existing target files are moved to `.bak` backups before the
           validated `.tmp` files are moved into place.
        3. If any replacement or post-replacement check fails, existing targets are restored
           from their `.bak` backups and newly created targets are removed. If an OS error
           prevents restoring a `.bak` file, the `.bak` file is left on disk so recoverable
           data is never deleted.
    """
    target_paths = list(split_frames.keys())
    _check_overwrite_paths(target_paths, force=force)

    for path in target_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    temp_paths: dict[Path, Path] = {
        target: target.with_name(f"{target.name}.tmp") for target in target_paths
    }
    backup_paths: dict[Path, Path] = {
        target: target.with_name(f"{target.name}.bak") for target in target_paths
    }
    existed_before: dict[Path, bool] = {target: target.exists() for target in target_paths}

    try:
        # Phase 1: Write and validate all temporary Parquet files before touching targets
        for target, df_part in split_frames.items():
            tmp_path = temp_paths[target]
            df_part.to_parquet(tmp_path, index=False)
            written_rows = validate_parquet_file(tmp_path)
            if written_rows != len(df_part):
                raise ValueError(
                    f"Row count mismatch in staged file {tmp_path}: "
                    f"expected {len(df_part)}, got {written_rows}."
                )

        # Phase 2a: Move any existing target files to .bak backups before publishing new files
        for target in target_paths:
            if existed_before[target] and target.exists():
                target.replace(backup_paths[target])

        # Phase 2b: Move validated .tmp files to target paths and validate final files
        for target, df_part in split_frames.items():
            tmp_path = temp_paths[target]
            tmp_path.replace(target)
            final_rows = validate_parquet_file(target)
            if final_rows != len(df_part):
                raise ValueError(
                    f"Row count mismatch in final file {target}: "
                    f"expected {len(df_part)}, got {final_rows}."
                )
    except Exception:
        # Roll back to pre-call state so we never leave a mix of old and new split files
        for target in target_paths:
            if existed_before[target]:
                bak_path = backup_paths[target]
                if bak_path.exists():
                    try:
                        bak_path.replace(target)
                    except OSError as restore_err:
                        logger.error(
                            "Failed to restore %s from backup %s (%s); keeping backup on disk.",
                            target,
                            bak_path,
                            restore_err,
                        )
            elif target.exists():
                try:
                    target.unlink()
                except OSError as unlink_err:
                    logger.error("Failed to remove partial file %s: %s", target, unlink_err)
        raise
    else:
        # Publication succeeded for all splits; remove backups
        for bak_path in backup_paths.values():
            if bak_path.exists():
                bak_path.unlink()
    finally:
        for tmp_path in temp_paths.values():
            if tmp_path.exists():
                tmp_path.unlink()


def prepare_data(
    input_file: Path,
    output_dir: Path,
    force: bool = False,
    timestamp_col: str = "lpep_pickup_datetime",
) -> PreparationSummary:
    """Clean raw taxi data, partition chronologically, validate, and save split Parquet files.

    Args:
        input_file: Path to the raw monthly Parquet file.
        output_dir: Directory where train.parquet, val.parquet, and test.parquet will be saved.
        force: If True, allows overwriting existing split files.
        timestamp_col: Column name used for chronological ordering.

    Returns:
        PreparationSummary describing the input, row counts, output paths, and split time ranges.
    """
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    # Fail fast before processing if target files already exist and force is False
    _check_overwrite_paths([train_path, val_path, test_path], force=force)

    validate_parquet_file(input_file)
    raw_df = pd.read_parquet(input_file)

    timed_splits: TimedDatasetSplits = split_with_timestamps(raw_df, timestamp_col=timestamp_col)
    splits = timed_splits.splits
    train_ts = timed_splits.train_timestamps
    val_ts = timed_splits.val_timestamps
    test_ts = timed_splits.test_timestamps

    train_df = _assemble_split_df(splits.X_train, splits.y_train)
    val_df = _assemble_split_df(splits.X_val, splits.y_val)
    test_df = _assemble_split_df(splits.X_test, splits.y_test)

    validate_prepared_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_ts=train_ts,
        val_ts=val_ts,
        test_ts=test_ts,
        expected_total_rows=timed_splits.cleaned_rows,
    )

    _write_splits_safely(
        {
            train_path: train_df,
            val_path: val_df,
            test_path: test_df,
        },
        force=force,
    )

    excluded_rows = timed_splits.total_raw_rows - timed_splits.cleaned_rows
    summary = PreparationSummary(
        input_file=input_file,
        total_raw_rows=timed_splits.total_raw_rows,
        cleaned_rows=timed_splits.cleaned_rows,
        excluded_rows=excluded_rows,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
        train_time_range=(train_ts.iloc[0], train_ts.iloc[-1]),
        val_time_range=(val_ts.iloc[0], val_ts.iloc[-1]),
        test_time_range=(test_ts.iloc[0], test_ts.iloc[-1]),
    )

    logger.info("Data preparation summary:")
    logger.info("  Input file:     %s", summary.input_file)
    logger.info(
        "  Row counts:     raw=%d, cleaned=%d, excluded=%d",
        summary.total_raw_rows,
        summary.cleaned_rows,
        summary.excluded_rows,
    )
    logger.info(
        "  Schema:         %d features (%s) + target '%s'",
        len(FEATURE_COLUMNS),
        ", ".join(FEATURE_COLUMNS),
        TARGET_COLUMN,
    )
    logger.info(
        "  Train split:    %s (%d rows, %s -> %s)",
        summary.train_path,
        summary.train_rows,
        summary.train_time_range[0],
        summary.train_time_range[1],
    )
    logger.info(
        "  Val split:      %s (%d rows, %s -> %s)",
        summary.val_path,
        summary.val_rows,
        summary.val_time_range[0],
        summary.val_time_range[1],
    )
    logger.info(
        "  Test split:     %s (%d rows, %s -> %s)",
        summary.test_path,
        summary.test_rows,
        summary.test_time_range[0],
        summary.test_time_range[1],
    )

    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for data preparation."""
    parser = argparse.ArgumentParser(
        description="Clean raw NYC Green Taxi data and write chronological train/val/test splits."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("data/raw/green_tripdata_2024-01.parquet"),
        help="Path to raw input Parquet file (default: data/raw/green_tripdata_2024-01.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for train.parquet, val.parquet, and test.parquet (default: data/processed)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing train/val/test Parquet files if present",
    )
    return parser


def main(argv: list[str] | None = None) -> PreparationSummary:
    """CLI entry point for preparing train, validation, and test Parquet files."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return prepare_data(
        input_file=args.input_file,
        output_dir=args.output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()
