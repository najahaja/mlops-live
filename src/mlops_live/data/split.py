"""Chronological, time-based dataset splitting to prevent temporal leakage."""

from __future__ import annotations

import logging
from typing import NamedTuple

import pandas as pd

from mlops_live.data.clean import CleanedDataset, clean_and_prepare_data

logger = logging.getLogger(__name__)


class DatasetSplits(NamedTuple):
    """Container for chronological train, validation, and test splits."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


class TimedDatasetSplits(NamedTuple):
    """Container for chronological splits with aligned pickup timestamps and row counts."""

    splits: DatasetSplits
    train_timestamps: pd.Series
    val_timestamps: pd.Series
    test_timestamps: pd.Series
    total_raw_rows: int
    cleaned_rows: int


def split_with_timestamps(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_col: str = "lpep_pickup_datetime",
) -> TimedDatasetSplits:
    """Clean, sort chronologically with stable tie-breaking, and return splits with timestamps.

    Args:
        df: Raw or partially cleaned DataFrame containing timestamp_col.
        train_ratio: Proportion of data allocated to training (default: 0.70).
        val_ratio: Proportion of data allocated to validation (default: 0.15).
        test_ratio: Proportion of data allocated to test (default: 0.15).
        timestamp_col: Column name containing pickup datetime for sorting.

    Returns:
        TimedDatasetSplits containing DatasetSplits, per-split pickup timestamp Series,
        and raw/cleaned row counts.

    Raises:
        ValueError: If split ratios do not sum to 1.0, are non-positive, timestamp_col is
            missing, or fewer than 10 valid rows remain after cleaning.
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.999 <= total_ratio <= 1.001):
        r_sum = f"{train_ratio} + {val_ratio} + {test_ratio}"
        raise ValueError(f"Split ratios must sum to 1.0, got {r_sum} = {total_ratio}")
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative, and train_ratio must be positive.")

    if timestamp_col not in df.columns:
        raise ValueError(f"Required timestamp column '{timestamp_col}' not found in DataFrame.")

    # Normalize index first so duplicate or non-contiguous indices in df cannot misalign .loc
    normalized_df = df.reset_index(drop=True)

    # 1. Clean the dataset first
    cleaned: CleanedDataset = clean_and_prepare_data(normalized_df)
    n = cleaned.cleaned_rows

    if n < 10:
        raise ValueError(f"Insufficient rows ({n}) to perform time-based splitting.")

    # 2. Extract pickup timestamps for cleaned rows and use stable mergesort for deterministic ties
    timestamps = pd.to_datetime(normalized_df.loc[cleaned.features.index, timestamp_col])
    sorted_indices = timestamps.sort_values(kind="mergesort").index

    sorted_features = cleaned.features.loc[sorted_indices].reset_index(drop=True)
    sorted_target = cleaned.target.loc[sorted_indices].reset_index(drop=True)
    sorted_timestamps = timestamps.loc[sorted_indices].reset_index(drop=True)

    # 3. Calculate chronological split boundaries
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    X_train = sorted_features.iloc[:n_train].copy()
    y_train = sorted_target.iloc[:n_train].copy()
    train_timestamps = sorted_timestamps.iloc[:n_train].reset_index(drop=True)

    X_val = sorted_features.iloc[n_train : n_train + n_val].copy()
    y_val = sorted_target.iloc[n_train : n_train + n_val].copy()
    val_timestamps = sorted_timestamps.iloc[n_train : n_train + n_val].reset_index(drop=True)

    X_test = sorted_features.iloc[n_train + n_val :].copy()
    y_test = sorted_target.iloc[n_train + n_val :].copy()
    test_timestamps = sorted_timestamps.iloc[n_train + n_val :].reset_index(drop=True)

    logger.info(
        "Time-based split completed: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(X_train),
        len(X_val),
        len(X_test),
        n,
    )

    splits = DatasetSplits(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    return TimedDatasetSplits(
        splits=splits,
        train_timestamps=train_timestamps,
        val_timestamps=val_timestamps,
        test_timestamps=test_timestamps,
        total_raw_rows=cleaned.total_raw_rows,
        cleaned_rows=cleaned.cleaned_rows,
    )


def time_based_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_col: str = "lpep_pickup_datetime",
) -> DatasetSplits:
    """Sort chronologically and partition into train, validation, and test sets.

    Args:
        df: Raw or partially cleaned DataFrame containing timestamp_col.
        train_ratio: Proportion of data allocated to training (default: 0.70).
        val_ratio: Proportion of data allocated to validation (default: 0.15).
        test_ratio: Proportion of data allocated to test (default: 0.15).
        timestamp_col: Column name containing pickup datetime for sorting.

    Returns:
        DatasetSplits with X and y partitioned chronologically without time leakage.

    Raises:
        ValueError: If split ratios do not sum to 1.0 or are non-positive.
    """
    return split_with_timestamps(
        df=df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        timestamp_col=timestamp_col,
    ).splits
