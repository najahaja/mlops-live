"""Data cleaning, validation, and feature preparation for NYC Green Taxi records."""

from __future__ import annotations

import logging
from typing import NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)

# Strictly leakage-safe predictor features available at ride-request/pickup time
FEATURE_COLUMNS = [
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "passenger_count",
    "pickup_hour",
    "pickup_day_of_week",
]

TARGET_COLUMN = "duration_minutes"


class CleanedDataset(NamedTuple):
    """Container for processed features and targets."""

    features: pd.DataFrame
    target: pd.Series
    total_raw_rows: int
    cleaned_rows: int


def compute_trip_duration(df: pd.DataFrame) -> pd.Series:
    """Calculate trip duration in minutes from pickup and dropoff datetimes.

    Args:
        df: DataFrame containing 'lpep_pickup_datetime' and 'lpep_dropoff_datetime'.

    Returns:
        Series of trip duration in minutes as float64.
    """
    pickup = pd.to_datetime(df["lpep_pickup_datetime"])
    dropoff = pd.to_datetime(df["lpep_dropoff_datetime"])
    duration = (dropoff - pickup).dt.total_seconds() / 60.0
    return duration


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract strictly leakage-safe features from raw taxi records.

    Args:
        df: Raw taxi trip DataFrame.

    Returns:
        DataFrame containing only engineered and approved model features.
    """
    pickup_dt = pd.to_datetime(df["lpep_pickup_datetime"])

    features = pd.DataFrame(index=df.index)
    features["PULocationID"] = df["PULocationID"].astype("int64")
    features["DOLocationID"] = df["DOLocationID"].astype("int64")
    features["trip_distance"] = df["trip_distance"].astype("float64")

    # Impute missing passenger_count with 1 (standard default for single rider)
    passenger_count = df.get("passenger_count", pd.Series(1, index=df.index))
    features["passenger_count"] = passenger_count.fillna(1.0).astype("float64")

    features["pickup_hour"] = pickup_dt.dt.hour.astype("int64")
    features["pickup_day_of_week"] = pickup_dt.dt.day_of_week.astype("int64")

    return features[FEATURE_COLUMNS]


def clean_and_prepare_data(
    df: pd.DataFrame,
    min_duration: float = 1.0,
    max_duration: float = 60.0,
    min_distance: float = 0.01,
    max_distance: float = 100.0,
) -> CleanedDataset:
    """Filter realistic trips, remove target outliers, and extract leakage-safe features.

    Args:
        df: Raw NYC Green Taxi DataFrame.
        min_duration: Minimum acceptable trip duration in minutes.
        max_duration: Maximum acceptable trip duration in minutes.
        min_distance: Minimum acceptable trip distance in miles.
        max_distance: Maximum acceptable trip distance in miles.

    Returns:
        CleanedDataset containing features DataFrame and target Series.
    """
    total_raw = len(df)
    if total_raw == 0:
        raise ValueError("Input DataFrame is empty.")

    # Calculate target label
    duration = compute_trip_duration(df)

    # Valid trip mask (removes zero-duration, negative times, extreme outliers, invalid distance)
    valid_mask = (
        (duration >= min_duration)
        & (duration <= max_duration)
        & (df["trip_distance"] >= min_distance)
        & (df["trip_distance"] <= max_distance)
        & (df["PULocationID"] > 0)
        & (df["DOLocationID"] > 0)
    )

    filtered_df = df.loc[valid_mask].copy()
    cleaned_target = duration.loc[valid_mask].astype("float64")
    cleaned_target.name = TARGET_COLUMN

    cleaned_features = extract_features(filtered_df)

    logger.info(
        "Cleaned dataset: kept %d of %d rows (%.2f%%)",
        len(filtered_df),
        total_raw,
        (len(filtered_df) / total_raw) * 100,
    )

    return CleanedDataset(
        features=cleaned_features,
        target=cleaned_target,
        total_raw_rows=total_raw,
        cleaned_rows=len(filtered_df),
    )
