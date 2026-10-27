"""Unit tests for data cleaning and feature extraction."""

import pandas as pd
import pytest

from mlops_live.data.clean import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_and_prepare_data,
    compute_trip_duration,
    extract_features,
)


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Fixture providing raw taxi records with intentional anomalies."""
    return pd.DataFrame(
        {
            "VendorID": [1, 2, 2, 1, 2],
            "lpep_pickup_datetime": [
                "2024-01-15 08:00:00",  # normal (15 mins)
                "2024-01-15 09:00:00",  # zero duration (0 mins, invalid)
                "2024-01-15 10:00:00",  # outlier duration (120 mins, invalid)
                "2024-01-15 11:00:00",  # zero distance (invalid)
                "2024-01-15 12:00:00",  # normal (30 mins), missing passenger count
            ],
            "lpep_dropoff_datetime": [
                "2024-01-15 08:15:00",
                "2024-01-15 09:00:00",
                "2024-01-15 12:00:00",
                "2024-01-15 11:20:00",
                "2024-01-15 12:30:00",
            ],
            "PULocationID": [74, 75, 41, 74, 166],
            "DOLocationID": [42, 75, 42, 42, 238],
            "trip_distance": [2.5, 1.2, 5.0, 0.0, 3.8],
            "passenger_count": [1.0, 2.0, 1.0, 1.0, None],
            # Leakage / post-trip columns that must never end up in feature set:
            "fare_amount": [12.5, 8.0, 45.0, 0.0, 19.5],
            "tip_amount": [2.5, 0.0, 5.0, 0.0, 3.0],
            "total_amount": [17.5, 10.0, 55.0, 0.0, 25.0],
        }
    )


def test_compute_trip_duration(sample_raw_df: pd.DataFrame) -> None:
    durations = compute_trip_duration(sample_raw_df)
    expected = [15.0, 0.0, 120.0, 20.0, 30.0]
    assert durations.tolist() == expected


def test_extract_features_no_leakage(sample_raw_df: pd.DataFrame) -> None:
    features = extract_features(sample_raw_df)

    # Exactly the expected leakage-safe feature columns
    assert list(features.columns) == FEATURE_COLUMNS

    # Ensure no post-trip columns leaked
    for leaked_col in ["fare_amount", "tip_amount", "total_amount", "lpep_dropoff_datetime"]:
        assert leaked_col not in features.columns

    # Verify time extractions for row 0 (2024-01-15 is Monday = 0, 08:00 = 8)
    assert features.loc[0, "pickup_hour"] == 8
    assert features.loc[0, "pickup_day_of_week"] == 0

    # Verify missing passenger_count is imputed with 1.0
    assert features.loc[4, "passenger_count"] == 1.0


def test_clean_and_prepare_data_filters_anomalies(sample_raw_df: pd.DataFrame) -> None:
    cleaned = clean_and_prepare_data(sample_raw_df)

    assert cleaned.total_raw_rows == 5
    # Only row 0 (15 min, 2.5 mi) and row 4 (30 min, 3.8 mi) are valid
    assert cleaned.cleaned_rows == 2
    assert len(cleaned.features) == 2
    assert len(cleaned.target) == 2
    assert cleaned.target.name == TARGET_COLUMN

    assert cleaned.target.tolist() == [15.0, 30.0]


def test_clean_and_prepare_data_empty() -> None:
    empty_df = pd.DataFrame(columns=["lpep_pickup_datetime", "lpep_dropoff_datetime"])
    with pytest.raises(ValueError, match="empty"):
        clean_and_prepare_data(empty_df)
