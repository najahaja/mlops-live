"""Unit tests for chronological time-based dataset splitting."""

import pandas as pd
import pytest

from mlops_live.data.split import split_with_timestamps, time_based_train_val_test_split


@pytest.fixture
def chronological_raw_df() -> pd.DataFrame:
    """Fixture with 20 trips ordered randomly in time."""
    dates = pd.date_range("2024-01-01 00:00:00", periods=20, freq="h")
    # shuffle timestamps intentionally to test sorting logic
    shuffled_dates = dates.take(
        [19, 0, 18, 1, 17, 2, 16, 3, 15, 4, 14, 5, 13, 6, 12, 7, 11, 8, 10, 9]
    )

    return pd.DataFrame(
        {
            "lpep_pickup_datetime": shuffled_dates,
            "lpep_dropoff_datetime": shuffled_dates + pd.Timedelta(minutes=15),
            "PULocationID": [74] * 20,
            "DOLocationID": [42] * 20,
            "trip_distance": [2.5] * 20,
            "passenger_count": [1.0] * 20,
        }
    )


def test_time_based_split_chronological_order(chronological_raw_df: pd.DataFrame) -> None:
    splits = time_based_train_val_test_split(
        chronological_raw_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # 20 rows total: 70% = 14 train, 15% = 3 val, 15% = 3 test
    assert len(splits.X_train) == 14
    assert len(splits.X_val) == 3
    assert len(splits.X_test) == 3

    assert len(splits.y_train) == 14
    assert len(splits.y_val) == 3
    assert len(splits.y_test) == 3


def test_time_based_split_invalid_ratios(chronological_raw_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        time_based_train_val_test_split(
            chronological_raw_df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2
        )

    with pytest.raises(ValueError, match="non-negative"):
        time_based_train_val_test_split(
            chronological_raw_df, train_ratio=-0.1, val_ratio=0.5, test_ratio=0.6
        )


def test_time_based_split_missing_timestamp_col(chronological_raw_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Required timestamp column"):
        time_based_train_val_test_split(chronological_raw_df, timestamp_col="missing_pickup_time")


def test_time_based_split_insufficient_rows() -> None:
    small_df = pd.DataFrame(
        {
            "lpep_pickup_datetime": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            "lpep_dropoff_datetime": ["2024-01-01 00:10:00", "2024-01-01 01:10:00"],
            "PULocationID": [1, 2],
            "DOLocationID": [2, 3],
            "trip_distance": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="Insufficient rows"):
        time_based_train_val_test_split(small_df)


def test_split_with_timestamps_handles_duplicate_timestamps_and_custom_index() -> None:
    """Verify stable tie-breaking on duplicate timestamps and duplicate DataFrame indices."""
    # Create 20 valid trips where several share identical timestamps, plus 1 invalid row
    pickup_times = pd.to_datetime(
        [
            "2024-01-01 08:00:00",  # invalid row (0 distance)
            *["2024-01-01 10:00:00"] * 5,
            *["2024-01-01 09:00:00"] * 10,
            *["2024-01-01 10:00:00"] * 5,
        ]
    )
    # Encode original position (1..20) in trip_distance and duration to verify stable tie order
    distances = [0.0] + [float(i) for i in range(1, 21)]
    durations = [10.0] + [float(10 + i) for i in range(1, 21)]
    dropoff_times = [
        p + pd.Timedelta(minutes=d) for p, d in zip(pickup_times, durations, strict=True)
    ]

    # Intentionally assign a non-unique index to verify index normalization
    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": pickup_times,
            "lpep_dropoff_datetime": dropoff_times,
            "PULocationID": [74] * 21,
            "DOLocationID": [42] * 21,
            "trip_distance": distances,
            "passenger_count": [1.0] * 21,
        },
        index=[99] * 21,
    )

    timed = split_with_timestamps(df)
    assert timed.total_raw_rows == 21
    assert timed.cleaned_rows == 20

    # First 10 chronologically are the 09:00:00 rows (positions 6..15),
    # next 10 are the 10:00:00 rows in their original relative order (1..5, then 16..20)
    expected_ordered_distances = [float(i) for i in range(6, 16)] + [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
    ]
    actual_distances = (
        timed.splits.X_train["trip_distance"].tolist()
        + timed.splits.X_val["trip_distance"].tolist()
        + timed.splits.X_test["trip_distance"].tolist()
    )
    assert actual_distances == expected_ordered_distances

    # Split boundaries (14 train, 3 val, 3 test) fall inside the 10:00:00 timestamp ties
    assert timed.train_timestamps.iloc[-1] == pd.Timestamp("2024-01-01 10:00:00")
    assert timed.val_timestamps.iloc[0] == pd.Timestamp("2024-01-01 10:00:00")
    assert timed.val_timestamps.iloc[-1] == pd.Timestamp("2024-01-01 10:00:00")
    assert timed.test_timestamps.iloc[0] == pd.Timestamp("2024-01-01 10:00:00")
