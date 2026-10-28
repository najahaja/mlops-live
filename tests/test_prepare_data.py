"""Unit and CLI integration tests for scripts/prepare_data.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from scripts.prepare_data import (
    EXPECTED_SPLIT_COLUMNS,
    main,
    prepare_data,
    validate_prepared_splits,
)

from mlops_live.data import FEATURE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def synthetic_raw_parquet(tmp_path: Path) -> Path:
    """Create a synthetic raw Parquet file with 20 valid shuffled trips and 3 invalid trips."""
    base_times = pd.date_range("2024-01-10 00:00:00", periods=20, freq="h")
    shuffled_order = [19, 0, 18, 1, 17, 2, 16, 3, 15, 4, 14, 5, 13, 6, 12, 7, 11, 8, 10, 9]
    valid_pickups = base_times.take(shuffled_order)
    # Encode chronological position into duration (10.0 to 29.0 mins) so we can verify split content
    valid_durations = [float(10 + idx) for idx in shuffled_order]
    valid_dropoffs = [
        pickup + pd.Timedelta(minutes=dur)
        for pickup, dur in zip(valid_pickups, valid_durations, strict=True)
    ]

    valid_df = pd.DataFrame(
        {
            "lpep_pickup_datetime": valid_pickups,
            "lpep_dropoff_datetime": valid_dropoffs,
            "PULocationID": [74 + (i % 5) for i in range(20)],
            "DOLocationID": [42 + (i % 5) for i in range(20)],
            "trip_distance": [2.5 + (i * 0.1) for i in range(20)],
            "passenger_count": [1.0 if i % 4 != 0 else None for i in range(20)],
            "fare_amount": [15.0] * 20,  # Post-trip leakage column that must be excluded
        }
    )

    # 3 invalid rows: zero duration, extreme duration (180 min), and zero distance
    invalid_df = pd.DataFrame(
        {
            "lpep_pickup_datetime": pd.to_datetime(
                [
                    "2024-01-09 01:00:00",
                    "2024-01-09 02:00:00",
                    "2024-01-09 03:00:00",
                ]
            ),
            "lpep_dropoff_datetime": pd.to_datetime(
                [
                    "2024-01-09 01:00:00",  # 0 min duration
                    "2024-01-09 05:00:00",  # 180 min duration
                    "2024-01-09 03:15:00",  # valid duration, but 0.0 distance below
                ]
            ),
            "PULocationID": [74, 74, 74],
            "DOLocationID": [42, 42, 42],
            "trip_distance": [1.5, 5.0, 0.0],
            "passenger_count": [1.0, 1.0, 1.0],
            "fare_amount": [0.0, 90.0, 10.0],
        }
    )

    raw_df = pd.concat([valid_df, invalid_df], ignore_index=True)
    input_path = tmp_path / "raw_sample.parquet"
    raw_df.to_parquet(input_path, index=False)
    return input_path


def test_prepare_data_end_to_end_via_main(synthetic_raw_parquet: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"

    summary = main(
        [
            "--input-file",
            str(synthetic_raw_parquet),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert summary.total_raw_rows == 23
    assert summary.cleaned_rows == 20
    assert summary.excluded_rows == 3
    assert summary.train_rows == 14
    assert summary.val_rows == 3
    assert summary.test_rows == 3

    # Verify chronological timestamp boundaries
    assert summary.train_time_range == (
        pd.Timestamp("2024-01-10 00:00:00"),
        pd.Timestamp("2024-01-10 13:00:00"),
    )
    assert summary.val_time_range == (
        pd.Timestamp("2024-01-10 14:00:00"),
        pd.Timestamp("2024-01-10 16:00:00"),
    )
    assert summary.test_time_range == (
        pd.Timestamp("2024-01-10 17:00:00"),
        pd.Timestamp("2024-01-10 19:00:00"),
    )

    train_df = pd.read_parquet(output_dir / "train.parquet")
    val_df = pd.read_parquet(output_dir / "val.parquet")
    test_df = pd.read_parquet(output_dir / "test.parquet")

    # Verify exact 6 features + target and absence of pickup timestamp or leakage columns
    for split_df in (train_df, val_df, test_df):
        assert list(split_df.columns) == EXPECTED_SPLIT_COLUMNS
        assert len(FEATURE_COLUMNS) == 6
        assert "lpep_pickup_datetime" not in split_df.columns
        assert "fare_amount" not in split_df.columns
        assert not split_df.isna().any().any()

    # Because duration was constructed as 10 + chronological_index, verify exact partition contents
    assert train_df[TARGET_COLUMN].tolist() == [float(10 + i) for i in range(0, 14)]
    assert val_df[TARGET_COLUMN].tolist() == [float(10 + i) for i in range(14, 17)]
    assert test_df[TARGET_COLUMN].tolist() == [float(10 + i) for i in range(17, 20)]


def test_prepare_data_cli_subprocess(synthetic_raw_parquet: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "cli_out"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--input-file",
            str(synthetic_raw_parquet),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "val.parquet").exists()
    assert (output_dir / "test.parquet").exists()


def test_overwrite_protection_blocks_when_any_output_exists(
    synthetic_raw_parquet: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "processed"
    output_dir.mkdir(parents=True)

    # Pre-create only val.parquet to verify all three paths are checked before writing anything
    existing_val = output_dir / "val.parquet"
    sentinel_bytes = b"EXISTING_VAL_FILE"
    existing_val.write_bytes(sentinel_bytes)

    with pytest.raises(FileExistsError, match="Pass --force to overwrite"):
        prepare_data(input_file=synthetic_raw_parquet, output_dir=output_dir, force=False)

    # Existing file must be untouched and neither train.parquet nor test.parquet should be created
    assert existing_val.read_bytes() == sentinel_bytes
    assert not (output_dir / "train.parquet").exists()
    assert not (output_dir / "test.parquet").exists()

    # With --force, all three files should be cleanly written
    summary = prepare_data(input_file=synthetic_raw_parquet, output_dir=output_dir, force=True)
    assert summary.train_path.exists()
    assert summary.val_path.exists()
    assert summary.test_path.exists()
    assert existing_val.read_bytes() != sentinel_bytes


def test_safe_write_cleans_up_temp_files_on_failure(
    synthetic_raw_parquet: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "failed_out"

    # Simulate failure on the second staged validation call
    call_count = 0
    real_validate = __import__(
        "mlops_live.data.ingest", fromlist=["validate_parquet_file"]
    ).validate_parquet_file

    def fail_on_third_validate(path: Path) -> int:
        nonlocal call_count
        call_count += 1
        # call 1 is input_file, call 2 is train.parquet.tmp, call 3 is val.parquet.tmp
        if call_count == 3:
            raise ValueError("Simulated staging validation failure")
        return real_validate(path)

    with (
        patch("scripts.prepare_data.validate_parquet_file", side_effect=fail_on_third_validate),
        pytest.raises(ValueError, match="Simulated staging validation failure"),
    ):
        prepare_data(input_file=synthetic_raw_parquet, output_dir=output_dir, force=False)

    # Neither temporary files nor partial final split files should remain
    if output_dir.exists():
        assert list(output_dir.iterdir()) == []


def test_forced_overwrite_rollback_preserves_existing_files_on_phase2_failure(
    synthetic_raw_parquet: Path, tmp_path: Path
) -> None:
    """Verify that if replacement fails mid-flight under --force, all prior files are restored."""
    output_dir = tmp_path / "overwrite_rollback"
    output_dir.mkdir(parents=True)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    old_train_bytes = b"ORIGINAL_TRAIN_PARQUET_CONTENT"
    old_val_bytes = b"ORIGINAL_VAL_PARQUET_CONTENT"
    old_test_bytes = b"ORIGINAL_TEST_PARQUET_CONTENT"

    train_path.write_bytes(old_train_bytes)
    val_path.write_bytes(old_val_bytes)
    test_path.write_bytes(old_test_bytes)

    call_count = 0
    real_validate = __import__(
        "mlops_live.data.ingest", fromlist=["validate_parquet_file"]
    ).validate_parquet_file

    def fail_on_final_test_validation(path: Path) -> int:
        nonlocal call_count
        call_count += 1
        # Call sequence in prepare_data:
        # 1: input_file
        # 2, 3, 4: staged train.parquet.tmp, val.parquet.tmp, test.parquet.tmp (Phase 1)
        # 5: final train.parquet (Phase 2b - train already replaced)
        # 6: final val.parquet (Phase 2b - val already replaced)
        # 7: final test.parquet (Phase 2b - simulate failure here after train & val were replaced)
        if call_count == 7:
            raise ValueError("Simulated post-replacement validation failure on test.parquet")
        return real_validate(path)

    with (
        patch(
            "scripts.prepare_data.validate_parquet_file",
            side_effect=fail_on_final_test_validation,
        ),
        pytest.raises(ValueError, match="Simulated post-replacement validation failure"),
    ):
        prepare_data(input_file=synthetic_raw_parquet, output_dir=output_dir, force=True)

    # All three original files must be restored intact without mixing old and new splits
    assert train_path.read_bytes() == old_train_bytes
    assert val_path.read_bytes() == old_val_bytes
    assert test_path.read_bytes() == old_test_bytes

    # No .tmp or .bak files should remain after successful rollback
    remaining_names = sorted(p.name for p in output_dir.iterdir())
    assert remaining_names == ["test.parquet", "train.parquet", "val.parquet"]


def test_prepare_data_with_duplicate_timestamps_at_split_boundaries(tmp_path: Path) -> None:
    """Verify prepare_data succeeds when duplicate timestamps span train/val/test boundaries."""
    # All 20 valid trips share only two timestamps (10 at 08:00, 10 at 09:00),
    # so the train (14) / val (3) / test (3) boundaries fall across identical 09:00 timestamps.
    pickups = pd.to_datetime(["2024-01-15 08:00:00"] * 10 + ["2024-01-15 09:00:00"] * 10)
    durations = [float(10 + i) for i in range(20)]
    dropoffs = [p + pd.Timedelta(minutes=d) for p, d in zip(pickups, durations, strict=True)]

    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": pickups,
            "lpep_dropoff_datetime": dropoffs,
            "PULocationID": [74] * 20,
            "DOLocationID": [42] * 20,
            "trip_distance": [2.5] * 20,
            "passenger_count": [1.0] * 20,
        }
    )
    raw_path = tmp_path / "dup_ts.parquet"
    df.to_parquet(raw_path, index=False)

    out_dir = tmp_path / "dup_out"
    summary = prepare_data(input_file=raw_path, output_dir=out_dir)

    assert summary.train_time_range[1] == pd.Timestamp("2024-01-15 09:00:00")
    assert summary.val_time_range[0] == pd.Timestamp("2024-01-15 09:00:00")
    assert summary.val_time_range[1] == pd.Timestamp("2024-01-15 09:00:00")
    assert summary.test_time_range[0] == pd.Timestamp("2024-01-15 09:00:00")


def test_missing_and_invalid_inputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"

    # 1. Non-existent input file
    with pytest.raises(FileNotFoundError):
        prepare_data(input_file=tmp_path / "does_not_exist.parquet", output_dir=output_dir)

    # 2. Corrupted input file
    corrupted = tmp_path / "corrupted.parquet"
    corrupted.write_bytes(b"NOT_A_PARQUET_FILE")
    with pytest.raises(ValueError, match="Failed to read parquet metadata"):
        prepare_data(input_file=corrupted, output_dir=output_dir)

    # 3. Dataset with too few valid rows after cleaning (< 10 rows)
    tiny_df = pd.DataFrame(
        {
            "lpep_pickup_datetime": pd.date_range("2024-01-01", periods=5, freq="h"),
            "lpep_dropoff_datetime": pd.date_range("2024-01-01 00:15:00", periods=5, freq="h"),
            "PULocationID": [74] * 5,
            "DOLocationID": [42] * 5,
            "trip_distance": [2.0] * 5,
            "passenger_count": [1.0] * 5,
        }
    )
    tiny_path = tmp_path / "tiny.parquet"
    tiny_df.to_parquet(tiny_path, index=False)
    with pytest.raises(ValueError, match="Insufficient rows"):
        prepare_data(input_file=tiny_path, output_dir=output_dir)


def test_validate_prepared_splits_catches_boundary_and_schema_errors() -> None:
    base_df = pd.DataFrame(
        {
            "PULocationID": [74, 75],
            "DOLocationID": [42, 43],
            "trip_distance": [2.0, 3.0],
            "passenger_count": [1.0, 2.0],
            "pickup_hour": [8, 9],
            "pickup_day_of_week": [0, 0],
            "duration_minutes": [12.0, 15.0],
        }
    )
    ts_train = pd.Series(pd.to_datetime(["2024-01-01 08:00:00", "2024-01-01 09:00:00"]))
    ts_val = pd.Series(pd.to_datetime(["2024-01-01 10:00:00", "2024-01-01 11:00:00"]))
    ts_test = pd.Series(pd.to_datetime(["2024-01-01 12:00:00", "2024-01-01 13:00:00"]))

    # Valid case passes
    validate_prepared_splits(base_df, base_df, base_df, ts_train, ts_val, ts_test, 6)

    # Boundary violation: train max > val min
    ts_val_overlap = pd.Series(pd.to_datetime(["2024-01-01 08:30:00", "2024-01-01 11:00:00"]))
    with pytest.raises(ValueError, match="Chronological boundary violation between train and val"):
        validate_prepared_splits(base_df, base_df, base_df, ts_train, ts_val_overlap, ts_test, 6)

    # Non-monotonic timestamps inside a split
    ts_unsorted = pd.Series(pd.to_datetime(["2024-01-01 09:00:00", "2024-01-01 08:00:00"]))
    with pytest.raises(ValueError, match="not sorted chronologically"):
        validate_prepared_splits(base_df, base_df, base_df, ts_unsorted, ts_val, ts_test, 6)

    # Extra column (e.g., accidentally including timestamp in features)
    bad_cols_df = base_df.copy()
    bad_cols_df["lpep_pickup_datetime"] = ts_train
    with pytest.raises(ValueError, match="unexpected columns"):
        validate_prepared_splits(bad_cols_df, base_df, base_df, ts_train, ts_val, ts_test, 6)
