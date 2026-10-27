"""Unit tests for data ingestion module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mlops_live.data.ingest import build_taxi_url, fetch_monthly_data, validate_parquet_file


def test_build_taxi_url_valid() -> None:
    url = build_taxi_url(2024, 1)
    assert url == "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-01.parquet"

    url_december = build_taxi_url(2023, 12, taxi_type="yellow")
    assert (
        url_december
        == "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-12.parquet"
    )


def test_build_taxi_url_invalid_month() -> None:
    with pytest.raises(ValueError, match="Month must be between 1 and 12"):
        build_taxi_url(2024, 0)
    with pytest.raises(ValueError, match="Month must be between 1 and 12"):
        build_taxi_url(2024, 13)


def test_build_taxi_url_invalid_year() -> None:
    with pytest.raises(ValueError, match="available from 2009 onwards"):
        build_taxi_url(2008, 5)


def test_validate_parquet_file_non_existent(tmp_path: Path) -> None:
    non_existent = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        validate_parquet_file(non_existent)


def test_validate_parquet_file_empty(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.parquet"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        validate_parquet_file(empty_file)


def test_validate_parquet_file_corrupted(tmp_path: Path) -> None:
    corrupt_file = tmp_path / "corrupted.parquet"
    corrupt_file.write_bytes(b"NOT_PARQUET_DATA")
    with pytest.raises(ValueError, match="Failed to read parquet metadata"):
        validate_parquet_file(corrupt_file)


def test_validate_parquet_file_valid(tmp_path: Path) -> None:
    valid_file = tmp_path / "sample.parquet"
    table = pa.Table.from_arrays([pa.array([1, 2, 3])], names=["col1"])
    pq.write_table(table, valid_file)

    num_rows = validate_parquet_file(valid_file)
    assert num_rows == 3


def test_fetch_monthly_data_skips_when_valid_exists(tmp_path: Path) -> None:
    dest_file = tmp_path / "green_tripdata_2024-01.parquet"
    table = pa.Table.from_arrays([pa.array([10, 20])], names=["trip_distance"])
    pq.write_table(table, dest_file)

    with patch("requests.get") as mock_get:
        result = fetch_monthly_data(2024, 1, output_dir=tmp_path, force=False)
        assert result == dest_file
        mock_get.assert_not_called()


def test_fetch_monthly_data_downloads_when_not_exists(tmp_path: Path) -> None:
    dest_file = tmp_path / "green_tripdata_2024-02.parquet"
    mock_table = pa.Table.from_arrays([pa.array([1, 2])], names=["trip_distance"])

    # Prepare raw parquet bytes to stream
    temp_mem_path = tmp_path / "temp.parquet"
    pq.write_table(mock_table, temp_mem_path)
    parquet_bytes = temp_mem_path.read_bytes()

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.iter_content.return_value = [parquet_bytes]
    mock_response.__enter__.return_value = mock_response

    with patch("requests.get", return_value=mock_response):
        result = fetch_monthly_data(2024, 2, output_dir=tmp_path)
        assert result == dest_file
        assert dest_file.exists()
        assert validate_parquet_file(dest_file) == 2
