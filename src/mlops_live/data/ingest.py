"""Data ingestion and validation utilities for NYC Green Taxi dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
import requests

logger = logging.getLogger(__name__)

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


def build_taxi_url(year: int, month: int, taxi_type: str = "green") -> str:
    """Build the official TLC CloudFront URL for a given year and month.

    Args:
        year: Trip record year (e.g., 2024).
        month: Trip record month (1-12).
        taxi_type: Taxi color/type (default: 'green').

    Returns:
        Canonical download URL string.
    """
    if not (1 <= month <= 12):
        raise ValueError(f"Month must be between 1 and 12, got: {month}")
    if year < 2009:
        raise ValueError(f"TLC records are only available from 2009 onwards, got: {year}")

    return f"{TLC_BASE_URL}/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"


def validate_parquet_file(file_path: Path) -> int:
    """Validate that the file exists and is a readable Parquet file.

    Args:
        file_path: Path to the parquet file.

    Returns:
        Number of rows in the Parquet file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or corrupted.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {file_path}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"Parquet file is empty (0 bytes): {file_path}")

    try:
        metadata = pq.read_metadata(file_path)
        num_rows = metadata.num_rows
        if num_rows <= 0:
            raise ValueError(f"Parquet file contains zero rows: {file_path}")
        return num_rows
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Failed to read parquet metadata from {file_path}: {exc}") from exc


def fetch_monthly_data(
    year: int,
    month: int,
    output_dir: Path,
    taxi_type: str = "green",
    force: bool = False,
    timeout: int = 60,
) -> Path:
    """Download a single monthly Parquet file if not already present and valid.

    Args:
        year: Year of records to download.
        month: Month of records to download.
        output_dir: Directory where the file should be saved.
        taxi_type: Taxi type ('green' by default).
        force: If True, re-downloads even if the file exists.
        timeout: HTTP request timeout in seconds.

    Returns:
        Path to the downloaded and validated Parquet file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    target_path = output_dir / filename

    if not force and target_path.exists():
        try:
            row_count = validate_parquet_file(target_path)
            logger.info("File already exists and is valid (%d rows): %s", row_count, target_path)
            return target_path
        except ValueError:
            logger.warning("Existing file at %s is corrupted; re-downloading...", target_path)

    url = build_taxi_url(year=year, month=month, taxi_type=taxi_type)
    logger.info("Downloading %s to %s", url, target_path)

    temp_path = target_path.with_suffix(".tmp")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        # Atomic replace so we don't leave partial files on disk
        temp_path.replace(target_path)
        validate_parquet_file(target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return target_path
