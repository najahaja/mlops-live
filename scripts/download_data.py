"""CLI script to download TLC Green Taxi data for specified year and month."""

import argparse
import logging
from pathlib import Path

from mlops_live.data.ingest import fetch_monthly_data, validate_parquet_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NYC Green Taxi monthly trip records.")
    parser.add_argument("--year", type=int, required=True, help="Year of trip data (e.g. 2024)")
    parser.add_argument("--month", type=int, required=True, help="Month of trip data (1-12)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Target folder (defaults to data/raw)",
    )
    parser.add_argument("--force", action="store_true", help="Force re-download if file exists")

    args = parser.parse_args()

    saved_file = fetch_monthly_data(
        year=args.year,
        month=args.month,
        output_dir=args.output_dir,
        force=args.force,
    )
    rows = validate_parquet_file(saved_file)
    logger.info("Successfully fetched %s (total rows: %d)", saved_file, rows)


if __name__ == "__main__":
    main()
