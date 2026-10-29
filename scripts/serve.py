"""CLI script to launch the FastAPI model serving service using Uvicorn."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from mlops_live.serving import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("serve_api")


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for the FastAPI serving server."""
    parser = argparse.ArgumentParser(description="Start the ML-Ops-Live FastAPI model server.")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)",
    )
    parser.add_argument(
        "--tracking-db",
        type=Path,
        default=Path("mlflow.db"),
        help="Path to MLflow SQLite database (default: mlflow.db)",
    )
    parser.add_argument(
        "--fallback-model",
        type=Path,
        default=Path("models/model.joblib"),
        help="Fallback joblib model path (default: models/model.joblib)",
    )
    parser.add_argument(
        "--predictions-db",
        type=Path,
        default=Path("predictions.db"),
        help="SQLite database path for prediction logging (default: predictions.db)",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="champion",
        help="MLflow registered model alias to serve (default: champion)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Start Uvicorn serving the FastAPI app."""
    parser = build_parser()
    args = parser.parse_args(argv)

    app = create_app(
        tracking_db=args.tracking_db,
        fallback_model_path=args.fallback_model,
        predictions_db=args.predictions_db,
        alias=args.alias,
    )
    logger.info("Starting FastAPI server on http://%s:%d ...", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

