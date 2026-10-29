"""CLI script for concurrent load and latency performance testing of the FastAPI prediction API."""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from mlops_live.serving import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("load_test")


def _generate_sample_payloads(num_requests: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate realistic NYC Green Taxi prediction request payloads."""
    rng = np.random.default_rng(seed)
    zones = [41, 42, 74, 75, 116, 151, 166, 238, 239, 244]
    payloads: list[dict[str, Any]] = []
    for _ in range(num_requests):
        payloads.append(
            {
                "PULocationID": int(rng.choice(zones)),
                "DOLocationID": int(rng.choice(zones)),
                "trip_distance": round(float(rng.uniform(0.5, 14.0)), 2),
                "passenger_count": float(rng.choice([1.0, 2.0, 3.0])),
                "pickup_hour": int(rng.integers(0, 24)),
                "pickup_day_of_week": int(rng.integers(0, 7)),
            }
        )
    return payloads


def run_load_test(
    num_requests: int = 100,
    concurrency: int = 4,
    tracking_db: Path = Path("mlflow.db"),
    fallback_model_path: Path = Path("models/model.joblib"),
    predictions_db: Path = Path("predictions.db"),
    output_json: Path = Path("reports/generated/load_test_report.json"),
    max_error_rate: float = 0.0,
    max_p95_latency_ms: float = 500.0,
) -> dict[str, Any]:
    """Execute concurrent requests against the FastAPI prediction service and report SLA metrics."""
    if num_requests <= 0 or concurrency <= 0:
        raise ValueError("num_requests and concurrency must be positive integers.")

    app = create_app(
        tracking_db=tracking_db,
        fallback_model_path=fallback_model_path,
        predictions_db=predictions_db,
    )
    payloads = _generate_sample_payloads(num_requests=num_requests)

    client_latencies_ms: list[float] = []
    server_latencies_ms: list[float] = []
    status_codes: list[int] = []

    with TestClient(app) as client:
        # Warm up a single request so model/connection initialization isn't counted in steady-state
        client.get("/health")

        def send_one(payload: dict[str, Any]) -> tuple[int, float, float]:
            t0 = time.perf_counter()
            resp = client.post("/predict", json=payload)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            srv_ms = float(resp.json().get("latency_ms", 0.0)) if resp.status_code == 200 else 0.0
            return resp.status_code, elapsed_ms, srv_ms

        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(send_one, p) for p in payloads]
            for fut in as_completed(futures):
                code, c_ms, s_ms = fut.result()
                status_codes.append(code)
                client_latencies_ms.append(c_ms)
                if code == 200:
                    server_latencies_ms.append(s_ms)
        total_wall_seconds = max(1e-6, time.perf_counter() - wall_start)

    success_count = sum(1 for c in status_codes if c == 200)
    failure_count = num_requests - success_count
    error_rate = failure_count / num_requests
    rps = num_requests / total_wall_seconds

    c_arr = np.asarray(client_latencies_ms, dtype=np.float64)
    s_arr = (
        np.asarray(server_latencies_ms, dtype=np.float64)
        if server_latencies_ms
        else np.zeros(1, dtype=np.float64)
    )

    p95_client_ms = float(np.percentile(c_arr, 95))
    sla_passed = bool(error_rate <= max_error_rate and p95_client_ms <= max_p95_latency_ms)

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "num_requests": num_requests,
        "concurrency": concurrency,
        "success_count": success_count,
        "failure_count": failure_count,
        "error_rate": round(float(error_rate), 6),
        "total_wall_seconds": round(float(total_wall_seconds), 4),
        "requests_per_second": round(float(rps), 2),
        "client_latency_ms": {
            "min": round(float(np.min(c_arr)), 3),
            "mean": round(float(np.mean(c_arr)), 3),
            "p50": round(float(np.percentile(c_arr, 50)), 3),
            "p95": round(p95_client_ms, 3),
            "p99": round(float(np.percentile(c_arr, 99)), 3),
            "max": round(float(np.max(c_arr)), 3),
        },
        "server_inference_latency_ms": {
            "mean": round(float(np.mean(s_arr)), 3),
            "p50": round(float(np.percentile(s_arr, 50)), 3),
            "p95": round(float(np.percentile(s_arr, 95)), 3),
        },
        "sla": {
            "max_error_rate": max_error_rate,
            "max_p95_latency_ms": max_p95_latency_ms,
            "sla_passed": sla_passed,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_json.with_name(f"{output_json.name}.tmp")
    try:
        tmp_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        tmp_path.replace(output_json)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    logger.info(
        "Load test finished: %d/%d succeeded (%.1f req/s) | Client P50=%.2fms, P95=%.2fms | SLA=%s",
        success_count,
        num_requests,
        rps,
        report["client_latency_ms"]["p50"],
        report["client_latency_ms"]["p95"],
        sla_passed,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for load testing."""
    parser = argparse.ArgumentParser(
        description="Run concurrent load and latency test against the FastAPI prediction service."
    )
    parser.add_argument(
        "--num-requests", type=int, default=100, help="Total requests (default: 100)"
    )
    parser.add_argument("--concurrency", type=int, default=4, help="Worker threads (default: 4)")
    parser.add_argument(
        "--tracking-db",
        type=Path,
        default=Path("mlflow.db"),
        help="MLflow SQLite database (default: mlflow.db)",
    )
    parser.add_argument(
        "--fallback-model",
        type=Path,
        default=Path("models/model.joblib"),
        help="Fallback joblib model (default: models/model.joblib)",
    )
    parser.add_argument(
        "--predictions-db",
        type=Path,
        default=Path("predictions.db"),
        help="SQLite database for prediction logs (default: predictions.db)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/generated/load_test_report.json"),
        help="Output JSON report path (default: reports/generated/load_test_report.json)",
    )
    parser.add_argument(
        "--max-p95-latency-ms",
        type=float,
        default=500.0,
        help="Maximum acceptable P95 latency in ms (default: 500.0)",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Execute load test CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_load_test(
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        tracking_db=args.tracking_db,
        fallback_model_path=args.fallback_model,
        predictions_db=args.predictions_db,
        output_json=args.output_json,
        max_p95_latency_ms=args.max_p95_latency_ms,
    )


if __name__ == "__main__":
    main()
