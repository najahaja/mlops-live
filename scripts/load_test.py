"""CLI script for in-process and real localhost HTTP load testing of the FastAPI service."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
from fastapi.testclient import TestClient
from requests.adapters import HTTPAdapter

from mlops_live.serving import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("load_test")

ALLOWED_LOCAL_HOSTS = {"127.0.0.1", "localhost"}
IN_PROCESS_DEFAULT_REPORT = Path("reports/generated/load_test_report.json")
MAX_ALLOWED_REQUESTS = 5000
MAX_ALLOWED_CONCURRENCY = 64


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


def _validate_load_params(
    num_requests: int,
    concurrency: int,
    timeout_seconds: float = 5.0,
) -> None:
    """Validate request count, concurrency, and timeout bounds."""
    if num_requests <= 0 or num_requests > MAX_ALLOWED_REQUESTS:
        raise ValueError(
            f"num_requests must be between 1 and {MAX_ALLOWED_REQUESTS}, got {num_requests}."
        )
    if concurrency <= 0 or concurrency > MAX_ALLOWED_CONCURRENCY:
        raise ValueError(
            f"concurrency must be between 1 and {MAX_ALLOWED_CONCURRENCY}, got {concurrency}."
        )
    if timeout_seconds <= 0:
        raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}.")


def _validate_localhost_url(base_url: str) -> str:
    """Ensure base_url uses http/https and targets localhost only."""
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"base_url must use http or https scheme, got: {base_url}")
    if parsed.hostname not in ALLOWED_LOCAL_HOSTS:
        raise ValueError(
            f"Refusing non-localhost target '{parsed.hostname}'. "
            f"Allowed hosts: {sorted(ALLOWED_LOCAL_HOSTS)}."
        )
    return base_url.rstrip("/")


def _is_port_in_use(host: str, port: int) -> bool:
    """Check whether a TCP port on host is already accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _collect_environment_versions() -> dict[str, str]:
    """Capture Python and key package versions from the active environment."""
    env_info: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in (
        "fastapi",
        "uvicorn",
        "requests",
        "httpx",
        "pydantic",
        "scikit-learn",
        "mlflow",
        "sqlalchemy",
    ):
        try:
            env_info[pkg] = version(pkg)
        except PackageNotFoundError:
            env_info[pkg] = "unavailable"
    return env_info


def _compute_latency_percentiles(values_ms: list[float]) -> dict[str, float] | None:
    """Compute latency summary statistics, or return None when values_ms is empty."""
    if not values_ms:
        return None
    arr = np.asarray(values_ms, dtype=np.float64)
    return {
        "min": round(float(np.min(arr)), 3),
        "mean": round(float(np.mean(arr)), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "max": round(float(np.max(arr)), 3),
    }


def _write_json_report_safely(
    report: dict[str, Any],
    output_json: Path,
    allow_overwrite: bool = False,
) -> Path:
    """Write a JSON report atomically, optionally refusing to overwrite existing files."""
    if not allow_overwrite and output_json.exists():
        raise FileExistsError(
            f"Report file already exists and will not be overwritten: {output_json}"
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_json.with_name(f"{output_json.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        json.loads(tmp_path.read_text(encoding="utf-8"))
        if not allow_overwrite and output_json.exists():
            raise FileExistsError(
                f"Report file already exists and will not be overwritten: {output_json}"
            )
        tmp_path.replace(output_json)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return output_json


@contextmanager
def managed_uvicorn_server(
    host: str = "127.0.0.1",
    port: int = 8010,
    tracking_db: Path = Path("mlflow.db"),
    fallback_model_path: Path = Path("models/model.joblib"),
    alias: str = "champion",
    startup_timeout_seconds: float = 30.0,
    server_log_path: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Start a local Uvicorn server in a child process with an isolated temporary predictions DB.

    Ensures:
    - Only localhost (`127.0.0.1` or `localhost`) is permitted.
    - Fails immediately if the port is already occupied.
    - Uses a temporary SQLite database for prediction logging so `predictions.db` is untouched.
    - Captures server stdout/stderr to a non-overwriting log file.
    - Terminates only the child process spawned by this context manager on exit or error.
    """
    if host not in ALLOWED_LOCAL_HOSTS:
        raise ValueError(
            f"Managed server must bind to localhost ({sorted(ALLOWED_LOCAL_HOSTS)}), got '{host}'."
        )
    if not (1024 <= port <= 65535):
        raise ValueError(f"port must be between 1024 and 65535, got {port}.")
    if startup_timeout_seconds <= 0:
        raise ValueError(
            f"startup_timeout_seconds must be positive, got {startup_timeout_seconds}."
        )
    if _is_port_in_use(host, port):
        raise RuntimeError(
            f"Port {port} on {host} is already in use. Choose an unused port or stop the existing "
            "process."
        )

    ts_slug = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    unique_suffix = uuid.uuid4().hex[:6]
    resolved_log_path = server_log_path or Path(
        f"reports/generated/uvicorn_http_load_{ts_slug}_{unique_suffix}.log"
    )
    if resolved_log_path.exists():
        raise FileExistsError(
            f"Server log file already exists and will not be overwritten: {resolved_log_path}"
        )
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="mlops_live_http_load_"))
    temp_predictions_db = temp_dir / "temp_predictions.db"

    cmd = [
        sys.executable,
        "scripts/serve.py",
        "--host",
        host,
        "--port",
        str(port),
        "--tracking-db",
        str(tracking_db),
        "--fallback-model",
        str(fallback_model_path),
        "--predictions-db",
        str(temp_predictions_db),
        "--alias",
        alias,
    ]

    proc: subprocess.Popen[bytes] | None = None
    log_handle = open(resolved_log_path, "wb")
    base_url = f"http://{host}:{port}"
    t_start = time.perf_counter()
    temp_cleaned = False

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        deadline = time.perf_counter() + startup_timeout_seconds
        ready_health: dict[str, Any] | None = None

        with requests.Session() as session:
            session.mount("http://", HTTPAdapter(max_retries=0))
            while time.perf_counter() < deadline:
                exit_code = proc.poll()
                if exit_code is not None:
                    raise RuntimeError(
                        f"Managed Uvicorn server exited prematurely with code {exit_code}. "
                        f"Inspect log: {resolved_log_path}"
                    )
                try:
                    resp = session.get(f"{base_url}/health", timeout=1.0)
                    if resp.status_code == 200:
                        body = resp.json()
                        if body.get("status") == "ok" and body.get("model_loaded") is True:
                            ready_health = body
                            break
                except requests.RequestException:
                    pass
                time.sleep(0.2)

        if ready_health is None:
            raise TimeoutError(
                f"Managed Uvicorn server at {base_url} did not become ready within "
                f"{startup_timeout_seconds}s. Inspect log: {resolved_log_path}"
            )

        readiness_elapsed = round(time.perf_counter() - t_start, 4)
        server_info: dict[str, Any] = {
            "spawned_by_test": True,
            "pid": proc.pid,
            "base_url": base_url,
            "readiness_elapsed_seconds": readiness_elapsed,
            "log_file": str(resolved_log_path),
            "isolated_predictions_db": str(temp_predictions_db),
            "temp_predictions_db_cleaned": False,
            "health": ready_health,
        }
        yield server_info
    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)
        log_handle.close()
        if proc is None or proc.poll() is not None:
            for attempt in range(10):
                try:
                    shutil.rmtree(temp_dir)
                    temp_cleaned = True
                    break
                except OSError as exc:
                    if attempt < 9:
                        time.sleep(0.2)
                    else:
                        logger.warning(
                            "Could not remove temporary directory %s (left intact): %s",
                            temp_dir,
                            exc,
                        )
        if "server_info" in locals():
            server_info["temp_predictions_db_cleaned"] = temp_cleaned


def _probe_server_metadata(base_url: str, timeout_seconds: float) -> dict[str, Any]:
    """Query GET /health once to verify server reachability and capture runtime model metadata."""
    with requests.Session() as session:
        session.mount("http://", HTTPAdapter(max_retries=0))
        session.mount("https://", HTTPAdapter(max_retries=0))
        try:
            resp = session.get(f"{base_url}/health", timeout=timeout_seconds)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "reachable": True,
                    "status_code": 200,
                    "model_name": str(data.get("model_name", "unavailable")),
                    "model_version": str(data.get("model_version", "unavailable")),
                    "model_alias": str(data.get("model_alias", "unavailable")),
                    "model_source": str(data.get("model_source", "unavailable")),
                }
            return {
                "reachable": True,
                "status_code": resp.status_code,
                "model_name": "unavailable",
                "model_version": "unavailable",
                "model_alias": "unavailable",
                "model_source": "unavailable",
            }
        except requests.RequestException as exc:
            return {
                "reachable": False,
                "status_code": None,
                "error": str(exc),
                "model_name": "unavailable",
                "model_version": "unavailable",
                "model_alias": "unavailable",
                "model_source": "unavailable",
            }


def _send_single_http_prediction(
    predict_url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> tuple[str, int | None, float, float | None]:
    """Send a single POST /predict over HTTP without retries and classify the outcome.

    Returns:
        Tuple of (outcome_category, status_code, client_elapsed_ms, server_latency_ms).
        outcome_category is one of: 'success', 'http_error', 'timeout', 'connection_error',
        'other_error'.
    """
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    t0 = time.perf_counter()
    try:
        resp = session.post(predict_url, json=payload, timeout=timeout_seconds)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code != 200:
            return "http_error", resp.status_code, elapsed_ms, None

        body = resp.json()
        pred_val = float(body["predicted_duration_minutes"])
        srv_ms = float(body["latency_ms"])
        req_id = str(body["request_id"])
        if not np.isfinite(pred_val) or not np.isfinite(srv_ms) or not req_id:
            return "other_error", resp.status_code, elapsed_ms, None
        return "success", 200, elapsed_ms, srv_ms
    except requests.Timeout:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return "timeout", None, elapsed_ms, None
    except requests.ConnectionError:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return "connection_error", None, elapsed_ms, None
    except Exception:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return "other_error", None, elapsed_ms, None
    finally:
        session.close()


def run_http_load_test(
    base_url: str = "http://127.0.0.1:8000",
    num_requests: int = 50,
    concurrency: int = 4,
    timeout_seconds: float = 5.0,
    output_json: Path | None = None,
    max_error_rate: float = 0.0,
    max_p95_latency_ms: float = 500.0,
    server_process_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute concurrent real-HTTP requests against a localhost Uvicorn server."""
    _validate_load_params(
        num_requests=num_requests,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
    )
    normalized_base_url = _validate_localhost_url(base_url)

    ts_slug = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    resolved_output = output_json or Path(f"reports/generated/http_load_test_report_{ts_slug}.json")
    if resolved_output.resolve() == IN_PROCESS_DEFAULT_REPORT.resolve():
        raise ValueError(
            f"HTTP load test refuses to overwrite the in-process report path: {resolved_output}"
        )
    if resolved_output.exists():
        raise FileExistsError(
            f"Report file already exists and will not be overwritten: {resolved_output}"
        )

    health_meta = _probe_server_metadata(normalized_base_url, timeout_seconds=timeout_seconds)
    predict_url = f"{normalized_base_url}/predict"
    payloads = _generate_sample_payloads(num_requests=num_requests)

    client_success_latencies_ms: list[float] = []
    server_success_latencies_ms: list[float] = []
    counts = {
        "success": 0,
        "http_error": 0,
        "timeout": 0,
        "connection_error": 0,
        "other_error": 0,
    }

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(_send_single_http_prediction, predict_url, p, timeout_seconds)
            for p in payloads
        ]
        for fut in as_completed(futures):
            category, _code, c_ms, s_ms = fut.result()
            counts[category] += 1
            if category == "success" and s_ms is not None:
                client_success_latencies_ms.append(c_ms)
                server_success_latencies_ms.append(s_ms)
    total_wall_seconds = max(1e-6, time.perf_counter() - wall_start)

    success_count = counts["success"]
    failure_count = num_requests - success_count
    success_rate = success_count / num_requests
    error_rate = failure_count / num_requests
    rps = num_requests / total_wall_seconds

    client_stats = _compute_latency_percentiles(client_success_latencies_ms)
    server_stats = _compute_latency_percentiles(server_success_latencies_ms)

    sla_passed = bool(
        success_count > 0
        and client_stats is not None
        and error_rate <= max_error_rate
        and client_stats["p95"] <= max_p95_latency_ms
    )

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "methodology": "local_uvicorn_localhost_tcp_http",
        "methodology_note": (
            "Measured over localhost TCP HTTP against a local Uvicorn FastAPI process. "
            "Results depend on local CPU load, model warm-up, concurrency, and OS scheduling, "
            "and do not establish cloud or production-scale capacity."
        ),
        "base_url": normalized_base_url,
        "endpoint": "/predict",
        "model_info": {
            "project": "mlops-live",
            "model_name": health_meta["model_name"],
            "model_version": health_meta["model_version"],
            "model_alias": health_meta["model_alias"],
            "model_source": health_meta["model_source"],
        },
        "num_requests": num_requests,
        "concurrency": concurrency,
        "timeout_seconds": timeout_seconds,
        "success_count": success_count,
        "failure_count": failure_count,
        "failure_breakdown": {
            "http_errors": counts["http_error"],
            "timeouts": counts["timeout"],
            "connection_errors": counts["connection_error"],
            "other_errors": counts["other_error"],
        },
        "success_rate": round(float(success_rate), 6),
        "error_rate": round(float(error_rate), 6),
        "total_wall_seconds": round(float(total_wall_seconds), 4),
        "requests_per_second": round(float(rps), 2),
        "client_latency_ms": client_stats,
        "server_inference_latency_ms": server_stats,
        "server_process": server_process_info
        or {
            "spawned_by_test": False,
            "health_reachable": health_meta["reachable"],
            "health_status_code": health_meta["status_code"],
        },
        "environment": _collect_environment_versions(),
        "sla": {
            "max_error_rate": max_error_rate,
            "max_p95_latency_ms": max_p95_latency_ms,
            "sla_passed": sla_passed,
        },
        "output_json": str(resolved_output),
    }

    _write_json_report_safely(report, resolved_output, allow_overwrite=False)

    p50_str = f"{client_stats['p50']:.2f}ms" if client_stats else "null"
    p95_str = f"{client_stats['p95']:.2f}ms" if client_stats else "null"
    logger.info(
        "HTTP load test finished (%s): %d/%d succeeded (%.1f req/s) | "
        "Client P50=%s, P95=%s | SLA=%s | Report=%s",
        predict_url,
        success_count,
        num_requests,
        rps,
        p50_str,
        p95_str,
        sla_passed,
        resolved_output,
    )
    return report


def run_load_test(
    num_requests: int = 100,
    concurrency: int = 4,
    tracking_db: Path = Path("mlflow.db"),
    fallback_model_path: Path = Path("models/model.joblib"),
    predictions_db: Path = Path("predictions.db"),
    output_json: Path = IN_PROCESS_DEFAULT_REPORT,
    max_error_rate: float = 0.0,
    max_p95_latency_ms: float = 500.0,
) -> dict[str, Any]:
    """Execute in-process FastAPI TestClient requests (functional/microbenchmark mode)."""
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

    _write_json_report_safely(report, output_json, allow_overwrite=True)

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
    """Create CLI argument parser for in-process and real-HTTP load testing."""
    parser = argparse.ArgumentParser(
        description="Run concurrent load and latency test against the FastAPI prediction service."
    )
    parser.add_argument(
        "--mode",
        choices=["in_process", "http"],
        default=None,
        help="Execution mode: 'in_process' (TestClient default) or 'http' (real TCP HTTP)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Target localhost base URL for real HTTP mode (e.g., http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--spawn-server",
        action="store_true",
        help="Spawn an isolated local Uvicorn child process for HTTP testing and stop it after",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface for --spawn-server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8010,
        help="Local port for --spawn-server (default: 8010)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-request HTTP timeout in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=30.0,
        help="Server readiness timeout in seconds for --spawn-server (default: 30.0)",
    )
    parser.add_argument(
        "--server-log",
        type=Path,
        default=None,
        help="Non-overwriting log file path for --spawn-server stdout/stderr",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Total requests (default: 50 in HTTP mode, 100 in in_process mode)",
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
        help="SQLite database for in-process mode prediction logs (default: predictions.db)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON report path (never overwrites existing files in HTTP mode)",
    )
    parser.add_argument(
        "--max-p95-latency-ms",
        type=float,
        default=500.0,
        help="Maximum acceptable P95 latency in ms (default: 500.0)",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Execute load test CLI in either in-process or real-HTTP mode."""
    parser = build_parser()
    args = parser.parse_args(argv)

    use_http = args.mode == "http" or args.base_url is not None or bool(args.spawn_server)

    if not use_http:
        num_reqs = args.num_requests if args.num_requests is not None else 100
        out_path = args.output_json if args.output_json is not None else IN_PROCESS_DEFAULT_REPORT
        return run_load_test(
            num_requests=num_reqs,
            concurrency=args.concurrency,
            tracking_db=args.tracking_db,
            fallback_model_path=args.fallback_model,
            predictions_db=args.predictions_db,
            output_json=out_path,
            max_p95_latency_ms=args.max_p95_latency_ms,
        )

    num_reqs = args.num_requests if args.num_requests is not None else 50
    if args.spawn_server:
        with managed_uvicorn_server(
            host=args.host,
            port=args.port,
            tracking_db=args.tracking_db,
            fallback_model_path=args.fallback_model,
            startup_timeout_seconds=args.startup_timeout,
            server_log_path=args.server_log,
        ) as srv_info:
            report = run_http_load_test(
                base_url=srv_info["base_url"],
                num_requests=num_reqs,
                concurrency=args.concurrency,
                timeout_seconds=args.timeout,
                output_json=args.output_json,
                max_p95_latency_ms=args.max_p95_latency_ms,
                server_process_info=srv_info,
            )
        # Update report on disk with final temp_predictions_db_cleaned status after context exit
        report["server_process"]["temp_predictions_db_cleaned"] = srv_info[
            "temp_predictions_db_cleaned"
        ]
        _write_json_report_safely(report, Path(report["output_json"]), allow_overwrite=True)
        return report

    target_url = args.base_url or f"http://{args.host}:{args.port}"
    return run_http_load_test(
        base_url=target_url,
        num_requests=num_reqs,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout,
        output_json=args.output_json,
        max_p95_latency_ms=args.max_p95_latency_ms,
    )


if __name__ == "__main__":
    main()
