"""Unit tests for real localhost HTTP load testing, safety guards, and server lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests
import scripts.load_test as load_test_mod
from scripts.load_test import (
    IN_PROCESS_DEFAULT_REPORT,
    _compute_latency_percentiles,
    managed_uvicorn_server,
    run_http_load_test,
)
from scripts.load_test import (
    main as load_test_main,
)


class _DummyResponse:
    """Lightweight mock HTTP response object for unit testing."""

    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload


def test_run_http_load_test_success_metrics_and_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify successful HTTP 200 workload computes separate client/server latencies and schema."""

    def fake_get(self: Any, url: str, timeout: float = 5.0) -> _DummyResponse:
        assert url.endswith("/health")
        return _DummyResponse(
            200,
            {
                "status": "ok",
                "model_loaded": True,
                "model_name": "nyc-green-taxi-duration",
                "model_version": "1",
                "model_alias": "champion",
                "model_source": "mlflow_registry",
            },
        )

    def fake_post(
        self: Any,
        url: str,
        json: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> _DummyResponse:
        assert url.endswith("/predict")
        assert json is not None
        assert "PULocationID" in json
        return _DummyResponse(
            200,
            {
                "request_id": "req-123",
                "predicted_duration_minutes": 13.75,
                "latency_ms": 2.45,
                "model_name": "nyc-green-taxi-duration",
                "model_version": "1",
                "model_alias": "champion",
                "model_source": "mlflow_registry",
            },
        )

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(requests.Session, "post", fake_post)

    report_path = tmp_path / "http_load_report.json"
    report = run_http_load_test(
        base_url="http://127.0.0.1:8000",
        num_requests=12,
        concurrency=3,
        timeout_seconds=3.0,
        output_json=report_path,
    )

    assert report["methodology"] == "local_uvicorn_localhost_tcp_http"
    assert "do not establish cloud or production-scale capacity" in report["methodology_note"]
    assert report["base_url"] == "http://127.0.0.1:8000"
    assert report["endpoint"] == "/predict"
    assert report["num_requests"] == 12
    assert report["concurrency"] == 3
    assert report["success_count"] == 12
    assert report["failure_count"] == 0
    assert report["success_rate"] == 1.0
    assert report["error_rate"] == 0.0
    assert report["requests_per_second"] > 0.0
    assert report["failure_breakdown"] == {
        "http_errors": 0,
        "timeouts": 0,
        "connection_errors": 0,
        "other_errors": 0,
    }
    assert report["model_info"] == {
        "project": "mlops-live",
        "model_name": "nyc-green-taxi-duration",
        "model_version": "1",
        "model_alias": "champion",
        "model_source": "mlflow_registry",
    }

    for key in ("min", "mean", "p50", "p95", "p99", "max"):
        assert report["client_latency_ms"][key] >= 0.0
        assert report["server_inference_latency_ms"][key] == pytest.approx(2.45)

    assert report["sla"]["sla_passed"] is True
    assert report_path.exists()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["success_count"] == 12


def test_run_http_load_test_mixed_failures_and_categories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify HTTP 500/422, timeouts, connection errors, and malformed payloads are categorized."""
    monkeypatch.setattr(
        requests.Session,
        "get",
        lambda self, url, timeout=5.0: _DummyResponse(200, {"status": "ok", "model_loaded": True}),
    )

    call_idx = {"n": 0}

    def fake_post(
        self: Any,
        url: str,
        json: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> _DummyResponse:
        call_idx["n"] += 1
        idx = call_idx["n"]
        if idx == 1:
            return _DummyResponse(
                200,
                {
                    "request_id": "ok-1",
                    "predicted_duration_minutes": 11.2,
                    "latency_ms": 3.1,
                },
            )
        if idx == 2:
            return _DummyResponse(500, {"detail": "internal server error"})
        if idx == 3:
            return _DummyResponse(422, {"detail": "validation error"})
        if idx == 4:
            raise requests.Timeout("request timed out")
        if idx == 5:
            raise requests.ConnectionError("connection refused")
        # idx == 6: malformed response missing required fields
        return _DummyResponse(200, {"unexpected": "schema"})

    monkeypatch.setattr(requests.Session, "post", fake_post)

    report_path = tmp_path / "mixed_failures_report.json"
    report = run_http_load_test(
        base_url="http://localhost:8000",
        num_requests=6,
        concurrency=1,
        timeout_seconds=2.0,
        output_json=report_path,
        max_error_rate=0.0,
    )

    assert report["num_requests"] == 6
    assert report["success_count"] == 1
    assert report["failure_count"] == 5
    assert report["failure_breakdown"] == {
        "http_errors": 2,
        "timeouts": 1,
        "connection_errors": 1,
        "other_errors": 1,
    }
    assert report["client_latency_ms"] is not None
    assert report["server_inference_latency_ms"]["p50"] == pytest.approx(3.1)
    assert report["sla"]["sla_passed"] is False


def test_run_http_load_test_all_failed_reports_null_latencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify zero successful requests produce null (None) latency summaries, not zeros."""
    assert _compute_latency_percentiles([]) is None

    def fail_conn(*args: Any, **kwargs: Any) -> Any:
        raise requests.ConnectionError("server down")

    monkeypatch.setattr(requests.Session, "get", fail_conn)
    monkeypatch.setattr(requests.Session, "post", fail_conn)

    report_path = tmp_path / "zero_success_report.json"
    report = run_http_load_test(
        base_url="http://127.0.0.1:8000",
        num_requests=4,
        concurrency=2,
        output_json=report_path,
    )

    assert report["success_count"] == 0
    assert report["failure_count"] == 4
    assert report["failure_breakdown"]["connection_errors"] == 4
    assert report["client_latency_ms"] is None
    assert report["server_inference_latency_ms"] is None
    assert report["sla"]["sla_passed"] is False

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["client_latency_ms"] is None
    assert persisted["server_inference_latency_ms"] is None


def test_report_protection_refuses_overwriting_existing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify existing report files are never overwritten and their contents stay unchanged."""
    existing_report = tmp_path / "existing_http_report.json"
    original_content = '{"sentinel": "must_not_change", "count": 99}\n'
    existing_report.write_text(original_content, encoding="utf-8")

    with pytest.raises(FileExistsError, match="Report file already exists"):
        run_http_load_test(
            base_url="http://127.0.0.1:8000",
            num_requests=5,
            concurrency=1,
            output_json=existing_report,
        )

    # Confirm original content is completely untouched
    assert existing_report.read_text(encoding="utf-8") == original_content

    # Also verify IN_PROCESS_DEFAULT_REPORT is explicitly blocked in HTTP mode
    with pytest.raises(ValueError, match="refuses to overwrite the in-process report path"):
        run_http_load_test(
            base_url="http://127.0.0.1:8000",
            num_requests=5,
            concurrency=1,
            output_json=IN_PROCESS_DEFAULT_REPORT,
        )


def test_validation_rejects_non_localhost_and_invalid_parameters(tmp_path: Path) -> None:
    """Verify non-localhost URLs and invalid numeric bounds raise ValueError immediately."""
    out = tmp_path / "unused.json"

    for bad_url in (
        "http://192.168.1.10:8000",
        "https://example.com",
        "http://0.0.0.0:8000",
        "ftp://127.0.0.1:8000",
    ):
        with pytest.raises(ValueError):
            run_http_load_test(base_url=bad_url, output_json=out)

    with pytest.raises(ValueError, match="num_requests"):
        run_http_load_test(base_url="http://127.0.0.1:8000", num_requests=0, output_json=out)

    with pytest.raises(ValueError, match="concurrency"):
        run_http_load_test(base_url="http://127.0.0.1:8000", concurrency=0, output_json=out)

    with pytest.raises(ValueError, match="timeout_seconds"):
        run_http_load_test(base_url="http://127.0.0.1:8000", timeout_seconds=-1.0, output_json=out)


def test_managed_uvicorn_server_rejects_occupied_port_and_cleans_up_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify port-in-use failure, premature exit handling, and readiness timeout cleanup."""
    # 1. Port already in use
    monkeypatch.setattr(load_test_mod, "_is_port_in_use", lambda host, port: True)
    with (
        pytest.raises(RuntimeError, match="already in use"),
        managed_uvicorn_server(
            host="127.0.0.1",
            port=8010,
            server_log_path=tmp_path / "port_busy.log",
        ),
    ):
        pass

    # 2. Readiness timeout terminates the child process and cleans up temp_dir
    monkeypatch.setattr(load_test_mod, "_is_port_in_use", lambda host, port: False)

    terminated = {"called": False, "waited": False}

    class _FakeHangingProc:
        pid = 43210

        def poll(self) -> int | None:
            return 0 if terminated["called"] else None

        def terminate(self) -> None:
            terminated["called"] = True

        def wait(self, timeout: float | None = None) -> int:
            terminated["waited"] = True
            return 0

        def kill(self) -> None:
            terminated["called"] = True

    monkeypatch.setattr(
        load_test_mod.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeHangingProc(),
    )
    monkeypatch.setattr(
        requests.Session,
        "get",
        lambda self, url, timeout=1.0: _DummyResponse(503, {"status": "starting"}),
    )

    timeout_log = tmp_path / "timeout_server.log"
    with (
        pytest.raises(TimeoutError, match="did not become ready"),
        managed_uvicorn_server(
            host="127.0.0.1",
            port=8011,
            startup_timeout_seconds=0.3,
            server_log_path=timeout_log,
        ),
    ):
        pass

    assert terminated["called"] is True
    assert terminated["waited"] is True


def test_spawn_server_cli_uses_isolated_predictions_db_and_stops_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify --spawn-server passes an isolated temporary predictions DB and cleans it up."""
    monkeypatch.setattr(load_test_mod, "_is_port_in_use", lambda host, port: False)

    captured_cmd: list[str] = []
    proc_state = {"terminated": False}

    class _FakeReadyProc:
        pid = 54321

        def poll(self) -> int | None:
            return 0 if proc_state["terminated"] else None

        def terminate(self) -> None:
            proc_state["terminated"] = True

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def kill(self) -> None:
            proc_state["terminated"] = True

    def fake_popen(cmd: list[str], **kwargs: Any) -> _FakeReadyProc:
        captured_cmd.extend(cmd)
        return _FakeReadyProc()

    monkeypatch.setattr(load_test_mod.subprocess, "Popen", fake_popen)

    def fake_get(self: Any, url: str, timeout: float = 1.0) -> _DummyResponse:
        return _DummyResponse(
            200,
            {
                "status": "ok",
                "model_loaded": True,
                "model_name": "nyc-green-taxi-duration",
                "model_version": "1",
                "model_alias": "champion",
                "model_source": "mlflow_registry",
            },
        )

    def fake_post(
        self: Any,
        url: str,
        json: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> _DummyResponse:
        return _DummyResponse(
            200,
            {
                "request_id": "spawn-req-1",
                "predicted_duration_minutes": 10.5,
                "latency_ms": 1.9,
            },
        )

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(requests.Session, "post", fake_post)

    report_file = tmp_path / "spawned_http_report.json"
    server_log = tmp_path / "spawned_server.log"

    report = load_test_main(
        [
            "--spawn-server",
            "--host",
            "127.0.0.1",
            "--port",
            "8012",
            "--num-requests",
            "5",
            "--concurrency",
            "2",
            "--output-json",
            str(report_file),
            "--server-log",
            str(server_log),
        ]
    )

    assert "--predictions-db" in captured_cmd
    db_arg_index = captured_cmd.index("--predictions-db") + 1
    isolated_db_arg = Path(captured_cmd[db_arg_index])
    assert isolated_db_arg.name == "temp_predictions.db"
    assert isolated_db_arg.resolve() != Path("predictions.db").resolve()
    assert not isolated_db_arg.parent.exists()

    assert proc_state["terminated"] is True
    assert report["server_process"]["spawned_by_test"] is True
    assert report["server_process"]["temp_predictions_db_cleaned"] is True

    saved_report = json.loads(report_file.read_text(encoding="utf-8"))
    assert saved_report["server_process"]["temp_predictions_db_cleaned"] is True
