"""Tests for the public Streamlit demo (webapp/) and its sanitized asset exporter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = REPO_ROOT / "webapp"


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


core = _load("demo_core", WEBAPP_DIR / "demo_core.py")
exporter = _load("export_streamlit_assets", REPO_ROOT / "scripts" / "export_streamlit_assets.py")

VALID_INPUT: dict[str, Any] = {
    "PULocationID": 74,
    "DOLocationID": 75,
    "trip_distance": 2.5,
    "passenger_count": 1.0,
    "pickup_hour": 17,
    "pickup_day_of_week": 2,
}


@pytest.fixture()
def trained_pipeline():
    from mlops_live.models.train import build_hist_gbdt_candidate

    rng = np.random.default_rng(0)
    n = 400
    features = pd.DataFrame(
        {
            "PULocationID": rng.integers(1, 20, n),
            "DOLocationID": rng.integers(1, 20, n),
            "trip_distance": rng.uniform(0.5, 10.0, n),
            "passenger_count": rng.integers(1, 4, n).astype(float),
            "pickup_hour": rng.integers(0, 24, n),
            "pickup_day_of_week": rng.integers(0, 7, n),
        }
    )
    target = 3.0 + 3.0 * features["trip_distance"] + rng.normal(0, 1, n)
    pipeline = build_hist_gbdt_candidate()
    pipeline.fit(features[core.FEATURE_COLUMNS], target)
    return pipeline


@pytest.fixture()
def asset_bundle(tmp_path: Path, trained_pipeline) -> Path:
    """Create a minimal fake repo and export it, so tests never touch real project files."""
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    joblib.dump(trained_pipeline, repo / "models" / "model.joblib")
    (repo / "models" / "metrics.json").write_text(
        json.dumps(
            {
                "selected_model": "hist_gbdt_candidate",
                "feature_columns": core.FEATURE_COLUMNS,
                "target_column": "duration_minutes",
                "mlflow": {
                    "tracking_uri": "sqlite:///C:/Users/someone/mlflow.db",
                    "registered_model_name": "nyc-green-taxi-duration-regressor",
                    "champion": {"version": "2", "alias": "champion"},
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "assets"
    exporter.export_assets(
        output_dir=out, repo_root=repo, include_zone_lookup=False, verify_holdout=False
    )
    return out


# --------------------------------------------------------------------------- validation


def test_valid_input_is_accepted_and_ordered():
    features, errors = core.validate_trip_input(VALID_INPUT)
    assert errors == []
    assert list(features) == core.FEATURE_COLUMNS


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trip_distance", 0.0),
        ("trip_distance", 150.0),
        ("passenger_count", 0.0),
        ("pickup_hour", 24),
        ("pickup_day_of_week", 7),
        ("PULocationID", 0),
    ],
)
def test_invalid_inputs_return_readable_errors(field: str, value: Any):
    features, errors = core.validate_trip_input({**VALID_INPUT, field: value})
    assert features is None
    assert errors and errors[0].startswith(field)


def test_unknown_fields_are_rejected():
    features, errors = core.validate_trip_input({**VALID_INPUT, "fare_amount": 10.0})
    assert features is None and errors


# --------------------------------------------------------------------------- model loading


def test_bundled_model_loads_and_predicts(asset_bundle: Path):
    pipeline, info = core.load_bundled_model(asset_bundle)
    assert info.sha256_verified is True
    assert info.model_name == "hist_gbdt_candidate"
    assert info.registry_version == "2"
    features, _ = core.validate_trip_input(VALID_INPUT)
    prediction = core.predict_duration_local(pipeline, features)
    assert prediction >= 1.0


def test_missing_model_raises_friendly_error(tmp_path: Path):
    with pytest.raises(core.DemoModelError, match="not available"):
        core.load_bundled_model(tmp_path)


def test_tampered_model_fails_integrity_check(asset_bundle: Path):
    with (asset_bundle / "model.joblib").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(core.DemoModelError, match="integrity"):
        core.load_bundled_model(asset_bundle)


def test_missing_and_malformed_reports_return_none(tmp_path: Path):
    assert core.load_json_asset("absent.json", tmp_path) is None
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert core.load_json_asset("bad.json", tmp_path) is None
    assert core.load_zone_lookup(tmp_path) == {}
    assert core.model_comparison_rows(None, None, None) == []
    assert core.drift_rows(None) == []


def test_local_inference_matches_fastapi_predict(tmp_path: Path, trained_pipeline):
    """The demo must return exactly what the FastAPI /predict endpoint returns."""
    from fastapi.testclient import TestClient

    from mlops_live.serving.app import create_app

    model_path = tmp_path / "model.joblib"
    joblib.dump(trained_pipeline, model_path)
    app = create_app(
        tracking_db=tmp_path / "absent_mlflow.db",
        fallback_model_path=model_path,
        fallback_metrics_path=tmp_path / "absent_metrics.json",
        predictions_db=tmp_path / "predictions.db",
    )
    features, _ = core.validate_trip_input(VALID_INPUT)
    with TestClient(app) as client:
        api_value = client.post("/predict", json=VALID_INPUT).json()["predicted_duration_minutes"]
    assert core.predict_duration_local(trained_pipeline, features) == api_value


# --------------------------------------------------------------------------- remote service


def test_api_url_resolution_and_validation():
    assert core.get_configured_api_url({"MLOPS_API_URL": "https://api.example.com/"}, {}) == (
        "https://api.example.com"
    )
    assert core.get_configured_api_url(None, {"MLOPS_API_URL": "http://localhost:8000"}) == (
        "http://localhost:8000"
    )
    assert core.get_configured_api_url(None, {"MLOPS_API_URL": "javascript:alert(1)"}) is None
    assert core.get_configured_api_url(None, {}) is None

    class RaisingSecrets(dict):
        def get(self, *_args, **_kwargs):
            raise FileNotFoundError("no secrets.toml")

    assert core.get_configured_api_url(RaisingSecrets(), {}) is None


def test_unreachable_remote_service_raises_friendly_error(monkeypatch):
    def boom(*_args, **_kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    features, _ = core.validate_trip_input(VALID_INPUT)
    with pytest.raises(core.RemoteServiceError, match="unavailable"):
        core.predict_duration_remote("http://127.0.0.1:9", features)


def test_remote_service_success(monkeypatch):
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "predicted_duration_minutes": 12.34567,
                "model_name": "hist_gbdt_candidate",
                "model_version": "2",
                "model_alias": "champion",
            }

    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: float):
        captured.update(url=url, json=json, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    features, _ = core.validate_trip_input(VALID_INPUT)
    result = core.predict_duration_remote("https://api.example.com/", features)
    assert captured["url"] == "https://api.example.com/predict"
    assert result["predicted_duration_minutes"] == 12.3457


# --------------------------------------------------------------------------- exporter safety


def test_sanitizer_drops_sensitive_keys_and_redacts_paths():
    payload = {
        "sandbox_directory": r"C:\Users\DELL\AppData\Local\Temp\x",
        "server_process": {"pid": 1},
        "base_url": "http://127.0.0.1:8010",
        "current_source": r"data\processed\val.parquet",
        "nested": [{"path": "/home/user/file.db"}, {"uri": "sqlite:///D:/mlflow.db"}],
    }
    clean = exporter.sanitize_payload(payload)
    assert "sandbox_directory" not in clean and "server_process" not in clean
    assert clean["base_url"] == "http://127.0.0.1:8010"
    assert clean["current_source"] == "data/processed/val.parquet"
    assert clean["nested"][0]["path"] == exporter.REDACTED
    assert clean["nested"][1]["uri"] == exporter.REDACTED


def test_exporter_never_overwrites_without_flag(asset_bundle: Path, tmp_path: Path):
    metrics = json.loads((asset_bundle / "metrics.json").read_text(encoding="utf-8"))
    assert "tracking_uri" not in metrics["mlflow"]
    with pytest.raises(FileExistsError):
        exporter.export_assets(
            output_dir=asset_bundle,
            repo_root=tmp_path / "repo",
            include_zone_lookup=False,
            verify_holdout=False,
        )


def test_published_assets_contain_no_local_paths():
    """Guard the committed webapp/assets bundle against leaking machine-specific details."""
    assets = WEBAPP_DIR / "assets"
    if not assets.is_dir():
        pytest.skip("webapp/assets not exported")
    for path in assets.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        for marker in ("AppData", "C:\\\\Users", "sqlite:///", "sandbox_directory", '"pid"'):
            assert marker not in text, f"{marker} leaked in {path.name}"


# --------------------------------------------------------------------------- headless UI


def test_streamlit_app_renders_and_predicts_without_errors():
    """Run the real entrypoint headlessly with Streamlit's AppTest harness."""
    pytest.importorskip("streamlit")
    if not (WEBAPP_DIR / "assets" / "model.joblib").is_file():
        pytest.skip("webapp/assets not exported")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(WEBAPP_DIR / "streamlit_app.py"), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    assert len(app.tabs) == 5

    submit_btn = [b for b in app.button if "Predict" in b.label][0]
    submit_btn.click().run()
    assert not app.exception, app.exception
    predicted = [m for m in app.metric if m.label == "Predicted Trip Duration"]
    assert predicted and predicted[0].value.endswith("minutes")


def test_streamlit_app_shows_validation_error_for_invalid_distance():
    pytest.importorskip("streamlit")
    if not (WEBAPP_DIR / "assets" / "model.joblib").is_file():
        pytest.skip("webapp/assets not exported")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(WEBAPP_DIR / "streamlit_app.py"), default_timeout=60)
    app.run()
    app.number_input[0].set_value(0.0)  # trip distance below the 0.01-mile minimum
    submit_btn = [b for b in app.button if "Predict" in b.label][0]
    submit_btn.click().run()
    assert not app.exception, app.exception
    assert any("correct the following" in e.value for e in app.error)
    assert not [m for m in app.metric if m.label == "Predicted Trip Duration"]


def test_no_mojibake_or_encoding_artifacts_in_webapp():
    """Verify webapp files are free of double-encoded UTF-8 or CP1252 mojibake."""
    suspicious = [
        "\u00e2\u20ac\u2014",  # â€”
        "\u00e2\u20ac\u2013",  # â€“
        "\u00e2\u20ac\u00a6",  # â€¦
        "\u00f0\u0178",  # ðŸ
        "\ufffd",  # replacement character
        "\u00c3\u00a2",  # double encoded C3 A2
        "\u00c3\u00b0",  # double encoded C3 B0
        "\u00c2\u00b2",  # double encoded Â²
        "\u00c2\u00b7",  # double encoded Â·
    ]
    for file_path in WEBAPP_DIR.glob("**/*.py"):
        content = file_path.read_text(encoding="utf-8")
        for pattern in suspicious:
            assert pattern not in content, f"Mojibake pattern {repr(pattern)} in {file_path.name}"


def test_streamlit_app_exact_title():
    """Verify exact title and taxi emoji in streamlit_app.py."""
    content = (WEBAPP_DIR / "streamlit_app.py").read_text(encoding="utf-8")
    assert "ML-Ops-Live — NYC Taxi Trip-Duration Prediction & MLOps Platform" in content
    assert 'page_icon="🚕"' in content


def test_root_streamlit_app_delegates_to_webapp():
    """Verify that root streamlit_app.py forwards execution to webapp/streamlit_app.py."""
    pytest.importorskip("streamlit")
    if not (WEBAPP_DIR / "assets" / "model.joblib").is_file():
        pytest.skip("webapp/assets not exported")
    from streamlit.testing.v1 import AppTest

    root_entry = REPO_ROOT / "streamlit_app.py"
    assert root_entry.is_file(), "Root streamlit_app.py missing"
    app = AppTest.from_file(str(root_entry), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    assert len(app.tabs) == 5
