"""Framework-independent logic for the ML-Ops-Live public Streamlit demo.

This module deliberately avoids importing Streamlit so it can be unit-tested directly. It reuses
the project's own contracts instead of re-implementing them:

* ``FEATURE_COLUMNS`` comes from ``mlops_live.data.clean`` (the training feature contract).
* Input validation uses the FastAPI ``TripPredictionRequest`` Pydantic schema, loaded directly
  from ``src/mlops_live/serving/schemas.py`` so the heavy serving package (FastAPI, MLflow,
  SQLAlchemy) is not required in the lightweight cloud environment.
* Local inference mirrors ``POST /predict`` in ``mlops_live.serving.app`` (feature ordering and the
  ``max(1.0, prediction)`` floor), and the parity is asserted in ``tests/test_webapp.py``.

All operations are read-only: nothing here writes to MLflow, SQLite databases, or model files.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlparse

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
SRC_DIR = REPO_ROOT / "src"
ASSETS_DIR = APP_DIR / "assets"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlops_live.data.clean import FEATURE_COLUMNS  # noqa: E402

logger = logging.getLogger(__name__)

API_URL_SETTING = "MLOPS_API_URL"
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class DemoModelError(RuntimeError):
    """Raised when the bundled demo model cannot be loaded or used."""


class RemoteServiceError(RuntimeError):
    """Raised when the optional external FastAPI prediction service cannot be used."""


@dataclass(frozen=True)
class ModelInfo:
    """Public, non-sensitive provenance metadata for the loaded model."""

    model_name: str
    registry_version: str
    registry_alias: str
    source: str
    sha256_verified: bool


# --------------------------------------------------------------------------- schema reuse


def _load_schemas_module() -> ModuleType:
    """Load the FastAPI Pydantic schemas without importing the full serving package."""
    module_name = "mlops_live_standalone_serving_schemas"
    if module_name in sys.modules:
        return sys.modules[module_name]
    schema_path = SRC_DIR / "mlops_live" / "serving" / "schemas.py"
    spec = importlib.util.spec_from_file_location(module_name, schema_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load serving schemas from {schema_path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


TripPredictionRequest = _load_schemas_module().TripPredictionRequest


def validate_trip_input(raw: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate raw form values against the API contract.

    Returns:
        (features, []) when valid, ordered by FEATURE_COLUMNS; otherwise (None, errors).
    """
    from pydantic import ValidationError

    try:
        request = TripPredictionRequest(**dict(raw))
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            field = ".".join(str(part) for part in err.get("loc", ())) or "input"
            errors.append(f"{field}: {err.get('msg', 'invalid value')}")
        return None, errors
    except TypeError as exc:
        return None, [f"input: {exc}"]

    payload = request.model_dump()
    return {column: payload[column] for column in FEATURE_COLUMNS}, []


# --------------------------------------------------------------------------- assets


def load_json_asset(name: str, assets_dir: Path = ASSETS_DIR) -> dict[str, Any] | None:
    """Load a JSON asset, returning None if it is missing or malformed."""
    path = assets_dir / name
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read asset %s: %s", name, exc)
        return None
    return payload if isinstance(payload, dict) else None


def load_zone_lookup(assets_dir: Path = ASSETS_DIR) -> dict[int, str]:
    """Return {LocationID: 'Borough - Zone'} from the bundled TLC lookup, or {} if absent."""
    path = assets_dir / "taxi_zone_lookup.csv"
    if not path.is_file():
        return {}
    zones: dict[int, str] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    location_id = int(row["LocationID"])
                except (KeyError, TypeError, ValueError):
                    continue
                borough = (row.get("Borough") or "").strip()
                zone = (row.get("Zone") or "").strip()
                zones[location_id] = f"{borough} - {zone}".strip(" -") or f"Zone {location_id}"
    except OSError as exc:
        logger.warning("Could not read taxi zone lookup: %s", exc)
        return {}
    return zones


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundled_model(assets_dir: Path = ASSETS_DIR) -> tuple[Any, ModelInfo]:
    """Load the exported champion pipeline and verify it against the manifest checksum."""
    model_path = assets_dir / "model.joblib"
    manifest = load_json_asset("manifest.json", assets_dir) or {}
    if not model_path.is_file():
        raise DemoModelError("The bundled model artifact is not available in this deployment.")

    expected_sha = manifest.get("model_sha256")
    sha_ok = bool(expected_sha) and _sha256(model_path) == expected_sha
    if expected_sha and not sha_ok:
        raise DemoModelError("Bundled model failed its integrity check and was not loaded.")

    try:
        import joblib

        pipeline = joblib.load(model_path)
    except Exception as exc:
        logger.exception("Model deserialization failed")
        raise DemoModelError("The bundled model could not be loaded.") from exc

    if not hasattr(pipeline, "predict"):
        raise DemoModelError("The bundled artifact is not a valid prediction pipeline.")

    info = ModelInfo(
        model_name=str(manifest.get("model_name", "unknown")),
        registry_version=str(manifest.get("registry_version", "unknown")),
        registry_alias=str(manifest.get("registry_alias", "champion")),
        source="bundled_champion_artifact",
        sha256_verified=sha_ok,
    )
    return pipeline, info


# --------------------------------------------------------------------------- inference


def predict_duration_local(pipeline: Any, features: Mapping[str, Any]) -> float:
    """Predict trip duration (minutes) exactly as ``POST /predict`` does."""
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "PULocationID": int(features["PULocationID"]),
                "DOLocationID": int(features["DOLocationID"]),
                "trip_distance": float(features["trip_distance"]),
                "passenger_count": float(features["passenger_count"]),
                "pickup_hour": int(features["pickup_hour"]),
                "pickup_day_of_week": int(features["pickup_day_of_week"]),
            }
        ]
    )[FEATURE_COLUMNS]
    try:
        raw_prediction = float(pipeline.predict(frame)[0])
    except Exception as exc:
        logger.exception("Prediction failed")
        raise DemoModelError("The model could not produce a prediction for this input.") from exc
    return round(max(1.0, raw_prediction), 4)


def normalize_api_url(url: str | None) -> str | None:
    """Return a cleaned http(s) base URL, or None when unset or invalid."""
    if not url:
        return None
    candidate = str(url).strip().rstrip("/")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return candidate


def get_configured_api_url(
    secrets: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve the optional FastAPI service URL from Streamlit secrets, then environment."""
    environ = os.environ if environ is None else environ
    value: Any = None
    if secrets is not None:
        try:
            value = secrets.get(API_URL_SETTING)
        except Exception:  # st.secrets raises when no secrets file exists
            value = None
    if not value:
        value = environ.get(API_URL_SETTING)
    return normalize_api_url(value)


def predict_duration_remote(
    api_url: str,
    features: Mapping[str, Any],
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Call an externally hosted ML-Ops-Live FastAPI ``POST /predict`` endpoint."""
    import requests

    base_url = normalize_api_url(api_url)
    if base_url is None:
        raise RemoteServiceError("The configured prediction service URL is invalid.")
    try:
        response = requests.post(f"{base_url}/predict", json=dict(features), timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        prediction = float(payload["predicted_duration_minutes"])
    except Exception as exc:
        raise RemoteServiceError("The external prediction service is unavailable.") from exc
    return {
        "predicted_duration_minutes": round(prediction, 4),
        "model_name": str(payload.get("model_name", "unknown")),
        "model_version": str(payload.get("model_version", "unknown")),
        "model_alias": str(payload.get("model_alias", "unknown")),
    }


# --------------------------------------------------------------------------- report shaping


def model_comparison_rows(
    metrics: Mapping[str, Any] | None,
    holdout: Mapping[str, Any] | None,
    out_of_time: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Flatten verified metrics into rows tagged with their dataset/evaluation split."""
    rows: list[dict[str, Any]] = []

    def add(split: str, rows_count: Any, model: str, payload: Mapping[str, Any] | None) -> None:
        if not payload:
            return
        rows.append(
            {
                "Evaluation split": split,
                "Rows": rows_count,
                "Model": model,
                "MAE (min)": payload.get("mae"),
                "RMSE (min)": payload.get("rmse"),
                "R²": payload.get("r2"),
            }
        )

    if metrics:
        models = metrics.get("models", {})
        for name in ("dummy_median", "ridge_baseline", "hist_gbdt_candidate"):
            val = models.get(name, {}).get("val_metrics")
            add("Validation (Jan 2024)", (val or {}).get("num_samples"), name, val)
    if holdout:
        for role in ("challenger", "champion"):
            entry = holdout.get(role, {})
            test_metrics = entry.get("test_metrics")
            add(
                "Isolated holdout test (Jan 2024)",
                (test_metrics or {}).get("num_samples"),
                entry.get("model_name", role),
                test_metrics,
            )
    if out_of_time:
        add(
            "Out-of-time (full Feb 2024)",
            out_of_time.get("feb_2024_cleaned_rows"),
            "ridge_baseline",
            out_of_time.get("challenger_v1_ridge_metrics"),
        )
        add(
            "Out-of-time (full Feb 2024)",
            out_of_time.get("feb_2024_cleaned_rows"),
            "hist_gbdt_candidate",
            out_of_time.get("champion_v2_hist_gbdt_metrics"),
        )
    return rows


def drift_rows(drift_report: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten a drift report into per-column PSI/KS rows (feature + prediction drift)."""
    if not drift_report:
        return []
    rows: list[dict[str, Any]] = []
    columns = dict(drift_report.get("feature_drift", {}))
    if drift_report.get("prediction_drift"):
        columns["predicted_duration_minutes"] = drift_report["prediction_drift"]
    for name, report in columns.items():
        rows.append(
            {
                "Column": name,
                "Type": "prediction" if name == "predicted_duration_minutes" else "feature",
                "PSI": report.get("psi"),
                "PSI severity": report.get("psi_severity"),
                "KS statistic": report.get("ks_statistic"),
                "KS p-value": report.get("ks_p_value"),
                "Reference mean": report.get("reference_mean"),
                "Current mean": report.get("current_mean"),
                "Status": report.get("evaluation_status", "evaluated"),
                "Drift detected": bool(report.get("drift_detected", False)),
            }
        )
    return rows
