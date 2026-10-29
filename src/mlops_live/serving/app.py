"""FastAPI application for serving NYC Green Taxi duration predictions and logging to SQLite."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from sklearn.pipeline import Pipeline

from mlops_live.data.clean import FEATURE_COLUMNS, TARGET_COLUMN
from mlops_live.serving.database import (
    count_logged_predictions,
    create_prediction_engine,
    log_prediction_record,
)
from mlops_live.serving.schemas import (
    HealthResponse,
    ModelInfoResponse,
    TripPredictionRequest,
    TripPredictionResponse,
)
from mlops_live.tracking.mlflow_tracker import (
    CHAMPION_ALIAS,
    DEFAULT_REGISTERED_MODEL_NAME,
    build_sqlite_tracking_uri,
    get_alias_version_info,
    load_registered_sklearn_model,
)

logger = logging.getLogger(__name__)


class LoadedModelBundle(NamedTuple):
    """Container for an active serving model pipeline and its provenance metadata."""

    pipeline: Pipeline
    model_name: str
    model_version: str
    model_alias: str
    model_source: str


def load_serving_model(
    tracking_db: Path = Path("mlflow.db"),
    fallback_model_path: Path = Path("models/model.joblib"),
    fallback_metrics_path: Path = Path("models/metrics.json"),
    alias: str = CHAMPION_ALIAS,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
) -> LoadedModelBundle:
    """Load the registered @champion model from MLflow, or fall back to models/model.joblib.

    Args:
        tracking_db: Path to local MLflow SQLite database.
        fallback_model_path: Path to local joblib model artifact.
        fallback_metrics_path: Path to local metrics JSON report.
        alias: Model registry alias to load (default: 'champion').
        registered_model_name: Registered model name in MLflow.

    Returns:
        LoadedModelBundle with fitted scikit-learn pipeline and metadata.

    Raises:
        FileNotFoundError: If neither a registered MLflow model nor a local fallback exists.
    """
    if tracking_db.exists():
        tracking_uri = build_sqlite_tracking_uri(tracking_db)
        alias_info = get_alias_version_info(
            alias=alias,
            tracking_uri=tracking_uri,
            registered_model_name=registered_model_name,
        )
        if alias_info is not None:
            version_str, model_name_tag = alias_info
            pipeline = load_registered_sklearn_model(
                tracking_uri=tracking_uri,
                alias=alias,
                registered_model_name=registered_model_name,
            )
            logger.info(
                "Loaded serving model from MLflow Registry: %s@%s (v%s, %s)",
                registered_model_name,
                alias,
                version_str,
                model_name_tag,
            )
            return LoadedModelBundle(
                pipeline=pipeline,
                model_name=model_name_tag,
                model_version=version_str,
                model_alias=alias,
                model_source="mlflow_registry",
            )

    if fallback_model_path.exists():
        pipeline = joblib.load(fallback_model_path)
        model_name = "local_joblib_model"
        model_version = "local"
        if fallback_metrics_path.exists():
            try:
                payload = json.loads(fallback_metrics_path.read_text(encoding="utf-8"))
                model_name = str(payload.get("selected_model", model_name))
            except Exception as exc:
                logger.warning("Could not parse fallback metrics JSON: %s", exc)

        logger.info(
            "Loaded serving model from local fallback artifact: %s (%s)",
            fallback_model_path,
            model_name,
        )
        return LoadedModelBundle(
            pipeline=pipeline,
            model_name=model_name,
            model_version=model_version,
            model_alias="fallback",
            model_source="local_joblib",
        )

    raise FileNotFoundError(
        f"No registered model '@{alias}' found in {tracking_db} and no fallback artifact "
        f"found at {fallback_model_path}."
    )


def create_app(
    tracking_db: Path = Path("mlflow.db"),
    fallback_model_path: Path = Path("models/model.joblib"),
    fallback_metrics_path: Path = Path("models/metrics.json"),
    predictions_db: Path = Path("predictions.db"),
    alias: str = CHAMPION_ALIAS,
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
) -> FastAPI:
    """Create and configure the FastAPI model serving application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.tracking_db = tracking_db
        app.state.fallback_model_path = fallback_model_path
        app.state.fallback_metrics_path = fallback_metrics_path
        app.state.alias = alias
        app.state.registered_model_name = registered_model_name

        app.state.db_engine = create_prediction_engine(predictions_db)
        app.state.model_bundle = load_serving_model(
            tracking_db=tracking_db,
            fallback_model_path=fallback_model_path,
            fallback_metrics_path=fallback_metrics_path,
            alias=alias,
            registered_model_name=registered_model_name,
        )
        try:
            yield
        finally:
            app.state.db_engine.dispose()

    app = FastAPI(
        title="ML-Ops-Live NYC Green Taxi Duration Service",
        version="0.1.0",
        description=(
            "Real-time trip duration prediction service backed by MLflow @champion "
            "and SQLite prediction logging."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        bundle: LoadedModelBundle = request.app.state.model_bundle
        total_logged = count_logged_predictions(request.app.state.db_engine)
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=bundle.model_name,
            model_version=bundle.model_version,
            model_alias=bundle.model_alias,
            model_source=bundle.model_source,
            predictions_logged=total_logged,
        )

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info(request: Request) -> ModelInfoResponse:
        bundle: LoadedModelBundle = request.app.state.model_bundle
        return ModelInfoResponse(
            model_name=bundle.model_name,
            model_version=bundle.model_version,
            model_alias=bundle.model_alias,
            model_source=bundle.model_source,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
        )

    @app.post("/predict", response_model=TripPredictionResponse)
    def predict(payload: TripPredictionRequest, request: Request) -> TripPredictionResponse:
        t0 = time.perf_counter()
        bundle: LoadedModelBundle = request.app.state.model_bundle

        features_df = pd.DataFrame(
            [
                {
                    "PULocationID": int(payload.PULocationID),
                    "DOLocationID": int(payload.DOLocationID),
                    "trip_distance": float(payload.trip_distance),
                    "passenger_count": float(payload.passenger_count),
                    "pickup_hour": int(payload.pickup_hour),
                    "pickup_day_of_week": int(payload.pickup_day_of_week),
                }
            ]
        )[FEATURE_COLUMNS]

        raw_pred = float(bundle.pipeline.predict(features_df)[0])
        predicted_minutes = round(max(1.0, raw_pred), 4)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 4)

        request_id = str(uuid.uuid4())
        created_at_utc = datetime.now(UTC).isoformat()

        log_prediction_record(
            engine=request.app.state.db_engine,
            record={
                "request_id": request_id,
                "created_at_utc": created_at_utc,
                "model_name": bundle.model_name,
                "model_version": bundle.model_version,
                "model_alias": bundle.model_alias,
                "model_source": bundle.model_source,
                "PULocationID": int(payload.PULocationID),
                "DOLocationID": int(payload.DOLocationID),
                "trip_distance": float(payload.trip_distance),
                "passenger_count": float(payload.passenger_count),
                "pickup_hour": int(payload.pickup_hour),
                "pickup_day_of_week": int(payload.pickup_day_of_week),
                "predicted_duration_minutes": predicted_minutes,
                "latency_ms": latency_ms,
            },
        )

        return TripPredictionResponse(
            request_id=request_id,
            predicted_duration_minutes=predicted_minutes,
            model_name=bundle.model_name,
            model_version=bundle.model_version,
            model_alias=bundle.model_alias,
            model_source=bundle.model_source,
            latency_ms=latency_ms,
            created_at_utc=created_at_utc,
        )

    @app.post("/reload-model", response_model=ModelInfoResponse)
    def reload_model(request: Request) -> ModelInfoResponse:
        state = request.app.state
        bundle = load_serving_model(
            tracking_db=state.tracking_db,
            fallback_model_path=state.fallback_model_path,
            fallback_metrics_path=state.fallback_metrics_path,
            alias=state.alias,
            registered_model_name=state.registered_model_name,
        )
        state.model_bundle = bundle
        return ModelInfoResponse(
            model_name=bundle.model_name,
            model_version=bundle.model_version,
            model_alias=bundle.model_alias,
            model_source=bundle.model_source,
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
        )

    return app
