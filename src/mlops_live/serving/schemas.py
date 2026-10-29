"""Pydantic v2 request and response schemas for the FastAPI model serving API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TripPredictionRequest(BaseModel):
    """Input payload for predicting NYC Green Taxi trip duration (minutes)."""

    model_config = ConfigDict(extra="forbid")

    PULocationID: int = Field(..., gt=0, description="TLC pickup taxi zone ID (> 0)")
    DOLocationID: int = Field(..., gt=0, description="TLC dropoff taxi zone ID (> 0)")
    trip_distance: float = Field(
        ..., ge=0.01, le=100.0, description="Trip distance in miles (0.01 to 100.0)"
    )
    passenger_count: float = Field(
        default=1.0, ge=1.0, le=9.0, description="Number of passengers (1.0 to 9.0)"
    )
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of pickup (0 to 23)")
    pickup_day_of_week: int = Field(
        ..., ge=0, le=6, description="Day of week of pickup (0=Monday to 6=Sunday)"
    )


class TripPredictionResponse(BaseModel):
    """Response payload returned by POST /predict."""

    request_id: str
    predicted_duration_minutes: float
    model_name: str
    model_version: str
    model_alias: str
    model_source: str
    latency_ms: float
    created_at_utc: str


class ModelInfoResponse(BaseModel):
    """Response payload returned by GET /model-info."""

    model_name: str
    model_version: str
    model_alias: str
    model_source: str
    feature_columns: list[str]
    target_column: str


class HealthResponse(BaseModel):
    """Response payload returned by GET /health."""

    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    model_alias: str
    model_source: str
    predictions_logged: int
