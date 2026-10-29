"""SQLite persistence layer for FastAPI prediction and application logs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import Engine, Float, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    """Declarative base for serving persistence models."""


class PredictionLog(Base):
    """Persistent record of a single model prediction request and response."""

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    created_at_utc: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_alias: Mapped[str] = mapped_column(String(64), nullable=False)
    model_source: Mapped[str] = mapped_column(String(64), nullable=False)

    PULocationID: Mapped[int] = mapped_column(Integer, nullable=False)
    DOLocationID: Mapped[int] = mapped_column(Integer, nullable=False)
    trip_distance: Mapped[float] = mapped_column(Float, nullable=False)
    passenger_count: Mapped[float] = mapped_column(Float, nullable=False)
    pickup_hour: Mapped[int] = mapped_column(Integer, nullable=False)
    pickup_day_of_week: Mapped[int] = mapped_column(Integer, nullable=False)

    predicted_duration_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)


def create_prediction_engine(db_path: Path = Path("predictions.db")) -> Engine:
    """Create a SQLite SQLAlchemy Engine and initialize the prediction_logs table."""
    resolved = db_path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{resolved.as_posix()}"
    engine = create_engine(uri, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


def log_prediction_record(engine: Engine, record: dict[str, Any]) -> None:
    """Persist a prediction log record to the SQLite database."""
    with Session(engine) as session, session.begin():
        entry = PredictionLog(**record)
        session.add(entry)


def count_logged_predictions(engine: Engine) -> int:
    """Return total number of prediction records stored in the database."""
    with Session(engine) as session:
        stmt = select(func.count()).select_from(PredictionLog)
        count = session.scalar(stmt)
        return int(count or 0)


def fetch_recent_predictions(engine: Engine, limit: int = 100) -> list[dict[str, Any]]:
    """Fetch the most recent prediction log records ordered by id descending."""
    with Session(engine) as session:
        stmt = select(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit)
        rows = session.scalars(stmt).all()
        return [
            {
                "id": r.id,
                "request_id": r.request_id,
                "created_at_utc": r.created_at_utc,
                "model_name": r.model_name,
                "model_version": r.model_version,
                "model_alias": r.model_alias,
                "model_source": r.model_source,
                "PULocationID": r.PULocationID,
                "DOLocationID": r.DOLocationID,
                "trip_distance": r.trip_distance,
                "passenger_count": r.passenger_count,
                "pickup_hour": r.pickup_hour,
                "pickup_day_of_week": r.pickup_day_of_week,
                "predicted_duration_minutes": r.predicted_duration_minutes,
                "latency_ms": r.latency_ms,
            }
            for r in rows
        ]

