"""Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) drift monitoring."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy import Boolean, Engine, Float, Integer, String, Text, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from mlops_live.data.clean import FEATURE_COLUMNS, TARGET_COLUMN
from mlops_live.serving.database import Base, create_prediction_engine, fetch_recent_predictions

logger = logging.getLogger(__name__)

CATEGORICAL_DRIFT_COLUMNS = {"PULocationID", "DOLocationID", "pickup_hour", "pickup_day_of_week"}
EPSILON = 1e-4


class DriftReportRecord(Base):
    """SQLite table storing historical data and prediction drift evaluation runs."""

    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at_utc: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    reference_source: Mapped[str] = mapped_column(String(256), nullable=False)
    current_source: Mapped[str] = mapped_column(String(256), nullable=False)
    reference_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    current_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    max_psi: Mapped[float] = mapped_column(Float, nullable=False)
    max_ks_stat: Mapped[float] = mapped_column(Float, nullable=False)
    overall_drift_detected: Mapped[bool] = mapped_column(Boolean, nullable=False)
    retraining_recommended: Mapped[bool] = mapped_column(Boolean, nullable=False)
    drifted_columns_csv: Mapped[str] = mapped_column(String(512), nullable=False)
    report_json: Mapped[str] = mapped_column(Text, nullable=False)


class ColumnDriftMetrics(NamedTuple):
    """Drift metrics for a single feature or prediction column."""

    column: str
    is_categorical: bool
    psi: float
    psi_severity: str
    ks_statistic: float
    ks_p_value: float
    reference_mean: float
    current_mean: float
    drift_detected: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize column drift metrics to a dictionary."""
        return {
            "column": self.column,
            "is_categorical": self.is_categorical,
            "psi": round(float(self.psi), 6),
            "psi_severity": self.psi_severity,
            "ks_statistic": round(float(self.ks_statistic), 6),
            "ks_p_value": round(float(self.ks_p_value), 6),
            "reference_mean": round(float(self.reference_mean), 4),
            "current_mean": round(float(self.current_mean), 4),
            "drift_detected": bool(self.drift_detected),
        }


class DriftReport(NamedTuple):
    """Complete data and prediction drift evaluation report."""

    created_at_utc: str
    reference_source: str
    current_source: str
    reference_rows: int
    current_rows: int
    psi_threshold: float
    ks_threshold: float
    ks_alpha: float
    feature_drift: dict[str, ColumnDriftMetrics]
    prediction_drift: ColumnDriftMetrics | None
    drifted_columns: list[str]
    overall_drift_detected: bool
    retraining_recommended: bool

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable drift report dictionary."""
        return {
            "created_at_utc": self.created_at_utc,
            "reference_source": self.reference_source,
            "current_source": self.current_source,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "thresholds": {
                "psi_threshold": self.psi_threshold,
                "ks_threshold": self.ks_threshold,
                "ks_alpha": self.ks_alpha,
            },
            "overall_drift_detected": self.overall_drift_detected,
            "retraining_recommended": self.retraining_recommended,
            "drifted_columns": self.drifted_columns,
            "feature_drift": {k: v.to_dict() for k, v in self.feature_drift.items()},
            "prediction_drift": (
                self.prediction_drift.to_dict() if self.prediction_drift is not None else None
            ),
        }


def classify_psi_severity(psi_value: float, significant_threshold: float = 0.25) -> str:
    """Classify PSI value into 'none', 'moderate', or 'significant'."""
    if psi_value >= significant_threshold:
        return "significant"
    if psi_value >= 0.10:
        return "moderate"
    return "none"


def compute_psi(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    num_bins: int = 10,
    is_categorical: bool = False,
    max_categories: int = 25,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current arrays.

    Args:
        reference: Reference baseline values.
        current: Current window values.
        num_bins: Number of quantile bins for continuous variables.
        is_categorical: Whether to bucket by discrete categories instead of quantiles.
        max_categories: Maximum top reference categories to track before grouping into OTHER.

    Returns:
        Non-negative PSI float value.
    """
    ref_arr = np.asarray(reference, dtype=np.float64)
    cur_arr = np.asarray(current, dtype=np.float64)

    if len(ref_arr) == 0 or len(cur_arr) == 0:
        raise ValueError("Reference and current arrays must be non-empty to compute PSI.")
    if not np.isfinite(ref_arr).all() or not np.isfinite(cur_arr).all():
        raise ValueError("Reference and current arrays must contain only finite values.")

    if is_categorical:
        ref_series = pd.Series(ref_arr)
        cur_series = pd.Series(cur_arr)
        top_cats = set(ref_series.value_counts().head(max_categories).index.tolist())

        ref_mapped = ref_series.where(ref_series.isin(top_cats), other=-1.0)
        cur_mapped = cur_series.where(cur_series.isin(top_cats), other=-1.0)

        categories = sorted(set(ref_mapped.unique()).union(set(cur_mapped.unique())))
        ref_counts = np.array([(ref_mapped == c).sum() for c in categories], dtype=np.float64)
        cur_counts = np.array([(cur_mapped == c).sum() for c in categories], dtype=np.float64)
    else:
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)
        edges = np.unique(np.quantile(ref_arr, quantiles))
        if len(edges) <= 1:
            # Constant reference array: use a small window around the constant value
            val = float(edges[0])
            bin_edges = np.array([-np.inf, val - 1e-6, val + 1e-6, np.inf])
        else:
            bin_edges = edges.copy()
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf

        ref_counts, _ = np.histogram(ref_arr, bins=bin_edges)
        cur_counts, _ = np.histogram(cur_arr, bins=bin_edges)

    ref_props = np.clip(ref_counts / len(ref_arr), EPSILON, None)
    cur_props = np.clip(cur_counts / len(cur_arr), EPSILON, None)
    ref_props = ref_props / ref_props.sum()
    cur_props = cur_props / cur_props.sum()

    psi = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    return max(0.0, psi)


def compute_ks_test(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
) -> tuple[float, float]:
    """Run two-sample Kolmogorov-Smirnov test and return (ks_statistic, p_value)."""
    ref_arr = np.asarray(reference, dtype=np.float64)
    cur_arr = np.asarray(current, dtype=np.float64)

    if len(ref_arr) == 0 or len(cur_arr) == 0:
        raise ValueError("Reference and current arrays must be non-empty for KS test.")

    res = ks_2samp(ref_arr, cur_arr)
    return float(res.statistic), float(res.pvalue)


def evaluate_column_drift(
    column: str,
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    is_categorical: bool = False,
    psi_threshold: float = 0.25,
    ks_threshold: float = 0.10,
    ks_alpha: float = 0.01,
) -> ColumnDriftMetrics:
    """Evaluate PSI and KS drift for a single column."""
    ref_arr = np.asarray(reference, dtype=np.float64)
    cur_arr = np.asarray(current, dtype=np.float64)

    psi_val = compute_psi(ref_arr, cur_arr, is_categorical=is_categorical)
    ks_stat, ks_pval = compute_ks_test(ref_arr, cur_arr)
    severity = classify_psi_severity(psi_val, significant_threshold=psi_threshold)

    psi_drifted = psi_val >= psi_threshold
    ks_drifted = (not is_categorical) and (ks_stat >= ks_threshold) and (ks_pval < ks_alpha)
    drift_detected = bool(psi_drifted or ks_drifted)

    return ColumnDriftMetrics(
        column=column,
        is_categorical=is_categorical,
        psi=psi_val,
        psi_severity=severity,
        ks_statistic=ks_stat,
        ks_p_value=ks_pval,
        reference_mean=float(np.mean(ref_arr)),
        current_mean=float(np.mean(cur_arr)),
        drift_detected=drift_detected,
    )


def evaluate_dataset_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    reference_source: str = "train.parquet",
    current_source: str = "current_window",
    reference_predictions: pd.Series | np.ndarray | None = None,
    current_predictions: pd.Series | np.ndarray | None = None,
    psi_threshold: float = 0.25,
    ks_threshold: float = 0.10,
    ks_alpha: float = 0.01,
) -> DriftReport:
    """Evaluate feature and prediction/target drift between reference and current datasets."""
    if reference_df.empty or current_df.empty:
        raise ValueError("Reference and current DataFrames must be non-empty.")

    for name, df_part in [("reference_df", reference_df), ("current_df", current_df)]:
        missing = [c for c in FEATURE_COLUMNS if c not in df_part.columns]
        if missing:
            raise ValueError(f"'{name}' is missing required feature columns: {missing}")

    feature_drift: dict[str, ColumnDriftMetrics] = {}
    drifted_columns: list[str] = []

    for col in FEATURE_COLUMNS:
        is_cat = col in CATEGORICAL_DRIFT_COLUMNS
        col_metrics = evaluate_column_drift(
            column=col,
            reference=reference_df[col],
            current=current_df[col],
            is_categorical=is_cat,
            psi_threshold=psi_threshold,
            ks_threshold=ks_threshold,
            ks_alpha=ks_alpha,
        )
        feature_drift[col] = col_metrics
        if col_metrics.drift_detected:
            drifted_columns.append(col)

    # Determine prediction / target series if provided or present in DataFrames
    ref_pred_arr: pd.Series | np.ndarray | None = reference_predictions
    cur_pred_arr: pd.Series | np.ndarray | None = current_predictions

    if ref_pred_arr is None and TARGET_COLUMN in reference_df.columns:
        ref_pred_arr = reference_df[TARGET_COLUMN]
    if cur_pred_arr is None:
        if "predicted_duration_minutes" in current_df.columns:
            cur_pred_arr = current_df["predicted_duration_minutes"]
        elif TARGET_COLUMN in current_df.columns:
            cur_pred_arr = current_df[TARGET_COLUMN]

    prediction_drift: ColumnDriftMetrics | None = None
    if ref_pred_arr is not None and cur_pred_arr is not None:
        prediction_drift = evaluate_column_drift(
            column="predicted_duration_minutes",
            reference=ref_pred_arr,
            current=cur_pred_arr,
            is_categorical=False,
            psi_threshold=psi_threshold,
            ks_threshold=ks_threshold,
            ks_alpha=ks_alpha,
        )
        if prediction_drift.drift_detected:
            drifted_columns.append("predicted_duration_minutes")

    overall_drift = len(drifted_columns) > 0
    # Recommend retraining if prediction drift is detected or >= 1 feature has significant drift
    retraining_recommended = overall_drift

    return DriftReport(
        created_at_utc=datetime.now(UTC).isoformat(),
        reference_source=reference_source,
        current_source=current_source,
        reference_rows=len(reference_df),
        current_rows=len(current_df),
        psi_threshold=psi_threshold,
        ks_threshold=ks_threshold,
        ks_alpha=ks_alpha,
        feature_drift=feature_drift,
        prediction_drift=prediction_drift,
        drifted_columns=drifted_columns,
        overall_drift_detected=overall_drift,
        retraining_recommended=retraining_recommended,
    )


def save_drift_report(
    report: DriftReport,
    output_json_path: Path = Path("reports/generated/drift_report.json"),
    db_engine: Engine | None = None,
) -> Path:
    """Save a DriftReport atomically to JSON and optionally log it to SQLite drift_reports."""
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload_str = json.dumps(report.to_dict(), indent=2)

    tmp_path = output_json_path.with_name(f"{output_json_path.name}.tmp")
    try:
        tmp_path.write_text(payload_str, encoding="utf-8")
        json.loads(tmp_path.read_text(encoding="utf-8"))
        tmp_path.replace(output_json_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    if db_engine is not None:
        Base.metadata.create_all(db_engine)
        all_cols = list(report.feature_drift.values())
        if report.prediction_drift is not None:
            all_cols.append(report.prediction_drift)
        max_psi = max((c.psi for c in all_cols), default=0.0)
        max_ks = max((c.ks_statistic for c in all_cols), default=0.0)

        with Session(db_engine) as session, session.begin():
            row = DriftReportRecord(
                created_at_utc=report.created_at_utc,
                reference_source=report.reference_source,
                current_source=report.current_source,
                reference_rows=report.reference_rows,
                current_rows=report.current_rows,
                max_psi=round(float(max_psi), 6),
                max_ks_stat=round(float(max_ks), 6),
                overall_drift_detected=report.overall_drift_detected,
                retraining_recommended=report.retraining_recommended,
                drifted_columns_csv=",".join(report.drifted_columns),
                report_json=payload_str,
            )
            session.add(row)

    return output_json_path


def fetch_recent_drift_reports(engine: Engine, limit: int = 20) -> list[dict[str, Any]]:
    """Fetch recent drift evaluation records from SQLite."""
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        stmt = select(DriftReportRecord).order_by(DriftReportRecord.id.desc()).limit(limit)
        rows = session.scalars(stmt).all()
        return [
            {
                "id": r.id,
                "created_at_utc": r.created_at_utc,
                "reference_source": r.reference_source,
                "current_source": r.current_source,
                "reference_rows": r.reference_rows,
                "current_rows": r.current_rows,
                "max_psi": r.max_psi,
                "max_ks_stat": r.max_ks_stat,
                "overall_drift_detected": r.overall_drift_detected,
                "retraining_recommended": r.retraining_recommended,
                "drifted_columns": (
                    r.drifted_columns_csv.split(",") if r.drifted_columns_csv else []
                ),
            }
            for r in rows
        ]


def load_current_window_from_sqlite(
    predictions_db: Path = Path("predictions.db"),
    limit: int = 5000,
) -> pd.DataFrame:
    """Load recent logged prediction requests from predictions.db as a DataFrame."""
    if not predictions_db.exists():
        raise FileNotFoundError(f"Predictions database does not exist: {predictions_db}")
    engine = create_prediction_engine(predictions_db)
    try:
        rows = fetch_recent_predictions(engine, limit=limit)
    finally:
        engine.dispose()

    if not rows:
        raise ValueError(f"No prediction records found in {predictions_db}.")
    return pd.DataFrame(rows)
