"""Regression evaluation metrics and baseline-vs-candidate comparison utilities."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


class RegressionMetrics(NamedTuple):
    """Container for regression evaluation metrics on a dataset split."""

    mae: float
    rmse: float
    r2: float
    num_samples: int

    def to_dict(self) -> dict[str, float | int]:
        """Return metrics as a JSON-serializable dictionary."""
        return {
            "mae": round(float(self.mae), 6),
            "rmse": round(float(self.rmse), 6),
            "r2": round(float(self.r2), 6),
            "num_samples": int(self.num_samples),
        }


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> RegressionMetrics:
    """Compute MAE, RMSE, and R2 for regression predictions.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predicted values.

    Returns:
        RegressionMetrics with MAE, RMSE, R2, and sample count.

    Raises:
        ValueError: If inputs are empty, differ in length, or contain NaN/infinite values.
    """
    true_arr = np.asarray(y_true, dtype=np.float64)
    pred_arr = np.asarray(y_pred, dtype=np.float64)

    if true_arr.ndim != 1 or pred_arr.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional arrays.")
    if len(true_arr) == 0:
        raise ValueError("Cannot evaluate metrics on an empty array.")
    if len(true_arr) != len(pred_arr):
        raise ValueError(
            f"Length mismatch between y_true ({len(true_arr)}) and y_pred ({len(pred_arr)})."
        )
    if not np.isfinite(true_arr).all() or not np.isfinite(pred_arr).all():
        raise ValueError("y_true and y_pred must contain only finite numeric values.")

    mae = float(mean_absolute_error(true_arr, pred_arr))
    rmse = float(root_mean_squared_error(true_arr, pred_arr))
    r2 = float(r2_score(true_arr, pred_arr))

    return RegressionMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        num_samples=len(true_arr),
    )


def compare_models(
    baseline: RegressionMetrics,
    candidate: RegressionMetrics,
) -> dict[str, float | bool]:
    """Compare candidate validation metrics against a baseline model.

    A candidate is considered strictly better if it lowers validation MAE and does not
    degrade validation RMSE.

    Args:
        baseline: Validation metrics of the baseline model.
        candidate: Validation metrics of the candidate model.

    Returns:
        Dictionary of absolute/relative improvements and a boolean promotion flag.
    """
    mae_delta = baseline.mae - candidate.mae
    rmse_delta = baseline.rmse - candidate.rmse
    r2_delta = candidate.r2 - baseline.r2

    mae_pct = (mae_delta / baseline.mae * 100.0) if baseline.mae > 0 else 0.0
    rmse_pct = (rmse_delta / baseline.rmse * 100.0) if baseline.rmse > 0 else 0.0

    candidate_is_better = bool(candidate.mae < baseline.mae and candidate.rmse <= baseline.rmse)

    return {
        "mae_reduction_minutes": round(float(mae_delta), 6),
        "mae_reduction_pct": round(float(mae_pct), 4),
        "rmse_reduction_minutes": round(float(rmse_delta), 6),
        "rmse_reduction_pct": round(float(rmse_pct), 4),
        "r2_gain": round(float(r2_delta), 6),
        "candidate_is_better": candidate_is_better,
    }
