"""Evaluation metrics for capacity-factor forecasts.

All metrics operate on plain numpy arrays. The reference forecast for skill is
same-hour-yesterday persistence (the `history_capacity_factor` feature), per
the project invariant: no model result is meaningful without that comparison.

Note: the feature builder drops NaN rows, and kt is NaN at night, so every
matrix — and therefore every metric here — covers daylight hours only.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def skill(y_true: np.ndarray, y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    """1 - MAE(model) / MAE(reference).

    > 0 means the model beats the reference; persistence itself scores 0.
    MAE (not RMSE) as the base so one bad outage hour can't dominate the ratio.
    """
    return 1.0 - mae(y_true, y_pred) / mae(y_true, y_ref)
