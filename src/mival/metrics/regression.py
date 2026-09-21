"""Regression metrics (category 'regression'): error, fit, and the derived classification."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ._common import expit
from .discrimination import auroc

METRICS = ("mae", "rmse", "r2")
METRIC_STEMS = ("auroc_below",)


def _arrays(y_true: Sequence, y_pred: Sequence):
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def mae(y_true: Sequence, y_pred: Sequence) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(p - y)))


def rmse(y_true: Sequence, y_pred: Sequence) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def r2(y_true: Sequence, y_pred: Sequence) -> float:
    y, p = _arrays(y_true, y_pred)
    total = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if total == 0 else 1.0 - float(np.sum((p - y) ** 2)) / total


def auroc_below(y_true: Sequence, y_pred: Sequence, cut: float) -> float:
    """AUROC for 'value <= cut', scoring by how far below the cut the prediction falls.

    ``discrimination.auroc`` requires its score in [0, 1] (it is meant for
    probabilities). ``cut - p`` is an unbounded score, so it is squashed
    through a sigmoid first; AUROC depends only on rank order, and a sigmoid
    is strictly monotonic, so the value is unchanged.
    """
    y, p = _arrays(y_true, y_pred)
    return auroc((y <= cut).astype(int), expit(cut - p))


def regression_metrics(y_true: Sequence, y_pred: Sequence, cuts: Sequence[float]) -> Dict[str, float]:
    values = {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred)}
    for cut in cuts:
        values[f"auroc_below@{cut:g}"] = auroc_below(y_true, y_pred, float(cut))
    return values
