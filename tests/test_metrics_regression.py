import numpy as np
import pytest

pytest.importorskip("sklearn")

from mival.metrics import category_of
from mival.metrics.regression import auroc_below, mae, r2, regression_metrics, rmse


def test_point_metrics():
    y = np.array([40.0, 50.0, 60.0])
    p = np.array([42.0, 50.0, 56.0])
    assert mae(y, p) == pytest.approx(2.0)
    assert rmse(y, p) == pytest.approx(np.sqrt(20 / 3))
    assert r2(y, p) == pytest.approx(1 - 20 / 200)


def test_auroc_below_scores_low_predictions_as_positive():
    y = np.array([30.0, 35.0, 55.0, 60.0])
    p = np.array([32.0, 45.0, 50.0, 58.0])
    assert auroc_below(y, p, 40.0) == pytest.approx(1.0)


def test_regression_metrics_are_registered():
    values = regression_metrics(np.array([30.0, 60.0]), np.array([35.0, 55.0]), cuts=(40.0,))
    assert set(values) == {"mae", "rmse", "r2", "auroc_below@40"}
    assert category_of("auroc_below@40") == "regression"
