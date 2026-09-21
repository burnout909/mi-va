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


def test_auroc_below_does_not_saturate_far_from_the_cut():
    # A sigmoid squash clips at +-40, which tied 85 and 95 against a cut of 40
    # and lost the rank order AUROC is made of.
    from sklearn.metrics import roc_auc_score

    y = np.array([30.0, 50.0, 60.0, 70.0])
    p = np.array([5.0, 45.0, 85.0, 95.0])
    assert auroc_below(y, p, 40.0) == pytest.approx(roc_auc_score(y <= 40.0, 40.0 - p))


def test_auroc_below_keeps_far_apart_predictions_distinct():
    from sklearn.metrics import roc_auc_score

    y = np.array([30.0, 50.0])
    p = np.array([95.0, 100.0])
    assert auroc_below(y, p, 40.0) == pytest.approx(roc_auc_score(y <= 40.0, 40.0 - p))
    assert auroc_below(y, p, 40.0) == pytest.approx(1.0)


def test_auroc_below_ranks_nothing_when_every_prediction_is_equal():
    # A constant score has no rank order; auroc reserves NaN for a one-class
    # label vector, so this is 0.5 and not NaN.
    y = np.array([30.0, 50.0])
    p = np.array([60.0, 60.0])
    assert auroc_below(y, p, 40.0) == pytest.approx(0.5)


def test_regression_metrics_are_registered():
    values = regression_metrics(np.array([30.0, 60.0]), np.array([35.0, 55.0]), cuts=(40.0,))
    assert set(values) == {"mae", "rmse", "r2", "auroc_below@40"}
    assert category_of("auroc_below@40") == "regression"
