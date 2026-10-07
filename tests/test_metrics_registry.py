"""The metric registry: every metric knows its category (spec §4.5)."""

import pytest

pytest.importorskip("sklearn")

import mival.metrics as metrics
from mival.stages.evaluate import CATEGORIES

Y = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
P = [0.9, 0.1, 0.7, 0.3, 0.6, 0.4, 0.2, 0.5, 0.8, 0.05]


def test_compute_metrics_covers_all_three_computed_categories():
    values = metrics.compute_metrics(Y, P, threshold=0.5)
    found = {metrics.category_of(name) for name in values}
    assert found == {metrics.DISCRIMINATION, metrics.CALIBRATION, metrics.CLINICAL_UTILITY}


def test_every_produced_metric_has_a_category():
    values = metrics.compute_metrics(Y, P, threshold=0.5)
    for name in values:
        assert metrics.category_of(name) in CATEGORIES


def test_brier_is_classified_as_calibration_not_as_overall_performance():
    # Spec §4.5: there is no "overall performance" layer. Brier is a composite
    # whose reliability term is a calibration quantity, so it lives here and
    # its decomposition is reported beside it.
    assert metrics.category_of("brier") == metrics.CALIBRATION
    assert metrics.category_of("brier_scaled") == metrics.CALIBRATION
    assert metrics.category_of("brier_reliability") == metrics.CALIBRATION
    assert "overall_performance" not in CATEGORIES


def test_net_benefit_is_its_own_category():
    # Not rank-based, so not discrimination; not an agreement between
    # predicted and observed risk, so not calibration.
    assert metrics.category_of("net_benefit@0.05") == metrics.CLINICAL_UTILITY


def test_a_parameterized_metric_is_looked_up_by_its_stem():
    # A new decision threshold must not need registering anywhere.
    assert metrics.category_of("net_benefit@0.123") == metrics.CLINICAL_UTILITY


def test_the_interpretation_category_has_a_slot_and_no_implementation():
    assert metrics.INTERPRETATION in CATEGORIES
    assert metrics.INTERPRETATION_METRICS == ()


def test_an_unknown_metric_name_is_an_error():
    with pytest.raises(KeyError):
        metrics.category_of("not_a_metric")
