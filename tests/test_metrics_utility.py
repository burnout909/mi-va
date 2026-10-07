"""Category 3 (spec §4.5): net benefit against its two reference strategies."""

import pytest

from mival.metrics import utility

# Two events ranked above two non-events.
Y = [1, 1, 0, 0]
P = [0.9, 0.8, 0.4, 0.1]


def test_net_benefit_at_a_hand_computed_point():
    # At p_t = 0.5 the weight is 1: two true positives, no false positives.
    assert utility.net_benefit(Y, P, 0.5) == pytest.approx(2 / 4 - 0.0)


def test_net_benefit_charges_false_positives_at_the_threshold_odds():
    # At p_t = 0.2 the weight is 0.25 and 0.4 is now flagged: tp = 2, fp = 1.
    assert utility.net_benefit(Y, P, 0.2) == pytest.approx(2 / 4 - (1 / 4) * 0.25)


def test_treat_all_is_prevalence_minus_weighted_false_positives():
    assert utility.net_benefit_treat_all(Y, P, 0.2) == pytest.approx(0.5 - 0.5 * 0.25)


def test_treat_all_has_zero_net_benefit_when_the_threshold_equals_prevalence():
    # A known fixed point: at p_t = prevalence, treating everyone is exactly
    # as good as treating nobody.
    y = [1, 0, 0, 0]
    assert utility.net_benefit_treat_all(y, [0.5] * 4, 0.25) == pytest.approx(0.0)


def test_a_model_that_flags_everyone_equals_treat_all():
    # Below the lowest score every patient is flagged, so the model *is* the
    # treat-all strategy. This anchors the low end of the decision curve.
    threshold = 0.05
    assert utility.net_benefit(Y, P, threshold) == pytest.approx(
        utility.net_benefit_treat_all(Y, P, threshold)
    )


def test_a_model_that_flags_nobody_equals_treat_none():
    assert utility.net_benefit(Y, P, 0.95) == pytest.approx(0.0)
    assert utility.net_benefit_treat_none(Y, P, 0.95) == 0.0


def test_decision_curve_reports_both_reference_strategies_per_threshold():
    rows = utility.decision_curve(Y, P, [0.05, 0.2, 0.5])
    assert [row["threshold"] for row in rows] == [0.05, 0.2, 0.5]
    assert rows[0]["net_benefit"] == pytest.approx(rows[0]["treat_all"])
    assert all(row["treat_none"] == 0.0 for row in rows)


def test_utility_metric_names_carry_the_threshold_as_a_parameter():
    values = utility.utility_metrics(Y, P, [0.01, 0.1])
    assert set(values) == {
        "net_benefit@0.01",
        "net_benefit_treat_all@0.01",
        "delta_net_benefit_vs_treat_all@0.01",
        "net_benefit@0.1",
        "net_benefit_treat_all@0.1",
        "delta_net_benefit_vs_treat_all@0.1",
    }
    assert values["delta_net_benefit_vs_treat_all@0.1"] == pytest.approx(
        values["net_benefit@0.1"] - values["net_benefit_treat_all@0.1"]
    )


def test_the_default_band_is_the_low_thresholds_the_spec_names_for_stemi():
    assert utility.DEFAULT_DECISION_THRESHOLDS == (0.01, 0.02, 0.05, 0.10)


def test_a_threshold_outside_the_open_unit_interval_is_rejected():
    # p_t = 1 implies an infinite exchange rate, and p_t = 0 implies that no
    # false positive costs anything; neither is a decision.
    for threshold in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            utility.net_benefit(Y, P, threshold)
