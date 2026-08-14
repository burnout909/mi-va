"""Threshold policies (spec §4.4).

Every expected value below is computed by hand from the vectors in the test,
not from the implementation. The ROC table for ``PROBS``/``LABELS`` is:

    threshold  tp fp  sensitivity  specificity  Youden J
    0.90        1  0         0.20         1.00      0.20
    0.80        2  0         0.40         1.00      0.40
    0.70        2  1         0.40         0.80      0.20
    0.60        3  1         0.60         0.80      0.40
    0.50        4  1         0.80         0.80      0.60   <- max J
    0.40        4  2         0.80         0.60      0.40
    0.30        4  3         0.80         0.40      0.20
    0.20        4  4         0.80         0.20      0.00
    0.10        5  4         1.00         0.20      0.20   <- first sens >= 0.95
    0.05        5  5         1.00         0.00      0.00
"""

import numpy as np
import pytest

from mival.threshold import (
    DEV_SPLIT,
    LEGACY,
    REFIT_SENS95,
    REFIT_YOUDEN,
    SENS95_TARGET_SENSITIVITY,
    SplitScores,
    ThresholdPolicyError,
    auroc,
    fit_thresholds,
    legacy_threshold,
    operating_point,
    policy_names,
    refit_sens95,
    refit_youden,
    roc_curve,
)

PROBS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
LABELS = [1, 1, 0, 1, 1, 0, 0, 0, 1, 0]

CARD_THRESHOLD = {"value": 0.0768, "provenance": "archived notebook", "status": "legacy"}


def dev(probs=PROBS, labels=LABELS):
    return SplitScores.of(DEV_SPLIT, probs, labels)


def test_policy_names_are_the_three_of_spec_4_4():
    assert policy_names() == (LEGACY, REFIT_YOUDEN, REFIT_SENS95)


def test_the_module_and_the_stage_agree_on_the_policy_vocabulary():
    # The stage pins the contract; this module implements it. They are two
    # files because a threshold module that imported a stage would be the
    # cross-stage coupling spec §3.1 forbids.
    from mival.stages.models import PRIMARY_THRESHOLD_POLICY, THRESHOLD_POLICIES

    assert set(THRESHOLD_POLICIES) == set(policy_names())
    assert PRIMARY_THRESHOLD_POLICY == REFIT_SENS95


def test_roc_curve_matches_the_hand_computed_table():
    curve = roc_curve(PROBS, LABELS)
    assert list(curve.thresholds) == PROBS
    assert curve.sensitivity == pytest.approx(
        [0.2, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0]
    )
    assert curve.specificity == pytest.approx(
        [1.0, 1.0, 0.8, 0.8, 0.8, 0.6, 0.4, 0.2, 0.2, 0.0]
    )


def test_ties_in_probability_collapse_to_one_operating_point():
    # Splitting records that carry the same probability would invent a
    # threshold no data supports.
    curve = roc_curve([0.7, 0.7, 0.2, 0.2], [1, 0, 1, 0])
    assert list(curve.thresholds) == [0.7, 0.2]


def test_refit_youden_takes_the_maximum_of_j():
    fit = refit_youden(dev())
    assert fit.value == pytest.approx(0.5)
    assert fit.sensitivity == pytest.approx(0.8)
    assert fit.specificity == pytest.approx(0.8)
    assert fit.youden_j == pytest.approx(0.6)
    assert fit.fitted_on == DEV_SPLIT
    assert (fit.n, fit.n_events) == (10, 5)


def test_refit_youden_breaks_ties_toward_the_more_sensitive_threshold():
    # Labels chosen so that J = 0.6 at both 0.5 and 0.3; the deterministic
    # rule keeps the smaller threshold, which misses fewer STEMIs.
    labels = [1, 1, 0, 1, 1, 0, 1, 0, 0, 0]
    curve = roc_curve(PROBS, labels)
    assert curve.youden_j[PROBS.index(0.5)] == pytest.approx(0.6)
    assert curve.youden_j[PROBS.index(0.3)] == pytest.approx(0.6)
    assert refit_youden(dev(PROBS, labels)).value == pytest.approx(0.3)


def test_refit_sens95_takes_the_most_specific_threshold_that_reaches_the_floor():
    fit = refit_sens95(dev())
    assert fit.value == pytest.approx(0.1)
    assert fit.sensitivity == pytest.approx(1.0)
    assert fit.specificity == pytest.approx(0.2)
    assert fit.fitted_on == DEV_SPLIT


def test_refit_sens95_is_deterministic_when_several_thresholds_reach_the_floor():
    # 0.10 and 0.05 both give sensitivity 1.0; 0.10 is more specific.
    curve = roc_curve(PROBS, LABELS)
    reaching = [t for t, s in zip(curve.thresholds, curve.sensitivity) if s >= 0.95]
    assert reaching == [0.1, 0.05]
    assert refit_sens95(dev()).value == pytest.approx(0.1)


def test_sens95_floor_is_the_documented_constant():
    assert SENS95_TARGET_SENSITIVITY == 0.95
    # The floor is a parameter, so a stricter study-level floor is expressible
    # without editing the policy.
    # The first row of the table reaching sensitivity 0.5 is threshold 0.60.
    assert refit_sens95(dev(), target_sensitivity=0.5).value == pytest.approx(0.6)


def test_legacy_reads_the_value_off_the_card_and_reports_where_it_lands():
    fit = legacy_threshold(CARD_THRESHOLD, "toy", dev())
    assert fit.value == pytest.approx(0.0768)
    assert fit.fitted_on == "model_card"
    assert fit.provenance == "archived notebook"
    # 0.0768 calls everything but the 0.05 record positive: 5/5 events caught,
    # 4 of 5 non-events called.
    assert fit.sensitivity == pytest.approx(1.0)
    assert fit.specificity == pytest.approx(0.2)


def test_legacy_without_a_card_value_is_an_error_not_an_invented_number():
    with pytest.raises(ThresholdPolicyError, match="no x-mival.threshold.value"):
        legacy_threshold({}, "toy", dev())


def test_a_card_without_a_legacy_value_still_gets_the_two_refit_policies():
    fits = fit_thresholds(policy_names(), dev(), {}, "toy")
    assert sorted(fits) == [REFIT_SENS95, REFIT_YOUDEN]


def test_fit_thresholds_returns_all_three_when_the_card_carries_one():
    fits = fit_thresholds(policy_names(), dev(), CARD_THRESHOLD, "toy")
    assert sorted(fits) == sorted(policy_names())
    assert fits[REFIT_SENS95].value == pytest.approx(0.1)
    assert fits[REFIT_YOUDEN].value == pytest.approx(0.5)
    assert fits[LEGACY].value == pytest.approx(0.0768)


@pytest.mark.parametrize("policy", [REFIT_YOUDEN, REFIT_SENS95])
def test_a_threshold_is_never_refit_on_the_test_split(policy):
    scores = SplitScores.of("test", PROBS, LABELS)
    with pytest.raises(ThresholdPolicyError, match="never selected on the test set"):
        fit_thresholds([policy], scores, CARD_THRESHOLD, "toy")


def test_the_split_guard_also_rejects_a_pooled_split():
    with pytest.raises(ThresholdPolicyError, match="'dev'"):
        refit_sens95(SplitScores.of("dev+test", PROBS, LABELS))


def test_unknown_policy_is_rejected():
    with pytest.raises(ThresholdPolicyError, match="unknown threshold policy"):
        fit_thresholds(["refit_f1"], dev(), CARD_THRESHOLD, "toy")


def test_auroc_matches_the_hand_counted_concordance():
    # positives {0.9, 0.8, 0.6, 0.5, 0.1} vs negatives {0.7, 0.4, 0.3, 0.2, 0.05}
    # concordant pairs: 5 + 5 + 4 + 4 + 1 = 19 of 25.
    assert auroc(PROBS, LABELS) == pytest.approx(19 / 25)


def test_auroc_gives_tied_scores_half_credit():
    assert auroc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)
    assert auroc([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0]) == pytest.approx(0.875)


def test_a_single_class_cannot_produce_a_threshold_or_an_auroc():
    with pytest.raises(ThresholdPolicyError, match="both classes"):
        refit_youden(dev([0.1, 0.9], [1, 1]))
    with pytest.raises(ThresholdPolicyError, match="undefined"):
        auroc([0.1, 0.9], [0, 0])


def test_malformed_scores_are_rejected_rather_than_silently_coerced():
    with pytest.raises(ThresholdPolicyError, match="differ in length"):
        SplitScores.of(DEV_SPLIT, [0.1, 0.2], [1])
    with pytest.raises(ThresholdPolicyError, match="NaN or infinity"):
        SplitScores.of(DEV_SPLIT, [0.1, np.nan], [1, 0])
    with pytest.raises(ThresholdPolicyError, match=r"\[0, 1\]"):
        SplitScores.of(DEV_SPLIT, [0.1, 1.4], [1, 0])
    with pytest.raises(ThresholdPolicyError, match="binary"):
        SplitScores.of(DEV_SPLIT, [0.1, 0.2], [1, 2])
    with pytest.raises(ThresholdPolicyError, match="binary"):
        SplitScores.of(DEV_SPLIT, [0.1, 0.2], [1, None])


def test_operating_point_of_a_fixed_threshold():
    sensitivity, specificity = operating_point(PROBS, LABELS, 0.55)
    assert sensitivity == pytest.approx(0.6)
    assert specificity == pytest.approx(0.8)
