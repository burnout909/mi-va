"""Category 1 against values worked out by hand (spec §4.5)."""

import math

import pytest

pytest.importorskip("sklearn")

from mival.metrics import discrimination as disc

# y, p and the 2x2 table at threshold 0.5:
#   0.9 -> positive, label 1  (tp)
#   0.4 -> negative, label 1  (fn)
#   0.6 -> positive, label 0  (fp)
#   0.1 -> negative, label 0  (tn)
#   0.5 -> positive, label 1  (tp; the threshold itself counts as positive)
HAND_Y = [1, 1, 0, 0, 1]
HAND_P = [0.9, 0.4, 0.6, 0.1, 0.5]


def test_auroc_is_one_for_perfect_separation():
    assert disc.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_auroc_is_zero_for_perfectly_reversed_ranking():
    assert disc.auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_auroc_is_one_half_when_every_score_is_tied():
    # Every pair is a tie, and a tie contributes half a concordant pair.
    assert disc.auroc([0, 0, 1, 1], [0.3, 0.3, 0.3, 0.3]) == 0.5


def test_auroc_counts_concordant_pairs():
    # Four (negative, positive) pairs, of which three are concordant.
    assert disc.auroc([0, 0, 1, 1], [0.1, 0.4, 0.2, 0.9]) == pytest.approx(0.75)


def test_auroc_is_nan_when_a_slice_has_one_class():
    # A rare-event subgroup can legitimately be one-class inside a bootstrap
    # replicate; that must not raise.
    assert math.isnan(disc.auroc([0, 0, 0], [0.1, 0.2, 0.3]))
    assert math.isnan(disc.auprc([1, 1], [0.1, 0.2]))


def test_auprc_matches_the_average_precision_computed_by_hand():
    # Ranked 0.9(1), 0.5(0), 0.1(1): precision at each positive is 1 and 2/3,
    # each contributing a recall step of 1/2.
    assert disc.auprc([1, 0, 1], [0.9, 0.5, 0.1]) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))


def test_confusion_counts_treat_the_threshold_as_positive():
    assert disc.confusion_counts(HAND_Y, HAND_P, 0.5) == (2, 1, 1, 1)


def test_contingency_metrics_match_the_hand_computed_table():
    values = disc.contingency_metrics(*disc.confusion_counts(HAND_Y, HAND_P, 0.5))
    assert values["sensitivity"] == pytest.approx(2 / 3)
    assert values["specificity"] == pytest.approx(1 / 2)
    assert values["ppv"] == pytest.approx(2 / 3)
    assert values["npv"] == pytest.approx(1 / 2)
    assert values["f1"] == pytest.approx(2 / 3)
    # (2*1 - 1*1) / sqrt(3 * 3 * 2 * 2)
    assert values["mcc"] == pytest.approx(1 / 6)
    assert values["balanced_accuracy"] == pytest.approx(7 / 12)


def test_contingency_metrics_agree_with_scikit_learn():
    from sklearn import metrics as skm

    y = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
    p = [0.91, 0.12, 0.44, 0.77, 0.63, 0.05, 0.58, 0.31, 0.49, 0.22]
    predicted = [int(value >= 0.5) for value in p]
    values = disc.contingency_metrics(*disc.confusion_counts(y, p, 0.5))
    assert values["sensitivity"] == pytest.approx(skm.recall_score(y, predicted))
    assert values["ppv"] == pytest.approx(skm.precision_score(y, predicted))
    assert values["f1"] == pytest.approx(skm.f1_score(y, predicted))
    assert values["mcc"] == pytest.approx(skm.matthews_corrcoef(y, predicted))
    assert values["balanced_accuracy"] == pytest.approx(skm.balanced_accuracy_score(y, predicted))


def test_undefined_ratios_are_nan_rather_than_zero():
    # Nothing is predicted positive, so PPV has no denominator. Reporting 0.0
    # would let a bootstrap average it in as if it were a measurement.
    values = disc.contingency_metrics(*disc.confusion_counts(HAND_Y, HAND_P, 1.1))
    assert math.isnan(values["ppv"])
    assert values["sensitivity"] == 0.0


def test_discrimination_slope_is_the_gap_between_mean_risks():
    assert disc.discrimination_slope([1, 1, 0, 0], [0.8, 0.6, 0.3, 0.1]) == pytest.approx(0.5)


def test_discrimination_metrics_returns_exactly_the_declared_names():
    values = disc.discrimination_metrics(HAND_Y, HAND_P, 0.5)
    assert tuple(sorted(values)) == tuple(sorted(disc.METRICS))


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        disc.auroc([0, 1, 2], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        disc.auroc([0, 1], [0.1, 1.5])
    with pytest.raises(ValueError):
        disc.auroc([0, 1], [0.1])
    with pytest.raises(ValueError):
        disc.auroc([], [])
