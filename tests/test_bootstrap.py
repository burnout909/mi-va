"""Resampling properties the spec depends on (spec §4.5).

The two that matter most are structural, not numeric: the resampling unit is
the person, and model comparison is paired rather than DeLong.
"""

import math

import numpy as np
import pytest

from mival.metrics.bootstrap import (
    Resampler,
    bootstrap_ci,
    bootstrap_distribution,
    event_strata,
    paired_bootstrap_difference,
    percentile_ci,
)


def clustered_data(n_persons=120, per_person=6, seed=3):
    """One person contributes several *identical* records.

    This is the extreme of what multiple ECGs per patient do: the records
    carry no information beyond the person, so a record-level bootstrap
    believes it has six times as much data as it does.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n_persons).astype(float)
    p = np.clip(0.5 + 0.25 * (2 * y - 1) + rng.normal(scale=0.1, size=n_persons), 0.01, 0.99)
    persons = np.arange(n_persons)
    return (
        np.repeat(y, per_person),
        np.repeat(p, per_person),
        np.repeat(persons, per_person),
    )


def mean_statistic(values):
    def statistic(indices):
        return {"mean": float(np.mean(values[indices]))}

    return statistic


def test_person_level_intervals_are_wider_than_record_level_ones():
    # Spec §4.5: "재표집 단위는 반드시 person이다." Resampling records treats
    # correlated recordings as independent evidence and understates the CI.
    y, p, persons = clustered_data()
    statistic = mean_statistic(p)
    by_record = bootstrap_ci(statistic, n_rows=y.size, n_replicates=400, seed=1)["mean"]
    by_person = bootstrap_ci(
        statistic, n_rows=y.size, n_replicates=400, seed=1, groups=persons
    )["mean"]
    record_width = by_record.ci_hi - by_record.ci_lo
    person_width = by_person.ci_hi - by_person.ci_lo
    assert person_width > record_width
    # With six identical copies per person the record-level standard error is
    # too small by roughly sqrt(6); the interval must widen by about as much.
    assert person_width / record_width > 2.0


def test_the_same_seed_reproduces_the_same_interval():
    y, p, persons = clustered_data()
    statistic = mean_statistic(p)
    first = bootstrap_ci(statistic, n_rows=y.size, n_replicates=200, seed=17, groups=persons)
    second = bootstrap_ci(statistic, n_rows=y.size, n_replicates=200, seed=17, groups=persons)
    assert first["mean"] == second["mean"]


def test_a_different_seed_gives_a_different_draw():
    y, p, persons = clustered_data()
    statistic = mean_statistic(p)
    first = bootstrap_ci(statistic, n_rows=y.size, n_replicates=200, seed=17, groups=persons)
    second = bootstrap_ci(statistic, n_rows=y.size, n_replicates=200, seed=18, groups=persons)
    assert first["mean"].ci_lo != second["mean"].ci_lo


def test_the_global_numpy_generator_is_left_alone():
    # A result that moves because something else drew a random number is not
    # reproducible, so the module must use its own Generator.
    np.random.seed(0)
    expected = np.random.random()
    np.random.seed(0)
    y, p, persons = clustered_data()
    bootstrap_ci(mean_statistic(p), n_rows=y.size, n_replicates=50, seed=2, groups=persons)
    assert np.random.random() == expected


def test_the_point_estimate_is_the_full_sample_value_not_the_replicate_mean():
    y, p, persons = clustered_data()
    result = bootstrap_ci(
        mean_statistic(p), n_rows=y.size, n_replicates=100, seed=5, groups=persons
    )
    assert result["mean"].value == pytest.approx(float(np.mean(p)))


def test_stratification_holds_the_number_of_events_constant():
    # Without stratification a rare-event resample can contain no events at
    # all, and every rank-based metric of that replicate is undefined.
    y, p, persons = clustered_data()
    strata = event_strata(y, persons)

    def count_events(indices):
        return {"events": float(np.sum(y[indices] == 1.0))}

    draws = bootstrap_distribution(
        count_events, n_rows=y.size, n_replicates=100, seed=6, groups=persons, strata=strata
    )["events"]
    assert np.all(draws == float(np.sum(y == 1.0)))


def test_event_strata_are_defined_at_the_level_of_the_resampling_unit():
    # A person with any event belongs to the event stratum, including the
    # person whose other recordings are negative.
    y = np.array([0.0, 1.0, 0.0, 0.0])
    persons = np.array(["a", "a", "b", "b"])
    assert list(event_strata(y, persons)) == [1, 1, 0, 0]


def test_a_unit_may_not_span_two_strata():
    with pytest.raises(ValueError, match="stratum"):
        Resampler(4, groups=["a", "a", "b", "b"], strata=[0, 1, 0, 0])


def test_percentile_ci_ignores_undefined_replicates():
    lo, hi, n_valid = percentile_ci([0.1, float("nan"), 0.3, 0.5])
    assert n_valid == 3
    assert lo == pytest.approx(0.11, abs=0.01)
    assert hi == pytest.approx(0.49, abs=0.01)


def test_percentile_ci_is_nan_when_almost_everything_is_undefined():
    lo, hi, n_valid = percentile_ci([float("nan"), 0.2])
    assert n_valid == 1
    assert math.isnan(lo) and math.isnan(hi)


def test_a_paired_comparison_of_a_model_with_itself_is_exactly_zero():
    # Pairing is what removes the shared between-person variance; comparing an
    # arm to itself must therefore leave nothing at all.
    y, p, persons = clustered_data()

    def statistic(y_values, p_values):
        return {"mean": float(np.mean(p_values))}

    result = paired_bootstrap_difference(
        statistic, y, p, p, n_replicates=100, seed=8, groups=persons
    )["mean"]
    assert result.value == 0.0
    assert result.ci_lo == 0.0 and result.ci_hi == 0.0


def test_a_paired_comparison_recovers_a_known_constant_difference():
    y, p, persons = clustered_data()

    def statistic(y_values, p_values):
        return {"mean": float(np.mean(p_values))}

    other = p - 0.005
    result = paired_bootstrap_difference(
        statistic, y, p, other, n_replicates=200, seed=9, groups=persons
    )["mean"]
    assert result.value == pytest.approx(0.005, abs=1e-12)
    assert result.ci_lo <= result.value <= result.ci_hi
    assert result.p_value is not None and result.p_value < 0.05


def test_pairing_is_tighter_than_comparing_two_independent_intervals():
    # The point of the paired bootstrap: two arms scored on the same patients
    # share their between-patient variance, and the difference does not carry
    # it. An unpaired comparison would.
    rng = np.random.default_rng(21)
    persons = np.arange(300)
    shared = rng.normal(size=300)
    a = 1.0 / (1.0 + np.exp(-(shared + 0.2)))
    b = 1.0 / (1.0 + np.exp(-shared))
    y = (rng.random(300) < 0.3).astype(float)

    def statistic(y_values, p_values):
        return {"mean": float(np.mean(p_values))}

    paired = paired_bootstrap_difference(
        statistic, y, a, b, n_replicates=300, seed=4, groups=persons
    )["mean"]
    lone_a = bootstrap_ci(
        lambda idx: {"mean": float(np.mean(a[idx]))},
        n_rows=300,
        n_replicates=300,
        seed=4,
        groups=persons,
    )["mean"]
    paired_width = paired.ci_hi - paired.ci_lo
    marginal_width = lone_a.ci_hi - lone_a.ci_lo
    assert paired_width < marginal_width


def test_bootstrapping_zero_rows_is_an_error_rather_than_an_empty_answer():
    with pytest.raises(ValueError):
        Resampler(0)
