"""Category 2 (spec §4.5): the Murphy identity, IPA, and Van Calster's levels."""

import math

import numpy as np
import pytest

from mival.metrics import calibration as calib


def test_brier_is_the_mean_squared_error():
    # (1 - 0.8)^2 = 0.04 and (0 - 0.3)^2 = 0.09
    assert calib.brier([1, 0], [0.8, 0.3]) == pytest.approx(0.065)


def test_scaled_brier_is_zero_for_the_prevalence_model():
    # The IPA baseline is "predict the observed prevalence for everyone".
    # Spec §4.5 requires it beside the Brier score precisely because at low
    # prevalence the raw Brier is dominated by the uncertainty term.
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    prevalence = sum(y) / len(y)
    assert calib.brier_scaled(y, [prevalence] * len(y)) == pytest.approx(0.0)


def test_scaled_brier_is_one_for_a_perfect_model():
    assert calib.brier_scaled([1, 0, 1, 0], [1.0, 0.0, 1.0, 0.0]) == pytest.approx(1.0)


def test_scaled_brier_is_negative_when_worse_than_prevalence():
    assert calib.brier_scaled([1, 0, 0, 0], [0.0, 1.0, 1.0, 1.0]) < 0.0


def test_murphy_decomposition_matches_a_hand_computed_two_bin_example():
    # Two forecast values, 0.8 (3 records, 2 events) and 0.2 (3 records, 1
    # event); overall event rate 0.5.
    y = [1, 1, 0, 0, 1, 0]
    p = [0.8, 0.8, 0.8, 0.2, 0.2, 0.2]
    parts = calib.murphy_decomposition(y, p)
    assert parts["reliability"] == pytest.approx(0.5 * (0.8 - 2 / 3) ** 2 + 0.5 * (0.2 - 1 / 3) ** 2)
    assert parts["resolution"] == pytest.approx(0.5 * (2 / 3 - 0.5) ** 2 + 0.5 * (1 / 3 - 0.5) ** 2)
    assert parts["uncertainty"] == pytest.approx(0.25)
    assert parts["brier"] == pytest.approx(0.24)


def test_the_murphy_identity_holds_exactly_on_unique_forecast_values():
    # Brier = reliability - resolution + uncertainty (spec §4.5). This is the
    # reason Brier sits under calibration and not in an "overall performance"
    # layer of its own.
    rng = np.random.default_rng(4)
    p = np.round(rng.random(400), 2)
    y = (rng.random(400) < p).astype(float)
    parts = calib.murphy_decomposition(y, p, bins=None)
    identity = parts["reliability"] - parts["resolution"] + parts["uncertainty"]
    assert identity == pytest.approx(parts["brier"], abs=1e-12)


def test_the_murphy_identity_holds_for_the_binned_forecast():
    rng = np.random.default_rng(5)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(float)
    parts = calib.murphy_decomposition(y, p, bins=10)
    identity = parts["reliability"] - parts["resolution"] + parts["uncertainty"]
    assert identity == pytest.approx(parts["brier_binned"], abs=1e-12)
    # Binning discards within-bin forecast variance, so the two Briers differ
    # a little; the point of returning both is that the gap is visible.
    assert parts["brier_binned"] == pytest.approx(parts["brier"], abs=0.01)
    assert parts["n_bins"] == 10


def test_binning_never_produces_an_empty_group():
    # Heavily tied scores collapse quantile edges; the weights must still sum
    # to one rather than yielding NaN for an empty bin.
    y = [1, 0, 1, 0]
    p = [0.5, 0.5, 0.5, 0.5]
    parts = calib.murphy_decomposition(y, p, bins=10)
    assert parts["n_bins"] == 1
    assert not math.isnan(parts["reliability"])


def test_calibration_in_the_large_is_observed_minus_predicted():
    assert calib.calibration_in_the_large([1, 0, 0, 0], [0.2, 0.2, 0.2, 0.2]) == pytest.approx(0.05)


def test_logistic_recalibration_solves_its_score_equations():
    # The MLE is defined by sum(y - phat) = 0 and sum(logit(p) * (y - phat)) = 0.
    # Checking the defining equations is exact, unlike comparing to a fitted
    # number from another library.
    rng = np.random.default_rng(7)
    x = rng.normal(size=800)
    p = 1.0 / (1.0 + np.exp(-(0.4 + 0.8 * x)))
    y = (rng.random(800) < p).astype(float)
    intercept, slope = calib.logistic_recalibration(y, p)
    fitted = 1.0 / (1.0 + np.exp(-(intercept + slope * np.log(p / (1 - p)))))
    residual = y - fitted
    assert float(np.sum(residual)) == pytest.approx(0.0, abs=1e-6)
    assert float(np.sum(np.log(p / (1 - p)) * residual)) == pytest.approx(0.0, abs=1e-6)


def test_a_well_calibrated_model_has_intercept_zero_and_slope_one():
    rng = np.random.default_rng(8)
    p = rng.random(20000) * 0.9 + 0.05
    y = (rng.random(20000) < p).astype(float)
    intercept, slope = calib.logistic_recalibration(y, p)
    assert intercept == pytest.approx(0.0, abs=0.1)
    assert slope == pytest.approx(1.0, abs=0.1)


def test_an_overconfident_model_has_a_slope_below_one():
    # True risk follows logit(x); the model reports logit(2x), i.e. it pushes
    # every prediction towards 0 and 1. The recalibration slope must halve.
    rng = np.random.default_rng(9)
    x = rng.normal(size=20000)
    truth = 1.0 / (1.0 + np.exp(-x))
    reported = 1.0 / (1.0 + np.exp(-2.0 * x))
    y = (rng.random(20000) < truth).astype(float)
    _, slope = calib.logistic_recalibration(y, reported)
    assert slope == pytest.approx(0.5, abs=0.05)


def test_separable_data_gives_nan_rather_than_a_fabricated_slope():
    assert math.isnan(calib.logistic_recalibration([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])[1])
    assert math.isnan(calib.logistic_recalibration([1, 1], [0.4, 0.6])[1])


def test_flexible_curve_of_a_calibrated_model_follows_the_diagonal():
    rng = np.random.default_rng(11)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(float)
    predicted, observed = calib.flexible_calibration_curve(y, p)
    assert predicted.size > 0
    assert float(np.max(np.abs(predicted - observed))) < 0.1


def test_eci_grows_when_predictions_are_shifted_off_the_diagonal():
    rng = np.random.default_rng(12)
    p = rng.random(4000) * 0.5
    y = (rng.random(4000) < p).astype(float)
    calibrated = calib.eci(y, p)
    shifted = calib.eci(y, np.clip(p + 0.2, 0.0, 1.0))
    assert calibrated < 1.0
    assert shifted > calibrated


def test_calibration_metrics_returns_exactly_the_declared_names():
    rng = np.random.default_rng(13)
    p = rng.random(200)
    y = (rng.random(200) < p).astype(float)
    values = calib.calibration_metrics(y, p)
    assert tuple(sorted(values)) == tuple(sorted(calib.METRICS))
