"""The attribution baseline contract (spec §4.4 ``attribute``, §4.6 overlay).

No backend: the baseline is the design decision, and it is decided in NumPy.
The property that makes an Integrated Gradients implementation checkable —
completeness — is pinned in ``test_torch_attribution.py``, which must stay
named ``test_torch_*`` so it collects after the "no backend has been imported"
guards elsewhere in the suite.
"""

import numpy as np
import pytest

from mival.adapters._attribution import (
    DEFAULT_STEPS,
    ZERO_BASELINE,
    AttributionError,
    completeness_gap,
    integration_points,
    resolve_baseline,
)

BATCH = np.zeros((3, 8, 20), dtype=np.float64)


# ---------------------------------------------------------------------------
# The baseline contract
# ---------------------------------------------------------------------------


def test_a_missing_baseline_is_refused_and_the_message_says_why():
    with pytest.raises(AttributionError, match="asystole"):
        resolve_baseline(None, BATCH)


def test_zeros_is_available_but_must_be_asked_for_by_name():
    assert resolve_baseline(ZERO_BASELINE, BATCH).shape == BATCH.shape
    assert not resolve_baseline(ZERO_BASELINE, BATCH).any()


def test_an_unknown_named_baseline_is_refused():
    with pytest.raises(AttributionError, match="only named baseline"):
        resolve_baseline("median", BATCH)


def test_one_value_per_lead_broadcasts_over_the_batch():
    per_lead = np.arange(8, dtype=np.float64)
    resolved = resolve_baseline(per_lead, BATCH)
    assert resolved.shape == BATCH.shape
    assert np.allclose(resolved[0, :, 0], per_lead)
    assert np.allclose(resolved[0, 3, :], 3.0)


def test_one_record_broadcasts_over_the_batch():
    record = np.random.default_rng(0).normal(size=(8, 20))
    resolved = resolve_baseline(record, BATCH)
    assert np.allclose(resolved[2], record)


def test_a_whole_batch_baseline_is_taken_as_given():
    other = np.random.default_rng(1).normal(size=BATCH.shape)
    assert np.allclose(resolve_baseline(other, BATCH), other)


def test_a_baseline_of_the_wrong_shape_names_the_shapes_that_would_fit():
    with pytest.raises(AttributionError, match=r"one value per lead"):
        resolve_baseline(np.zeros((5, 5)), BATCH)


def test_the_integration_points_are_midpoints_and_never_reach_the_baseline():
    points = integration_points(4)
    assert np.allclose(points, [0.125, 0.375, 0.625, 0.875])
    assert points.min() > 0.0 and points.max() < 1.0


def test_at_least_one_step_is_required():
    with pytest.raises(AttributionError, match="at least 1"):
        integration_points(0)


def test_completeness_gap_is_zero_for_an_exact_attribution():
    attribution = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    assert np.allclose(completeness_gap(attribution, np.array([10.0]), np.array([0.0])), 0.0)
