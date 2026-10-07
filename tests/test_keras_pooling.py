import numpy as np
import pytest

from mival.adapters.keras_adapter import pool_ensemble


def member_outputs():
    # Deliberately asymmetric: every member differs, and the negative and
    # positive columns differ, so a wrong axis or a wrong column shows up.
    return [
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        np.array([[0.7, 0.3], [0.4, 0.6]]),
        np.array([[0.5, 0.5], [0.6, 0.4]]),
    ]


def test_mean_probability_averages_members_then_selects_positive_column():
    probs = pool_ensemble(member_outputs(), "mean_probability", 1)
    # column 1 per member: [0.1, 0.8], [0.3, 0.6], [0.5, 0.4]
    np.testing.assert_allclose(probs, [0.3, 0.6], rtol=1e-12)


def test_positive_index_selects_the_declared_column():
    probs = pool_ensemble(member_outputs(), "mean_probability", 0)
    np.testing.assert_allclose(probs, [0.7, 0.4], rtol=1e-12)


def test_none_method_uses_the_first_member_only():
    probs = pool_ensemble(member_outputs(), "none", 1)
    np.testing.assert_allclose(probs, [0.1, 0.8], rtol=1e-12)


def test_unsupported_method_is_rejected():
    with pytest.raises(ValueError, match="unsupported ensemble method"):
        pool_ensemble(member_outputs(), "median_probability", 1)
