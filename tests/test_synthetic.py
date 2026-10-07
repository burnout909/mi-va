import numpy as np

from mival.signal import LEADS_12
from mival.synthetic import synthetic_ecg


def test_same_index_produces_identical_signal():
    a = synthetic_ecg(3)
    b = synthetic_ecg(3)
    np.testing.assert_array_equal(a.data, b.data)


def test_different_index_produces_different_signal():
    assert not np.allclose(synthetic_ecg(0).data, synthetic_ecg(1).data)


def test_shape_and_metadata_match_mimic_style_source():
    sig = synthetic_ecg(0)
    assert sig.data.shape == (12, 5000)
    assert sig.leads == LEADS_12
    assert sig.sampling_rate_hz == 500.0
    assert sig.unit == "mV"


def test_values_are_finite_and_physiologically_scaled():
    data = synthetic_ecg(7).data
    assert np.isfinite(data).all()
    assert np.abs(data).max() < 10.0
