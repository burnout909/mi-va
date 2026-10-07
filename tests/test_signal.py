import numpy as np
import pytest

from mival.signal import LEADS_12, Signal, SourceMetadata


def test_signal_reports_shape_and_duration():
    sig = Signal(
        data=np.zeros((12, 5000), dtype=np.float32),
        leads=LEADS_12,
        sampling_rate_hz=500.0,
        unit="mV",
    )
    assert sig.n_leads == 12
    assert sig.n_samples == 5000
    assert sig.duration_s == pytest.approx(10.0)


def test_signal_rejects_lead_count_mismatch():
    with pytest.raises(ValueError, match="lead count"):
        Signal(
            data=np.zeros((3, 100), dtype=np.float32),
            leads=("I", "II"),
            sampling_rate_hz=500.0,
            unit="mV",
        )


def test_signal_rejects_non_2d_data():
    with pytest.raises(ValueError, match="2-D"):
        Signal(
            data=np.zeros((2, 3, 4), dtype=np.float32),
            leads=("I", "II"),
            sampling_rate_hz=500.0,
            unit="mV",
        )


def test_signal_rejects_non_float32_data():
    with pytest.raises(ValueError, match="float32"):
        Signal(
            data=np.zeros((2, 100), dtype=np.float64),
            leads=("I", "II"),
            sampling_rate_hz=500.0,
            unit="mV",
        )


def test_source_metadata_allows_missing_unit():
    meta = SourceMetadata(
        leads=LEADS_12, sampling_rate_hz=500.0, n_samples=5000, unit=None
    )
    assert meta.unit is None
    assert meta.duration_s == pytest.approx(10.0)


def test_leads_12_is_standard_order():
    assert LEADS_12[:6] == ("I", "II", "III", "aVR", "aVL", "aVF")
    assert LEADS_12[6:] == ("V1", "V2", "V3", "V4", "V5", "V6")
