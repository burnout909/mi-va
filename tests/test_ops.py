import numpy as np
import pytest

from mival.ops import Crop, OpChain, Pad, Resample
from mival.signal import Signal


def make_signal(n_leads=2, n_samples=1000, fs=500.0, unit="mV"):
    t = np.arange(n_samples, dtype=np.float32) / fs
    rows = [np.sin(2 * np.pi * (1.0 + i) * t) for i in range(n_leads)]
    return Signal(
        data=np.stack(rows).astype(np.float32),
        leads=tuple(f"L{i}" for i in range(n_leads)),
        sampling_rate_hz=fs,
        unit=unit,
    )


def test_resample_halves_sample_count():
    out = Resample(250.0).apply(make_signal(n_samples=1000, fs=500.0))
    assert out.sampling_rate_hz == 250.0
    assert out.n_samples == 500
    assert out.data.dtype == np.float32


def test_resample_is_identity_at_same_rate():
    sig = make_signal()
    out = Resample(500.0).apply(sig)
    np.testing.assert_array_equal(out.data, sig.data)


def test_resample_preserves_lead_names():
    out = Resample(125.0).apply(make_signal(n_leads=3))
    assert out.leads == ("L0", "L1", "L2")


def test_resample_4096_to_500_is_exact():
    # 500/4096 reduces to 125/1024; a denominator cap of 1000 would have
    # silently approximated this to 99/811 and mislabelled the result.
    sig = make_signal(n_leads=2, n_samples=8192, fs=4096.0)
    out = Resample(500.0).apply(sig)
    assert out.sampling_rate_hz == 500.0
    assert out.n_samples == 1000
    assert out.data.dtype == np.float32


def test_resample_rejects_ratio_beyond_the_polyphase_limit():
    sig = make_signal(n_leads=1, n_samples=1000, fs=333.33)
    with pytest.raises(ValueError, match="polyphase factor limit"):
        Resample(500.0).apply(sig)


def test_crop_from_start_keeps_leading_samples():
    sig = make_signal(n_samples=1000)
    out = Crop(400).apply(sig)
    assert out.n_samples == 400
    np.testing.assert_array_equal(out.data, sig.data[:, :400])


def test_crop_center_is_symmetric():
    sig = make_signal(n_samples=1000)
    out = Crop(400, anchor="center").apply(sig)
    np.testing.assert_array_equal(out.data, sig.data[:, 300:700])


def test_crop_rejects_longer_than_source():
    with pytest.raises(ValueError, match="shorter"):
        Crop(2000).apply(make_signal(n_samples=1000))


def test_pad_zero_appends_at_end():
    sig = make_signal(n_samples=100)
    out = Pad(150).apply(sig)
    assert out.n_samples == 150
    np.testing.assert_array_equal(out.data[:, :100], sig.data)
    assert np.all(out.data[:, 100:] == 0.0)


def test_pad_rejects_shorter_than_source():
    with pytest.raises(ValueError, match="longer"):
        Pad(50).apply(make_signal(n_samples=100))


def test_opchain_applies_in_order_and_describes_itself():
    chain = OpChain((Resample(250.0), Crop(300)))
    out = chain.apply(make_signal(n_samples=1000, fs=500.0))
    assert out.n_samples == 300
    assert out.sampling_rate_hz == 250.0
    assert chain.describe() == [
        {"name": "resample", "params": {"target_hz": 250.0}},
        {"name": "crop", "params": {"n_samples": 300, "anchor": "start"}},
    ]
