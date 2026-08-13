import numpy as np
import pytest

from mival.ops import BandFilter, Crop, Normalize, OpChain, Pad, ReconstructLeads, Resample, ScaleUnit, SelectLeads
from mival.signal import LEADS_12, Signal


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


def make_named(leads, n_samples=1000, fs=500.0, unit="mV"):
    t = np.arange(n_samples, dtype=np.float32) / fs
    rows = [np.sin(2 * np.pi * (1.0 + i) * t) for i in range(len(leads))]
    return Signal(
        data=np.stack(rows).astype(np.float32),
        leads=tuple(leads),
        sampling_rate_hz=fs,
        unit=unit,
    )


def test_scale_unit_uv_to_mv_divides_by_1000():
    sig = make_named(["I"], unit="uV")
    out = ScaleUnit("uV", "mV").apply(sig)
    assert out.unit == "mV"
    np.testing.assert_allclose(out.data, sig.data / 1000.0, rtol=1e-6)


def test_scale_unit_same_unit_is_identity():
    sig = make_named(["I"], unit="mV")
    out = ScaleUnit("mV", "mV").apply(sig)
    np.testing.assert_array_equal(out.data, sig.data)


def test_scale_unit_rejects_unknown_unit():
    with pytest.raises(ValueError, match="unsupported unit"):
        ScaleUnit("V", "mV").apply(make_named(["I"], unit="V"))


def test_select_leads_reorders_and_subsets():
    sig = make_named(LEADS_12)
    order = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
    out = SelectLeads(order).apply(sig)
    assert out.leads == order
    np.testing.assert_array_equal(out.data[2], sig.data[LEADS_12.index("V1")])


def test_select_leads_rejects_missing_lead():
    with pytest.raises(KeyError, match="V6"):
        SelectLeads(("I", "V6")).apply(make_named(["I", "II"]))


def test_reconstruct_derives_lead_iii_from_i_and_ii():
    sig = make_named(["I", "II"])
    out = ReconstructLeads(("I", "II", "III")).apply(sig)
    assert out.leads == ("I", "II", "III")
    np.testing.assert_allclose(out.data[2], sig.data[1] - sig.data[0], rtol=1e-6)


def test_reconstruct_derives_avr():
    sig = make_named(["I", "II"])
    out = ReconstructLeads(("aVR",)).apply(sig)
    np.testing.assert_allclose(
        out.data[0], -0.5 * sig.data[0] - 0.5 * sig.data[1], rtol=1e-6
    )


def test_reconstruct_rejects_underivable_lead():
    with pytest.raises(KeyError, match="V3"):
        ReconstructLeads(("V3",)).apply(make_named(["I", "II"]))


def test_normalize_global_zscore_uses_whole_array():
    sig = make_named(["I", "II"])
    out = Normalize("global_zscore").apply(sig)
    assert out.data.mean() == pytest.approx(0.0, abs=1e-5)
    assert out.data.std() == pytest.approx(1.0, abs=1e-5)


def test_normalize_per_lead_zscore_normalizes_each_row():
    sig = make_named(["I", "II"])
    out = Normalize("per_lead_zscore").apply(sig)
    for row in out.data:
        assert row.mean() == pytest.approx(0.0, abs=1e-5)
        assert row.std() == pytest.approx(1.0, abs=1e-5)


def test_normalize_none_is_identity():
    sig = make_named(["I"])
    np.testing.assert_array_equal(Normalize("none").apply(sig).data, sig.data)


def test_normalize_handles_constant_lead_without_nan():
    sig = Signal(
        data=np.ones((1, 100), dtype=np.float32),
        leads=("I",),
        sampling_rate_hz=500.0,
        unit="mV",
    )
    out = Normalize("per_lead_zscore").apply(sig)
    assert np.isfinite(out.data).all()


def test_highpass_filter_removes_dc_offset():
    sig = make_named(["I"], n_samples=5000)
    offset = Signal(
        data=sig.data + 5.0,
        leads=sig.leads,
        sampling_rate_hz=sig.sampling_rate_hz,
        unit=sig.unit,
    )
    out = BandFilter("highpass", 0.5).apply(offset)
    assert abs(float(out.data.mean())) < 0.05
