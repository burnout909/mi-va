import numpy as np
import pytest

from mival.compiler import compile_recipe
from mival.contract import CompileError, InputContract
from mival.signal import LEADS_12, Signal, SourceMetadata

PROPHECG = InputContract.from_dict(
    {
        "leads": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
        "sampling_rate_hz": 500,
        "duration_s": 10,
        "unit": "mV",
        "scaling": "none",
        "layout": "time_lead",
        "dtype": "float32",
    }
)

ECGFOUNDER_DICT = {
    "leads": list(LEADS_12),
    "sampling_rate_hz": 500,
    "duration_s": 10,
    "unit": "mV",
    "scaling": "global_zscore",
    "layout": "lead_time",
    "dtype": "float32",
}

ECGFOUNDER = InputContract.from_dict(ECGFOUNDER_DICT)


def mimic_source(unit="mV", fs=500.0, n=5000, leads=LEADS_12):
    return SourceMetadata(
        leads=tuple(leads), sampling_rate_hz=fs, n_samples=n, unit=unit
    )


def make_signal(meta):
    t = np.arange(meta.n_samples, dtype=np.float32) / meta.sampling_rate_hz
    rows = [np.sin(2 * np.pi * (1.0 + i) * t) for i in range(len(meta.leads))]
    return Signal(
        data=np.stack(rows).astype(np.float32),
        leads=meta.leads,
        sampling_rate_hz=meta.sampling_rate_hz,
        unit=meta.unit,
    )


def test_prophecg_chain_has_canonical_order():
    chain = compile_recipe(PROPHECG, mimic_source())
    assert [op["name"] for op in chain.describe()] == [
        "scale_unit",
        "select_leads",
        "normalize",
    ]


def test_crop_is_inserted_only_when_source_is_longer():
    chain = compile_recipe(PROPHECG, mimic_source(n=6000))
    assert [op["name"] for op in chain.describe()] == [
        "scale_unit",
        "select_leads",
        "crop",
        "normalize",
    ]


def test_prophecg_chain_produces_contract_shape():
    meta = mimic_source()
    out = compile_recipe(PROPHECG, meta).apply(make_signal(meta))
    assert out.data.shape == (8, 5000)
    assert out.leads == PROPHECG.leads
    assert out.sampling_rate_hz == 500.0


def test_ecgfounder_chain_produces_contract_shape():
    meta = mimic_source()
    out = compile_recipe(ECGFOUNDER, meta).apply(make_signal(meta))
    assert out.data.shape == (12, 5000)
    assert out.data.mean() == pytest.approx(0.0, abs=1e-5)


def test_resample_is_inserted_when_rates_differ():
    chain = compile_recipe(PROPHECG, mimic_source(fs=1000.0, n=10000))
    assert "resample" in [op["name"] for op in chain.describe()]


def test_reconstruct_is_used_when_limb_leads_are_derivable():
    source = mimic_source(leads=("I", "II", "V1", "V2", "V3", "V4", "V5", "V6"))
    chain = compile_recipe(ECGFOUNDER, source)
    assert "reconstruct_leads" in [op["name"] for op in chain.describe()]


def test_missing_unit_raises_unit_missing():
    with pytest.raises(CompileError) as excinfo:
        compile_recipe(PROPHECG, mimic_source(unit=None))
    assert excinfo.value.reason_code == "unit_missing"


def test_underivable_precordial_lead_raises_lead_unavailable():
    source = mimic_source(leads=("I", "II", "V1"))
    with pytest.raises(CompileError) as excinfo:
        compile_recipe(PROPHECG, source)
    assert excinfo.value.reason_code == "lead_unavailable"
    assert "V2" in excinfo.value.detail


def test_upsampling_is_rejected_by_default():
    with pytest.raises(CompileError) as excinfo:
        compile_recipe(PROPHECG, mimic_source(fs=250.0, n=2500))
    assert excinfo.value.reason_code == "upsample_required"


def test_upsampling_is_allowed_when_explicitly_enabled():
    chain = compile_recipe(
        PROPHECG, mimic_source(fs=250.0, n=2500), allow_upsample=True
    )
    assert "resample" in [op["name"] for op in chain.describe()]


def test_short_record_raises_duration_short():
    with pytest.raises(CompileError) as excinfo:
        compile_recipe(PROPHECG, mimic_source(n=2500))
    assert excinfo.value.reason_code == "duration_short"


def test_short_record_is_padded_when_policy_allows():
    chain = compile_recipe(PROPHECG, mimic_source(n=2500), pad_policy="zero")
    assert "pad" in [op["name"] for op in chain.describe()]


def test_unrecognised_unit_raises_unit_missing():
    with pytest.raises(CompileError) as excinfo:
        compile_recipe(PROPHECG, mimic_source(unit="V"))
    assert excinfo.value.reason_code == "unit_missing"
    assert "V" in excinfo.value.detail


def test_post_resample_length_matches_what_resample_actually_yields():
    # 9989 samples at 999 Hz is ~10.0 s. round() predicts 4999 and would
    # wrongly reject the record; resample_poly yields exactly 5000.
    meta = mimic_source(fs=999.0, n=9989)
    chain = compile_recipe(PROPHECG, meta)
    out = chain.apply(make_signal(meta))
    assert out.n_samples == 5000
    assert [op["name"] for op in chain.describe()] == [
        "scale_unit",
        "resample",
        "select_leads",
        "normalize",
    ]


def test_compiled_chain_length_prediction_agrees_with_application():
    for fs, n in ((1000.0, 10000), (999.0, 9989), (2000.0, 20000)):
        meta = mimic_source(fs=fs, n=n)
        out = compile_recipe(PROPHECG, meta).apply(make_signal(meta))
        assert out.n_samples == PROPHECG.n_samples, (fs, n, out.n_samples)


def test_unconvertible_rate_raises_rate_unsupported():
    with pytest.raises(CompileError) as excinfo:
        compile_recipe(PROPHECG, mimic_source(fs=1024.7, n=10247))
    assert excinfo.value.reason_code == "rate_unsupported"


def test_irrational_rate_raises_rate_unsupported():
    with pytest.raises(CompileError) as excinfo:
        compile_recipe(PROPHECG, mimic_source(fs=500.0000001, n=5000))
    assert excinfo.value.reason_code == "rate_unsupported"


def test_genuine_upsample_still_reports_upsample_required():
    # Ordering guard: a low-rate source must report the rate policy, not the
    # conversion-feasibility failure.
    with pytest.raises(CompileError) as excinfo:
        compile_recipe(PROPHECG, mimic_source(fs=250.0, n=2500))
    assert excinfo.value.reason_code == "upsample_required"


def test_unknown_pad_policy_is_rejected_at_compile_time():
    with pytest.raises(ValueError, match="pad_policy"):
        compile_recipe(PROPHECG, mimic_source(n=2500), pad_policy="constant")


def test_gain_is_applied_after_unit_conversion_and_before_normalize():
    contract = InputContract.from_dict({**ECGFOUNDER_DICT, "gain": 208.3333, "scaling": "none"})
    meta = mimic_source(unit="uV")
    chain = compile_recipe(contract, meta)
    names = [op["name"] for op in chain.describe()]
    assert names.index("scale_unit") < names.index("gain") < names.index("normalize")
    out = chain.apply(make_signal(meta))
    assert np.isclose(out.data[0, 0], make_signal(meta).data[0, 0] * 1e-3 * 208.3333, rtol=1e-5)
