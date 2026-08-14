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

ECGFOUNDER = InputContract.from_dict(
    {
        "leads": list(LEADS_12),
        "sampling_rate_hz": 500,
        "duration_s": 10,
        "unit": "mV",
        "scaling": "global_zscore",
        "layout": "lead_time",
        "dtype": "float32",
    }
)


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
