"""The input contract refuses to guess anything that changes the samples."""

import pytest

from mival.contract import InputContract



# --------------------------------------------------------------------------
# Filter order is a model property, never a default
# --------------------------------------------------------------------------


def _contract_body(filters):
    return {
        "leads": ["I", "II"],
        "sampling_rate_hz": 500,
        "duration_s": 10,
        "unit": "mV",
        "scaling": "none",
        "layout": "lead_time",
        "dtype": "float32",
        "filters": filters,
    }


def test_a_filter_without_an_order_is_refused():
    """Same argument as `unit`: it changes the samples and cannot be detected later."""
    with pytest.raises(ValueError, match="declares no 'order'"):
        InputContract.from_dict(_contract_body([{"kind": "highpass", "cutoff_hz": 0.5}]))


def test_a_filter_with_an_order_is_accepted():
    contract = InputContract.from_dict(
        _contract_body([{"kind": "highpass", "cutoff_hz": 0.5, "order": 2}])
    )
    assert contract.filters[0]["order"] == 2


def test_a_nonsensical_filter_order_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        InputContract.from_dict(
            _contract_body([{"kind": "lowpass", "cutoff_hz": 40, "order": 0}])
        )
