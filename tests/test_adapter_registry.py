import numpy as np
import pytest

from mival.adapters import available_adapters, get_adapter
from mival.adapters.base import to_model_layout


def test_lead_time_layout_is_identity():
    batch = np.zeros((2, 12, 5000), dtype=np.float32)
    assert to_model_layout(batch, "lead_time").shape == (2, 12, 5000)


def test_time_lead_layout_transposes_last_two_axes():
    batch = np.zeros((2, 8, 5000), dtype=np.float32)
    assert to_model_layout(batch, "time_lead").shape == (2, 5000, 8)


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match="unknown layout"):
        to_model_layout(np.zeros((1, 2, 3), dtype=np.float32), "channels_middle")


def test_registry_lists_both_backends():
    assert set(available_adapters()) == {"torch", "keras"}


def test_unknown_adapter_name_is_rejected():
    with pytest.raises(KeyError, match="no adapter named"):
        get_adapter("jax")
