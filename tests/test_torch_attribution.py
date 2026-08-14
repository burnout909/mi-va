"""Integrated Gradients in the torch adapter (spec §4.4, §4.6).

The property under test is **completeness**: ``sum(attribution) == F(x) −
F(baseline)``. It is the defining property of the method and the only thing
that separates a correct implementation from a plausible-looking picture — a
saliency map is not falsifiable by looking at it.

The baseline contract itself is tested backend-free in ``test_attribution.py``.
"""

import json

import numpy as np
import pytest

from mival.adapters._attribution import (
    DEFAULT_STEPS,
    AttributionError,
    completeness_gap,
)

# torch is imported inside these helpers, never at module scope: several tests
# elsewhere assert that no backend has leaked into sys.modules, and a module
# level import here would break them for the whole session.
pytestmark = pytest.mark.torch

from mival.adapters._training import LinearHead  # noqa: E402
from mival.adapters.torch_adapter import TorchAdapter, TorchHandle  # noqa: E402
from mival.modelcard import load_card  # noqa: E402


def make_module(transpose_input=False):
    """A small differentiable stand-in: (logits, features) from a real signal."""
    import torch

    class Net(torch.nn.Module):
        def __init__(self, n_leads=4, width=6):
            super().__init__()
            torch.manual_seed(0)
            self.body = torch.nn.Sequential(
                torch.nn.Conv1d(n_leads, width, kernel_size=3, padding=1),
                torch.nn.Tanh(),
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
            )
            self.dense = torch.nn.Linear(width, 2)

        def forward(self, x):
            if transpose_input:
                x = x.transpose(1, 2)
            features = self.body(x)
            return self.dense(features), features

    return Net().double().eval()


def card(tmp_path, layout="lead_time", positive_index=1, output_type="softmax"):
    body = {
        "model_id": "synthetic",
        "Name": "Synthetic",
        "x-mival": {
            "adapter": "torch",
            "weights": [{"uri": "unused", "sha256": "0" * 64, "role": "backbone"}],
            "ensemble": {"method": "none", "members": 1},
            "input_contract": {
                "leads": ["I", "II", "III", "aVR"],
                "sampling_rate_hz": 500,
                "duration_s": 10,
                "unit": "mV",
                "filters": [],
                "scaling": "none",
                "layout": layout,
                "dtype": "float64",
            },
            "output": {
                "type": output_type,
                "n_outputs": 2,
                "positive_index": positive_index,
            },
            "feature_layer": "body",
            "threshold": {},
            "runtime": {},
            "training_modes_supported": ["inference_only", "linear_probe"],
            "pretraining_corpora": [],
        },
    }
    path = tmp_path / f"synthetic-{layout}-{positive_index}-{output_type}.json"
    path.write_text(json.dumps(body))
    return load_card(path)


def handle_of(tmp_path, transpose_input=False, **kwargs):
    return TorchHandle(
        module=make_module(transpose_input), device="cpu", card=card(tmp_path, **kwargs)
    )


def signals(n=2, n_leads=4, n_samples=16):
    return np.random.default_rng(7).normal(size=(n, n_leads, n_samples))


def probability_of(adapter, handle, batch):
    return adapter.forward(handle, batch)


def test_attribution_satisfies_completeness_for_a_published_head(tmp_path):
    adapter = TorchAdapter()
    handle = handle_of(tmp_path)
    batch = signals()
    baseline = np.zeros(4)

    attribution = adapter.attribute(handle, batch, baseline=baseline, steps=256)
    reference = np.broadcast_to(baseline[:, None], batch.shape).copy()
    gap = completeness_gap(
        attribution,
        probability_of(adapter, handle, batch),
        probability_of(adapter, handle, reference),
    )
    assert np.abs(gap).max() < 1e-4


def test_attribution_satisfies_completeness_for_a_fitted_head(tmp_path):
    """A probed handle attributes through the same path its forward scores on."""
    adapter = TorchAdapter()
    handle = handle_of(tmp_path)
    features = adapter.features(handle, signals(6))
    head = LinearHead(
        weights=np.linspace(-1.0, 1.0, features.shape[1]),
        bias=0.25,
        mean=features.mean(axis=0),
        scale=features.std(axis=0) + 1e-6,
    )
    fitted = adapter.with_head(handle, head, {"mode": "linear_probe"})

    batch = signals()
    baseline = np.zeros(4)
    attribution = adapter.attribute(fitted, batch, baseline=baseline, steps=256)
    reference = np.broadcast_to(baseline[:, None], batch.shape).copy()
    gap = completeness_gap(
        attribution,
        probability_of(adapter, fitted, batch),
        probability_of(adapter, fitted, reference),
    )
    assert np.abs(gap).max() < 1e-4


def test_attribution_is_returned_in_canonical_layout_whatever_the_card_declares(tmp_path):
    """A caller plotting lead 3 must not have to know how the model wanted its input."""
    adapter = TorchAdapter()
    batch = signals()
    lead_time = adapter.attribute(
        handle_of(tmp_path, layout="lead_time"), batch, baseline="zeros", steps=32
    )
    assert lead_time.shape == batch.shape

    handle = handle_of(tmp_path, transpose_input=True, layout="time_lead")
    time_lead = adapter.attribute(handle, batch, baseline="zeros", steps=32)
    assert time_lead.shape == batch.shape


def test_a_baseline_equal_to_the_input_attributes_nothing(tmp_path):
    # F(x) - F(x) = 0, so completeness forces every attribution to vanish.
    adapter = TorchAdapter()
    batch = signals()
    attribution = adapter.attribute(handle_of(tmp_path), batch, baseline=batch, steps=16)
    assert np.allclose(attribution, 0.0)


def test_the_baseline_changes_the_map(tmp_path):
    """The point of requiring it: the same input attributes differently."""
    adapter = TorchAdapter()
    batch = signals()
    zeros = adapter.attribute(handle_of(tmp_path), batch, baseline="zeros", steps=64)
    typical = adapter.attribute(
        handle_of(tmp_path), batch, baseline=batch.mean(axis=(0, 2)), steps=64
    )
    assert not np.allclose(zeros, typical)


def test_attribute_refuses_without_a_baseline(tmp_path):
    with pytest.raises(AttributionError, match="asystole"):
        TorchAdapter().attribute(handle_of(tmp_path), signals())


def test_a_card_with_no_positive_index_has_nothing_to_attribute(tmp_path):
    handle = handle_of(tmp_path, positive_index=None)
    with pytest.raises(NotImplementedError, match="positive_index"):
        TorchAdapter().attribute(handle, signals(), baseline="zeros")


def test_attribution_leaves_the_module_in_eval_mode_it_found_it_in(tmp_path):
    adapter = TorchAdapter()
    handle = handle_of(tmp_path)
    handle.module.train()
    adapter.attribute(handle, signals(), baseline="zeros", steps=4)
    assert handle.module.training


def test_the_default_step_count_is_enough_for_a_readable_map(tmp_path):
    adapter = TorchAdapter()
    handle = handle_of(tmp_path)
    batch = signals()
    attribution = adapter.attribute(handle, batch, baseline="zeros", steps=DEFAULT_STEPS)
    gap = completeness_gap(
        attribution,
        probability_of(adapter, handle, batch),
        probability_of(adapter, handle, np.zeros_like(batch)),
    )
    assert np.abs(gap).max() < 1e-3
