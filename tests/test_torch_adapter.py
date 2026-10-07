import numpy as np
import pytest

from mival.adapters import get_adapter
from mival.modelcard import load_card

pytestmark = [pytest.mark.torch, pytest.mark.weights("ecgfounder")]


def synthetic_batch(batch_size=2, n_leads=12, n_samples=5000, fs=500.0):
    t = np.arange(n_samples, dtype=np.float32) / fs
    rows = [np.sin(2 * np.pi * (1.0 + i) * t) for i in range(n_leads)]
    one = np.stack(rows).astype(np.float32)
    return np.repeat(one[None, ...], batch_size, axis=0)


def test_load_returns_handle_with_module_and_device(registry_dir):
    card = load_card(registry_dir / "ecgfounder.json")
    handle = get_adapter("torch").load(card)
    assert handle.module is not None
    assert handle.card.model_id == "ecgfounder"


def test_features_have_declared_width(registry_dir):
    card = load_card(registry_dir / "ecgfounder.json")
    adapter = get_adapter("torch")
    feats = adapter.features(adapter.load(card), synthetic_batch())
    assert feats.shape == (2, 1024)
    assert np.isfinite(feats).all()


def test_trainable_groups_are_named(registry_dir):
    card = load_card(registry_dir / "ecgfounder.json")
    adapter = get_adapter("torch")
    groups = adapter.trainable_groups(adapter.load(card))
    assert groups
    assert all(isinstance(name, str) for name in groups)


def test_forward_is_unavailable_without_a_head(registry_dir):
    card = load_card(registry_dir / "ecgfounder.json")
    adapter = get_adapter("torch")
    with pytest.raises(NotImplementedError, match="declares no output.positive_index"):
        adapter.forward(adapter.load(card), synthetic_batch())
