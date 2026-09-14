import numpy as np
import pytest

from mival.adapters import get_adapter
from mival.modelcard import load_card

pytestmark = [pytest.mark.keras, pytest.mark.weights("prophecg-stemi")]


def synthetic_batch(batch_size=3, n_leads=8, n_samples=5000, fs=500.0):
    t = np.arange(n_samples, dtype=np.float32) / fs
    rows = [np.sin(2 * np.pi * (1.0 + i) * t) for i in range(n_leads)]
    one = np.stack(rows).astype(np.float32)
    return np.repeat(one[None, ...], batch_size, axis=0)


def test_load_returns_five_members(registry_dir):
    card = load_card(registry_dir / "prophecg-stemi.json")
    handle = get_adapter("keras").load(card)
    assert len(handle.members) == 5


def test_forward_returns_one_probability_per_record(registry_dir):
    card = load_card(registry_dir / "prophecg-stemi.json")
    adapter = get_adapter("keras")
    probs = adapter.forward(adapter.load(card), synthetic_batch())
    assert probs.shape == (3,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_forward_is_deterministic(registry_dir):
    card = load_card(registry_dir / "prophecg-stemi.json")
    adapter = get_adapter("keras")
    handle = adapter.load(card)
    batch = synthetic_batch()
    np.testing.assert_allclose(
        adapter.forward(handle, batch), adapter.forward(handle, batch), rtol=0, atol=0
    )


def test_identical_rows_produce_identical_probabilities(registry_dir):
    card = load_card(registry_dir / "prophecg-stemi.json")
    adapter = get_adapter("keras")
    probs = adapter.forward(adapter.load(card), synthetic_batch(batch_size=3))
    assert probs[0] == pytest.approx(probs[1]) == pytest.approx(probs[2])


def test_features_have_declared_width(registry_dir):
    """``feature_layer`` names the layer feeding the output head: ``dense_1``,
    64 wide, identical across the five members (read from the H5 files on
    2026-09-14). ``dropout_1`` sits between it and the head but is the
    identity at inference, so the two are the same representation."""
    card = load_card(registry_dir / "prophecg-stemi.json")
    adapter = get_adapter("keras")
    feats = adapter.features(adapter.load(card), synthetic_batch())
    assert feats.shape == (3, 64)
    assert np.isfinite(feats).all()
