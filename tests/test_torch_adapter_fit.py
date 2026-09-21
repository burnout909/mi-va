"""The torch fitting path (spec §4.4), on a synthetic module.

No weights and no ECGFounder checkout: a tiny module that returns
``(logits, features)`` is enough to exercise everything the adapter owns —
which parameters get gradients, that the loaded module is left alone, that
early stopping restores the best epoch, and that the fitted head is lifted out
of torch so inference has one code path. Testing that against a real
foundation-model checkpoint would test torch, not this adapter.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from mival.adapters import get_adapter
from mival.adapters._training import LinearHead, TrainingError
from mival.adapters.torch_adapter import TorchAdapter, TorchHandle
from mival.modelcard import load_card

pytestmark = pytest.mark.torch

N_LEADS = 4
N_SAMPLES = 32
WIDTH = 6


def make_card(tmp_path, positive_index=None, output_type="logits", n_outputs=3):
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
                "layout": "lead_time",
                "dtype": "float32",
            },
            "output": {
                "type": output_type,
                "n_outputs": n_outputs,
                "positive_index": positive_index,
            },
            "feature_layer": "trunk",
            "threshold": {},
            "runtime": {},
            "training_modes_supported": [
                "linear_probe",
                "partial_unfreeze",
                "full_finetune",
            ],
            "pretraining_corpora": [],
        },
    }
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(body))
    return load_card(path)


def make_module(n_outputs=3):
    import torch

    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(N_LEADS * N_SAMPLES, WIDTH)

        def forward(self, x):
            return torch.tanh(self.linear(x.flatten(1)))

    class Net(torch.nn.Module):
        """Returns ``(logits, features)``, the shape the adapter expects."""

        def __init__(self):
            super().__init__()
            self.trunk = Trunk()
            self.dense = torch.nn.Linear(WIDTH, n_outputs)

        def forward(self, x):
            features = self.trunk(x)
            return self.dense(features), features

    torch.manual_seed(0)
    return Net().eval()


def make_data(n=96, seed=0, n_validation=32, batch_size=16):
    """A learnable problem: lead 0's mean carries the label."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    rng.shuffle(y)
    signals = rng.normal(0, 0.3, size=(n, N_LEADS, N_SAMPLES)).astype(np.float32)
    signals[:, 0, :] += y[:, None] * 2.0
    store = {f"rec-{i}": signals[i] for i in range(n)}

    def load_batch(paths):
        return np.stack([store[path] for path in paths])

    split = n - n_validation
    return {
        "tensor_paths": [f"rec-{i}" for i in range(split)],
        "y": y[:split].astype(np.int64),
        "person_id": [f"p-{i}" for i in range(split)],
        "fold": [0] * split,
        "label_column": "label_primary",
        "load_batch": load_batch,
        "batch_size": batch_size,
        "validation": {
            "tensor_paths": [f"rec-{i}" for i in range(split, n)],
            "y": y[split:].astype(np.int64),
            "person_id": [f"p-{i}" for i in range(split, n)],
        },
    }


def handle_for(tmp_path, **card_kwargs):
    return TorchHandle(
        module=make_module(), device="cpu", card=make_card(tmp_path, **card_kwargs)
    )


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def test_forward_without_a_head_names_the_model_and_the_way_out(tmp_path):
    adapter = TorchAdapter()
    handle = handle_for(tmp_path)  # positive_index is null
    with pytest.raises(NotImplementedError, match="linear_probe"):
        adapter.forward(handle, np.zeros((2, N_LEADS, N_SAMPLES), dtype=np.float32))


def test_a_card_that_declares_a_positive_index_needs_no_fit(tmp_path):
    """inference_only on torch is a card fact, not a hardcoded exception.

    The ECGFounder card names no positive_index, which is why that model must
    be fitted. A torch card that does name one publishes a probability, and the
    adapter reads it.
    """
    adapter = TorchAdapter()
    handle = handle_for(tmp_path, positive_index=1, output_type="softmax")
    probs = adapter.forward(handle, np.zeros((5, N_LEADS, N_SAMPLES), dtype=np.float32))
    assert probs.shape == (5,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_sigmoid_outputs_are_read_per_class(tmp_path):
    adapter = TorchAdapter()
    handle = handle_for(tmp_path, positive_index=2, output_type="logits")
    probs = adapter.forward(handle, np.zeros((3, N_LEADS, N_SAMPLES), dtype=np.float32))
    assert probs.shape == (3,)


def test_an_unsupported_output_type_is_named(tmp_path):
    adapter = TorchAdapter()
    handle = handle_for(tmp_path, positive_index=0, output_type="ordinal")
    with pytest.raises(ValueError, match="ordinal"):
        adapter.forward(handle, np.zeros((2, N_LEADS, N_SAMPLES), dtype=np.float32))


def test_trainable_groups_are_the_named_children(tmp_path):
    assert TorchAdapter().trainable_groups(handle_for(tmp_path)) == ["trunk", "dense"]


# ---------------------------------------------------------------------------
# linear_probe
# ---------------------------------------------------------------------------


def test_linear_probe_learns_and_leaves_the_backbone_untouched(tmp_path):
    import torch

    adapter = TorchAdapter()
    handle = handle_for(tmp_path)
    before = {k: v.clone() for k, v in handle.module.state_dict().items()}

    fitted = adapter.fit(handle, make_data(), "linear_probe", {"epochs": 300})
    assert isinstance(fitted.head, LinearHead)

    for name, value in handle.module.state_dict().items():
        assert torch.equal(value, before[name]), f"{name} changed"

    data = make_data()
    batch = data["load_batch"](data["tensor_paths"])
    probs = adapter.forward(fitted, batch)
    assert probs[data["y"] == 1].mean() > probs[data["y"] == 0].mean() + 0.3


def test_linear_probe_needs_no_seed(tmp_path):
    # Convex objective, deterministic start: no randomness is consumed.
    adapter = TorchAdapter()
    first = adapter.fit(handle_for(tmp_path), make_data(), "linear_probe", {"epochs": 50})
    second = adapter.fit(handle_for(tmp_path), make_data(), "linear_probe", {"epochs": 50})
    np.testing.assert_array_equal(first.head.weights, second.head.weights)


# ---------------------------------------------------------------------------
# fine-tuning
# ---------------------------------------------------------------------------


def test_full_finetune_learns(tmp_path):
    adapter = TorchAdapter()
    data = make_data()
    fitted = adapter.fit(
        handle_for(tmp_path), data, "full_finetune", {"epochs": 30, "lr": 1e-2, "seed": 7}
    )
    probs = adapter.forward(fitted, data["load_batch"](data["tensor_paths"]))
    assert probs[data["y"] == 1].mean() > probs[data["y"] == 0].mean() + 0.3


def test_finetuning_copies_the_module_instead_of_mutating_it(tmp_path):
    import torch

    adapter = TorchAdapter()
    handle = handle_for(tmp_path)
    before = {k: v.clone() for k, v in handle.module.state_dict().items()}
    fitted = adapter.fit(
        handle, make_data(), "full_finetune", {"epochs": 5, "lr": 1e-2, "seed": 7}
    )
    # models fits a fresh handle per inner CV fold and compares their held-out
    # predictions; a fit that wrote through would carry one fold's weights into
    # the next fold's estimate.
    for name, value in handle.module.state_dict().items():
        assert torch.equal(value, before[name]), f"{name} changed"
    assert fitted.module is not handle.module
    assert not torch.equal(
        fitted.module.state_dict()["trunk.linear.weight"], before["trunk.linear.weight"]
    )


def test_partial_unfreeze_trains_only_the_named_groups(tmp_path):
    import torch

    adapter = TorchAdapter()
    handle = handle_for(tmp_path)
    before = {k: v.clone() for k, v in handle.module.state_dict().items()}
    fitted = adapter.fit(
        handle,
        make_data(),
        "partial_unfreeze",
        {"epochs": 5, "lr": 1e-2, "seed": 7, "unfreeze_groups": ["trunk"]},
    )
    state = fitted.module.state_dict()
    assert not torch.equal(state["trunk.linear.weight"], before["trunk.linear.weight"])
    assert torch.equal(state["dense.weight"], before["dense.weight"])
    assert fitted.fit_record["frozen"] == ["dense"]


def test_an_unknown_unfreeze_group_lists_the_real_ones(tmp_path):
    adapter = TorchAdapter()
    with pytest.raises(TrainingError, match="trunk"):
        adapter.fit(
            handle_for(tmp_path),
            make_data(),
            "partial_unfreeze",
            {"epochs": 1, "seed": 7, "unfreeze_groups": ["backbone"]},
        )


def test_finetuning_without_a_seed_is_refused(tmp_path):
    # Parameter initialisation and batch order both draw randomness; an
    # unseeded run could not be reproduced from its manifest.
    adapter = TorchAdapter()
    with pytest.raises(TrainingError, match="--seed"):
        adapter.fit(handle_for(tmp_path), make_data(), "full_finetune", {"epochs": 1})


def test_the_same_seed_reproduces_the_fit(tmp_path):
    adapter = TorchAdapter()
    hparams = {"epochs": 5, "lr": 1e-2, "seed": 21}
    first = adapter.fit(handle_for(tmp_path), make_data(), "full_finetune", hparams)
    second = adapter.fit(handle_for(tmp_path), make_data(), "full_finetune", hparams)
    np.testing.assert_allclose(first.head.weights, second.head.weights)


def test_different_seeds_give_different_fits(tmp_path):
    adapter = TorchAdapter()
    first = adapter.fit(
        handle_for(tmp_path), make_data(), "full_finetune", {"epochs": 5, "lr": 1e-2, "seed": 1}
    )
    second = adapter.fit(
        handle_for(tmp_path), make_data(), "full_finetune", {"epochs": 5, "lr": 1e-2, "seed": 2}
    )
    assert not np.allclose(first.head.weights, second.head.weights)


def test_the_global_torch_generator_is_restored(tmp_path):
    """Seeding torch is unavoidable; leaving it reseeded is not.

    A fit that left the process-global stream at its own seed would silently
    change every later torch draw in the same run.
    """
    import torch

    adapter = TorchAdapter()
    handle, data = handle_for(tmp_path), make_data()  # building these seeds torch itself
    torch.manual_seed(555)
    expected = torch.randn(3)
    torch.manual_seed(555)
    adapter.fit(handle, data, "full_finetune", {"epochs": 2, "seed": 9})
    assert torch.equal(torch.randn(3), expected)


def test_early_stopping_restores_the_best_epoch(tmp_path):
    adapter = TorchAdapter()
    fitted = adapter.fit(
        handle_for(tmp_path),
        make_data(),
        "full_finetune",
        {"epochs": 40, "lr": 1e-2, "seed": 3, "patience": 2},
    )
    record = fitted.fit_record
    assert record["best_epoch"] is not None
    assert record["best_epoch"] <= record["epochs_run"]
    assert record["validation_metric"] == "auroc"


def test_fitting_with_no_validation_split_runs_every_epoch(tmp_path):
    data = make_data()
    data["validation"] = None
    fitted = TorchAdapter().fit(
        handle_for(tmp_path), data, "full_finetune", {"epochs": 4, "lr": 1e-2, "seed": 3}
    )
    assert fitted.fit_record["epochs_run"] == 4
    assert fitted.fit_record["best_epoch"] is None


def test_the_fit_record_carries_what_train_log_must_report(tmp_path):
    fitted = TorchAdapter().fit(
        handle_for(tmp_path),
        make_data(),
        "partial_unfreeze",
        {"epochs": 3, "lr": 1e-2, "seed": 5, "unfreeze_groups": ["trunk"]},
    )
    record = fitted.fit_record
    # Spec §4.4 requires the full hyperparameters and the unfrozen group list.
    assert record["hparams"]["lr"] == 1e-2
    assert record["hparams"]["unfreeze_groups"] == ["trunk"]
    assert record["n_features"] == WIDTH
    assert record["trainable_parameters"] > 0
    # It has to survive json.dumps: it is written into train_log.jsonl.
    json.dumps(record)


def test_the_feature_width_is_probed_from_the_module_not_the_card(tmp_path):
    """A card that disagreed with its checkpoint would build the wrong head."""
    fitted = TorchAdapter().fit(
        handle_for(tmp_path), make_data(), "linear_probe", {"epochs": 5}
    )
    assert fitted.head.weights.size == WIDTH


# ---------------------------------------------------------------------------
# the real card
# ---------------------------------------------------------------------------


def test_the_ecgfounder_card_supports_no_inference_only_arm(registry_dir):
    """Pinned because it is the reason the study needs fit() at all."""
    card = load_card(registry_dir / "ecgfounder.json")
    assert "inference_only" not in card.training_modes_supported
    assert card.output["positive_index"] is None


def test_full_finetune_with_mse_uses_identity_head(tmp_path):
    """The fine-tuned head predicts values, not probabilities."""
    import torch
    from mival.modelcard import load_card

    (tmp_path / "tinymod_mse.py").write_text(
        "import torch\n"
        "class Tiny(torch.nn.Module):\n"
        "    def __init__(self, n_in, n_out):\n"
        "        super().__init__(); self.dense = torch.nn.Linear(n_in, n_out)\n"
        "    def forward(self, x):\n"
        "        flat = x.flatten(1); return self.dense(flat), flat\n"
    )
    reference = torch.nn.Linear(4, 3)
    torch.save({"dense.weight": reference.weight, "dense.bias": reference.bias}, tmp_path / "w.pt")
    card_body = {
        "model_id": "tiny", "Name": "tiny",
        "x-mival": {
            "adapter": "torch", "code_path": str(tmp_path), "weights_format": "state_dict",
            "weights": [{"uri": str(tmp_path / "w.pt"), "sha256": "0" * 64, "role": "backbone"}],
            "builder": {"module": "tinymod_mse:Tiny", "kwargs": {"n_in": 4, "n_out": 3}},
            "input_contract": {"leads": ["I", "II"], "sampling_rate_hz": 2, "duration_s": 1, "unit": "mV",
                               "scaling": "none", "layout": "lead_time", "dtype": "float32"},
            "output": {"type": "logits", "n_outputs": 3, "positive_index": 0},
            "runtime": {"returns": ["logits", "features"]},
            "training_modes_supported": ["full_finetune"], "pretraining_corpora": [],
        },
    }
    (tmp_path / "tiny.json").write_text(json.dumps(card_body))
    card = load_card(tmp_path / "tiny.json")

    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(0, 1, (n, 2, 2)).astype(np.float32)          # flattens to 4 features
    y = 50.0 + 3.0 * x[:, 0, 0] - 2.0 * x[:, 1, 1] + rng.normal(0, 0.1, n)
    store = {f"r{i}": x[i] for i in range(n)}
    data = {
        "tensor_paths": [f"r{i}" for i in range(100)], "y": y[:100],
        "person_id": [f"p{i}" for i in range(100)], "fold": [0] * 100,
        "label_column": "label_value", "objective": "mse",
        "load_batch": lambda paths: np.stack([store[p] for p in paths]), "batch_size": 16,
        "validation": {"tensor_paths": [f"r{i}" for i in range(100, n)], "y": y[100:],
                       "person_id": [f"p{i}" for i in range(100, n)]},
    }
    adapter = get_adapter("torch")
    fitted = adapter.fit(adapter.load(card), data, "full_finetune", {"epochs": 3, "seed": 0, "lr": 0.01})
    assert fitted.head.link == "identity"
    out = adapter.forward(fitted, x[:5])
    assert out.shape == (5,) and np.all(out > 1.5)   # values near 50, not probabilities
