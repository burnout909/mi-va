"""Builder declarations on the ModelCard replace the hardcoded module constructor."""

import json

import numpy as np
import pytest

pytestmark = pytest.mark.torch

from mival.adapters import get_adapter  # noqa: E402
from mival.modelcard import load_card  # noqa: E402

TINY_MODULE = '''
import torch
from typing import Final

class Tiny(torch.nn.Module):
    # Final tells TorchScript this branch is fixed at construction time, so
    # scripting can settle on one return type instead of rejecting the
    # (Tensor, Tensor) vs Tensor union the plain ternary would otherwise have.
    with_features: Final[bool]
    def __init__(self, n_in, n_out, with_features=True):
        super().__init__()
        self.dense = torch.nn.Linear(n_in, n_out)
        self.with_features = with_features
    def forward(self, x):
        flat = x.flatten(1)
        out = self.dense(flat)
        return (out, flat) if self.with_features else out
'''


def write_code(tmp_path):
    (tmp_path / "tinymod.py").write_text(TINY_MODULE)
    return str(tmp_path)


def write_card(tmp_path, weights, weights_format, builder=None, extra=None, returns=None):
    body = {
        "model_id": "tiny",
        "Name": "tiny",
        "x-mival": {
            "adapter": "torch",
            "weights": [{"uri": str(weights), "sha256": "0" * 64, "role": "backbone"}],
            "weights_format": weights_format,
            "code_path": write_code(tmp_path),
            "input_contract": {
                "leads": ["I", "II"], "sampling_rate_hz": 2, "duration_s": 1,
                "unit": "mV", "scaling": "none", "layout": "lead_time", "dtype": "float32",
            },
            "output": {"type": "logits", "n_outputs": 3, "positive_index": 0, "weight_key": "dense.weight"},
            "runtime": {"returns": returns or ["logits", "features"]},
            "training_modes_supported": ["inference_only"],
            "pretraining_corpora": [],
        },
    }
    if builder is not None:
        body["x-mival"]["builder"] = builder
    body["x-mival"].update(extra or {})
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(body))
    return load_card(path)


def batch():
    return np.ones((2, 2, 2), dtype=np.float32)


def test_state_dict_weights_are_built_from_the_declared_module(tmp_path):
    import torch
    torch.manual_seed(0)
    reference = torch.nn.Linear(4, 3)
    torch.save({"dense.weight": reference.weight, "dense.bias": reference.bias}, tmp_path / "w.pt")
    card = write_card(tmp_path, tmp_path / "w.pt", "state_dict",
                      builder={"module": "tinymod:Tiny", "kwargs": {"n_in": 4, "n_out": 3}})
    handle = get_adapter("torch").load(card)
    assert torch.equal(handle.module.dense.weight.cpu(), reference.weight)


def test_checkpoint_weights_use_state_dict_key_and_rename_keys(tmp_path):
    import torch
    reference = torch.nn.Linear(4, 3)
    torch.save({"epoch": 1, "state_dict": {"fc1.weight": reference.weight, "fc1.bias": reference.bias}},
               tmp_path / "w.pt")
    card = write_card(tmp_path, tmp_path / "w.pt", "checkpoint",
                      builder={"module": "tinymod:Tiny", "kwargs": {"n_in": 4, "n_out": 3}},
                      extra={"state_dict_key": "state_dict", "rename_keys": {"fc1.": "dense."}})
    handle = get_adapter("torch").load(card)
    assert torch.equal(handle.module.dense.bias.cpu(), reference.bias)


def test_jit_weights_need_no_builder(tmp_path):
    import torch
    import sys
    sys.path.insert(0, write_code(tmp_path))
    from tinymod import Tiny
    scripted = torch.jit.script(Tiny(4, 3, with_features=False))
    scripted.save(str(tmp_path / "w.pt"))
    card = write_card(tmp_path, tmp_path / "w.pt", "jit", returns=["logits"])
    handle = get_adapter("torch").load(card)
    probs = get_adapter("torch").forward(handle, batch())
    assert probs.shape == (2,)


def test_returns_declaration_names_each_output(tmp_path):
    import torch
    reference = torch.nn.Linear(4, 3)
    torch.save({"dense.weight": reference.weight, "dense.bias": reference.bias}, tmp_path / "w.pt")
    card = write_card(tmp_path, tmp_path / "w.pt", "state_dict",
                      builder={"module": "tinymod:Tiny", "kwargs": {"n_in": 4, "n_out": 3}},
                      returns=["ignore", "features"])
    adapter = get_adapter("torch")
    features = adapter.features(adapter.load(card), batch())
    assert features.shape == (2, 4)


def test_n_outputs_mismatch_is_rejected(tmp_path):
    import torch
    reference = torch.nn.Linear(4, 5)
    torch.save({"dense.weight": reference.weight, "dense.bias": reference.bias}, tmp_path / "w.pt")
    card = write_card(tmp_path, tmp_path / "w.pt", "state_dict",
                      builder={"module": "tinymod:Tiny", "kwargs": {"n_in": 4, "n_out": 5}})
    with pytest.raises(ValueError, match="declares 3 outputs"):
        get_adapter("torch").load(card)


def test_unknown_returns_name_is_rejected(tmp_path):
    import torch
    reference = torch.nn.Linear(4, 3)
    torch.save({"dense.weight": reference.weight, "dense.bias": reference.bias}, tmp_path / "w.pt")
    card = write_card(tmp_path, tmp_path / "w.pt", "state_dict",
                      builder={"module": "tinymod:Tiny", "kwargs": {"n_in": 4, "n_out": 3}},
                      returns=["logits", "festures"])
    with pytest.raises(ValueError, match="'tiny'.*'festures'"):
        get_adapter("torch").load(card)
