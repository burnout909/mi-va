"""Torch ensembles and output transforms (L2-14, L2-8; plan 2026-10-07 stage 3)."""
import json

import numpy as np
import pytest

MODULE = '''
import torch
class Const(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.value.expand(x.shape[0]).unsqueeze(1) + 0 * x.mean(dim=(1, 2)).unsqueeze(1)
'''


def card(tmp_path, values, output, ensemble=None):
    import torch

    (tmp_path / "constmod.py").write_text(MODULE)
    weights = []
    for i, value in enumerate(values):
        path = tmp_path / f"m{i}.pt"
        torch.save({"value": torch.tensor([float(value)])}, path)
        weights.append({"uri": str(path), "sha256": "0" * 64, "role": "ensemble_member" if len(values) > 1 else "full"})
    body = {"model_id": "c", "Name": "c", "x-mival": {
        "adapter": "torch", "weights": weights, "weights_format": "state_dict",
        "builder": {"module": "constmod:Const"}, "code_path": str(tmp_path),
        "ensemble": ensemble or {"method": "none", "members": 1},
        "input_contract": {"leads": ["II"], "sampling_rate_hz": 10, "duration_s": 1, "unit": "mV",
                           "scaling": "none", "layout": "lead_time", "dtype": "float32"},
        "output": output, "runtime": {"returns": ["logits"]},
        "training_modes_supported": ["inference_only"], "pretraining_corpora": []}}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(body))
    from mival.modelcard import load_card
    return load_card(path)


def run(tmp_path, *args, **kwargs):
    from mival.adapters import get_adapter
    adapter = get_adapter("torch")
    handle = adapter.load(card(tmp_path, *args, **kwargs))
    return adapter.forward(handle, np.zeros((2, 1, 10), dtype=np.float32))


@pytest.mark.torch
def test_torch_ensemble_averages_members(tmp_path):
    out = run(tmp_path, [1.0, 2.0, 6.0], {"type": "regression", "value_index": 0},
              {"method": "mean", "members": 3})
    np.testing.assert_allclose(out, [3.0, 3.0])


@pytest.mark.torch
def test_torch_ensemble_member_count_must_match(tmp_path):
    with pytest.raises(ValueError, match="members"):
        run(tmp_path, [1.0, 2.0], {"type": "regression", "value_index": 0}, {"method": "mean", "members": 3})


@pytest.mark.torch
def test_single_weight_card_ignores_ensemble_none(tmp_path):
    np.testing.assert_allclose(run(tmp_path, [4.0], {"type": "regression", "value_index": 0}), [4.0, 4.0])


@pytest.mark.torch
def test_affine_transform_inverts_a_zscore(tmp_path):
    out = run(tmp_path, [2.0], {"type": "regression", "value_index": 0,
                                "transform": {"kind": "affine", "scale": 0.5, "offset": 3.99}})
    np.testing.assert_allclose(out, [4.99, 4.99])


@pytest.mark.torch
def test_exp_transform_after_ensemble_mean(tmp_path):
    out = run(tmp_path, [1.0, 3.0], {"type": "regression", "value_index": 0, "transform": {"kind": "exp"}},
              {"method": "mean", "members": 2})
    np.testing.assert_allclose(out, np.exp([2.0, 2.0]))


def test_unknown_transform_is_rejected():
    from mival.decode.transforms import apply_transform
    with pytest.raises(ValueError, match="transform"):
        apply_transform(np.ones(2), {"kind": "sqrt"})
