"""Interval decoder and the new torch output types (plan 2026-10-07, Task 3)."""
import json

import numpy as np
import pytest

from mival.decode.intervals import INTERVAL_ORDER, intervals_from_mask


def beat_mask(fs=250, beats=((50, 70, 95, 120, 170), (300, 320, 345, 370, 420)), length=500):
    """(p_on, qrs_on, qrs_off, t_on, t_off) per beat, classes 1=P 2=QRS 3=T."""
    mask = np.zeros(length, dtype=np.int64)
    for p_on, qrs_on, qrs_off, t_on, t_off in beats:
        mask[p_on:qrs_on - 5] = 1
        mask[qrs_on:qrs_off] = 2
        mask[t_on:t_off] = 3
    return mask


def test_intervals_from_synthetic_mask():
    out = intervals_from_mask(beat_mask()[None, :], fs=250)
    assert INTERVAL_ORDER == ("pr", "qrs", "qt")
    np.testing.assert_allclose(out[0], [80.0, 100.0, 400.0])


def test_intervals_nan_without_p():
    mask = beat_mask()
    mask[mask == 1] = 0
    out = intervals_from_mask(mask[None, :], fs=250)
    assert np.isnan(out[0, 0]) and out[0, 1] == 100.0 and out[0, 2] == 400.0


def test_intervals_ignore_beats_cut_by_the_window():
    mask = beat_mask(beats=((300, 320, 345, 370, 420),))
    mask[0:20] = 2  # a QRS already under way when the window opens
    mask[40:80] = 3
    np.testing.assert_allclose(intervals_from_mask(mask[None, :], fs=250)[0], [80.0, 100.0, 400.0])


def test_intervals_all_nan_on_empty_mask():
    assert np.isnan(intervals_from_mask(np.zeros((1, 100), dtype=int), fs=250)).all()



MODULE = '''
import torch
class Risk(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2)).unsqueeze(1) * 2.0
class Seg(torch.nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask), persistent=False)
    def forward(self, x):
        onehot = torch.nn.functional.one_hot(self.mask, 4).T.float()
        return onehot.unsqueeze(0).repeat(x.shape[0], 1, 1)
'''


def card(tmp_path, name, output, n_samples, builder_kwargs=None):
    torch = pytest.importorskip("torch")
    (tmp_path / "outmods.py").write_text(MODULE)
    weights = tmp_path / f"{name}.pt"
    torch.save({}, weights)
    body = {
        "model_id": name, "Name": name,
        "x-mival": {
            "adapter": "torch",
            "weights": [{"uri": str(weights), "sha256": "0" * 64, "role": "full"}],
            "weights_format": "state_dict",
            "builder": {"module": f"outmods:{name.capitalize()}", "kwargs": builder_kwargs or {}},
            "code_path": str(tmp_path),
            "input_contract": {"leads": ["II"], "sampling_rate_hz": 250, "duration_s": n_samples / 250,
                               "unit": "mV", "scaling": "none", "layout": "lead_time", "dtype": "float32"},
            "output": output,
            "runtime": {"returns": ["logits"]},
            "training_modes_supported": ["inference_only"],
            "pretraining_corpora": [],
        },
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(body))
    from mival.modelcard import load_card
    return load_card(path)


@pytest.mark.torch
def test_torch_forward_risk_score(tmp_path):
    from mival.adapters import get_adapter
    adapter = get_adapter("torch")
    handle = adapter.load(card(tmp_path, "risk", {"type": "risk_score", "n_outputs": 1}, 10))
    batch = np.ones((3, 1, 10), dtype=np.float32)
    np.testing.assert_allclose(adapter.forward(handle, batch), [2.0, 2.0, 2.0])


@pytest.mark.torch
def test_torch_forward_segmentation_mask(tmp_path):
    from mival.adapters import get_adapter
    adapter = get_adapter("torch")
    mask = beat_mask().tolist()
    handle = adapter.load(card(tmp_path, "seg", {"type": "segmentation_mask", "n_outputs": 4},
                               500, {"mask": mask}))
    out = adapter.forward(handle, np.zeros((2, 1, 500), dtype=np.float32))
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out[1], [80.0, 100.0, 400.0])
