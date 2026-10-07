"""Crop and pad anchors on the input contract (plan 2026-10-07, Task 1)."""
import json
from pathlib import Path

import numpy as np
import pytest

from mival.compiler import compile_recipe
from mival.contract import InputContract
from mival.modelcard import load_card
from mival.ops import Crop, Pad
from mival.signal import LEADS_12, Signal, SourceMetadata

ROOT = Path(__file__).resolve().parents[1]


def _contract(**extra):
    body = {
        "leads": list(LEADS_12), "sampling_rate_hz": 400, "duration_s": 10.24,
        "unit": "mV", "scaling": "none", "layout": "lead_time", "dtype": "float32",
    }
    body.update(extra)
    return InputContract.from_dict(body)


def _ones(n_samples):
    return Signal(data=np.ones((1, n_samples), dtype=np.float32), leads=("II",),
                  sampling_rate_hz=400.0, unit="mV")


def test_pad_center_splits_deficit():
    out = Pad(10, anchor="center").apply(_ones(5))
    np.testing.assert_array_equal(out.data[0], [0, 0, 1, 1, 1, 1, 1, 0, 0, 0])


def test_contract_anchors_default_to_start():
    contract = _contract()
    assert (contract.crop_anchor, contract.pad_anchor) == ("start", "start")


def test_contract_rejects_unknown_anchor():
    with pytest.raises(ValueError, match="pad_anchor"):
        _contract(pad_anchor="middle")
    with pytest.raises(ValueError, match="crop_anchor"):
        _contract(crop_anchor="end")


def test_compiler_passes_anchors():
    source = SourceMetadata(tuple(LEADS_12), 400.0, 4000, "mV")
    chain = compile_recipe(_contract(pad_anchor="center"), source, pad_policy="zero")
    pads = [op for op in chain.ops if isinstance(op, Pad)]
    assert pads and pads[0].anchor == "center"
    short = _contract(duration_s=7, crop_anchor="center")
    crops = [op for op in compile_recipe(short, source).ops if isinstance(op, Crop)]
    assert crops and crops[0].anchor == "center"


def test_lvef_recipes_unchanged():
    golden = json.loads((ROOT / "tests/golden/lvef_recipes.json").read_text())
    source = SourceMetadata(tuple(LEADS_12), 500.0, 5000, "mV")
    now = {}
    for path in sorted((ROOT / "registry/models").glob("*.json")):
        card = load_card(path)
        ops = compile_recipe(card.input_contract, source).ops
        now[card.model_id] = json.loads(json.dumps([[op.name, op.params] for op in ops], default=str))
    assert now == golden
