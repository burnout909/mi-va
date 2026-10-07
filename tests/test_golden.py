import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mival.adapters import get_adapter
from mival.compiler import compile_recipe
from mival.modelcard import load_card
from mival.synthetic import GOLDEN_SOURCE, synthetic_ecg

GOLDEN = Path(__file__).parent / "golden" / "expected.json"


def _probabilities(model_id, registry_dir):
    card = load_card(registry_dir / f"{model_id}.json")
    chain = compile_recipe(card.input_contract, GOLDEN_SOURCE)
    batch = np.stack(
        [chain.apply(synthetic_ecg(i)).data for i in range(10)]
    ).astype(np.float32)
    adapter = get_adapter(card.adapter)
    return adapter.forward(adapter.load(card), batch)


@pytest.mark.keras
@pytest.mark.weights("prophecg-stemi")
def test_prophecg_output_matches_golden(registry_dir):
    expected = json.loads(GOLDEN.read_text())["prophecg-stemi"]
    probs = np.round(_probabilities("prophecg-stemi", registry_dir), 6)
    assert hashlib.sha256(probs.tobytes()).hexdigest() == expected["sha256"]


@pytest.mark.keras
@pytest.mark.weights("prophecg-stemi")
def test_golden_file_records_every_model_with_a_forward_pass():
    body = json.loads(GOLDEN.read_text())
    assert "prophecg-stemi" in body
    assert len(body["prophecg-stemi"]["probabilities"]) == 10
