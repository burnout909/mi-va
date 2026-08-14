import json
from pathlib import Path

import pytest

from mival.contract import REASON_CODES, CompileError, InputContract
from mival.modelcard import load_card, load_registry

REGISTRY = Path(__file__).resolve().parents[1] / "registry" / "models"


def test_input_contract_computes_sample_count():
    contract = InputContract.from_dict(
        {
            "leads": ["I", "II"],
            "sampling_rate_hz": 500,
            "duration_s": 10,
            "unit": "mV",
            "scaling": "none",
            "layout": "lead_time",
            "dtype": "float32",
        }
    )
    assert contract.n_samples == 5000
    assert contract.filters == ()


def test_reason_codes_are_the_agreed_vocabulary():
    assert REASON_CODES == frozenset(
        {"lead_unavailable", "upsample_required", "duration_short", "unit_missing"}
    )


def test_compile_error_rejects_unknown_reason_code():
    with pytest.raises(ValueError, match="unknown reason_code"):
        CompileError("something_else", "detail")


def test_compile_error_carries_reason_and_detail():
    err = CompileError("unit_missing", "no sensitivity in DICOM")
    assert err.reason_code == "unit_missing"
    assert "sensitivity" in str(err)


def test_prophecg_card_matches_verified_contract():
    card = load_card(REGISTRY / "prophecg-stemi.json")
    assert card.model_id == "prophecg-stemi"
    assert card.adapter == "keras"
    assert card.input_contract.leads == ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
    assert card.input_contract.n_samples == 5000
    assert card.input_contract.layout == "time_lead"
    assert len(card.weights) == 5
    assert card.ensemble["method"] == "mean_probability"
    assert card.output["positive_index"] == 1
    assert card.threshold["status"] == "legacy"


def test_ecgfounder_card_matches_verified_contract():
    card = load_card(REGISTRY / "ecgfounder.json")
    assert card.adapter == "torch"
    assert len(card.input_contract.leads) == 12
    assert card.input_contract.layout == "lead_time"
    assert card.input_contract.scaling == "global_zscore"
    assert "linear_probe" in card.training_modes_supported


def test_every_card_declares_pretraining_corpora():
    cards = load_registry(REGISTRY)
    assert cards
    for card in cards.values():
        assert isinstance(card.pretraining_corpora, tuple)


def test_card_without_pretraining_corpora_is_rejected(tmp_path):
    body = json.loads((REGISTRY / "ecgfounder.json").read_text())
    del body["x-mival"]["pretraining_corpora"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="pretraining_corpora"):
        load_card(bad)


def test_card_without_top_level_name_is_rejected(tmp_path):
    body = json.loads((REGISTRY / "ecgfounder.json").read_text())
    del body["Name"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="Name"):
        load_card(bad)
