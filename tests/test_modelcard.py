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
        {
            "lead_unavailable",
            "upsample_required",
            "duration_short",
            "unit_missing",
            "rate_unsupported",
            "read_failed",
        }
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


def test_contract_rejects_unsupported_unit():
    with pytest.raises(ValueError, match="unit"):
        InputContract.from_dict(
            {
                "leads": ["I", "II"],
                "sampling_rate_hz": 500,
                "duration_s": 10,
                "unit": "V",
                "scaling": "none",
                "layout": "lead_time",
                "dtype": "float32",
            }
        )


def test_contract_rejects_unsupported_scaling():
    with pytest.raises(ValueError, match="scaling"):
        InputContract.from_dict(
            {
                "leads": ["I", "II"],
                "sampling_rate_hz": 500,
                "duration_s": 10,
                "unit": "mV",
                "scaling": "minmax",
                "layout": "lead_time",
                "dtype": "float32",
            }
        )


def test_contract_rejects_unsupported_filter_kind():
    with pytest.raises(ValueError, match="bandpass"):
        InputContract.from_dict(
            {
                "leads": ["I", "II"],
                "sampling_rate_hz": 500,
                "duration_s": 10,
                "unit": "mV",
                "scaling": "none",
                "layout": "lead_time",
                "dtype": "float32",
                "filters": [{"kind": "bandpass", "cutoff_hz": 40}],
            }
        )


def test_both_registry_cards_still_load():
    cards = load_registry(REGISTRY)
    assert {"ecgfounder", "prophecg-stemi"} <= set(cards)


def test_every_registry_card_loads(registry_dir):
    from mival.modelcard import load_registry

    cards = load_registry(registry_dir)
    assert {"ecgfounder", "prophecg-stemi", "xecg", "heartwise-lvef-binary", "heartwise-lvef-regression"} <= set(cards)


def test_lvef_study_perturbation_axes_pass_every_registry_card(registry_dir):
    """studies/lvef declares baseline-only resample/duration axes because the
    study's cards span 500/250/100 Hz; every card must clear the check with
    warnings only, never a GridContractError (see study.yaml's comment)."""
    from mival.perturbation import GridContractError, check_grid_against_contract
    from mival.pipeline.study import load_study

    study_path = Path(__file__).resolve().parents[1] / "studies" / "lvef"
    study = load_study(study_path)
    axes = study.stage_spec("preprocess")["perturbation"]["axes"]

    cards = load_registry(registry_dir)
    for card in cards.values():
        try:
            check_grid_against_contract(axes, card.input_contract, card.model_id)
        except GridContractError as exc:
            pytest.fail(f"{card.model_id}: {exc}")


def _card_with_paths(tmp_path, weight_uri, code_path):
    body = json.loads((REGISTRY / "ecgfounder.json").read_text())
    body["x-mival"]["weights"][0]["uri"] = weight_uri
    body["x-mival"]["code_path"] = code_path
    body["x-mival"]["runtime"]["env"] = "/abs/env"
    path = tmp_path / "card.json"
    path.write_text(json.dumps(body))
    return path


def test_card_paths_expand_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MIVAL_HOME", "/site/mival")
    path = _card_with_paths(tmp_path, "${MIVAL_HOME}/models/w.pth", "${MIVAL_HOME}/code/x")
    card = load_card(path)
    assert card.weights[0]["uri"] == "/site/mival/models/w.pth"
    assert card.raw["x-mival"]["code_path"] == "/site/mival/code/x"


def test_card_with_unset_placeholder_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("MIVAL_HOME", raising=False)
    path = _card_with_paths(tmp_path, "${MIVAL_HOME}/models/w.pth", "/abs/code")
    with pytest.raises(ValueError, match="MIVAL_HOME is not set"):
        load_card(path)


def test_card_without_placeholder_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("MIVAL_HOME", raising=False)
    path = _card_with_paths(tmp_path, "/abs/w.pth", "/abs/code")
    assert load_card(path).weights[0]["uri"] == "/abs/w.pth"
