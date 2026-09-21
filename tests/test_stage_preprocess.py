import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from mival.contract import REASON_CODES  # noqa: E402
from mival.pipeline.ledger import read_exclusions  # noqa: E402
from mival.pipeline.layout import exclusions_root  # noqa: E402
from mival.pipeline.manifest import RunManifest  # noqa: E402
from mival.pipeline.stage import execute, get_stage, prepare  # noqa: E402
from mival.pipeline.tables import read_table, write_table  # noqa: E402
from mival.signal import LEADS_12  # noqa: E402
from mival.stages.preprocess import (  # noqa: E402
    BASELINE_PERTURBATION_ID,
    COHORT_INDEX_COLUMNS,
    PERTURBATION_GRID,
    PREPROCESS_INDEX,
    PREPROCESS_INDEX_COLUMNS,
    RECIPES,
    TENSOR_DIR,
    PreprocessStage,
    load_npz_record,
    recipe_id,
)
from mival.synthetic import synthetic_ecg  # noqa: E402

COHORT_COLUMNS = COHORT_INDEX_COLUMNS + ("label_primary",)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def write_record(
    path,
    record_index=0,
    leads=LEADS_12,
    sampling_rate_hz=500.0,
    n_samples=5000,
    unit="mV",
    declared_rate=None,
):
    """A stand-in for one DICOM waveform, in the loader's .npz form."""
    signal = synthetic_ecg(
        record_index, leads=leads, sampling_rate_hz=sampling_rate_hz, n_samples=n_samples
    )
    payload = {
        "data": signal.data,
        "leads": np.array(list(leads)),
        "sampling_rate_hz": np.float64(
            sampling_rate_hz if declared_rate is None else declared_rate
        ),
    }
    if unit is not None:
        payload["unit"] = np.array(unit)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **payload)
    return Path(path)


def write_cohort(path, records):
    return write_table(records, path, COHORT_COLUMNS)


def write_card(directory, model_id, leads=LEADS_12, sampling_rate_hz=500, duration_s=10,
               scaling="none", unit="mV", adapter="torch"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "model_id": model_id,
        "Name": model_id,
        "x-mival": {
            "adapter": adapter,
            "weights": [{"uri": f"/nonexistent/{model_id}.bin", "sha256": "0" * 64, "role": "backbone"}],
            "input_contract": {
                "leads": list(leads),
                "sampling_rate_hz": sampling_rate_hz,
                "duration_s": duration_s,
                "unit": unit,
                "filters": [],
                "scaling": scaling,
                "layout": "lead_time",
                "dtype": "float32",
            },
            "output": {"type": "logits", "n_outputs": 1, "positive_index": 0},
            "training_modes_supported": ["inference_only"],
            "pretraining_corpora": [],
        },
    }
    path = directory / f"{model_id}.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def context(tmp_path, spec, cohort_path, stage=None, seed=13):
    return prepare(
        stage=stage or PreprocessStage(),
        study_id="s1",
        site="mimic",
        spec=spec,
        inputs={"cohort_index": cohort_path},
        runs_root=tmp_path / "runs",
        seed=seed,
    )


@pytest.fixture
def study(tmp_path):
    """Two cards with the same input contract, one with a different one."""
    registry = tmp_path / "registry"
    write_card(registry, "alpha", leads=LEADS_12)
    write_card(registry, "bravo", leads=LEADS_12)
    write_card(registry, "charlie", leads=("I", "II", "V1", "V2", "V3", "V4", "V5", "V6"))

    records = []
    for index in range(3):
        source = write_record(tmp_path / "raw" / f"r{index}.npz", record_index=index)
        records.append(
            {
                "image_occurrence_id": f"img-{index}",
                "person_id": f"p-{index}",
                "local_path": str(source),
                "label_primary": index % 2,
            }
        )
    cohort = write_cohort(tmp_path / "cohort_index.parquet", records)
    return {"registry": registry, "cohort": cohort, "spec": {"registry": str(registry)}}


def run_stage(tmp_path, study, spec_extra=None, stage=None):
    spec = dict(study["spec"])
    spec.update(spec_extra or {})
    stage = stage or PreprocessStage()
    ctx = context(tmp_path, spec, study["cohort"], stage=stage)
    ctx.layout.create_dirs()
    result = stage.run(ctx)
    return ctx, result


# --------------------------------------------------------------------------
# the index schema stage 4 joins on
# --------------------------------------------------------------------------


def test_index_columns_match_the_pinned_contract_exactly(tmp_path, study):
    ctx, result = run_stage(tmp_path, study)
    frame = read_table(ctx.layout.artifact(PREPROCESS_INDEX))
    assert tuple(frame.columns) == PREPROCESS_INDEX_COLUMNS
    assert len(frame) == 3 * 3  # 3 records x 3 cards
    assert result.counts["in"] == 3
    assert result.counts["out"] == 3


def test_every_stored_tensor_row_is_unperturbed(tmp_path, study):
    # Spec §4.3: perturbed tensors are never written; the sweep is stage 4's.
    ctx, _ = run_stage(tmp_path, study)
    frame = read_table(ctx.layout.artifact(PREPROCESS_INDEX))
    assert set(frame["perturbation_id"]) == {BASELINE_PERTURBATION_ID}


def test_tensor_path_is_relative_to_the_run_dir_and_resolves(tmp_path, study):
    ctx, _ = run_stage(tmp_path, study)
    frame = read_table(ctx.layout.artifact(PREPROCESS_INDEX))
    for row in frame.to_dict("records"):
        # Absolute paths would not survive the move from a laptop to DICOM-MIVA.
        assert not Path(row["tensor_path"]).is_absolute()
        tensor = ctx.layout.run_dir / row["tensor_path"]
        assert tensor.is_file()
        data = np.load(tensor)
        assert data.shape == (row["n_leads"], row["n_samples"])
        assert data.dtype == np.float32


def test_the_stored_tensor_is_the_compiled_recipe_output(tmp_path, study):
    from mival.compiler import compile_recipe
    from mival.modelcard import load_card
    from mival.signal import Signal

    ctx, _ = run_stage(tmp_path, study)
    frame = read_table(ctx.layout.artifact(PREPROCESS_INDEX))
    row = frame[frame["model_id"] == "charlie"].iloc[0].to_dict()

    card = load_card(study["registry"] / "charlie.json")
    data, source = load_npz_record(Path(str(tmp_path / "raw" / "r0.npz")))
    chain = compile_recipe(card.input_contract, source)
    expected = chain.apply(
        Signal(data=data, leads=source.leads, sampling_rate_hz=source.sampling_rate_hz, unit=source.unit)
    )
    stored = np.load(ctx.layout.run_dir / row["tensor_path"])
    assert np.array_equal(stored, expected.data)
    assert row["n_leads"] == 8
    assert row["n_samples"] == 5000
    assert row["sampling_rate_hz"] == 500.0


def test_a_missing_cohort_column_names_what_is_missing(tmp_path, study):
    bad = write_table(
        [{"image_occurrence_id": "img-0", "person_id": "p-0"}],
        tmp_path / "bad.parquet",
        ("image_occurrence_id", "person_id"),
    )
    stage = PreprocessStage()
    ctx = context(tmp_path, study["spec"], bad, stage=stage)
    ctx.layout.create_dirs()
    with pytest.raises(ValueError, match="local_path"):
        stage.run(ctx)


# --------------------------------------------------------------------------
# recipe_id
# --------------------------------------------------------------------------


def test_two_cards_with_the_same_input_contract_share_a_recipe_and_a_tensor(tmp_path, study):
    ctx, result = run_stage(tmp_path, study)
    frame = read_table(ctx.layout.artifact(PREPROCESS_INDEX))
    by_model = frame.groupby("model_id")["recipe_id"].unique()

    assert by_model["alpha"].tolist() == by_model["bravo"].tolist()
    assert by_model["alpha"].tolist() != by_model["charlie"].tolist()

    # Two contracts, three records: six tensors, not nine.
    assert result.counts["recipes"] == 2
    assert result.counts["tensors"] == 6
    assert result.counts["index_rows"] == 9

    shared = by_model["alpha"][0]
    alpha_paths = set(frame[frame["model_id"] == "alpha"]["tensor_path"])
    bravo_paths = set(frame[frame["model_id"] == "bravo"]["tensor_path"])
    assert alpha_paths == bravo_paths
    assert all(f"{TENSOR_DIR}/{shared}/" in path for path in alpha_paths)


def test_recipe_id_is_a_function_of_the_op_chain_not_of_the_model(tmp_path):
    from mival.compiler import compile_recipe
    from mival.modelcard import load_card

    registry = tmp_path / "registry"
    write_card(registry, "one", leads=LEADS_12)
    write_card(registry, "two", leads=LEADS_12)
    write_card(registry, "three", leads=LEADS_12, scaling="global_zscore")
    _, source = load_npz_record(write_record(tmp_path / "r.npz"))

    ids = {
        model_id: recipe_id(compile_recipe(load_card(registry / f"{model_id}.json").input_contract, source))
        for model_id in ("one", "two", "three")
    }
    assert ids["one"] == ids["two"]
    assert ids["one"] != ids["three"]


def test_recipe_id_is_a_legal_path_component(tmp_path, study):
    import re

    ctx, _ = run_stage(tmp_path, study)
    frame = read_table(ctx.layout.artifact(PREPROCESS_INDEX))
    for value in frame["recipe_id"].unique():
        assert re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", value)


def test_a_heterogeneous_cohort_yields_more_than_one_recipe_per_model(tmp_path):
    # Not a bug: recipe_id names the transform, and a 250 Hz source needs a
    # different one than a 500 Hz source. Spec §3.4 wants that on the axis.
    registry = tmp_path / "registry"
    write_card(registry, "alpha", leads=LEADS_12)
    records = []
    for index, rate in enumerate((500.0, 1000.0)):
        source = write_record(
            tmp_path / "raw" / f"r{index}.npz",
            record_index=index,
            sampling_rate_hz=rate,
            n_samples=int(10 * rate),
        )
        records.append(
            {
                "image_occurrence_id": f"img-{index}",
                "person_id": f"p-{index}",
                "local_path": str(source),
                "label_primary": 0,
            }
        )
    cohort = write_cohort(tmp_path / "cohort_index.parquet", records)
    stage = PreprocessStage()
    ctx = context(tmp_path, {"registry": str(registry)}, cohort, stage=stage)
    ctx.layout.create_dirs()
    stage.run(ctx)
    frame = read_table(ctx.layout.artifact(PREPROCESS_INDEX))
    assert frame["recipe_id"].nunique() == 2


# --------------------------------------------------------------------------
# the input contract gate: all five reason codes reach the ledger
# --------------------------------------------------------------------------


def _single_record_run(tmp_path, card_kwargs, record_kwargs, spec_extra=None):
    registry = tmp_path / "registry"
    write_card(registry, "solo", **card_kwargs)
    source = write_record(tmp_path / "raw" / "r0.npz", **record_kwargs)
    cohort = write_cohort(
        tmp_path / "cohort_index.parquet",
        [
            {
                "image_occurrence_id": "img-0",
                "person_id": "p-0",
                "local_path": str(source),
                "label_primary": 1,
            }
        ],
    )
    # The perturbation grid degrades the *contract's* rate and duration, so a
    # fixture card that departs from the defaults needs a grid that fits it —
    # exactly what check_grid_against_contract enforces in the stage.
    axes = {}
    if "duration_s" in card_kwargs:
        axes["duration"] = [card_kwargs["duration_s"], 5, 2.5]
    if "sampling_rate_hz" in card_kwargs:
        axes["resample"] = [card_kwargs["sampling_rate_hz"], 250, 125, 100]
    spec = {"registry": str(registry)}
    if axes:
        spec["perturbation"] = {"axes": axes}
    spec.update(spec_extra or {})
    stage = PreprocessStage()
    ctx = context(tmp_path, spec, cohort, stage=stage)
    ctx.layout.create_dirs()
    result = stage.run(ctx)
    return ctx, result


GATE_CASES = {
    # A source with only two leads cannot supply or derive the precordials.
    "lead_unavailable": (
        {"leads": LEADS_12},
        {"leads": ("I", "II"), "n_samples": 5000},
    ),
    # 250 Hz source, 500 Hz contract, allow_upsample defaults to false.
    "upsample_required": (
        {"leads": LEADS_12, "sampling_rate_hz": 500},
        {"sampling_rate_hz": 250.0, "n_samples": 2500},
    ),
    # 10 s of source, 20 s contract, pad_policy defaults to reject.
    "duration_short": (
        {"leads": LEADS_12, "duration_s": 20},
        {"n_samples": 5000},
    ),
    # No unit in the bundle: spec §4.3 forbids guessing one.
    "unit_missing": (
        {"leads": LEADS_12},
        {"unit": None},
    ),
    # 100003 Hz has no polyphase ratio to 500 Hz within the factor limit.
    "rate_unsupported": (
        {"leads": LEADS_12},
        {"declared_rate": 100003.0},
    ),
}


@pytest.mark.parametrize("reason_code", sorted(GATE_CASES))
def test_each_compile_failure_reaches_the_ledger_with_its_own_reason_code(
    tmp_path, reason_code
):
    card_kwargs, record_kwargs = GATE_CASES[reason_code]
    ctx, result = _single_record_run(tmp_path, card_kwargs, record_kwargs)

    assert ctx.ledger.counts() == {reason_code: 1}
    row = ctx.ledger.rows[0]
    assert row["image_occurrence_id"] == "img-0"
    assert row["person_id"] == "p-0"
    assert row["stage"] == "preprocess"
    assert "solo" in row["detail"]

    assert result.counts["out"] == 0
    assert result.counts["index_rows"] == 0
    assert result.counts["tensors"] == 0


def test_the_five_gate_cases_cover_the_whole_reason_code_vocabulary():
    assert set(GATE_CASES) == set(REASON_CODES)
    assert PreprocessStage.reason_codes == REASON_CODES


def test_allow_upsample_and_pad_policy_are_the_documented_escape_hatches(tmp_path):
    card_kwargs, record_kwargs = GATE_CASES["upsample_required"]
    _, result = _single_record_run(
        tmp_path, card_kwargs, record_kwargs, spec_extra={"allow_upsample": True}
    )
    assert result.counts["out"] == 1

    card_kwargs, record_kwargs = GATE_CASES["duration_short"]
    _, result = _single_record_run(
        tmp_path, card_kwargs, record_kwargs, spec_extra={"pad_policy": "zero"}
    )
    assert result.counts["out"] == 1


def test_one_ledger_row_per_record_and_reason_not_per_model(tmp_path):
    # The STARD flow of spec §3.6 counts records; three cards failing the same
    # way must not inflate it threefold.
    registry = tmp_path / "registry"
    for model_id in ("alpha", "bravo", "charlie"):
        write_card(registry, model_id, leads=LEADS_12)
    source = write_record(tmp_path / "raw" / "r0.npz", unit=None)
    cohort = write_cohort(
        tmp_path / "cohort_index.parquet",
        [
            {
                "image_occurrence_id": "img-0",
                "person_id": "p-0",
                "local_path": str(source),
                "label_primary": 1,
            }
        ],
    )
    stage = PreprocessStage()
    ctx = context(tmp_path, {"registry": str(registry)}, cohort, stage=stage)
    ctx.layout.create_dirs()
    stage.run(ctx)
    assert ctx.ledger.counts() == {"unit_missing": 1}
    assert "alpha, bravo, charlie" in ctx.ledger.rows[0]["detail"]


def test_a_record_excluded_for_one_model_only_is_flagged_in_the_warnings(tmp_path):
    # The two cards differ in leads rather than in duration: a perturbation
    # grid degrades the *contract's* rate and duration, so two cards whose
    # contracts differ in either cannot share one grid (see
    # check_grid_against_contract). Leads are the axis-free way to make one
    # card exclude a record while the other keeps it.
    registry = tmp_path / "registry"
    write_card(registry, "two_lead", leads=("I", "II"))
    write_card(registry, "twelve_lead", leads=LEADS_12)
    source = write_record(tmp_path / "raw" / "r0.npz", leads=("I", "II"), n_samples=5000)
    cohort = write_cohort(
        tmp_path / "cohort_index.parquet",
        [
            {
                "image_occurrence_id": "img-0",
                "person_id": "p-0",
                "local_path": str(source),
                "label_primary": 1,
            }
        ],
    )
    stage = PreprocessStage()
    ctx = context(tmp_path, {"registry": str(registry)}, cohort, stage=stage)
    ctx.layout.create_dirs()
    result = stage.run(ctx)

    assert result.counts["in"] == 1
    assert result.counts["out"] == 1
    assert ctx.ledger.counts() == {"lead_unavailable": 1}
    assert any("record-model pairs" in warning for warning in result.warnings)


def test_a_missing_person_id_is_carried_through_as_null(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "alpha", leads=LEADS_12)
    source = write_record(tmp_path / "raw" / "r0.npz")
    cohort = write_cohort(
        tmp_path / "cohort_index.parquet",
        [
            {
                "image_occurrence_id": "img-0",
                "person_id": None,
                "local_path": str(source),
                "label_primary": 1,
            }
        ],
    )
    stage = PreprocessStage()
    ctx = context(tmp_path, {"registry": str(registry)}, cohort, stage=stage)
    ctx.layout.create_dirs()
    stage.run(ctx)
    frame = read_table(ctx.layout.artifact(PREPROCESS_INDEX))
    assert frame["person_id"].isna().all()


def test_a_duplicate_image_occurrence_id_fails_the_run(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "alpha", leads=LEADS_12)
    source = write_record(tmp_path / "raw" / "r0.npz")
    row = {
        "image_occurrence_id": "img-0",
        "person_id": "p-0",
        "local_path": str(source),
        "label_primary": 1,
    }
    cohort = write_cohort(tmp_path / "cohort_index.parquet", [row, dict(row)])
    stage = PreprocessStage()
    ctx = context(tmp_path, {"registry": str(registry)}, cohort, stage=stage)
    ctx.layout.create_dirs()
    with pytest.raises(ValueError, match="more than once"):
        stage.run(ctx)


def test_an_unsafe_image_occurrence_id_fails_the_run(tmp_path):
    # Not a ledger exclusion: it is outside the closed vocabulary of spec §4.3
    # and means the upstream table is malformed.
    registry = tmp_path / "registry"
    write_card(registry, "alpha", leads=LEADS_12)
    source = write_record(tmp_path / "raw" / "r0.npz")
    cohort = write_cohort(
        tmp_path / "cohort_index.parquet",
        [
            {
                "image_occurrence_id": "../escape",
                "person_id": "p-0",
                "local_path": str(source),
                "label_primary": 1,
            }
        ],
    )
    stage = PreprocessStage()
    ctx = context(tmp_path, {"registry": str(registry)}, cohort, stage=stage)
    ctx.layout.create_dirs()
    with pytest.raises(ValueError, match="path component"):
        stage.run(ctx)


# --------------------------------------------------------------------------
# sidecars
# --------------------------------------------------------------------------


def test_the_perturbation_grid_is_published_for_stage_four(tmp_path, study):
    ctx, result = run_stage(tmp_path, study)
    grid = json.loads(ctx.layout.artifact(PERTURBATION_GRID).read_text())

    assert grid["mode"] == "ofat"
    assert grid["seed"] == 13
    assert grid["axes"]["resample"] == [500, 250, 125, 100]
    ids = [item["perturbation_id"] for item in grid["perturbations"]]
    assert ids[0] == BASELINE_PERTURBATION_ID
    assert "resample-125" in ids
    assert len(ids) == result.counts["perturbations"] == 13


def test_cartesian_mode_is_opt_in_from_the_study_spec(tmp_path, study):
    ctx, result = run_stage(tmp_path, study, spec_extra={"perturbation": {"mode": "cartesian"}})
    grid = json.loads(ctx.layout.artifact(PERTURBATION_GRID).read_text())
    assert grid["mode"] == "cartesian"
    assert result.counts["perturbations"] == 432


def test_recipes_json_records_the_op_chain_and_the_contract_scaling(tmp_path, study):
    # Stage 4 needs `scaling` to restore contract form after a perturbation.
    ctx, _ = run_stage(tmp_path, study)
    recipes = json.loads(ctx.layout.artifact(RECIPES).read_text())["recipes"]
    assert len(recipes) == 2
    for entry in recipes:
        assert [op["name"] for op in entry["ops"]][0] == "scale_unit"
        assert [op["name"] for op in entry["ops"]][-1] == "normalize"
        assert entry["contract"]["scaling"] == "none"
        assert entry["model_ids"]
    shared = [entry for entry in recipes if len(entry["model_ids"]) == 2]
    assert shared and sorted(shared[0]["model_ids"]) == ["alpha", "bravo"]


# --------------------------------------------------------------------------
# wired into the runner
# --------------------------------------------------------------------------


def test_execute_writes_the_manifest_ledger_and_artifacts(tmp_path, study):
    stage = PreprocessStage()
    ctx = context(tmp_path, study["spec"], study["cohort"], stage=stage)
    summary = execute(stage, ctx)

    assert summary["status"] == "ok"
    manifest = RunManifest.read(ctx.layout.manifest_path)
    assert manifest.counts["in"] == 3
    assert manifest.counts["out"] == 3
    assert manifest.counts["excluded"] == 0
    assert sorted(ref.path for ref in manifest.output_artifacts) == [
        f"artifacts/{PERTURBATION_GRID}",
        f"artifacts/{PREPROCESS_INDEX}",
        f"artifacts/{RECIPES}",
    ]

    ledger = read_exclusions(exclusions_root(tmp_path / "runs", "s1"))
    assert len(ledger) == 0


def test_the_stage_registry_resolves_this_class():
    stage = get_stage("preprocess")
    assert isinstance(stage, PreprocessStage)
    assert stage.required_inputs() == ("cohort_index",)


def test_the_real_registry_compiles_without_importing_any_backend(tmp_path):
    registry = Path(__file__).resolve().parents[1] / "registry" / "models"
    source = write_record(tmp_path / "raw" / "r0.npz")
    cohort = write_cohort(
        tmp_path / "cohort_index.parquet",
        [
            {
                "image_occurrence_id": "img-0",
                "person_id": "p-0",
                "local_path": str(source),
                "label_primary": 1,
            }
        ],
    )
    stage = PreprocessStage()
    ctx = context(tmp_path, {"registry": str(registry)}, cohort, stage=stage)
    ctx.layout.create_dirs()
    result = stage.run(ctx)

    # Preprocess never touches weights, so it must run in an environment that
    # has neither torch nor TensorFlow.
    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules
    frame = read_table(ctx.layout.artifact(PREPROCESS_INDEX))
    assert set(frame["model_id"]) == {"ecgfounder", "prophecg-stemi"}
    assert result.counts["models"] == 2
    assert result.counts["recipes"] == 2  # 12-lead zscore vs 8-lead unscaled


def test_no_model_is_named_anywhere_in_this_stage_or_the_perturbation_library():
    # Claim C1: registering a model is adding one JSON card and nothing else.
    from mival.modelcard import load_registry

    registry = Path(__file__).resolve().parents[1] / "registry" / "models"
    model_ids = set(load_registry(registry))
    assert model_ids

    src = Path(__file__).resolve().parents[1] / "src" / "mival"
    for module in (src / "stages" / "preprocess.py", src / "perturbation.py"):
        text = module.read_text(encoding="utf-8")
        for model_id in model_ids:
            assert model_id not in text, f"{module.name} names the model {model_id}"
        assert "model_id ==" not in text


def test_a_registry_without_cards_fails_loudly(tmp_path, study):
    empty = tmp_path / "empty"
    empty.mkdir()
    stage = PreprocessStage()
    ctx = context(tmp_path, {"registry": str(empty)}, study["cohort"], stage=stage)
    ctx.layout.create_dirs()
    with pytest.raises(ValueError, match="no ModelCard"):
        stage.run(ctx)

    stage = PreprocessStage()
    ctx = context(tmp_path, {"registry": str(tmp_path / "nope")}, study["cohort"], stage=stage)
    ctx.layout.create_dirs()
    with pytest.raises(ValueError, match="not a directory"):
        stage.run(ctx)


def test_the_loader_is_injectable_because_dicom_reading_is_a_later_plan(tmp_path, study):
    calls = []

    def loader(path):
        calls.append(path)
        return load_npz_record(path)

    ctx, result = run_stage(tmp_path, study, stage=PreprocessStage(loader=loader))
    assert len(calls) == 3
    assert result.counts["in"] == 3


def test_loader_is_chosen_by_name_from_the_spec(tmp_path):
    stage = PreprocessStage()
    assert stage.resolve_loader({"loader": "npz"}) is load_npz_record
    with pytest.raises(ValueError, match="loader"):
        stage.resolve_loader({"loader": "xml"})
