"""Models stage (spec §4.4).

Everything here runs against a fake adapter. torch and TensorFlow are absent
from this machine by design (Plan 1's dual-environment constraint), and the
stage is supposed to be indifferent to which backend is behind the card — so a
fake adapter is not a compromise, it is the property under test.

The cohort fixture is small and hand-checkable. Dev scores and labels are:

    img  person  prob  label  fold
    0    p0      0.90    1      0
    1    p1      0.80    1      1
    2    p2      0.70    0      0
    3    p3      0.60    1      1
    4    p4      0.50    1      0
    5    p5      0.40    0      1
    6    p6      0.30    0      0
    7    p7      0.15    1      1

giving 5 events among 8 dev records, Youden's J maximal at 0.50 and the
sensitivity-95% point at 0.15. Test records (p8..p11) are deliberately scored
so that pooling them in would change the AUROC but must never reach a
threshold fit.
"""

import json
import sys

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from mival.adapters.base import Adapter  # noqa: E402
from mival.gates import LeakageError  # noqa: E402
from mival.pipeline.manifest import RunManifest  # noqa: E402
from mival.pipeline.runkey import RunKey  # noqa: E402
from mival.pipeline.stage import execute, prepare  # noqa: E402
from mival.pipeline.tables import read_table, write_table  # noqa: E402
from mival.stages import models as models_module  # noqa: E402
from mival.stages.models import (  # noqa: E402
    PREDICTION_COLUMNS,
    PRIMARY_THRESHOLD_POLICY,
    TRAINING_MODES,
    AdapterCapabilityError,
    ModelsStage,
    _TestSet,
)

DEV_ROWS = [
    ("img0", "p0", 0.90, 1, "dev", 0),
    ("img1", "p1", 0.80, 1, "dev", 1),
    ("img2", "p2", 0.70, 0, "dev", 0),
    ("img3", "p3", 0.60, 1, "dev", 1),
    ("img4", "p4", 0.50, 1, "dev", 0),
    ("img5", "p5", 0.40, 0, "dev", 1),
    ("img6", "p6", 0.30, 0, "dev", 0),
    ("img7", "p7", 0.15, 1, "dev", 1),
]
TEST_ROWS = [
    ("img8", "p8", 0.55, 1, "test", None),
    ("img9", "p9", 0.45, 0, "test", None),
    ("img10", "p10", 0.35, 1, "test", None),
    ("img11", "p11", 0.25, 0, "test", None),
]
ROWS = DEV_ROWS + TEST_ROWS

RECIPE_ID = "r1"
PERTURBATION_ID = "baseline"
CARD_THRESHOLD_VALUE = 0.42


# ---------------------------------------------------------------------------
# fake backend
# ---------------------------------------------------------------------------


class FakeHandle:
    def __init__(self, card, flip=False):
        self.card = card
        self.flip = flip


class FakeAdapter:
    """A backend whose output is a known function of the tensor it is given.

    Each fixture tensor is constant, so ``forward`` returns exactly the score
    the row table declares. That makes every probability in these tests
    hand-predictable without any model.
    """

    name = "fake"

    def __init__(self, groups=("stem", "trunk", "head")):
        self.groups = tuple(groups)
        self.loads = 0
        self.fits = []
        self.forward_calls = []  # one entry per batch, each a list of scores

    def load(self, card):
        self.loads += 1
        return FakeHandle(card)

    def forward(self, handle, batch):
        scores = np.asarray(batch, dtype=np.float64).mean(axis=(1, 2))
        self.forward_calls.append([round(float(score), 6) for score in scores])
        values = 1.0 - scores if handle.flip else scores
        return 100.0 * values if handle.card.output.get("type") == "regression" else values

    def features(self, handle, batch, index=0):
        return np.asarray(batch, dtype=np.float64).reshape(len(batch), -1)

    def trainable_groups(self, handle):
        return list(self.groups)

    def fit(self, handle, data, mode, hparams):
        validation = data["validation"]
        objective = data.get("objective", "bce")
        self.fits.append(
            {
                "mode": mode,
                "hparams": dict(hparams),
                "train_persons": list(data["person_id"]),
                "train_y": [float(value) if objective == "mse" else int(value) for value in data["y"]],
                "label_column": data["label_column"],
                "val_persons": None if validation is None else list(validation["person_id"]),
                "data": {"objective": objective},
            }
        )
        return FakeHandle(handle.card, flip=bool(hparams.get("flip")))

    @property
    def scores_seen(self):
        return [score for call in self.forward_calls for score in call]


class NoFitAdapter(FakeAdapter):
    """A backend that never grew a fit() — spec §4.4's interface, minus one."""

    fit = None


# ---------------------------------------------------------------------------
# fixture construction
# ---------------------------------------------------------------------------


def write_card(directory, model_id, modes, corpora=("toy-corpus",), threshold=None,
               adapter="fake", feature_layer="trunk", output=None):
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        "model_id": model_id,
        "Name": model_id.upper(),
        "x-mival": {
            "adapter": adapter,
            "weights": [
                {"uri": f"/nowhere/{model_id}.bin", "sha256": "ab" * 32, "role": "backbone"}
            ],
            "ensemble": {"method": "none", "members": 1},
            "input_contract": {
                "leads": ["I", "II"],
                "sampling_rate_hz": 500,
                "duration_s": 10,
                "unit": "mV",
                "scaling": "none",
                "layout": "lead_time",
                "dtype": "float32",
                "filters": [],
            },
            "output": {"type": "softmax", "positive_index": 1} if output is None else output,
            "feature_layer": feature_layer,
            "threshold": {} if threshold is None else threshold,
            "runtime": {"framework": "fake", "python": "3.9", "device": "cpu"},
            "training_modes_supported": list(modes),
            "pretraining_corpora": list(corpora),
        },
    }
    path = directory / f"{model_id}.json"
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def build_inputs(tmp_path, model_ids=("toy-a",), rows=ROWS, skip_tensor=(), split_rows=None,
                  label_value=None):
    """Write the preprocess_index, cohort_split and cohort_index this stage joins.

    ``label_value`` is a ``score -> float`` function for the regression label
    column; it defaults to ``100 * score`` so a regression arm has a target
    that is trivially related to the fake adapter's score.
    """
    label_value = label_value or (lambda score: 100.0 * score)
    tensors = tmp_path / "tensors"
    tensors.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for image_id, person_id, score, _label, _split, _fold in rows:
        tensor_path = tensors / f"{image_id}.npy"
        if image_id not in skip_tensor:
            np.save(tensor_path, np.full((2, 4), score, dtype=np.float64))
        for model_id in model_ids:
            index_rows.append(
                {
                    "image_occurrence_id": image_id,
                    "person_id": person_id,
                    "model_id": model_id,
                    "recipe_id": RECIPE_ID,
                    "perturbation_id": PERTURBATION_ID,
                    "tensor_path": str(tensor_path),
                    "n_leads": 2,
                    "n_samples": 4,
                    "sampling_rate_hz": 500.0,
                }
            )
    index_path = tmp_path / "preprocess_index.parquet"
    write_table(
        index_rows,
        index_path,
        (
            "image_occurrence_id",
            "person_id",
            "model_id",
            "recipe_id",
            "perturbation_id",
            "tensor_path",
            "n_leads",
            "n_samples",
            "sampling_rate_hz",
        ),
    )

    if split_rows is None:
        split_rows = [
            {"person_id": person_id, "split": split, "fold": fold}
            for _image, person_id, _score, _label, split, fold in rows
        ]
    split_path = tmp_path / "cohort_split.parquet"
    write_table(split_rows, split_path, ("person_id", "split", "fold"))

    cohort_path = tmp_path / "cohort_index.parquet"
    write_table(
        [
            {
                "image_occurrence_id": image_id,
                "person_id": person_id,
                "label_primary": label,
                "label_sens1": 1 - label,
                "label_sens2": label,
                "label_sens3": 0,
                "label_value": label_value(score),
            }
            for image_id, person_id, score, label, _split, _fold in rows
        ],
        cohort_path,
        ("image_occurrence_id", "person_id") + tuple(
            column for column in PREDICTION_COLUMNS if column.startswith("label_")
        ),
    )
    return {
        "preprocess_index": index_path,
        "cohort_split": split_path,
        "cohort_index": cohort_path,
    }


def arm(model_id="toy-a", training_mode="inference_only", **extra):
    body = {"model_id": model_id, "training_mode": training_mode}
    body.update(extra)
    return body


def make_spec(registry, arms, cohort_sources=("evaluation-corpus",), **extra):
    body = {
        "registry": str(registry),
        "cohort_sources": list(cohort_sources),
        "arms": list(arms),
        "batch_size": 3,
    }
    body.update(extra)
    return body


def make_context(tmp_path, spec, inputs, adapter, seed=7, site="mimic"):
    stage = ModelsStage(adapter_factory={"fake": adapter}.__getitem__)
    ctx = prepare(
        stage=stage,
        study_id="s1",
        site=site,
        spec=spec,
        inputs=inputs,
        runs_root=tmp_path / "runs",
        seed=seed,
    )
    ctx.layout.create_dirs()
    return stage, ctx


def run_stage(tmp_path, spec, inputs, adapter, **kwargs):
    stage, ctx = make_context(tmp_path, spec, inputs, adapter, **kwargs)
    return stage.run(ctx), ctx


def train_log(ctx):
    path = ctx.layout.artifact("train_log.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def simple_run(tmp_path, training_mode="inference_only", modes=None, adapter=None, **arm_kwargs):
    registry = tmp_path / "registry"
    write_card(
        registry,
        "toy-a",
        modes=modes or list(TRAINING_MODES),
        threshold={"value": CARD_THRESHOLD_VALUE, "provenance": "toy paper", "status": "legacy"},
    )
    inputs = build_inputs(tmp_path)
    adapter = adapter or FakeAdapter()
    spec = make_spec(registry, [arm(training_mode=training_mode, **arm_kwargs)])
    result, ctx = run_stage(tmp_path, spec, inputs, adapter)
    return result, ctx, adapter


# ---------------------------------------------------------------------------
# claim C1: a model is a card, not a code change
# ---------------------------------------------------------------------------


def test_importing_the_stage_loads_no_backend():
    # The torch environment has no TensorFlow and vice versa; the stage module
    # must import in both, so the backend import happens in get_adapter only.
    import mival.stages.models  # noqa: F401

    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules


OWNED_MODULES = ["mival/stages/models.py", "mival/threshold.py", "mival/gates.py"]


@pytest.mark.parametrize("module", OWNED_MODULES)
def test_no_model_is_named_and_nothing_branches_on_a_model_name(module):
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src" / module).read_text(encoding="utf-8")
    code = _executable_source(source)
    lowered = code.lower()
    # Prose may cite a model or a corpus as rationale; executable code may not.
    for name in ("ecgfounder", "prophecg", "net1d", "mimic", "harvard", "ptb"):
        assert name not in lowered, f"{module} names {name} in code"
    # Comparing two model_id *values* is a join; comparing one against a
    # literal would be the branch that falsifies C1.
    assert not re.search(r"""model_id\s*(==|!=|\sin\s)\s*[\[({'"]""", code)


def _executable_source(source):
    """The module's source with comments and string literals removed."""
    import io
    import tokenize

    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def test_two_models_run_side_by_side_from_two_cards(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"], threshold={"value": 0.42})
    write_card(registry, "toy-b", modes=["inference_only"], corpora=["other-corpus"])
    inputs = build_inputs(tmp_path, model_ids=("toy-a", "toy-b"))
    adapter = FakeAdapter()
    spec = make_spec(
        registry,
        [arm("toy-a", "inference_only"), arm("toy-b", "inference_only")],
    )
    result, ctx = run_stage(tmp_path, spec, inputs, adapter)

    logged = {entry["model_id"] for entry in train_log(ctx)}
    assert logged == {"toy-a", "toy-b"}
    # toy-b's card publishes no threshold, so it simply has no legacy policy —
    # no code in the stage knows the difference between the two models.
    by_model = {entry["model_id"]: entry for entry in train_log(ctx)}
    assert sorted(by_model["toy-a"]["thresholds"]) == ["legacy", "refit_sens95", "refit_youden"]
    assert sorted(by_model["toy-b"]["thresholds"]) == ["refit_sens95", "refit_youden"]
    assert result.counts["out"] == 2 * len(ROWS)


def test_the_adapter_is_looked_up_by_the_name_on_the_card(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"], adapter="not-registered")
    inputs = build_inputs(tmp_path)
    spec = make_spec(registry, [arm()])
    stage, ctx = make_context(tmp_path, spec, inputs, FakeAdapter())
    with pytest.raises(KeyError, match="not-registered"):
        stage.run(ctx)


# ---------------------------------------------------------------------------
# training modes (spec §4.4 table)
# ---------------------------------------------------------------------------


def test_inference_only_fits_nothing_and_records_the_weights_checksum(tmp_path):
    result, ctx, adapter = simple_run(tmp_path, "inference_only")
    assert adapter.fits == []
    entry = train_log(ctx)[0]
    assert entry["training"]["mode"] == "inference_only"
    assert [weight["sha256"] for weight in entry["training"]["weights"]] == ["ab" * 32]
    assert "internal_cv" not in entry["training"]
    # dev is not trained on, but it is still scored: the refit policies need it.
    assert result.counts["out"] == len(ROWS)


def test_linear_probe_fits_and_records_the_feature_layer_and_head_hparams(tmp_path):
    _result, ctx, adapter = simple_run(
        tmp_path, "linear_probe", hparams={"lr": 0.01, "epochs": 3}
    )
    assert [fit["mode"] for fit in adapter.fits] == ["linear_probe"] * 3  # 2 CV folds + final
    entry = train_log(ctx)[0]
    assert entry["training"]["feature_layer"] == "trunk"
    assert entry["training"]["hparams"]["lr"] == 0.01
    assert entry["training"]["hparams"]["epochs"] == 3
    # The seed travels with the hyperparameters: the adapter owns the fit and
    # therefore owns its randomness.
    assert entry["training"]["hparams"]["seed"] == 7


def test_partial_unfreeze_records_the_unfrozen_group_list(tmp_path):
    _result, ctx, adapter = simple_run(
        tmp_path, "partial_unfreeze", unfreeze_groups=["trunk", "head"]
    )
    entry = train_log(ctx)[0]
    assert entry["training"]["unfreeze_groups"] == ["trunk", "head"]
    assert adapter.fits[0]["hparams"]["unfreeze_groups"] == ["trunk", "head"]


def test_partial_unfreeze_without_a_group_list_is_refused(tmp_path):
    with pytest.raises(ValueError, match="requires unfreeze_groups"):
        simple_run(tmp_path, "partial_unfreeze")


def test_partial_unfreeze_checks_the_groups_against_the_adapter(tmp_path):
    with pytest.raises(ValueError, match="not trainable groups"):
        simple_run(tmp_path, "partial_unfreeze", unfreeze_groups=["no_such_block"])


def test_full_finetune_records_the_whole_hyperparameter_set(tmp_path):
    _result, ctx, _adapter = simple_run(
        tmp_path, "full_finetune", hparams={"lr": 0.0003, "epochs": 10, "weight_decay": 0.01}
    )
    hparams = train_log(ctx)[0]["training"]["hparams"]
    assert hparams["lr"] == 0.0003
    assert hparams["epochs"] == 10
    assert hparams["weight_decay"] == 0.01


def test_a_mode_the_card_does_not_support_is_refused_before_anything_runs(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])
    inputs = build_inputs(tmp_path)
    adapter = FakeAdapter()
    spec = make_spec(registry, [arm(training_mode="full_finetune")])
    stage, ctx = make_context(tmp_path, spec, inputs, adapter)
    with pytest.raises(ValueError, match="training_modes_supported"):
        stage.run(ctx)
    assert adapter.loads == 0


def test_an_unknown_training_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown training_mode"):
        simple_run(tmp_path, "prompt_tuning")


def test_an_adapter_without_fit_says_so_instead_of_failing_obscurely(tmp_path):
    with pytest.raises(AdapterCapabilityError, match="does not implement fit"):
        simple_run(tmp_path, "linear_probe", adapter=NoFitAdapter())


# ---------------------------------------------------------------------------
# internal CV: selection happens inside dev
# ---------------------------------------------------------------------------


def test_hyperparameters_are_selected_by_dev_auroc_out_of_fold(tmp_path):
    # The 'flip' candidate inverts the ranking, so dev AUROC must reject it.
    _result, ctx, adapter = simple_run(
        tmp_path, "linear_probe", hparam_grid={"flip": [False, True]}
    )
    entry = train_log(ctx)[0]["training"]["internal_cv"]
    assert entry["selection_split"] == "dev"
    assert entry["selection_metric"] == "auroc"
    scores = {candidate["hparams"]["flip"]: candidate["auroc"] for candidate in entry["candidates"]}
    # dev concordance: 10 of 15 pairs with the identity transform.
    assert scores[False] == pytest.approx(10 / 15)
    assert scores[True] == pytest.approx(5 / 15)
    assert entry["selected"]["flip"] is False
    # 2 candidates x 2 folds, then one final fit.
    assert len(adapter.fits) == 5


def test_every_cv_fit_is_person_disjoint_from_the_fold_it_predicts(tmp_path):
    _result, _ctx, adapter = simple_run(tmp_path, "linear_probe")
    for fit in adapter.fits:
        if fit["val_persons"] is None:
            continue
        assert not set(fit["train_persons"]) & set(fit["val_persons"])


def test_the_final_fit_reserves_a_dev_fold_for_early_stopping(tmp_path):
    _result, ctx, adapter = simple_run(tmp_path, "linear_probe")
    internal = train_log(ctx)[0]["training"]["internal_cv"]
    assert internal["folds"] == [0, 1]
    assert internal["early_stopping_fold"] == 1
    final = adapter.fits[-1]
    assert set(final["train_persons"]) == {"p0", "p2", "p4", "p6"}
    assert set(final["val_persons"]) == {"p1", "p3", "p5", "p7"}
    # Early stopping never sees test.
    assert not {"p8", "p9", "p10", "p11"} & set(final["val_persons"])


def test_early_stopping_fold_null_fits_the_final_model_on_all_of_dev(tmp_path):
    _result, ctx, adapter = simple_run(tmp_path, "linear_probe", early_stopping_fold=None)
    assert train_log(ctx)[0]["training"]["internal_cv"]["early_stopping_fold"] is None
    final = adapter.fits[-1]
    assert len(final["train_persons"]) == len(DEV_ROWS)
    assert final["val_persons"] is None


def test_a_trained_mode_needs_the_dev_folds_frozen_upstream(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=list(TRAINING_MODES))
    rows = [row[:5] + (None,) if row[4] == "dev" else row for row in ROWS]
    inputs = build_inputs(tmp_path, rows=rows)
    spec = make_spec(registry, [arm(training_mode="linear_probe")])
    stage, ctx = make_context(tmp_path, spec, inputs, FakeAdapter())
    with pytest.raises(ValueError, match="k-fold assignment"):
        stage.run(ctx)


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def test_leakage_gate_stops_the_run_when_a_person_is_in_dev_and_test(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])
    # p0 already holds a dev assignment; the split table also puts it in test.
    # Splits are person-level and frozen upstream (spec §2.4), so this is the
    # only shape the defect can take — and the run must stop, not drop a row.
    split_rows = [
        {"person_id": person, "split": split, "fold": fold}
        for _image, person, _score, _label, split, fold in ROWS
    ] + [{"person_id": "p0", "split": "test", "fold": None}]
    inputs = build_inputs(tmp_path, split_rows=split_rows)
    spec = make_spec(registry, [arm()])
    stage, ctx = make_context(tmp_path, spec, inputs, FakeAdapter())
    with pytest.raises(LeakageError, match="p0"):
        stage.run(ctx)
    assert not list(ctx.layout.artifact("predictions").glob("*.parquet"))


def test_the_leakage_gate_is_re_checked_on_the_records_reaching_each_run(tmp_path):
    """The gate is applied three times, and the last two can only ever pass.

    Once on ``cohort_split`` (the test above, where it can fail), then on the
    records of each run, then on each inner CV fold. Because the split is
    person-keyed, the latter two cannot fail once the first has passed — they
    are assertions that the join did not corrupt the split, not additional
    filters. Keeping them costs nothing and makes the invariant local to the
    code that depends on it.
    """
    import inspect

    from mival.stages import models as module

    source = inspect.getsource(module.ModelsStage)
    assert source.count("check_leakage(") == 3


def test_a_dev_record_and_a_test_record_never_share_a_person(tmp_path):
    from mival.stages.models import ModelsSpec

    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])
    inputs = build_inputs(tmp_path)
    spec = make_spec(registry, [arm()])
    stage, ctx = make_context(tmp_path, spec, inputs, FakeAdapter())
    records, _n_in = stage._collect(ctx, {"toy-a"}, ModelsSpec.from_mapping(spec))
    dev_persons = {record.person_id for record in records if record.split == "dev"}
    test_persons = {record.person_id for record in records if record.split == "test"}
    assert dev_persons and test_persons
    assert not dev_persons & test_persons


def test_contamination_marks_the_run_and_lets_it_finish(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"], corpora=["evaluation-corpus"])
    inputs = build_inputs(tmp_path)
    spec = make_spec(registry, [arm()], cohort_sources=["evaluation-corpus"])
    result, ctx = run_stage(tmp_path, spec, inputs, FakeAdapter())

    assert result.contamination == {"flag": True, "overlapping_corpora": ["evaluation-corpus"]}
    assert any("contaminated" in warning for warning in result.warnings)
    # Not stopping is the whole point: the predictions still exist.
    assert result.counts["out"] == len(ROWS)
    assert train_log(ctx)[0]["contamination"]["flag"] is True


def test_an_uncontaminated_run_is_not_marked(tmp_path):
    result, _ctx, _adapter = simple_run(tmp_path)
    assert result.contamination == {"flag": False, "overlapping_corpora": []}


def test_a_study_without_declared_sources_is_warned_that_the_gate_is_blind(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])
    inputs = build_inputs(tmp_path)
    spec = make_spec(registry, [arm()], cohort_sources=[])
    result, _ctx = run_stage(tmp_path, spec, inputs, FakeAdapter())
    assert any("not checked" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------


def test_all_three_policies_are_reported_with_the_hand_computed_values(tmp_path):
    _result, ctx, _adapter = simple_run(tmp_path)
    thresholds = train_log(ctx)[0]["thresholds"]
    assert sorted(thresholds) == ["legacy", "refit_sens95", "refit_youden"]
    assert thresholds["legacy"]["value"] == pytest.approx(CARD_THRESHOLD_VALUE)
    assert thresholds["refit_youden"]["value"] == pytest.approx(0.50)
    assert thresholds["refit_youden"]["youden_j"] == pytest.approx(0.8 - 1 / 3)
    assert thresholds["refit_sens95"]["value"] == pytest.approx(0.15)
    assert thresholds["refit_sens95"]["sensitivity"] == pytest.approx(1.0)


def test_the_primary_policy_is_refit_sens95(tmp_path):
    _result, ctx, _adapter = simple_run(tmp_path)
    entry = train_log(ctx)[0]
    assert entry["primary_threshold_policy"] == PRIMARY_THRESHOLD_POLICY == "refit_sens95"
    assert entry["primary_threshold_policy"] in entry["thresholds"]


def test_thresholds_are_estimated_on_dev_and_only_on_dev(tmp_path):
    _result, ctx, _adapter = simple_run(tmp_path)
    thresholds = train_log(ctx)[0]["thresholds"]
    for policy in ("refit_youden", "refit_sens95"):
        assert thresholds[policy]["fitted_on"] == "dev"
        # 8 dev records with 5 events; the 4 test records are not in the fit.
        assert thresholds[policy]["n"] == len(DEV_ROWS)
        assert thresholds[policy]["n_events"] == 5
    assert thresholds["legacy"]["fitted_on"] == "model_card"


def test_the_legacy_value_comes_from_the_card_not_from_the_data(tmp_path):
    registry = tmp_path / "registry"
    write_card(
        registry, "toy-a", modes=["inference_only"], threshold={"value": 0.123, "status": "legacy"}
    )
    inputs = build_inputs(tmp_path)
    spec = make_spec(registry, [arm()])
    _result, ctx = run_stage(tmp_path, spec, inputs, FakeAdapter())
    assert train_log(ctx)[0]["thresholds"]["legacy"]["value"] == pytest.approx(0.123)


# ---------------------------------------------------------------------------
# the test split is touched once
# ---------------------------------------------------------------------------


def test_the_test_split_is_handed_out_exactly_once(tmp_path):
    result, ctx, adapter = simple_run(tmp_path)
    assert result.counts["test_contacts"] == 1
    assert train_log(ctx)[0]["test"] == {"contacts": 1, "n": len(TEST_ROWS)}
    # Each test score is forwarded exactly once across the whole run.
    for _image, _person, score, _label, split, _fold in TEST_ROWS:
        assert adapter.scores_seen.count(round(score, 6)) == 1


def test_a_second_contact_with_the_test_split_is_impossible_not_merely_counted():
    held = _TestSet([], "arm toy/inference_only")
    assert held.take() == ()
    with pytest.raises(models_module.TestSetContactError, match="already been used once"):
        held.take()


# ---------------------------------------------------------------------------
# output schema
# ---------------------------------------------------------------------------


def test_prediction_files_are_named_by_run_key_and_carry_exactly_the_schema(tmp_path):
    result, ctx, _adapter = simple_run(tmp_path)
    expected = {
        RunKey(
            site="mimic",
            model_id="toy-a",
            training_mode="inference_only",
            recipe_id=RECIPE_ID,
            perturbation_id=PERTURBATION_ID,
            label_def="primary",
            split=split,
            fold=fold,
        )
        for split, fold in [("dev", 0), ("dev", 1), ("test", None)]
    }
    assert set(result.run_keys) == expected
    for key in expected:
        path = ctx.layout.artifact("predictions", key.to_string() + ".parquet")
        assert path.is_file()
        frame = read_table(path)
        assert tuple(frame.columns) == PREDICTION_COLUMNS
        assert set(frame["split"]) == {key.split}
        assert set(frame["model_id"]) == {"toy-a"}
        assert set(frame["site"]) == {"mimic"}
        assert set(frame["seed"]) == {7}


def test_probabilities_and_labels_land_on_the_right_records(tmp_path):
    _result, ctx, _adapter = simple_run(tmp_path)
    key = RunKey(
        site="mimic",
        model_id="toy-a",
        training_mode="inference_only",
        recipe_id=RECIPE_ID,
        perturbation_id=PERTURBATION_ID,
        label_def="primary",
        split="test",
        fold=None,
    )
    frame = read_table(ctx.layout.artifact("predictions", key.to_string() + ".parquet"))
    by_image = frame.set_index("image_occurrence_id")
    for image_id, person_id, score, label, _split, _fold in TEST_ROWS:
        assert by_image.loc[image_id, "prob"] == pytest.approx(score)
        assert by_image.loc[image_id, "person_id"] == person_id
        assert by_image.loc[image_id, "label_primary"] == label
        # label_sens1 is the complement in the fixture: the columns are carried
        # independently rather than copied from the primary label.
        assert by_image.loc[image_id, "label_sens1"] == 1 - label
        assert by_image.loc[image_id, "logit"] == pytest.approx(
            np.log(score) - np.log1p(-score)
        )


def test_dev_predictions_are_split_by_fold_because_fold_is_a_run_key_axis(tmp_path):
    _result, ctx, _adapter = simple_run(tmp_path)
    for fold, persons in [(0, {"p0", "p2", "p4", "p6"}), (1, {"p1", "p3", "p5", "p7"})]:
        key = RunKey(
            site="mimic",
            model_id="toy-a",
            training_mode="inference_only",
            recipe_id=RECIPE_ID,
            perturbation_id=PERTURBATION_ID,
            label_def="primary",
            split="dev",
            fold=fold,
        )
        frame = read_table(ctx.layout.artifact("predictions", key.to_string() + ".parquet"))
        assert set(frame["person_id"]) == persons
        assert set(frame["fold"]) == {fold}


def test_label_def_is_a_run_key_axis_and_selects_the_training_label(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])
    inputs = build_inputs(tmp_path)
    spec = make_spec(registry, [arm(label_def="sens1")])
    result, ctx = run_stage(tmp_path, spec, inputs, FakeAdapter())
    assert {key.label_def for key in result.run_keys} == {"sens1"}
    # label_sens1 is the complement of label_primary in the fixture, so the
    # threshold refit lands somewhere else entirely.
    assert train_log(ctx)[0]["thresholds"]["refit_sens95"]["value"] != pytest.approx(0.15)


def test_an_unknown_label_def_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="names no label column"):
        simple_run(tmp_path, label_def="mortality")


# ---------------------------------------------------------------------------
# exclusion ledger (spec §3.6)
# ---------------------------------------------------------------------------


def test_a_person_absent_from_the_split_is_excluded_as_split_unassigned(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])
    split_rows = [
        {"person_id": person, "split": split, "fold": fold}
        for _image, person, _score, _label, split, fold in ROWS
        if person != "p9"
    ]
    inputs = build_inputs(tmp_path, split_rows=split_rows)
    spec = make_spec(registry, [arm()])
    result, ctx = run_stage(tmp_path, spec, inputs, FakeAdapter())
    assert ctx.ledger.counts() == {"split_unassigned": 1}
    assert result.counts["out"] == len(ROWS) - 1


def test_a_record_without_the_arms_label_is_excluded_as_label_missing(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])
    inputs = build_inputs(tmp_path)
    frame = read_table(inputs["cohort_index"])
    frame.loc[frame["image_occurrence_id"] == "img9", "label_primary"] = None
    frame.to_parquet(inputs["cohort_index"], index=False)
    spec = make_spec(registry, [arm()])
    result, ctx = run_stage(tmp_path, spec, inputs, FakeAdapter())
    assert ctx.ledger.counts() == {"label_missing": 1}
    assert result.counts["out"] == len(ROWS) - 1


def test_a_missing_tensor_file_is_excluded_as_tensor_missing(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])
    inputs = build_inputs(tmp_path, skip_tensor={"img9"})
    spec = make_spec(registry, [arm()])
    result, ctx = run_stage(tmp_path, spec, inputs, FakeAdapter())
    assert ctx.ledger.counts() == {"tensor_missing": 1}
    assert result.counts["out"] == len(ROWS) - 1


def test_labels_are_required_and_the_error_says_how_to_supply_them(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])
    inputs = build_inputs(tmp_path)
    inputs.pop("cohort_index")
    spec = make_spec(registry, [arm()])
    stage, ctx = make_context(tmp_path, spec, inputs, FakeAdapter())
    with pytest.raises(KeyError, match="cohort_index"):
        stage.run(ctx)


# ---------------------------------------------------------------------------
# spec validation and wrapper integration
# ---------------------------------------------------------------------------


def test_a_model_without_a_card_is_named_in_the_error(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])
    inputs = build_inputs(tmp_path, model_ids=("toy-a",))
    spec = make_spec(registry, [arm("toy-z")])
    stage, ctx = make_context(tmp_path, spec, inputs, FakeAdapter())
    with pytest.raises(ValueError, match="no ModelCard for 'toy-z'"):
        stage.run(ctx)


def test_two_identical_arms_are_rejected_rather_than_overwriting_one_run_key(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])
    inputs = build_inputs(tmp_path)
    spec = make_spec(registry, [arm(), arm()])
    stage, ctx = make_context(tmp_path, spec, inputs, FakeAdapter())
    with pytest.raises(ValueError, match="same run_key twice"):
        stage.run(ctx)


def test_the_registry_path_is_required(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])
    inputs = build_inputs(tmp_path)
    stage, ctx = make_context(tmp_path, {"arms": [arm()]}, inputs, FakeAdapter())
    with pytest.raises(ValueError, match="models.registry is required"):
        stage.run(ctx)


def test_a_grid_axis_must_be_a_list_of_candidates(tmp_path):
    with pytest.raises(ValueError, match="must be a list of candidate values"):
        simple_run(tmp_path, "linear_probe", hparam_grid={"lr": 0.01})


def test_the_wrapper_writes_the_contamination_block_and_the_test_contact_count(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"], corpora=["evaluation-corpus"])
    inputs = build_inputs(tmp_path)
    spec = make_spec(registry, [arm()], cohort_sources=["evaluation-corpus"])
    stage, ctx = make_context(tmp_path, spec, inputs, FakeAdapter())
    summary = execute(stage, ctx)

    assert summary["status"] == "ok"
    manifest = RunManifest.read(ctx.layout.manifest_path)
    assert manifest.contamination == {"flag": True, "overlapping_corpora": ["evaluation-corpus"]}
    # Spec §4.4: the fact that test was contacted is itself recorded.
    assert manifest.counts["test_contacts"] == 1
    assert manifest.counts["in"] == len(ROWS)
    assert manifest.counts["out"] == len(ROWS)
    written = {ref.path for ref in manifest.output_artifacts}
    assert "artifacts/train_log.jsonl" in written
    assert sum(path.startswith("artifacts/predictions/") for path in written) == 3


# ---------------------------------------------------------------------------
# the real Adapter.fit, driven by the stage
# ---------------------------------------------------------------------------


class ProbeHandle(FakeHandle):
    """What a real adapter's handle looks like once a head has been fitted."""

    def __init__(self, card, head=None, fit_record=None):
        super().__init__(card)
        self.head = head
        self.fit_record = fit_record


class ProbeAdapter(Adapter):
    """A backend that inherits ``fit`` instead of faking it.

    Every other test in this file supplies its own ``fit`` so that it can
    assert on exactly what the stage handed the adapter. This one does the
    opposite: it exercises the real ``linear_probe`` implementation end to end,
    so the two halves of spec §4.4 — what the stage passes and what an adapter
    does with it — are known to fit together.
    """

    name = "fake"

    def load(self, card):
        return ProbeHandle(card)

    def features(self, handle, batch, index=0):
        return np.asarray(batch, dtype=np.float64).reshape(len(batch), -1)

    def forward(self, handle, batch):
        if handle.head is None:
            # Stands in for a checkpoint that publishes its own head, which is
            # what `inference_only` scores through.
            return np.clip(np.asarray(batch, dtype=np.float64).mean(axis=(1, 2)), 0.0, 1.0)
        return handle.head.probabilities(self.features(handle, batch))

    def with_head(self, handle, head, record):
        return ProbeHandle(handle.card, head=head, fit_record=dict(record))


def test_a_linear_probe_arm_runs_the_real_fit(tmp_path):
    result, ctx, _adapter = simple_run(
        tmp_path, "linear_probe", adapter=ProbeAdapter(), hparams={"epochs": 40}
    )
    assert result.counts["out"] == len(ROWS)
    probs = read_table(ctx.layout.artifact("predictions"))["prob"]
    assert probs.between(0.0, 1.0).all()


def test_the_train_log_records_what_the_adapter_actually_did(tmp_path):
    """Spec §4.4 requires the full hyperparameters on record.

    ``hparams`` alone holds only what the study asked for; the epochs the fit
    stopped at and the defaults the adapter filled in live in ``fit_record``.
    """
    _result, ctx, _adapter = simple_run(
        tmp_path, "linear_probe", adapter=ProbeAdapter(), hparams={"epochs": 40}
    )
    training = train_log(ctx)[0]["training"]
    record = training["fit_record"]
    assert record["mode"] == "linear_probe"
    assert record["hparams"]["epochs"] == 40
    assert record["hparams"]["pos_weight"] == 1.0
    assert record["n_train"] == 4  # dev fold 0; fold 1 is reserved for stopping
    assert record["n_validation"] == 4
    assert record["best_epoch"] is not None


def test_the_head_is_refit_per_inner_cv_fold(tmp_path):
    """Each fold's held-out probabilities must come from a fold-specific head.

    A fit that mutated the loaded handle instead of returning a new one would
    leak fold 0's head into fold 1's estimate, and the out-of-fold dev
    probabilities that the threshold refit depends on would no longer be out of
    fold.
    """
    _result, ctx, _adapter = simple_run(
        tmp_path, "linear_probe", adapter=ProbeAdapter(), hparams={"epochs": 40}
    )
    internal = train_log(ctx)[0]["training"]["internal_cv"]
    assert internal["folds"] == [0, 1]
    assert internal["early_stopping_fold"] == 1
    assert internal["final_fit_n"] == 4


# ---------------------------------------------------------------------------
# Fitted heads survive the run (decisions: Models 4)
# ---------------------------------------------------------------------------


def test_a_fitted_head_is_written_beside_the_predictions(tmp_path):
    """A head that lives only in memory cannot be re-applied or interpreted."""
    from mival.stages.models import HEADS_DIR

    _result, ctx, _adapter = simple_run(
        tmp_path, "linear_probe", adapter=ProbeAdapter(), hparams={"epochs": 40}
    )
    entry = train_log(ctx)[0]
    assert entry["fitted_head"], "linear_probe must record where its head was written"
    path = ctx.layout.run_dir / entry["fitted_head"]
    assert path.is_file()
    assert HEADS_DIR in str(path)

    stored = np.load(path)
    assert {"head0.weights", "head0.bias", "head0.mean", "head0.scale"} <= set(stored.files)


def test_an_inference_only_arm_writes_no_head(tmp_path):
    _result, ctx, _adapter = simple_run(
        tmp_path, "inference_only", adapter=ProbeAdapter()
    )
    assert train_log(ctx)[0]["fitted_head"] is None


# ---------------------------------------------------------------------------
# regression arms (label_def=value, spec addendum: LVEF)
# ---------------------------------------------------------------------------

REGRESSION_OUTPUT = {"type": "regression", "n_outputs": 1, "value_index": 0}


def test_regression_arm_writes_values_and_no_thresholds(tmp_path):
    """label_def=value: pred_value filled, prob and logit null, no threshold fit."""
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"], output=REGRESSION_OUTPUT)
    inputs = build_inputs(tmp_path, label_value=lambda score: 100.0 * score)
    spec = make_spec(registry, [arm(label_def="value")])
    _result, ctx = run_stage(tmp_path, spec, inputs, FakeAdapter())

    files = sorted(ctx.layout.artifact("predictions").glob("*.parquet"))
    assert files and all("label_def=value" in f.name for f in files)
    frame = read_table(files[0])
    assert frame["prob"].isna().all() and frame["logit"].isna().all()
    assert frame["pred_value"].notna().all()
    assert list(frame.columns) == list(PREDICTION_COLUMNS)
    entry = train_log(ctx)[0]
    assert entry["thresholds"] == {} and entry["primary_threshold_policy"] is None


def test_regression_arm_on_a_classification_card_is_rejected(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])  # softmax card
    inputs = build_inputs(tmp_path)
    spec = make_spec(registry, [arm(label_def="value")])
    stage, ctx = make_context(tmp_path, spec, inputs, FakeAdapter())
    with pytest.raises(ValueError, match="output.type"):
        stage.run(ctx)


def test_linear_probe_regression_selects_by_neg_mae(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["linear_probe"], output=REGRESSION_OUTPUT)
    inputs = build_inputs(tmp_path, label_value=lambda score: 100.0 * score)
    adapter = FakeAdapter()
    spec = make_spec(
        registry,
        [
            arm(
                training_mode="linear_probe",
                label_def="value",
                hparam_grid={"lr": [0.1, 0.01]},
            )
        ],
    )
    _result, ctx = run_stage(tmp_path, spec, inputs, adapter)
    assert all(fit["data"]["objective"] == "mse" for fit in adapter.fits)
    entry = train_log(ctx)[0]
    assert entry["training"]["internal_cv"]["selection_metric"] == "neg_mae"
    assert all("neg_mae" in c for c in entry["training"]["internal_cv"]["candidates"])
