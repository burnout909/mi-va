"""The misclassification stage end to end (spec §4.6).

The properties under test are the ones that make this a query engine rather
than a comparison module: a selector never names an axis, the same `contrast`
code path serves `model_id` and `label_def`, an ambiguous pattern is refused
instead of guessed at, `case_id` survives a change of selector parameters, and
no human review is ever silently dropped.
"""

import csv
import json
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pandas as pd

from mival.pipeline.runkey import AXES, RunKey
from mival.pipeline.stage import prepare
from mival.stages.misclassify import (
    CASE_COLUMNS,
    CASES,
    DEFAULT_VERDICTS,
    REVIEW_TEMPLATE,
    REVIEW_UNMATCHED,
    SELECTOR_LOG,
    UNREVIEWED,
    MisclassifySpec,
    MisclassifyStage,
    SelectorError,
)

N_RECORDS = 30
MODELS = ("modela", "modelb")
PERTURBATIONS = ("baseline", "downsample125")
LABEL_DEFS = ("primary", "sens1")


def _labels(index: int):
    """`sens1` disagrees with `primary` on two records — label noise, on purpose."""
    primary = 1 if index < 10 else 0
    sens1 = 1 if index < 12 else 0
    return {
        "label_primary": primary,
        "label_sens1": sens1,
        "label_sens2": primary,
        "label_sens3": primary,
    }


def _prob(model: str, perturbation: str, index: int) -> float:
    """Deterministic scores. modelb is modela pushed across the middle."""
    base = 0.92 - 0.055 * index if index < 10 else 0.06 + 0.030 * (index - 10)
    if perturbation != "baseline":
        base = 0.5 + 0.6 * (base - 0.5)
    if model == "modelb":
        base = base + (0.55 if index % 2 == 0 else -0.35)
    return float(min(max(base, 0.01), 0.99))


def _split(index: int) -> str:
    return "test" if index % 4 == 0 else "dev"


def build_predictions(directory: Path, seeds=(1,)):
    """One parquet per run_key, over model x perturbation x label_def x split."""
    directory.mkdir(parents=True, exist_ok=True)
    import numpy as np

    for model in MODELS:
        for perturbation in PERTURBATIONS:
            for label_def in LABEL_DEFS:
                for split in ("dev", "test"):
                    key = RunKey(
                        site="mimic",
                        model_id=model,
                        training_mode="inference_only",
                        recipe_id="r1",
                        perturbation_id=perturbation,
                        label_def=label_def,
                        split=split,
                        fold=None,
                    )
                    rows = []
                    for index in range(N_RECORDS):
                        if _split(index) != split:
                            continue
                        prob = _prob(model, perturbation, index)
                        for seed in seeds:
                            rows.append(
                                {
                                    "image_occurrence_id": f"r{index:02d}",
                                    "person_id": f"p{index:02d}",
                                    "split": split,
                                    "fold": None,
                                    **_labels(index),
                                    "prob": prob if seed == 1 else min(prob + 0.4, 0.99),
                                    "logit": float(np.log(prob / (1 - prob))),
                                    "model_id": model,
                                    "training_mode": "inference_only",
                                    "recipe_id": "r1",
                                    "perturbation_id": perturbation,
                                    "seed": seed,
                                    "site": "mimic",
                                }
                            )
                    pd.DataFrame(rows).to_parquet(
                        directory / f"{key.to_string()}.parquet", index=False
                    )
    return directory


REFERENCE = {
    "site": "mimic",
    "model_id": "modela",
    "training_mode": "inference_only",
    "recipe_id": "r1",
    "perturbation_id": "baseline",
    "label_def": "primary",
}


def run_stage(tmp_path: Path, spec, inputs=None, subdirectory="case", seeds=(1,)):
    root = tmp_path / subdirectory
    predictions = build_predictions(root / "predictions", seeds=seeds)
    stage = MisclassifyStage()
    ctx = prepare(
        stage=stage,
        study_id="test-study",
        site="mimic",
        spec=spec,
        inputs=dict(inputs or {}),
        runs_root=root / "runs",
        seed=1,
    )
    ctx.inputs = dict(ctx.inputs)
    ctx.inputs["predictions"] = predictions
    ctx.layout.create_dirs()
    result = stage.run(ctx)
    cases = pd.read_parquet(ctx.layout.artifact(CASES))
    return ctx, result, cases


ERROR_SPEC = {
    "figures": False,
    "selectors": [{"name": "errors", "type": "error", "pattern": REFERENCE, "k": 3}],
}


# ---------------------------------------------------------------------------
# Schema and the review loop
# ---------------------------------------------------------------------------


def test_the_case_columns_are_the_guaranteed_prefix(tmp_path):
    _, _, cases = run_stage(tmp_path, ERROR_SPEC)
    assert tuple(cases.columns)[: len(CASE_COLUMNS)] == CASE_COLUMNS


def test_every_run_key_axis_is_a_case_column(tmp_path):
    # A case must be locatable in run_key space, or the audit cannot say which
    # run produced the failure it is describing.
    assert all(axis in CASE_COLUMNS for axis in AXES)


def test_the_case_id_is_readable_and_names_its_selector(tmp_path):
    _, _, cases = run_stage(tmp_path, ERROR_SPEC)
    assert all(case_id.startswith("errors~") for case_id in cases["case_id"])
    assert all(
        case_id.split("~", 1)[1] == image
        for case_id, image in zip(cases["case_id"], cases["image_occurrence_id"])
    )


def test_the_case_id_survives_a_change_of_selector_parameters(tmp_path):
    """Tightening `k` must not orphan the review already done on surviving cases."""
    _, _, wide = run_stage(tmp_path, ERROR_SPEC, subdirectory="wide")
    narrow_spec = {
        "figures": False,
        "selectors": [{"name": "errors", "type": "error", "pattern": REFERENCE, "k": 1}],
    }
    _, _, narrow = run_stage(tmp_path, narrow_spec, subdirectory="narrow")
    assert set(narrow["case_id"]) < set(wide["case_id"])


def test_a_review_file_is_merged_by_case_id(tmp_path):
    ctx, _, cases = run_stage(tmp_path, ERROR_SPEC, subdirectory="first")
    reviewed = sorted(cases["case_id"])[0]
    review = tmp_path / "review.csv"
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "verdict", "note"])
        writer.writeheader()
        writer.writerow({"case_id": reviewed, "verdict": "stemi_mimic", "note": "LBBB"})

    _, result, merged = run_stage(
        tmp_path, ERROR_SPEC, inputs={"review": review}, subdirectory="second"
    )
    row = merged[merged["case_id"] == reviewed].iloc[0]
    assert row["verdict"] == "stemi_mimic"
    assert row["note"] == "LBBB"
    assert set(merged[merged["case_id"] != reviewed]["verdict"]) == {UNREVIEWED}
    assert not result.warnings or all("not in this run" not in w for w in result.warnings)


def test_review_of_a_case_that_no_longer_exists_is_preserved_not_dropped(tmp_path):
    review = tmp_path / "review.csv"
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "verdict", "note"])
        writer.writeheader()
        writer.writerow({"case_id": "errors~does-not-exist", "verdict": "label_error", "note": "x"})

    ctx, result, _ = run_stage(tmp_path, ERROR_SPEC, inputs={"review": review})
    unmatched = ctx.layout.artifact(REVIEW_UNMATCHED)
    assert unmatched.exists()
    kept = list(csv.DictReader(unmatched.open(encoding="utf-8")))
    assert kept == [{"case_id": "errors~does-not-exist", "verdict": "label_error", "note": "x"}]
    assert any("preserved" in warning for warning in result.warnings)


def test_a_verdict_outside_the_declared_vocabulary_warns(tmp_path):
    review = tmp_path / "review.csv"
    ctx, _, cases = run_stage(tmp_path, ERROR_SPEC, subdirectory="probe")
    with review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "verdict", "note"])
        writer.writeheader()
        writer.writerow(
            {"case_id": sorted(cases["case_id"])[0], "verdict": "looks_odd", "note": ""}
        )
    _, result, merged = run_stage(
        tmp_path, ERROR_SPEC, inputs={"review": review}, subdirectory="merged"
    )
    assert any("looks_odd" in warning for warning in result.warnings)
    # Kept as written: an unaggregatable verdict is still a human judgement.
    assert "looks_odd" in set(merged["verdict"])


def test_the_review_template_lists_every_case(tmp_path):
    ctx, _, cases = run_stage(tmp_path, ERROR_SPEC)
    template = list(csv.DictReader(ctx.layout.artifact(REVIEW_TEMPLATE).open(encoding="utf-8")))
    assert [row["case_id"] for row in template] == list(cases["case_id"])
    assert all(row["verdict"] == "" for row in template)


def test_the_declared_verdict_vocabulary_is_recorded_with_the_run(tmp_path):
    ctx, _, _ = run_stage(tmp_path, ERROR_SPEC)
    body = json.loads(ctx.layout.artifact(SELECTOR_LOG).read_text())
    assert body["verdicts"] == list(DEFAULT_VERDICTS)


def test_the_unreviewed_marker_cannot_also_be_a_verdict():
    with pytest.raises(SelectorError, match="nobody has looked at"):
        MisclassifySpec.from_spec({"selectors": [], "verdicts": [UNREVIEWED, "label_error"]})


# ---------------------------------------------------------------------------
# error
# ---------------------------------------------------------------------------


def test_error_selects_false_positives_and_false_negatives_only(tmp_path):
    _, _, cases = run_stage(tmp_path, ERROR_SPEC)
    assert set(cases["outcome_class"]) <= {"fp", "fn"}
    assert {"fp", "fn"} & set(cases["outcome_class"])


def test_error_ranks_false_positives_by_descending_probability(tmp_path):
    _, _, cases = run_stage(tmp_path, ERROR_SPEC)
    false_positives = cases[cases["outcome_class"] == "fp"].sort_values("rank")
    assert list(false_positives["prob"]) == sorted(false_positives["prob"], reverse=True)
    assert len(false_positives) <= 3


def test_error_ranks_false_negatives_by_ascending_probability(tmp_path):
    _, _, cases = run_stage(tmp_path, ERROR_SPEC)
    false_negatives = cases[cases["outcome_class"] == "fn"].sort_values("rank")
    assert list(false_negatives["prob"]) == sorted(false_negatives["prob"])


def test_the_outcome_class_agrees_with_the_recorded_threshold(tmp_path):
    _, _, cases = run_stage(tmp_path, ERROR_SPEC)
    for _, row in cases.iterrows():
        predicted = row["prob"] >= row["threshold"]
        assert (row["outcome_class"] == "fp") == (predicted and row["label_primary"] == 0)
        assert (row["outcome_class"] == "fn") == (not predicted and row["label_primary"] == 1)


# ---------------------------------------------------------------------------
# boundary
# ---------------------------------------------------------------------------


def test_boundary_selects_only_within_epsilon_of_the_threshold(tmp_path):
    spec = {
        "figures": False,
        "selectors": [
            {"name": "band", "type": "boundary", "pattern": REFERENCE, "epsilon": 0.2}
        ],
    }
    _, _, cases = run_stage(tmp_path, spec)
    assert len(cases) > 0
    assert ((cases["prob"] - cases["threshold"]).abs() < 0.2).all()
    assert list(cases["rank"]) == sorted(cases["rank"])


def test_boundary_needs_an_epsilon():
    with pytest.raises(SelectorError, match="needs 'epsilon'"):
        MisclassifySpec.from_spec(
            {"selectors": [{"name": "b", "type": "boundary", "pattern": REFERENCE}]}
        )


def test_both_stages_parse_one_threshold_policy(tmp_path):
    """The audit's operating point is the evaluation's, by construction."""
    from mival.metrics import DEFAULT_DECISION_THRESHOLDS, DEFAULT_MURPHY_BINS
    from mival.stages.evaluate import _Settings
    from mival.stages._predictions import ThresholdPolicy

    spec = {"threshold_policy": "refit_youden", "sensitivity_target": 0.9, "dev_split": "dev"}
    evaluation = _Settings(spec, DEFAULT_DECISION_THRESHOLDS, DEFAULT_MURPHY_BINS)
    assert evaluation.threshold_policy_settings() == ThresholdPolicy.from_spec(spec)


def test_an_explicit_threshold_in_the_spec_is_what_the_band_centres_on(tmp_path):
    spec = {
        "figures": False,
        "threshold": 0.5,
        "selectors": [
            {"name": "band", "type": "boundary", "pattern": REFERENCE, "epsilon": 0.1}
        ],
    }
    _, _, cases = run_stage(tmp_path, spec)
    assert set(cases["threshold"]) == {0.5}


# ---------------------------------------------------------------------------
# contrast — one code path, whichever axis separates the two patterns
# ---------------------------------------------------------------------------


def _contrast(a_axis_value, b_axis_value, axis, criterion="prediction_flip", **extra):
    a = dict(REFERENCE, **{axis: a_axis_value})
    b = dict(REFERENCE, **{axis: b_axis_value})
    return {
        "figures": False,
        "selectors": [
            dict(
                {"name": "flip", "type": "contrast", "a": a, "b": b, "criterion": criterion},
                **extra,
            )
        ],
    }


def test_contrast_across_model_id_finds_prediction_flips(tmp_path):
    _, _, cases = run_stage(tmp_path, _contrast("modela", "modelb", "model_id"))
    assert len(cases) > 0
    assert set(cases["model_id"]) == {"modela"}
    assert all("model_id=modelb" in key for key in cases["peer_run_key"])
    for _, row in cases.iterrows():
        assert (row["prob"] >= row["threshold"]) != (row["peer_prob"] >= row["threshold"]) or (
            row["outcome_class"] != row["peer_outcome_class"]
        )


def test_contrast_across_perturbation_uses_the_same_code_path(tmp_path):
    _, _, cases = run_stage(
        tmp_path, _contrast("baseline", "downsample125", "perturbation_id")
    )
    assert set(cases["perturbation_id"]) <= {"baseline"}
    assert all("perturbation_id=downsample125" in key for key in cases["peer_run_key"])


def test_contrast_across_label_def_finds_cases_whose_reference_standard_changes(tmp_path):
    """Spec §4.6: label noise made concrete, from the same selector as model disagreement."""
    _, _, cases = run_stage(
        tmp_path, _contrast("primary", "sens1", "label_def", criterion="truth_flip")
    )
    assert len(cases) == 2  # exactly the two records the two definitions disagree on
    assert set(cases["image_occurrence_id"]) == {"r10", "r11"}


def test_truth_flip_and_prediction_flip_are_different_questions(tmp_path):
    """The spec's word "label_flip" is ambiguous on the label_def axis.

    Holding the operating point fixed isolates the two meanings. The two label
    definitions score identical probabilities, so at one shared threshold no
    prediction can flip — yet the reference standard flips for two records, and
    those two are exactly the cases the spec wants ("정의에 따라 정답이 바뀌는
    사례"). A selector named for "label" alone would return the wrong set.
    """
    fixed = {"threshold": 0.5}
    _, _, truth = run_stage(
        tmp_path,
        dict(_contrast("primary", "sens1", "label_def", criterion="truth_flip"), **fixed),
        subdirectory="truth",
    )
    _, _, prediction = run_stage(
        tmp_path,
        dict(_contrast("primary", "sens1", "label_def", criterion="prediction_flip"), **fixed),
        subdirectory="prediction",
    )
    assert len(prediction) == 0
    assert set(truth["image_occurrence_id"]) == {"r10", "r11"}


def test_the_operating_point_is_refit_per_label_definition(tmp_path):
    """A threshold is a property of the arm, and label_def is part of the arm.

    So changing the reference standard moves the operating point even though
    the model's probabilities are untouched — which is why a `prediction_flip`
    contrast across `label_def` returns cases at all.
    """
    _, _, cases = run_stage(
        tmp_path, _contrast("primary", "sens1", "label_def", criterion="prediction_flip")
    )
    assert len(cases) > 0
    assert (cases["prob"] == cases["peer_prob"]).all()


def test_label_flip_is_accepted_as_the_specs_word_for_prediction_flip(tmp_path):
    spec = MisclassifySpec.from_spec(
        _contrast("modela", "modelb", "model_id", criterion="label_flip")
    )
    assert spec.selectors[0].criterion == "prediction_flip"


def test_delta_prob_needs_a_tau():
    with pytest.raises(SelectorError, match="needs 'tau'"):
        MisclassifySpec.from_spec(_contrast("modela", "modelb", "model_id", "delta_prob"))


def test_delta_prob_selects_only_above_tau(tmp_path):
    _, _, cases = run_stage(
        tmp_path, _contrast("modela", "modelb", "model_id", "delta_prob", tau=0.5)
    )
    assert len(cases) > 0
    assert ((cases["prob"] - cases["peer_prob"]).abs() > 0.5).all()
    assert list(cases["score"]) == sorted(cases["score"], reverse=True)


def test_contrast_reports_when_the_two_patterns_share_no_record(tmp_path):
    spec = {
        "figures": False,
        "selectors": [
            {
                "name": "flip",
                "type": "contrast",
                "a": dict(REFERENCE, split="dev"),
                "b": dict(REFERENCE, split="test"),
            }
        ],
    }
    _, result, cases = run_stage(tmp_path, spec)
    assert len(cases) == 0
    assert any("no record in common" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# instability
# ---------------------------------------------------------------------------


INSTABILITY_SPEC = {
    "figures": False,
    "selectors": [
        {
            "name": "unstable",
            "type": "instability",
            "pattern": REFERENCE,
            "over": ["seed"],
            "tau": 0.1,
        }
    ],
}


def test_instability_needs_repeated_predictions_and_says_so_when_there_are_none(tmp_path):
    _, result, cases = run_stage(tmp_path, INSTABILITY_SPEC, seeds=(1,))
    assert len(cases) == 0
    assert any("two or more predictions" in warning for warning in result.warnings)


def test_instability_scores_the_spread_across_seeds(tmp_path):
    _, _, cases = run_stage(tmp_path, INSTABILITY_SPEC, seeds=(1, 2))
    assert len(cases) > 0
    assert set(cases["score_name"]) == {"prob_std"}
    assert (cases["score"] > 0.1).all()
    assert (cases["prob"] >= cases["peer_prob"]).all()


def test_instability_must_declare_what_varies():
    with pytest.raises(SelectorError, match="needs 'over'"):
        MisclassifySpec.from_spec(
            {"selectors": [{"name": "u", "type": "instability", "pattern": REFERENCE, "tau": 0.1}]}
        )


def test_an_axis_cannot_be_both_pinned_and_free():
    with pytest.raises(SelectorError, match="both 'pattern' and 'over'"):
        MisclassifySpec.from_spec(
            {
                "selectors": [
                    {
                        "name": "u",
                        "type": "instability",
                        "pattern": REFERENCE,
                        "over": ["perturbation_id"],
                        "tau": 0.1,
                    }
                ]
            }
        )


def test_over_must_name_an_axis_or_the_seed():
    with pytest.raises(SelectorError, match="neither run_key axes nor"):
        MisclassifySpec.from_spec(
            {
                "selectors": [
                    {
                        "name": "u",
                        "type": "instability",
                        "pattern": REFERENCE,
                        "over": ["subgroup"],
                        "tau": 0.1,
                    }
                ]
            }
        )


# ---------------------------------------------------------------------------
# Refusing to guess
# ---------------------------------------------------------------------------


def test_a_pattern_that_leaves_an_axis_free_is_refused_with_the_axis_named(tmp_path):
    """The failure this prevents: perturbation sensitivity reported as error."""
    loose = {axis: value for axis, value in REFERENCE.items() if axis != "perturbation_id"}
    spec = {
        "figures": False,
        "selectors": [{"name": "errors", "type": "error", "pattern": loose, "k": 3}],
    }
    with pytest.raises(SelectorError, match="perturbation_id"):
        run_stage(tmp_path, spec)


def test_a_pattern_may_only_name_run_key_axes():
    with pytest.raises(SelectorError, match="not run_key axes"):
        MisclassifySpec.from_spec(
            {"selectors": [{"name": "e", "type": "error", "pattern": {"subgroup": "F"}}]}
        )


def test_an_unknown_selector_type_is_refused():
    with pytest.raises(SelectorError, match="spec §4.6"):
        MisclassifySpec.from_spec({"selectors": [{"name": "e", "type": "interesting"}]})


def test_two_selectors_may_not_share_a_name():
    with pytest.raises(SelectorError, match="used more than once"):
        MisclassifySpec.from_spec(
            {
                "selectors": [
                    {"name": "e", "type": "error", "pattern": REFERENCE},
                    {"name": "e", "type": "boundary", "pattern": REFERENCE, "epsilon": 0.1},
                ]
            }
        )


def test_a_selector_name_must_be_path_safe():
    with pytest.raises(SelectorError, match="figure filename"):
        MisclassifySpec.from_spec(
            {"selectors": [{"name": "top 10 errors", "type": "error", "pattern": REFERENCE}]}
        )


def test_a_selector_that_matches_no_run_is_a_warning_not_a_crash(tmp_path):
    spec = {
        "figures": False,
        "selectors": [
            {
                "name": "absent",
                "type": "error",
                "pattern": dict(REFERENCE, model_id="modelc"),
                "k": 3,
            }
        ],
    }
    _, result, cases = run_stage(tmp_path, spec)
    assert len(cases) == 0
    assert any("matched no predictions" in warning for warning in result.warnings)


def test_fold_matches_whether_the_pattern_writes_it_as_a_number_or_a_string(tmp_path):
    numeric = MisclassifySpec.from_spec(
        {"selectors": [{"name": "e", "type": "error", "pattern": dict(REFERENCE, fold=1)}]}
    )
    textual = MisclassifySpec.from_spec(
        {"selectors": [{"name": "e", "type": "error", "pattern": dict(REFERENCE, fold="1")}]}
    )
    assert numeric.selectors[0].pattern == textual.selectors[0].pattern


# ---------------------------------------------------------------------------
# Companion information and figures (spec §4.6)
# ---------------------------------------------------------------------------


def test_context_tables_join_without_being_named_in_code(tmp_path):
    context = tmp_path / "acquisition.parquet"
    pd.DataFrame(
        [
            {
                "image_occurrence_id": f"r{index:02d}",
                "device_manufacturer": "GE" if index % 2 else "Philips",
            }
            for index in range(N_RECORDS)
        ]
    ).to_parquet(context, index=False)

    _, _, cases = run_stage(tmp_path, ERROR_SPEC, inputs={"acquisition": context})
    assert "device_manufacturer" in cases.columns
    assert set(cases["device_manufacturer"]) <= {"GE", "Philips"}
    # The guaranteed prefix is unchanged; context is appended.
    assert tuple(cases.columns)[: len(CASE_COLUMNS)] == CASE_COLUMNS


def test_a_case_figure_is_drawn_from_the_tensor_the_model_saw(tmp_path):
    pytest.importorskip("matplotlib")
    import numpy as np

    root = tmp_path / "figs"
    tensors = root / "tensors"
    tensors.mkdir(parents=True)
    rows = []
    for index in range(N_RECORDS):
        path = tensors / f"r{index:02d}.npy"
        np.save(path, np.linspace(-1.0, 1.0, 8 * 40).reshape(8, 40))
        rows.append(
            {
                "image_occurrence_id": f"r{index:02d}",
                "person_id": f"p{index:02d}",
                "model_id": "modela",
                "recipe_id": "r1",
                "perturbation_id": "baseline",
                "tensor_path": str(path),
                "n_leads": 8,
                "n_samples": 40,
                "sampling_rate_hz": 500.0,
            }
        )
    index_path = root / "preprocess_index.parquet"
    pd.DataFrame(rows).to_parquet(index_path, index=False)
    (root / "recipes.json").write_text(
        json.dumps(
            {"recipes": [{"recipe_id": "r1", "leads": [f"L{i}" for i in range(8)]}]}
        ),
        encoding="utf-8",
    )

    spec = dict(ERROR_SPEC, figures=True)
    ctx, _, cases = run_stage(
        tmp_path, spec, inputs={"preprocess_index": index_path}, subdirectory="figs2"
    )
    assert cases["tensor_path"].notna().all()
    assert cases["figure_path"].notna().all()
    for path in cases["figure_path"]:
        assert Path(path).exists()


def test_a_missing_tensor_index_costs_the_figure_not_the_run(tmp_path):
    _, result, cases = run_stage(tmp_path, dict(ERROR_SPEC, figures=True))
    assert len(cases) > 0
    assert cases["figure_path"].isna().all()


# ---------------------------------------------------------------------------
# Attribution overlay (spec §4.6) — off unless declared, and honest when it
# cannot be computed
# ---------------------------------------------------------------------------


def _tensor_inputs(tmp_path, subdirectory):
    """A preprocess_index and tensors for every record."""
    import numpy as np

    root = tmp_path / subdirectory
    tensors = root / "tensors"
    tensors.mkdir(parents=True)
    rows = []
    for index in range(N_RECORDS):
        path = tensors / f"r{index:02d}.npy"
        np.save(path, np.linspace(-1.0, 1.0, 4 * 16).reshape(4, 16))
        rows.append(
            {
                "image_occurrence_id": f"r{index:02d}",
                "person_id": f"p{index:02d}",
                "model_id": "modela",
                "recipe_id": "r1",
                "perturbation_id": "baseline",
                "tensor_path": str(path),
                "n_leads": 4,
                "n_samples": 16,
                "sampling_rate_hz": 500.0,
            }
        )
    index_path = root / "preprocess_index.parquet"
    pd.DataFrame(rows).to_parquet(index_path, index=False)
    return index_path


def _registry(tmp_path):
    directory = tmp_path / "registry"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "modela.json").write_text(
        json.dumps(
            {
                "model_id": "modela",
                "Name": "Model A",
                "x-mival": {
                    "adapter": "stub",
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
                        "dtype": "float64",
                    },
                    "output": {"type": "sigmoid", "n_outputs": 1, "positive_index": 0},
                    "feature_layer": None,
                    "threshold": {},
                    "runtime": {},
                    "training_modes_supported": ["inference_only"],
                    "pretraining_corpora": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return directory


class StubAdapter:
    """Stands in for a backend so the audit's wiring is testable without weights."""

    def __init__(self):
        self.calls = []

    def load(self, card):
        return {"card": card}

    def attribute(self, handle, batch, baseline, steps):
        import numpy as np

        self.calls.append((np.asarray(batch).shape, steps))
        return np.asarray(batch) * 0.5


def _run_with_attribution(tmp_path, attribution, subdirectory):
    index_path = _tensor_inputs(tmp_path, subdirectory + "-data")
    adapter = StubAdapter()
    spec = dict(ERROR_SPEC, figures=True, attribution=attribution)

    root = tmp_path / subdirectory
    predictions = build_predictions(root / "predictions")
    stage = MisclassifyStage(adapter_factory=lambda name: adapter)
    ctx = prepare(
        stage=stage,
        study_id="test-study",
        site="mimic",
        spec=spec,
        inputs={"preprocess_index": index_path},
        runs_root=root / "runs",
        seed=1,
    )
    ctx.inputs = dict(ctx.inputs)
    ctx.inputs["predictions"] = predictions
    ctx.layout.create_dirs()
    result = stage.run(ctx)
    cases = pd.read_parquet(ctx.layout.artifact(CASES))
    return result, cases, adapter


def test_no_attribution_is_computed_unless_the_study_declares_one(tmp_path):
    _, _, cases = run_stage(
        tmp_path, dict(ERROR_SPEC, figures=True), inputs={"preprocess_index": _tensor_inputs(tmp_path, "plain")}
    )
    assert cases["attribution_status"].isna().all()


def test_a_declared_attribution_without_a_baseline_draws_nothing_and_says_why(tmp_path):
    result, cases, adapter = _run_with_attribution(
        tmp_path, {"registry": str(_registry(tmp_path))}, "nobaseline"
    )
    assert adapter.calls == []
    assert any("asystole" in warning for warning in result.warnings)
    assert cases["attribution_status"].isna().all()


def test_a_declared_attribution_reaches_the_adapter(tmp_path):
    pytest.importorskip("matplotlib")
    result, cases, adapter = _run_with_attribution(
        tmp_path,
        {"registry": str(_registry(tmp_path)), "baseline": "zeros", "steps": 8},
        "declared",
    )
    assert len(cases) > 0
    assert set(cases["attribution_status"]) == {"ok"}
    assert adapter.calls and all(shape == (1, 4, 16) for shape, _ in adapter.calls)
    assert all(steps == 8 for _, steps in adapter.calls)


def test_a_fitted_arm_cannot_be_attributed_and_the_case_records_that(tmp_path):
    """A probed head lives in the models stage's memory, never on disk."""
    from mival.stages.misclassify import _Attributor

    attributor = _Attributor.of(
        MisclassifySpec.from_spec(
            dict(ERROR_SPEC, attribution={"registry": "unused", "baseline": "zeros"})
        ),
        [],
        lambda name: StubAdapter(),
    )
    attribution, status = attributor.of_case({"training_mode": "linear_probe"}, None)
    assert attribution is None
    assert status == "unavailable_fitted_head_not_persisted"
