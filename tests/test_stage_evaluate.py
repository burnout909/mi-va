"""The evaluation stage end to end (spec §4.5).

The properties under test are the structural ones: the long schema is exactly
what Plan 2 pinned, the axes are a groupby and nothing more, suppression marks
rather than deletes, and the whole stage is reproducible from its seed.
"""

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

import numpy as np
import pandas as pd

from mival.pipeline.runkey import AXES, REPORT_AXES, RunKey
from mival.pipeline.stage import execute, prepare
from mival.stages.evaluate import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_UNIT,
    CATEGORIES,
    COMPARISONS,
    METRICS_LONG,
    METRICS_LONG_COLUMNS,
    SUBGROUP_SUPPRESS_MIN_EVENTS,
    EvaluateStage,
)
from mival.stages.models import REGRESSION_LABEL_DEF

MODELS = (("prophecg", "inference_only"), ("ecgfounder", "linear_probe"))


def build_predictions(
    directory: Path, splits=("dev", "test"), perturbations=("none",), regression=False
):
    """A small cohort with a rare-event subgroup, written one file per run_key.

    50 male persons carrying 20 events and 10 female persons carrying 3, so
    `sex=F` sits below the suppression floor and `sex=M` above it. Every
    person contributes two ECGs, which is what makes the person-level
    bootstrap the correct unit.

    ``regression=True`` writes a regression arm instead of a classification
    one: `label_def="value"`, `prob`/`logit` null, and continuous
    `label_value`/`pred_value` built from the same binary label so the LVEF
    cut at 40 still separates the two groups.
    """
    rng = np.random.default_rng(2)
    people = []
    for index in range(60):
        sex = "M" if index < 50 else "F"
        events = 20 if sex == "M" else 3
        threshold = events / (50 if sex == "M" else 10)
        people.append(
            {
                "person_id": f"p{index:03d}",
                "sex": sex,
                "label": int(rng.random() < threshold),
                "split": "test" if index % 3 == 0 else "dev",
            }
        )
    cohort = pd.DataFrame(people)

    records = []
    for person in people:
        for recording in range(2):
            records.append((person, f"{person['person_id']}-{recording}"))

    directory.mkdir(parents=True, exist_ok=True)
    for model_id, mode in MODELS:
        skill = 1.4 if model_id == "ecgfounder" else 1.0
        for perturbation in perturbations:
            damage = 1.0 if perturbation == "none" else 0.5
            scores = {}
            label_values = {}
            pred_values = {}
            for person, image in records:
                signal = rng.normal(loc=skill * damage * person["label"], scale=1.0)
                scores[image] = float(1.0 / (1.0 + np.exp(-(signal - 1.0))))
                if regression:
                    label_values[image] = 60.0 - 20.0 * person["label"] + float(rng.normal(scale=2.0))
                    pred_values[image] = label_values[image] + float(rng.normal(scale=2.0))
            for split in splits:
                key = RunKey(
                    site="mimic",
                    model_id=model_id,
                    training_mode=mode,
                    recipe_id="r1",
                    perturbation_id=perturbation,
                    label_def=REGRESSION_LABEL_DEF if regression else "primary",
                    split=split,
                    fold=None,
                )
                rows = [
                    {
                        "image_occurrence_id": image,
                        "person_id": person["person_id"],
                        "split": person["split"],
                        "fold": None,
                        "label_primary": person["label"],
                        "label_sens1": person["label"],
                        "label_sens2": person["label"],
                        "label_sens3": person["label"],
                        "label_value": label_values.get(image) if regression else None,
                        "pred_value": pred_values.get(image) if regression else None,
                        "prob": None if regression else scores[image],
                        "logit": (
                            None
                            if regression
                            else float(np.log(scores[image] / (1 - scores[image])))
                        ),
                        "model_id": model_id,
                        "training_mode": mode,
                        "recipe_id": "r1",
                        "perturbation_id": perturbation,
                        "seed": 1,
                        "site": "mimic",
                    }
                    for person, image in records
                    if person["split"] == split
                ]
                pd.DataFrame(rows).to_parquet(directory / f"{key.to_string()}.parquet", index=False)
    return cohort


def run_stage(
    tmp_path: Path, spec=None, subdirectory="case", cohort_columns=("sex",), regression=False
):
    """Run the stage directly on a directory of prediction files."""
    root = tmp_path / subdirectory
    predictions = root / "predictions"
    cohort = build_predictions(predictions, regression=regression)
    cohort_path = root / "cohort_split.parquet"
    cohort[["person_id", "split"] + [column for column in cohort_columns]].to_parquet(
        cohort_path, index=False
    )

    stage = EvaluateStage()
    ctx = prepare(
        stage=stage,
        study_id="test-study",
        site="mimic",
        spec=spec or {},
        inputs={"cohort_split": cohort_path},
        runs_root=root / "runs",
        seed=1,
    )
    ctx.inputs = dict(ctx.inputs)
    ctx.inputs["predictions"] = predictions
    ctx.layout.create_dirs()
    result = stage.run(ctx)
    table = pd.read_parquet(ctx.layout.artifact(METRICS_LONG))
    return ctx, result, table


BASE_SPEC = {"subgroups": ["sex"], "bootstrap_replicates": 20, "figures": False}


def test_regression_arm_reports_regression_metrics_only(tmp_path):
    _, _, table = run_stage(tmp_path, BASE_SPEC, regression=True)
    reg = table[table["label_def"] == "value"]
    assert set(reg["metric"]) == {"mae", "rmse", "r2", "auroc_below@40"}
    assert set(reg["category"]) == {"regression"}
    assert reg["ci_lo"].notna().any()
    assert not (table[table["label_def"] == "primary"]["category"] == "regression").any()


def test_regression_cut_comes_from_the_spec(tmp_path):
    _, _, table = run_stage(tmp_path, {**BASE_SPEC, "regression_cuts": [35, 50]}, regression=True)
    reg = table[table["label_def"] == "value"]
    assert {"auroc_below@35", "auroc_below@50"} <= set(reg["metric"])


def test_the_output_columns_are_exactly_the_pinned_schema(tmp_path):
    _, _, table = run_stage(tmp_path, BASE_SPEC)
    assert tuple(table.columns) == METRICS_LONG_COLUMNS


def test_the_pinned_schema_is_the_axes_plus_the_report_axes(tmp_path):
    # Long format exists so that a new axis is a new column value, never a new
    # column (spec §4.5). This asserts the schema really is built that way.
    assert METRICS_LONG_COLUMNS[: len(AXES)] == AXES
    assert METRICS_LONG_COLUMNS[len(AXES) : len(AXES) + len(REPORT_AXES)] == REPORT_AXES


def test_every_row_carries_a_category_from_the_declared_four(tmp_path):
    _, _, table = run_stage(tmp_path, BASE_SPEC)
    assert set(table["category"]) <= set(CATEGORIES)
    # Categories 1-3 are computed; category 4 (interpretation) holds a schema
    # slot only and therefore emits no rows in this plan.
    assert set(table["category"]) == {"discrimination", "calibration", "clinical_utility"}


def test_the_three_computed_categories_are_all_present_for_every_slice(tmp_path):
    _, _, table = run_stage(tmp_path, BASE_SPEC)
    per_slice = table.groupby(list(AXES) + list(REPORT_AXES), dropna=False)["category"].nunique()
    assert set(per_slice) == {3}


def test_a_subgroup_below_the_event_floor_is_marked_but_still_computed(tmp_path):
    # Spec §4.5: n_events < 10 is computed, flagged, and excluded from
    # conclusions — not deleted, because a suppressed subgroup is itself a
    # finding about how much of the cohort is unassessable.
    _, _, table = run_stage(tmp_path, BASE_SPEC)
    rare = table[(table["subgroup"] == "sex=F") & (table["metric"] == "auroc")]
    assert not rare.empty
    assert (rare["n_events"] < SUBGROUP_SUPPRESS_MIN_EVENTS).all()
    assert rare["suppressed"].all()
    # Marked, but still computed wherever the metric is defined at all. A
    # one-class slice has no AUROC and is null for a different reason.
    computed = rare[rare["n_events"] > 0]
    assert not computed.empty
    assert computed["value"].notna().all()

    common = table[(table["subgroup"] == "sex=M") & (table["metric"] == "auroc")]
    assert (common["n_events"] >= SUBGROUP_SUPPRESS_MIN_EVENTS).all()
    assert not common["suppressed"].any()


def test_adding_a_report_axis_leaves_the_existing_rows_untouched(tmp_path):
    # The report axes re-cut existing predictions (spec §3.4). Slicing by sex
    # must therefore add rows and change nothing that was already there.
    without = run_stage(tmp_path, dict(BASE_SPEC, subgroups=[]), subdirectory="a")[2]
    with_sex = run_stage(tmp_path, BASE_SPEC, subdirectory="b")[2]
    key = list(AXES) + ["metric"]
    left = without[without["subgroup"] == "all"].set_index(key)["value"]
    right = with_sex[with_sex["subgroup"] == "all"].set_index(key)["value"]
    assert len(right) > 0
    pd.testing.assert_series_equal(left.sort_index(), right.sort_index())
    assert len(with_sex) > len(without)


def test_contamination_is_flagged_per_model_and_not_blocked(tmp_path):
    # Spec §3.5: a contaminated arm is still evaluated and still reported.
    spec = dict(BASE_SPEC, contamination={"ecgfounder": True})
    _, _, table = run_stage(tmp_path, spec)
    assert table[table["model_id"] == "ecgfounder"]["contaminated"].all()
    assert not table[table["model_id"] == "prophecg"]["contaminated"].any()


def test_the_same_seed_reproduces_the_whole_table(tmp_path):
    spec = dict(BASE_SPEC, seed=5)
    first = run_stage(tmp_path, spec, subdirectory="first")[2]
    second = run_stage(tmp_path, spec, subdirectory="second")[2]
    pd.testing.assert_frame_equal(first, second)


def test_a_different_seed_moves_the_intervals_but_not_the_estimates(tmp_path):
    one = run_stage(tmp_path, dict(BASE_SPEC, seed=5), subdirectory="one")[2]
    two = run_stage(tmp_path, dict(BASE_SPEC, seed=6), subdirectory="two")[2]
    pd.testing.assert_series_equal(one["value"], two["value"])
    assert not one["ci_lo"].equals(two["ci_lo"])


def test_intervals_are_present_and_bracket_the_estimate(tmp_path):
    _, _, table = run_stage(tmp_path, BASE_SPEC)
    auroc = table[(table["metric"] == "auroc") & (table["subgroup"] == "all")]
    assert auroc["ci_lo"].notna().all()
    assert (auroc["ci_lo"] <= auroc["value"]).all()
    assert (auroc["value"] <= auroc["ci_hi"]).all()


def test_bootstrap_can_be_switched_off_leaving_estimates_without_intervals(tmp_path):
    _, _, table = run_stage(tmp_path, dict(BASE_SPEC, bootstrap=False))
    assert table["value"].notna().any()
    assert table["ci_lo"].isna().all()


def test_the_resampling_unit_and_replicate_count_are_the_pinned_ones():
    assert BOOTSTRAP_UNIT == "person_id"
    assert BOOTSTRAP_REPLICATES == 2000


def test_paired_comparisons_go_to_their_own_table_with_a_confirmatory_flag(tmp_path):
    spec = dict(
        BASE_SPEC,
        comparisons=[
            {
                "name": "asis_vs_probe",
                "confirmatory": True,
                "a": {"model_id": "prophecg", "split": "test"},
                "b": {"model_id": "ecgfounder", "split": "test"},
            }
        ],
    )
    ctx, _, _ = run_stage(tmp_path, spec)
    comparisons = pd.read_parquet(ctx.layout.artifact(COMPARISONS))
    assert set(comparisons["metric"]) == {"delta_auroc", "delta_auprc"}
    assert comparisons["confirmatory"].all()
    # Spec §2.5 pre-specifies the confirmatory pairs; everything else is
    # exploratory, and the flag is what keeps the two apart in the report.
    assert comparisons["arm_a"].str.contains("model_id=prophecg").all()
    assert comparisons["arm_b"].str.contains("model_id=ecgfounder").all()
    assert comparisons["ci_lo"].notna().all()
    assert (comparisons["ci_lo"] <= comparisons["value"]).all()


def test_an_ambiguous_comparison_arm_is_an_error_not_a_silent_pool(tmp_path):
    spec = dict(
        BASE_SPEC,
        comparisons=[
            {
                "name": "ambiguous",
                # No split filter, so this matches both the dev and the test
                # run_key. Pooling them would measure something nobody asked for.
                "a": {"model_id": "prophecg"},
                "b": {"model_id": "ecgfounder", "split": "test"},
            }
        ],
    )
    with pytest.raises(ValueError, match="exactly one"):
        run_stage(tmp_path, spec)


def _fitted_thresholds(tmp_path, spec, drop_test=False):
    from mival.metrics import DEFAULT_DECISION_THRESHOLDS, DEFAULT_MURPHY_BINS
    from mival.stages.evaluate import (
        _Settings,
        _attach_cohort_columns,
        _fit_thresholds,
        _load_predictions,
    )

    root = tmp_path / ("dropped" if drop_test else "full")
    cohort = build_predictions(root / "predictions")
    frame = _attach_cohort_columns(
        _load_predictions(root / "predictions"), cohort[["person_id", "split", "sex"]]
    )
    if drop_test:
        frame = frame[frame["split"] != "test"]
    settings = _Settings.from_spec(spec, DEFAULT_DECISION_THRESHOLDS, DEFAULT_MURPHY_BINS)
    warnings = []
    return _fit_thresholds(frame, settings, warnings), frame, settings, warnings


def test_the_operating_threshold_reaches_the_target_sensitivity_on_dev(tmp_path):
    # Spec §4.4: the primary policy is refit_sens95, because Youden prices a
    # missed STEMI the same as a false alarm.
    from mival.stages.evaluate import _arm_of, _group_by_run_key

    thresholds, frame, _, _ = _fitted_thresholds(tmp_path, {})
    development = frame[frame["split"] != "test"]
    for key, group in _group_by_run_key(development):
        y = group["label_primary"].to_numpy(dtype=float)
        p = group["prob"].to_numpy(dtype=float)
        flagged = p >= thresholds[_arm_of(key)]
        sensitivity = np.sum(flagged & (y == 1)) / np.sum(y == 1)
        assert sensitivity >= 0.95


def test_the_threshold_is_the_same_whether_or_not_the_test_split_is_present(tmp_path):
    # Spec §4.4: thresholds are never selected on the test set. Removing the
    # held-out rows must therefore not move the operating point.
    with_test, _, _, _ = _fitted_thresholds(tmp_path, {})
    without_test, _, _, _ = _fitted_thresholds(tmp_path, {}, drop_test=True)
    assert with_test == without_test


def test_an_explicit_threshold_in_the_spec_overrides_refitting(tmp_path):
    thresholds, _, _, _ = _fitted_thresholds(tmp_path, {"threshold": 0.42})
    assert set(thresholds.values()) == {0.42}


def test_a_missing_development_split_falls_back_and_says_so(tmp_path):
    # Everything is test, so there is nothing to fit on. The neutral 0.5 is
    # used and the manifest carries the warning; it is never passed off as a
    # refitted operating point.
    from mival.metrics import DEFAULT_DECISION_THRESHOLDS, DEFAULT_MURPHY_BINS
    from mival.stages.evaluate import (
        _Settings,
        _fit_thresholds,
        _load_predictions,
    )

    root = tmp_path / "alltest"
    build_predictions(root / "predictions")
    frame = _load_predictions(root / "predictions")
    frame["split"] = "test"
    settings = _Settings.from_spec({}, DEFAULT_DECISION_THRESHOLDS, DEFAULT_MURPHY_BINS)
    warnings = []
    thresholds = _fit_thresholds(frame, settings, warnings)
    assert set(thresholds.values()) == {0.5}
    assert warnings and all("development split" in message for message in warnings)


def _write_train_log(path: Path, value: float, contaminated: bool):
    """The shape the models stage writes: one JSON object per arm."""
    import json

    lines = []
    for model_id, mode in MODELS:
        run_keys = [
            RunKey(
                site="mimic",
                model_id=model_id,
                training_mode=mode,
                recipe_id="r1",
                perturbation_id="none",
                label_def="primary",
                split=split,
                fold=None,
            ).to_string()
            for split in ("dev", "test")
        ]
        lines.append(
            json.dumps(
                {
                    "model_id": model_id,
                    "training_mode": mode,
                    "label_def": "primary",
                    "thresholds": {
                        "refit_sens95": {"policy": "refit_sens95", "value": value, "fitted_on": "dev"}
                    },
                    "primary_threshold_policy": "refit_sens95",
                    "contamination": {
                        "flag": contaminated and model_id == "ecgfounder",
                        "overlapping_corpora": ["mimic-iv-ecg"] if contaminated else [],
                    },
                    "run_keys": run_keys,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_the_models_stage_train_log_supplies_the_operating_threshold(tmp_path):
    # Spec §4.4: the threshold was fitted by the stage that held the ModelCard
    # and the out-of-fold dev probabilities. Refitting it here from the saved
    # predictions would be a second, slightly different answer to a question
    # that has already been answered.
    from mival.metrics import DEFAULT_DECISION_THRESHOLDS, DEFAULT_MURPHY_BINS
    from mival.stages.evaluate import (
        _Settings,
        _fit_thresholds,
        _load_predictions,
        _recorded_thresholds_and_contamination,
    )

    root = tmp_path / "trainlog"
    build_predictions(root / "predictions")
    frame = _load_predictions(root / "predictions")
    log = root / "train_log.jsonl"
    _write_train_log(log, value=0.1234, contaminated=True)

    settings = _Settings.from_spec({}, DEFAULT_DECISION_THRESHOLDS, DEFAULT_MURPHY_BINS)
    warnings = []
    recorded, contamination = _recorded_thresholds_and_contamination(log, settings, warnings)
    thresholds = _fit_thresholds(frame, settings, warnings, recorded)
    assert set(thresholds.values()) == {0.1234}
    assert warnings == []
    assert any(contamination.values()) and not all(contamination.values())


def test_the_train_log_is_the_authority_on_contamination(tmp_path):
    # Spec §3.5: the gate runs in the models stage, against the card's
    # pretraining corpora. The study spec's restatement is only a fallback.
    root = tmp_path / "contam"
    predictions = root / "predictions"
    cohort = build_predictions(predictions)
    cohort_path = root / "cohort_split.parquet"
    cohort[["person_id", "split", "sex"]].to_parquet(cohort_path, index=False)
    log = root / "train_log.jsonl"
    _write_train_log(log, value=0.3, contaminated=True)

    stage = EvaluateStage()
    ctx = prepare(
        stage=stage,
        study_id="contam",
        site="mimic",
        spec=dict(BASE_SPEC, contamination={"ecgfounder": False}),
        inputs={"cohort_split": cohort_path, "train_log": log},
        runs_root=root / "runs",
        seed=1,
    )
    ctx.inputs = dict(ctx.inputs)
    ctx.inputs["predictions"] = predictions
    ctx.layout.create_dirs()
    stage.run(ctx)
    table = pd.read_parquet(ctx.layout.artifact(METRICS_LONG))
    assert table[table["model_id"] == "ecgfounder"]["contaminated"].all()
    assert not table[table["model_id"] == "prophecg"]["contaminated"].any()


def test_metric_code_never_mentions_an_axis(tmp_path):
    # Spec §4.5: evaluation is one function computing four categories, and the
    # rest is a groupby. If an axis name appears as a column reference inside
    # mival.metrics, adding an axis stops being free.
    import mival.metrics as metrics_package

    package_dir = Path(metrics_package.__file__).parent
    offenders = []
    for source in sorted(package_dir.glob("*.py")):
        text = source.read_text(encoding="utf-8")
        for axis in AXES + REPORT_AXES:
            if f'"{axis}"' in text or f"'{axis}'" in text:
                offenders.append(f"{source.name}: {axis}")
    assert offenders == []


def test_the_stage_runs_through_the_wrapper_on_a_single_prediction_file(tmp_path):
    # The CLI addresses inputs as files, because an input is hashed into the
    # config_hash. A concatenated prediction table must therefore work too.
    root = tmp_path / "wrapped"
    predictions_dir = root / "predictions"
    cohort = build_predictions(predictions_dir)
    frames = []
    for part in sorted(predictions_dir.glob("*.parquet")):
        frame = pd.read_parquet(part)
        for axis, value in RunKey.from_string(part.stem).to_dict().items():
            frame[axis] = value
        frames.append(frame)
    flat = root / "predictions.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(flat, index=False)
    cohort_path = root / "cohort_split.parquet"
    cohort[["person_id", "split", "sex"]].to_parquet(cohort_path, index=False)

    stage = EvaluateStage()
    ctx = prepare(
        stage=stage,
        study_id="wrapped",
        site="mimic",
        spec=BASE_SPEC,
        inputs={"predictions": flat, "cohort_split": cohort_path},
        runs_root=root / "runs",
        seed=1,
    )
    summary = execute(stage, ctx)
    assert summary["status"] == "ok"
    assert ctx.layout.manifest_path.is_file()
    assert summary["counts"]["out"] > 0
    # Evaluation aggregates and excludes nothing, so it adds no ledger rows.
    assert summary["counts"]["excluded"] == 0
    table = pd.read_parquet(ctx.layout.artifact(METRICS_LONG))
    assert tuple(table.columns) == METRICS_LONG_COLUMNS
    assert set(table["label_def"]) == {"primary"}


def test_a_single_file_without_the_label_def_axis_fails_loudly(tmp_path):
    root = tmp_path / "nolabeldef"
    predictions_dir = root / "predictions"
    cohort = build_predictions(predictions_dir)
    frames = []
    for part in sorted(predictions_dir.glob("*.parquet")):
        frames.append(pd.read_parquet(part))
    flat = root / "predictions.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(flat, index=False)
    cohort_path = root / "cohort_split.parquet"
    cohort[["person_id", "split"]].to_parquet(cohort_path, index=False)

    stage = EvaluateStage()
    ctx = prepare(
        stage=stage,
        study_id="nolabeldef",
        site="mimic",
        spec={"figures": False, "bootstrap": False},
        inputs={"predictions": flat, "cohort_split": cohort_path},
        runs_root=root / "runs",
        seed=1,
    )
    ctx.layout.create_dirs()
    with pytest.raises(ValueError, match="label_def"):
        stage.run(ctx)


def test_a_prediction_file_that_contradicts_its_run_key_name_is_rejected(tmp_path):
    # The file name is where label_def comes from, so a name that disagrees
    # with the contents would misattribute a whole run's results.
    from mival.stages.evaluate import _load_predictions

    root = tmp_path / "mismatch"
    build_predictions(root / "predictions")
    part = sorted((root / "predictions").glob("*.parquet"))[0]
    frame = pd.read_parquet(part)
    frame["model_id"] = "someone_else"
    frame.to_parquet(part, index=False)
    with pytest.raises(ValueError, match="model_id"):
        _load_predictions(root / "predictions")


def test_an_outcome_whose_label_is_absent_is_warned_about_and_skipped(tmp_path):
    spec = dict(BASE_SPEC, outcomes={"stemi": None, "mortality_30d": "label_death_30d"})
    _, result, table = run_stage(tmp_path, spec)
    assert set(table["outcome"]) == {"stemi"}
    assert any("mortality_30d" in message for message in result.warnings)


def test_an_extra_outcome_column_becomes_another_value_of_the_outcome_axis(tmp_path):
    # Spec §3.4: prognostic validity is a value of the outcome axis, joined
    # onto predictions that already exist — no second inference pass.
    spec = dict(BASE_SPEC, outcomes={"stemi": None, "mortality_30d": "label_sens3"})
    _, _, table = run_stage(tmp_path, spec)
    assert set(table["outcome"]) == {"stemi", "mortality_30d"}


def test_figures_are_written_as_code_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    spec = dict(BASE_SPEC, figures=True)
    ctx, result, _ = run_stage(tmp_path, spec)
    written = {path.name for path in ctx.layout.artifact("figures").glob("*.png")}
    assert {"roc_pr.png", "calibration.png", "decision_curve.png"} <= written
    assert "subgroup_forest.png" in written
    assert all(path.exists() for path in result.outputs)
