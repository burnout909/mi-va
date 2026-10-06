"""Interval and survival arms in the models stage (plan 2026-10-07, Task 4)."""
import numpy as np
import pytest

pytest.importorskip("pyarrow")

from mival.pipeline.tables import read_table  # noqa: E402
from mival.stages.models import PREDICTION_COLUMNS, Arm  # noqa: E402

from test_stage_models import (  # noqa: E402
    ROWS, FakeAdapter, arm, build_inputs, make_context, make_spec, run_stage, train_log, write_card,
)


def _with_columns(inputs, **columns):
    frame = read_table(inputs["cohort_index"])
    for name, function in columns.items():
        frame[name] = [function(i) for i in range(len(frame))]
    frame.to_parquet(inputs["cohort_index"], index=False)
    return inputs


class SegAdapter(FakeAdapter):
    """Returns (N, 3) intervals: score, 2*score, 3*score."""

    def forward(self, handle, batch):
        scores = np.asarray(batch, dtype=np.float64).mean(axis=(1, 2))
        return np.stack([scores, 2 * scores, 3 * scores], axis=1)


def _predictions(ctx):
    assert sorted(ctx.layout.artifact("predictions").glob("*.parquet"))
    return read_table(ctx.layout.artifact("predictions"))


def test_new_label_columns_are_in_the_prediction_contract():
    for column in ("label_event", "label_time_days", "label_pr", "label_qrs", "label_qt"):
        assert column in PREDICTION_COLUMNS


def test_arm_survival_requires_inference_only():
    with pytest.raises(ValueError, match="survival"):
        Arm.from_mapping({"model_id": "m", "training_mode": "linear_probe", "label_def": "survival"}, 0, "primary")
    survival = Arm.from_mapping({"model_id": "m", "training_mode": "inference_only", "label_def": "survival"}, 0, "primary")
    assert survival.is_survival and not survival.is_regression
    assert survival.required_label_columns == ("label_event", "label_time_days")
    assert Arm.from_mapping({"model_id": "m", "training_mode": "inference_only", "label_def": "qrs"}, 0, "primary").is_regression


def test_regression_arm_on_interval_label(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "seg", modes=["inference_only"], output={"type": "segmentation_mask", "n_outputs": 4})
    inputs = _with_columns(build_inputs(tmp_path, model_ids=("seg",)),
                           label_pr=lambda i: 150.0, label_qrs=lambda i: 90.0, label_qt=lambda i: 400.0)
    _result, ctx = run_stage(tmp_path, make_spec(registry, [arm("seg", label_def="qrs")]), inputs, SegAdapter())
    frame = _predictions(ctx)
    scores = {image: score for image, _p, score, *_ in ROWS}
    np.testing.assert_allclose(frame["pred_value"], [2 * scores[i] for i in frame["image_occurrence_id"]])
    assert frame["prob"].isna().all() and (frame["label_qrs"] == 90.0).all()
    assert train_log(ctx)[0]["thresholds"] == {}


def test_survival_rows_have_no_prob(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "risk", modes=["inference_only"], output={"type": "risk_score", "n_outputs": 1})
    inputs = _with_columns(build_inputs(tmp_path, model_ids=("risk",)),
                           label_event=lambda i: i % 2, label_time_days=lambda i: 100.0 + i)
    _result, ctx = run_stage(tmp_path, make_spec(registry, [arm("risk", label_def="survival")]), inputs, FakeAdapter())
    frame = _predictions(ctx)
    assert frame["prob"].isna().all() and frame["logit"].isna().all()
    assert frame["pred_value"].notna().all()
    assert set(frame["label_event"]) == {0, 1} and frame["label_time_days"].notna().all()
    entry = train_log(ctx)[0]
    assert entry["thresholds"] == {} and entry["primary_threshold_policy"] is None


@pytest.mark.parametrize("label_def,output", [
    ("survival", {"type": "softmax", "positive_index": 1}),
    ("survival", {"type": "regression", "value_index": 0}),
    ("qrs", {"type": "risk_score"}),
    ("primary", {"type": "segmentation_mask"}),
])
def test_inference_only_pairing_table(tmp_path, label_def, output):
    registry = tmp_path / "registry"
    write_card(registry, "m", modes=["inference_only"], output=output)
    inputs = _with_columns(build_inputs(tmp_path, model_ids=("m",)), label_event=lambda i: 0,
                           label_time_days=lambda i: 10.0, label_qrs=lambda i: 90.0)
    stage, ctx = make_context(tmp_path, make_spec(registry, [arm("m", label_def=label_def)]), inputs, FakeAdapter())
    with pytest.raises(ValueError, match="output.type"):
        stage.run(ctx)


def test_lvef_cohort_without_new_columns_still_loads(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "toy-a", modes=["inference_only"])
    inputs = build_inputs(tmp_path)
    frame = read_table(inputs["cohort_index"])
    frame = frame.drop(columns=[c for c in ("label_event", "label_time_days", "label_pr", "label_qrs", "label_qt") if c in frame])
    frame.to_parquet(inputs["cohort_index"], index=False)
    _result, ctx = run_stage(tmp_path, make_spec(registry, [arm()]), inputs, FakeAdapter())
    predictions = _predictions(ctx)
    assert predictions["prob"].notna().all() and predictions["label_qrs"].isna().all()


def test_interval_arm_runs_on_a_cohort_without_label_value(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "seg", modes=["inference_only"], output={"type": "segmentation_mask", "n_outputs": 4})
    inputs = _with_columns(build_inputs(tmp_path, model_ids=("seg",), label_value=lambda score: None),
                           label_pr=lambda i: 150.0, label_qrs=lambda i: 90.0, label_qt=lambda i: 400.0)
    _result, ctx = run_stage(tmp_path, make_spec(registry, [arm("seg", label_def="qt")]), inputs, SegAdapter())
    assert _predictions(ctx)["pred_value"].notna().all()


def test_regression_arm_without_its_label_column_names_it(tmp_path):
    registry = tmp_path / "registry"
    write_card(registry, "seg", modes=["inference_only"], output={"type": "segmentation_mask", "n_outputs": 4})
    inputs = build_inputs(tmp_path, model_ids=("seg",))
    stage, ctx = make_context(tmp_path, make_spec(registry, [arm("seg", label_def="qt")]), inputs, SegAdapter())
    with pytest.raises(ValueError, match="label_qt"):
        stage.run(ctx)
