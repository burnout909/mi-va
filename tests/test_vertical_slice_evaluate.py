"""Survival metrics and per-label regression cuts (plan 2026-10-07, Task 5)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from mival.metrics import category_of  # noqa: E402
from mival.metrics.survival import auroc_at_horizon, harrell_c  # noqa: E402
from mival.pipeline.runkey import RunKey  # noqa: E402
from mival.pipeline.stage import execute, prepare  # noqa: E402
from mival.stages.evaluate import METRICS_LONG, EvaluateStage, _Settings  # noqa: E402
from mival.pipeline.tables import read_table  # noqa: E402


def _naive_c(time, event, risk):
    num = den = 0.0
    for i in range(len(time)):
        if not event[i]:
            continue
        for j in range(len(time)):
            if j != i and (time[j] > time[i] or (time[j] == time[i] and not event[j])):
                den += 1
                num += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
    return num / den


def test_harrell_c_known_example():
    time = np.array([5, 10, 10, 20, 30])
    event = np.array([1, 1, 0, 1, 0])
    risk = np.array([0.9, 0.5, 0.7, 0.5, 0.1])
    assert harrell_c(time, event, risk) == pytest.approx(_naive_c(time, event, risk))
    rng = np.random.default_rng(0)
    t = rng.integers(1, 50, 300)
    e = rng.integers(0, 2, 300)
    r = np.round(rng.normal(size=300), 1)
    assert harrell_c(t, e, r) == pytest.approx(_naive_c(t, e, r))
    assert harrell_c(t, e, -t.astype(float)) == pytest.approx(_naive_c(t, e, -t.astype(float)))


def test_auroc_at_horizon_excludes_early_censored():
    time = np.array([10, 365, 100, 365])
    event = np.array([1, 0, 0, 0])
    risk = np.array([0.9, 0.1, 0.95, 0.2])  # the early-censored record would break perfect ranking
    assert auroc_at_horizon(time, event, risk, 365) == pytest.approx(1.0)


def test_survival_metrics_are_categorised():
    assert category_of("c_index") == "survival"
    assert category_of("auroc_horizon@365") == "survival"


def test_regression_cuts_mapping():
    settings = _Settings({"regression_cuts": {"value": [65], "qrs": [120]}}, (0.05,), 10)
    assert settings.cuts_for("value") == (65.0,) and settings.cuts_for("qrs") == (120.0,)
    with pytest.raises(ValueError, match="regression_cuts"):
        settings.cuts_for("pr")
    assert _Settings({"regression_cuts": [40]}, (0.05,), 10).cuts_for("pr") == (40.0,)
    with pytest.raises(ValueError, match="regression_cuts"):
        _Settings({"regression_cuts": {"value": []}}, (0.05,), 10)


def _write(directory: Path, label_def, frame):
    directory.mkdir(parents=True, exist_ok=True)
    key = RunKey(site="mimic", model_id="m", training_mode="inference_only", recipe_id="r1",
                 perturbation_id="none", label_def=label_def, split="test", fold=None)
    frame.to_parquet(directory / f"{key.to_string()}.parquet", index=False)


def _base(n, rng):
    return pd.DataFrame({
        "image_occurrence_id": [f"i{k}" for k in range(n)], "person_id": [f"p{k}" for k in range(n)],
        "split": "test", "fold": None, "label_primary": np.nan, "label_sens1": np.nan,
        "label_sens2": np.nan, "label_sens3": np.nan, "label_value": np.nan,
        "prob": np.nan, "logit": np.nan, "model_id": "m", "training_mode": "inference_only",
        "recipe_id": "r1", "perturbation_id": "none", "seed": 1, "site": "mimic",
    })


def _run(tmp_path, frame, label_def, spec):
    predictions = tmp_path / "predictions"
    _write(predictions, label_def, frame)
    split = tmp_path / "split.parquet"
    frame[["person_id", "split"]].to_parquet(split, index=False)
    stage = EvaluateStage()
    ctx = prepare(stage, "s", "mimic", spec, {"cohort_split": split}, tmp_path / "runs", seed=3)
    ctx.inputs = dict(ctx.inputs)
    ctx.inputs["predictions"] = predictions
    ctx.layout.create_dirs()
    stage.run(ctx)
    return read_table(ctx.layout.artifact(METRICS_LONG))


def test_evaluate_survival_rows(tmp_path):
    rng = np.random.default_rng(1)
    n = 200
    frame = _base(n, rng)
    risk = rng.normal(size=n)
    event = (rng.random(n) < 1 / (1 + np.exp(-2 * risk))).astype(int)
    frame["label_event"] = event
    frame["label_time_days"] = np.where(event == 1, rng.integers(1, 365, n), 365).astype(float)
    frame["pred_value"] = risk
    table = _run(tmp_path, frame, "survival", {"outcomes": {"death_1y": None}, "bootstrap_replicates": 50,
                                               "figures": True, "horizon_days": 365})
    assert set(table["category"]) == {"survival"}
    c = table[table["metric"] == "c_index"].iloc[0]
    assert 0.6 < c["value"] < 1.0 and c["ci_lo"] < c["value"] < c["ci_hi"]
    assert c["n_events"] == int(event.sum())
    assert "auroc_horizon@365" in set(table["metric"])


def test_evaluate_interval_arm_uses_its_own_label_and_cut(tmp_path):
    rng = np.random.default_rng(2)
    n = 120
    frame = _base(n, rng)
    frame["label_qrs"] = rng.normal(100, 15, n)
    frame["pred_value"] = frame["label_qrs"] + rng.normal(5, 3, n)
    table = _run(tmp_path, frame, "qrs", {"outcomes": {"qrs": None}, "bootstrap_replicates": 20,
                                          "regression_cuts": {"qrs": [120]}, "figures": False})
    metrics = dict(zip(table["metric"], table["value"]))
    assert metrics["bias"] == pytest.approx(5.0, abs=1.0)
    assert "auroc_below@120" in metrics


def test_harrell_c_counts_same_time_censoring_as_outliving_the_event():
    # lifelines / survival::concordance: a record censored at the event's time outlived it.
    assert harrell_c(np.array([1, 1, 2]), np.array([1, 0, 1]), np.array([2.0, 1.0, 3.0])) == pytest.approx(0.5)
    time = np.array([3, 3, 3, 5])
    event = np.array([1, 1, 0, 0])
    risk = np.array([0.9, 0.1, 0.5, 0.0])
    # events at 3 vs censored at 3 (risk .5) and censored at 5 (risk 0); event-event ties excluded
    assert harrell_c(time, event, risk) == pytest.approx(3 / 4)


def test_rank_auroc_matches_the_discrimination_auroc():
    from mival.metrics.discrimination import auroc
    from mival.metrics.survival import rank_auroc

    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500), 2)  # ties on purpose
    assert rank_auroc(y, s) == pytest.approx(auroc(y, s))
