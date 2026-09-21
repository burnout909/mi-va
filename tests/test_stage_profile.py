"""Profile stage (spec §4.2): a frozen person-level split and the event-count gate."""

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from mival.gates import EventCountError  # noqa: E402
from mival.pipeline.stage import execute, get_stage, prepare  # noqa: E402
from mival.pipeline.tables import read_table  # noqa: E402
from mival.stages.profile import COHORT_SPLIT, SUMMARY, ProfileStage, assign_splits  # noqa: E402


def cohort(n_persons=100, ecgs_per_person=2, positive_every=4):
    rows = []
    for person in range(n_persons):
        for recording in range(ecgs_per_person):
            value = 30.0 if person % positive_every == 0 else 60.0
            rows.append({
                "image_occurrence_id": person * 10 + recording, "person_id": person,
                "local_path": f"/x/{person}_{recording}.dcm", "index_datetime": None,
                "label_datetime": None, "label_delta_days": 0, "label_value": value,
                "label_primary": int(value <= 40), "label_sens1": int(value < 50),
                "label_sens2": int(value <= 40), "label_sens3": None,
            })
    return pd.DataFrame(rows)


def run(tmp_path, frame, spec=None, seed=7):
    index = tmp_path / "cohort_index.parquet"
    frame.to_parquet(index, index=False)
    stage = ProfileStage()
    ctx = prepare(stage, "s", "site-a", spec or {"min_test_positives": 5}, {"cohort_index": index}, tmp_path / "runs", seed=seed)
    execute(stage, ctx)
    return ctx, read_table(ctx.layout.artifact(COHORT_SPLIT))


def test_split_is_person_level_and_stratified(tmp_path):
    frame = cohort()
    _, split = run(tmp_path, frame)
    assert set(split.columns) == {"person_id", "split", "fold"}
    assert split["person_id"].is_unique and len(split) == 100
    test = split[split["split"] == "test"]
    assert len(test) == 20
    positives = frame.groupby("person_id")["label_primary"].max()
    assert positives[test["person_id"]].sum() == 5          # 25 positive persons, 20% of them
    assert split[split["split"] == "dev"]["fold"].nunique() == 5
    assert split[split["split"] == "test"]["fold"].isna().all()


def test_same_seed_reproduces_and_another_seed_differs(tmp_path):
    a = assign_splits(pd.DataFrame({"person_id": range(50), "label_primary": [i % 3 == 0 for i in range(50)]}), "label_primary", 0.2, 5, seed=1)
    b = assign_splits(pd.DataFrame({"person_id": range(50), "label_primary": [i % 3 == 0 for i in range(50)]}), "label_primary", 0.2, 5, seed=1)
    c = assign_splits(pd.DataFrame({"person_id": range(50), "label_primary": [i % 3 == 0 for i in range(50)]}), "label_primary", 0.2, 5, seed=2)
    assert a.equals(b) and not a.equals(c)


def test_event_count_gate_fails_the_run(tmp_path):
    with pytest.raises(EventCountError, match="test split holds"):
        run(tmp_path, cohort(), spec={"min_test_positives": 1000})


def test_summary_reports_counts(tmp_path):
    ctx, _ = run(tmp_path, cohort())
    text = ctx.layout.artifact(SUMMARY).read_text()
    assert '"test"' in text and '"n_positive_ecgs"' in text


def test_stage_is_registered():
    assert get_stage("profile").name == "profile"
