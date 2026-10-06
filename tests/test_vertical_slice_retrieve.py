"""Derived label kinds and one ECG per person (plan 2026-10-07, Task 2)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from mival.pipeline.layout import exclusions_root  # noqa: E402
from mival.pipeline.ledger import read_exclusions  # noqa: E402
from mival.pipeline.stage import execute, get_stage, prepare  # noqa: E402
from mival.pipeline.tables import read_table  # noqa: E402
from mival.stages.profile import ProfileStage  # noqa: E402
from mival.stages.retrieve import COHORT_INDEX, RetrieveStage  # noqa: E402

PREFIX = "/x/mimic-iv-ecg-dcm/files/p1000/p10000032"


def ecg(image_id, person, day, birth=1950, death=None, last_visit=None):
    return {
        "image_occurrence_id": image_id, "person_id": person,
        "local_path": f"{PREFIX}/s{40000000 + image_id}/{40000000 + image_id}.dcm",
        "index_datetime": day, "year_of_birth": birth,
        "death_date": death, "last_visit_end": last_visit,
    }


def query_of(rows):
    def query(sql, params):
        assert "year_of_birth" in sql and "%(modality_concept_id)s" in sql
        return pd.DataFrame(rows)
    return query


def spec(tmp_path, kind, **extra):
    body = {"dsn_env": "/nonexistent", "modality_concept_id": 4145308,
            "local_path_root": str(tmp_path / "dicom"), "require_local_file": False,
            "label_source": {"kind": kind, **extra.pop("source", {})}}
    body.update(extra)
    return body


def run(tmp_path, body, rows):
    stage = RetrieveStage(query=query_of(rows))
    ctx = prepare(stage, "s", "site", body, {}, tmp_path / "runs", seed=7)
    execute(stage, ctx)
    ledger = read_exclusions(exclusions_root(tmp_path / "runs", "s"))
    return read_table(ctx.layout.artifact(COHORT_INDEX)), dict(zip(ledger["image_occurrence_id"], ledger["reason_code"])) if len(ledger) else {}


def test_age_label(tmp_path):
    index, _ = run(tmp_path, spec(tmp_path, "age_at_ecg"), [ecg(1, 10, date(2180, 7, 1), birth=2120)])
    assert index["label_value"].tolist() == [60.0]


def test_derived_kind_leaves_lvef_cutoffs_nan(tmp_path):
    index, _ = run(tmp_path, spec(tmp_path, "age_at_ecg"), [ecg(1, 10, date(2180, 7, 1))])
    for column in ("label_primary", "label_sens1", "label_sens2", "label_sens3", "label_delta_days"):
        assert index[column].isna().all(), column


def test_death_within_event_and_censoring(tmp_path):
    d0 = date(2180, 1, 1)
    rows = [
        ecg(1, 10, d0, death=date(2180, 6, 1), last_visit=date(2180, 5, 1)),     # dies inside the year
        ecg(2, 11, d0, death=date(2181, 1, 1), last_visit=date(2180, 12, 1)),    # dies on day 366: censored at 365
        ecg(3, 12, d0, death=None, last_visit=date(2180, 2, 1)),                 # follow-up ends at last visit + 365
        ecg(4, 13, d0, death=None, last_visit=date(2179, 3, 1)),                 # last visit + 365 is 59 days after the ECG
        ecg(5, 14, d0, death=None, last_visit=None),                             # never followed up
        ecg(6, 15, d0, death=date(2180, 12, 31), last_visit=date(2180, 1, 2)),  # death on day 365 counts
    ]
    index, excluded = run(tmp_path, spec(tmp_path, "death_within"), rows)
    got = {int(r.image_occurrence_id): (int(r.label_event), int(r.label_time_days)) for r in index.itertuples()}
    assert got == {1: (1, 152), 2: (0, 365), 3: (0, 365), 4: (0, 59), 6: (1, 365)}
    assert excluded == {"5": "label_implausible"}


def test_machine_measurement_join_and_exclusion(tmp_path):
    csv = tmp_path / "mm.csv"
    pd.DataFrame({
        "study_id": [40000001, 40000002, 40000003],
        "p_onset": [40, 29999, 40], "qrs_onset": [200, 200, 200],
        "qrs_end": [300, 300, 300], "t_end": [600, 600, 29999],
    }).to_csv(csv, index=False)
    rows = [ecg(1, 10, date(2180, 1, 1)), ecg(2, 11, date(2180, 1, 1)), ecg(3, 12, date(2180, 1, 1)), ecg(4, 13, date(2180, 1, 1))]
    index, excluded = run(tmp_path, spec(tmp_path, "machine_measurement", source={"path": str(csv)}), rows)
    assert index["image_occurrence_id"].tolist() == [1]
    assert (index["label_pr"].iloc[0], index["label_qrs"].iloc[0], index["label_qt"].iloc[0]) == (160.0, 100.0, 400.0)
    assert excluded == {"2": "label_implausible", "3": "label_implausible", "4": "label_missing"}


def test_machine_measurement_file_enters_config_hash(tmp_path):
    csv = tmp_path / "mm.csv"
    csv.write_text("study_id,p_onset,qrs_onset,qrs_end,t_end\n")
    body = spec(tmp_path, "machine_measurement", source={"path": str(csv)})
    assert set(RetrieveStage().config_inputs(body).values()) == {csv}
    assert RetrieveStage().config_inputs(spec(tmp_path, "age_at_ecg")) == {}


def test_one_per_person_is_seeded(tmp_path):
    rows = [ecg(i, 10 + i % 3, date(2180, 1, 1 + i)) for i in range(1, 13)]
    body = spec(tmp_path, "age_at_ecg", one_per_person=True)
    first, excluded = run(tmp_path, body, rows)
    assert first["person_id"].is_unique and len(first) == 3
    assert set(excluded.values()) == {"not_selected"} and len(excluded) == 9
    second, _ = run(tmp_path / "again", body, rows)
    assert first["image_occurrence_id"].tolist() == second["image_occurrence_id"].tolist()


def test_profile_stratify_none(tmp_path):
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({"image_occurrence_id": range(50), "person_id": range(50),
                  "label_primary": [np.nan] * 50, "label_value": np.arange(50.0)}).to_parquet(cohort)
    stage = ProfileStage()
    ctx = prepare(stage, "s", "site", {"stratify_on": "none", "min_test_positives": 0},
                  {"cohort_index": cohort}, tmp_path / "runs", seed=1)
    execute(stage, ctx)
    split = read_table(ctx.layout.artifact("cohort_split.parquet"))
    assert len(split) == 50 and (split["split"] == "test").sum() == 10


def test_measurement_kind_honours_one_per_person(tmp_path):
    from test_stage_retrieve import fake_query, row

    rows = [row(i, 10 + i % 2, 1 + i, 4.0 + i / 10, 0) for i in range(1, 7)]
    stage = RetrieveStage(query=fake_query({1: rows, 2: rows}))
    body = {"dsn_env": "/x", "modality_concept_id": 4145308, "label_concept_id": 3023103,
            "local_path_root": str(tmp_path / "dicom"), "require_local_file": False,
            "window_days": 1, "window_days_sens2": 2, "implausible_below": 1.5, "one_per_person": True}
    ctx = prepare(stage, "s", "site", body, {}, tmp_path / "runs", seed=3)
    execute(stage, ctx)
    index = read_table(ctx.layout.artifact(COHORT_INDEX))
    assert index["person_id"].is_unique and len(index) == 2
