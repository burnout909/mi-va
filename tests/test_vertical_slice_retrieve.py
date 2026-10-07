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


def test_derived_kinds_write_age_and_sex_covariates(tmp_path):
    rows = [dict(ecg(1, 10, date(2180, 1, 1), birth=2120, last_visit=date(2180, 6, 1)), gender_concept_id=8507),
            dict(ecg(2, 11, date(2180, 1, 1), birth=2150, last_visit=date(2180, 6, 1)), gender_concept_id=8532)]
    index, _ = run(tmp_path, spec(tmp_path, "death_within"), rows)
    assert index["cov_age_years"].tolist() == [60.0, 30.0]
    assert index["cov_sex_male"].tolist() == [1.0, 0.0]


def test_measurement_file_takes_the_nearest_value_in_window(tmp_path):
    labs = tmp_path / "labs.csv"
    pd.DataFrame({
        "person_id": [10, 10, 10, 11, 12],
        "measurement_datetime": ["2180-01-01 08:00", "2180-01-02 08:00", "2179-12-31 20:00", "2180-01-05 00:00", "2180-01-01 09:00"],
        "value": [500.0, 900.0, 700.0, 300.0, 0.0],
    }).to_csv(labs, index=False)
    rows = [ecg(1, 10, date(2180, 1, 1)), ecg(2, 11, date(2180, 1, 1)), ecg(3, 12, date(2180, 1, 1))]
    body = spec(tmp_path, "measurement_file", source={"path": str(labs)}, window_days=1, implausible_below=0)
    index, excluded = run(tmp_path, body, rows)
    assert index["image_occurrence_id"].tolist() == [1]
    assert index["label_value"].tolist() == [500.0] and index["label_delta_days"].tolist() == [0]
    assert excluded == {"2": "label_missing", "3": "label_implausible"}
    assert set(RetrieveStage().config_inputs(body).values()) == {labs}


def test_death_on_the_ecg_day_is_an_event_at_time_zero(tmp_path):
    d0 = date(2180, 1, 1)
    index, excluded = run(tmp_path, spec(tmp_path, "death_within"),
                          [ecg(1, 10, d0, death=d0, last_visit=d0), ecg(2, 11, d0, death=None, last_visit=date(2178, 1, 1))])
    assert index["image_occurrence_id"].tolist() == [1]
    assert (index["label_event"].iloc[0], index["label_time_days"].iloc[0]) == (1.0, 0.0)
    assert excluded == {"2": "label_implausible"}


# --- retrieve.ecg_selection: per_person all|one and the rule that picks the one ---


def _selected(tmp_path, selection, rows, kind="age_at_ecg"):
    index, excluded = run(tmp_path, spec(tmp_path, kind, ecg_selection=selection), rows)
    return dict(zip(index["person_id"], index["image_occurrence_id"])), excluded


def test_ecg_selection_first_and_last_by_date_then_id(tmp_path):
    rows = [ecg(3, 10, date(2180, 1, 5)), ecg(1, 10, date(2180, 1, 9)), ecg(2, 10, date(2180, 1, 5)),
            ecg(4, 11, date(2181, 2, 1))]
    first, excluded = _selected(tmp_path / "f", {"per_person": "one", "rule": "first"}, rows)
    assert first == {10: 2, 11: 4}  # same day: lower image_occurrence_id wins
    assert excluded == {"1": "not_selected", "3": "not_selected"}
    last, _ = _selected(tmp_path / "l", {"per_person": "one", "rule": "last"}, rows)
    assert last == {10: 1, 11: 4}


def test_ecg_selection_all_keeps_every_ecg(tmp_path):
    rows = [ecg(i, 10, date(2180, 1, i)) for i in range(1, 4)]
    index, excluded = run(tmp_path, spec(tmp_path, "age_at_ecg", ecg_selection={"per_person": "all"}), rows)
    assert sorted(index["image_occurrence_id"]) == [1, 2, 3] and excluded == {}


def test_ecg_selection_random_matches_legacy_one_per_person(tmp_path):
    rows = [ecg(i, 10 + i % 3, date(2180, 1, 1 + i)) for i in range(1, 13)]
    legacy, _ = run(tmp_path / "a", spec(tmp_path, "age_at_ecg", one_per_person=True), rows)
    new, _ = _selected(tmp_path / "b", {"per_person": "one", "rule": "random"}, rows)
    assert dict(zip(legacy["person_id"], legacy["image_occurrence_id"])) == new


def test_ecg_selection_nearest_label(tmp_path):
    from test_stage_retrieve import fake_query, row

    # person 10: |delta| 3, 0, 1 -> image 2; person 11: tie |1| on two days -> earlier ECG (image 4)
    rows = [row(1, 10, 1, 4.0, 3), row(2, 10, 2, 4.1, 0), row(3, 10, 3, 4.2, -1),
            row(4, 11, 4, 4.0, -1), row(5, 11, 5, 4.0, 1)]
    stage = RetrieveStage(query=fake_query({3: rows, 4: rows}))
    body = {"dsn_env": "/x", "modality_concept_id": 4145308, "label_concept_id": 3023103,
            "local_path_root": str(tmp_path / "dicom"), "require_local_file": False,
            "window_days": 3, "window_days_sens2": 4, "implausible_below": 1.5,
            "ecg_selection": {"per_person": "one", "rule": "nearest_label"}}
    ctx = prepare(stage, "s", "site", body, {}, tmp_path / "runs", seed=3)
    execute(stage, ctx)
    index = read_table(ctx.layout.artifact(COHORT_INDEX))
    assert dict(zip(index["person_id"], index["image_occurrence_id"])) == {10: 2, 11: 4}


@pytest.mark.parametrize("selection, kind, message", [
    ({"per_person": "some"}, "age_at_ecg", "per_person"),
    ({"per_person": "one"}, "age_at_ecg", "rule"),
    ({"per_person": "one", "rule": "median"}, "age_at_ecg", "rule"),
    ({"per_person": "all", "rule": "first"}, "age_at_ecg", "rule"),
    ({"per_person": "one", "rule": "nearest_label"}, "age_at_ecg", "nearest_label"),
])
def test_ecg_selection_rejects_bad_settings(tmp_path, selection, kind, message):
    from mival.stages.retrieve import RetrieveSpec

    with pytest.raises(ValueError, match=message):
        RetrieveSpec.from_mapping(spec(tmp_path, kind, ecg_selection=selection))


def test_ecg_selection_and_one_per_person_together_is_an_error(tmp_path):
    from mival.stages.retrieve import RetrieveSpec

    body = spec(tmp_path, "age_at_ecg", one_per_person=True, ecg_selection={"per_person": "all"})
    with pytest.raises(ValueError, match="one_per_person"):
        RetrieveSpec.from_mapping(body)
