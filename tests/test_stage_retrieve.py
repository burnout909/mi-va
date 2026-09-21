"""Retrieve stage (spec §4.1) against an injected query, so no database is needed."""

from datetime import date, datetime

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from mival.pipeline.ledger import read_exclusions  # noqa: E402
from mival.pipeline.layout import exclusions_root  # noqa: E402
from mival.pipeline.stage import execute, get_stage, prepare  # noqa: E402
from mival.pipeline.tables import read_table  # noqa: E402
from mival.stages.retrieve import (  # noqa: E402
    COHORT_INDEX,
    COHORT_INDEX_COLUMNS,
    SUMMARY,
    RetrieveStage,
    resolve_local_path,
)

CDM_PREFIX = "/elsewhere/mimic-iv-ecg-dcm/files/p1000/p10000032"


def row(image_id, person, day, value, delta, path=None):
    return {
        "image_occurrence_id": image_id,
        "person_id": person,
        "local_path": path or f"{CDM_PREFIX}/s{image_id}/{image_id}.dcm",
        "index_datetime": date(2180, 7, day),
        "label_datetime": None if value is None else datetime(2180, 7, day + delta, 9, 0),
        "label_delta_days": None if value is None else delta,
        "label_value": value,
    }


def fake_query(rows_by_window):
    """Return the frame for the window the SQL was run with."""
    calls = []

    def query(sql, params):
        calls.append((sql, dict(params)))
        return pd.DataFrame(rows_by_window[params["window_days"]])

    query.calls = calls
    return query


def make_spec(tmp_path, **overrides):
    spec = {
        "dsn_env": "/nonexistent/micdm.env",
        "schema": "cdm",
        "label_concept_id": 3027172,
        "modality_concept_id": 4145308,
        "window_days": 7,
        "window_days_sens2": 30,
        "implausible_below": 5,
        "local_path_root": str(tmp_path / "dicom"),
        "require_local_file": False,
    }
    spec.update(overrides)
    return spec


def run(tmp_path, query, **overrides):
    stage = RetrieveStage(query=query)
    ctx = prepare(stage, "s", "site-a", make_spec(tmp_path, **overrides), {}, tmp_path / "runs", seed=1)
    execute(stage, ctx)
    return ctx, read_table(ctx.layout.artifact(COHORT_INDEX))


def test_local_path_is_rerooted_under_the_site_root():
    cdm = f"{CDM_PREFIX}/s1/1.dcm"
    assert resolve_local_path(cdm, "/scratch/dicom") == "/scratch/dicom/files/p1000/p10000032/s1/1.dcm"
    assert resolve_local_path(cdm, "/other/root") == "/other/root/files/p1000/p10000032/s1/1.dcm"
    assert resolve_local_path("/no/segment.dcm", "/scratch/dicom") is None


def test_labels_are_thresholded_and_sens2_uses_the_wide_window(tmp_path):
    narrow = [row(1, 10, 1, 35.0, 2), row(2, 11, 4, 55.0, -3), row(3, 12, 1, None, None)]
    wide = [row(1, 10, 1, 35.0, 2), row(2, 11, 4, 55.0, -3), row(3, 12, 1, 45.0, 20)]
    _, index = run(tmp_path, fake_query({7: narrow, 30: wide}))
    assert list(index.columns) == list(COHORT_INDEX_COLUMNS)
    assert index["image_occurrence_id"].tolist() == [1, 2]
    assert index["label_primary"].tolist() == [1, 0]
    assert index["label_sens1"].tolist() == [1, 0]
    assert index["label_sens2"].tolist() == [1, 0]
    assert index["label_sens3"].isna().all()
    assert index["label_value"].tolist() == [35.0, 55.0]


def test_exclusions_are_recorded_in_order(tmp_path):
    rows = [row(1, 10, 1, None, None), row(2, 11, 1, 0.0, 0), row(3, 12, 1, 50.0, 1, path="/no/segment.dcm"), row(4, 13, 1, 50.0, 1)]
    ctx, index = run(tmp_path, fake_query({7: rows, 30: rows}))
    ledger = read_exclusions(exclusions_root(tmp_path / "runs", "s"))
    assert dict(zip(ledger["image_occurrence_id"], ledger["reason_code"])) == {
        "1": "label_missing", "2": "label_implausible", "3": "path_unresolvable"}
    assert index["image_occurrence_id"].tolist() == [4]


def test_missing_files_are_excluded_only_when_required(tmp_path):
    rows = [row(1, 10, 1, 30.0, 0), row(2, 11, 1, 30.0, 0)]
    present = tmp_path / "dicom" / "files" / "p1000" / "p10000032" / "s1" / "1.dcm"
    present.parent.mkdir(parents=True)
    present.write_bytes(b"")
    _, index = run(tmp_path, fake_query({7: rows, 30: rows}), require_local_file=True)
    assert index["image_occurrence_id"].tolist() == [1]
    _, index = run(tmp_path, fake_query({7: rows, 30: rows}), require_local_file=False)
    assert len(index) == 2


def test_no_labels_at_all_fails_loudly(tmp_path):
    rows = [row(1, 10, 1, None, None)]
    with pytest.raises(ValueError, match="no label"):
        run(tmp_path, fake_query({7: rows, 30: rows}))


def test_sql_is_copied_and_summary_written(tmp_path):
    rows = [row(1, 10, 1, 30.0, 0)]
    query = fake_query({7: rows, 30: rows})
    ctx, _ = run(tmp_path, query)
    assert query.calls[0][1]["window_days"] == 7 and query.calls[1][1]["window_days"] == 30
    assert query.calls[0][1]["modality_concept_id"] == 4145308
    assert "cdm.image_occurrence" in query.calls[0][0]
    assert (ctx.layout.artifact("sql", "window_7.sql")).is_file()
    summary = ctx.layout.artifact(SUMMARY)
    assert summary.is_file() and '"n_out": 1' in summary.read_text()


def test_stage_is_registered():
    assert get_stage("retrieve").name == "retrieve"
