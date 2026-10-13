"""Retrieve stage (spec §4.1) against an injected query, so no database is needed."""

import decimal
import json
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
    _decimals_to_float,
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


SNAPSHOT = {"n_rows": 1234, "max_measurement_id": 99}


def fake_query(rows_by_window, snapshot=None):
    """Return the frame for the window the SQL was run with, or the snapshot frame."""
    calls = []
    snapshot = SNAPSHOT if snapshot is None else snapshot

    def query(sql, params):
        calls.append((sql, dict(params)))
        if "window_days" not in params:
            return pd.DataFrame([snapshot])
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


def test_the_summary_carries_the_label_table_snapshot(tmp_path):
    # Spec §4.1: a cohort is reproducible only against the snapshot it read.
    rows = [row(1, 10, 1, 30.0, 0)]
    query = fake_query({7: rows, 30: rows}, snapshot={
        "n_rows": decimal.Decimal("4321"), "max_measurement_id": decimal.Decimal("777")})
    ctx, _ = run(tmp_path, query)

    snapshot_sql, snapshot_params = query.calls[2]
    assert "cdm.measurement" in snapshot_sql and "window_days" not in snapshot_params
    assert snapshot_params == {"label_concept_id": 3027172}

    payload = json.loads(ctx.layout.artifact(SUMMARY).read_text())
    assert payload["snapshot"] == {"label_rows": 4321, "max_measurement_id": 777}
    assert isinstance(payload["snapshot"]["label_rows"], int)
    copied = ctx.layout.artifact("sql", "label_snapshot.sql")
    assert copied.is_file() and "3027172" in copied.read_text()


def test_stage_is_registered():
    assert get_stage("retrieve").name == "retrieve"


def test_decimals_to_float_converts_decimal_columns_only():
    frame = pd.DataFrame({
        "label_value": [decimal.Decimal("35.0"), None, decimal.Decimal("55.5")],
        "image_occurrence_id": [1, 2, 3],
    })
    out = _decimals_to_float(frame)
    assert out["label_value"].dtype == "float64"
    assert out["label_value"].tolist()[0] == 35.0
    assert pd.isna(out["label_value"].tolist()[1])
    assert out["label_value"].tolist()[2] == 55.5
    assert out["image_occurrence_id"].dtype == frame["image_occurrence_id"].dtype
    assert out["image_occurrence_id"].tolist() == [1, 2, 3]


def test_without_a_root_the_cdm_path_is_used_as_it_is():
    assert resolve_local_path("/mnt/ecg/2024/01/a.dcm", None) == "/mnt/ecg/2024/01/a.dcm"
    assert resolve_local_path("  /mnt/ecg/a.dcm ", None) == "/mnt/ecg/a.dcm"
    assert resolve_local_path("", None) is None
    assert resolve_local_path(None, None) is None


def test_null_root_keeps_paths_without_a_files_segment(tmp_path):
    present = tmp_path / "site" / "2024" / "a.dcm"
    present.parent.mkdir(parents=True)
    present.write_bytes(b"")
    rows = [row(1, 10, 1, 30.0, 0, path=str(present)), row(2, 11, 1, 30.0, 0, path=str(tmp_path / "gone.dcm")),
            row(3, 12, 1, 30.0, 0, path="  ")]
    _, index = run(tmp_path, fake_query({7: rows, 30: rows}), local_path_root=None, require_local_file=True)
    assert index["local_path"].tolist() == [str(present)]
    ledger = read_exclusions(exclusions_root(tmp_path / "runs", "s"))
    assert dict(zip(ledger["image_occurrence_id"], ledger["reason_code"])) == {
        "2": "file_missing", "3": "path_unresolvable"}


def test_sqlserver_uses_its_own_label_query(tmp_path):
    rows = [row(1, 10, 1, 30.0, 0)]
    query = fake_query({7: rows, 30: rows})
    run(tmp_path, query, dbms="sqlserver", schema="CDM2025.dbo")
    label_sql = query.calls[0][0]
    assert "outer apply (" in label_sql and "join lateral" not in label_sql
    assert "CDM2025.dbo.image_occurrence" in label_sql
    snapshot_sql = query.calls[2][0]
    assert "CDM2025.dbo.measurement" in snapshot_sql


def test_postgresql_is_the_default_and_keeps_the_lateral_query(tmp_path):
    rows = [row(1, 10, 1, 30.0, 0)]
    query = fake_query({7: rows, 30: rows})
    run(tmp_path, query)
    assert "join lateral" in query.calls[0][0]


def test_dbms_and_schema_are_checked():
    from mival.stages.retrieve import RetrieveSpec

    base = {"dsn_env": "x.env", "modality_concept_id": 4145308}
    with pytest.raises(ValueError, match="retrieve.dbms"):
        RetrieveSpec.from_mapping({**base, "dbms": "oracle"})
    with pytest.raises(ValueError, match="plain identifier"):
        RetrieveSpec.from_mapping({**base, "schema": "CDM.dbo"})
    with pytest.raises(ValueError, match="plain identifier"):
        RetrieveSpec.from_mapping({**base, "dbms": "sqlserver", "schema": "cdm; drop table x"})
    spec = RetrieveSpec.from_mapping({**base, "dbms": "sqlserver", "schema": "CDM.dbo"})
    assert spec.schema == "CDM.dbo" and spec.local_path_root is None


def test_connection_file_accepts_pg_and_db_names(tmp_path):
    from mival.stages.retrieve import read_connection_settings

    pg = tmp_path / "pg.env"
    pg.write_text("PGHOST=h\nPGPORT=5432\nPGDATABASE=d\nPGUSER=u\nPGPASSWORD='p'\n")
    db = tmp_path / "db.env"
    db.write_text("export DB_HOST=h\nDB_PORT=1433\nDB_NAME=d\nDB_USER=u\nDB_PASSWORD=\"p\"\n# comment\n")
    assert read_connection_settings(pg) == {"host": "h", "port": "5432", "dbname": "d", "user": "u", "password": "p"}
    assert read_connection_settings(db) == {"host": "h", "port": "1433", "dbname": "d", "user": "u", "password": "p"}


def test_null_implausible_below_excludes_nothing_as_implausible(tmp_path):
    rows = [row(1, 10, 1, 2.0, 0), row(2, 11, 1, 35.0, 0)]
    _, index = run(tmp_path, fake_query({7: rows, 30: rows}), implausible_below=None)
    assert index["image_occurrence_id"].tolist() == [1, 2]
    ledger_root = exclusions_root(tmp_path / "runs", "s")
    assert not ledger_root.exists() or "label_implausible" not in set(read_exclusions(ledger_root)["reason_code"])


def test_implausible_below_defaults_to_five(tmp_path):
    rows = [row(1, 10, 1, 5.0, 0), row(2, 11, 1, 35.0, 0)]
    _, index = run(tmp_path, fake_query({7: rows, 30: rows}), implausible_below=5)
    assert index["image_occurrence_id"].tolist() == [2]
    from mival.stages.retrieve import RetrieveSpec
    assert RetrieveSpec.from_mapping({"dsn_env": "x", "modality_concept_id": 1}).implausible_below == 5.0
