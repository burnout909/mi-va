import pytest

from mival.contract import REASON_CODES
from mival.pipeline.ledger import COLUMNS, ExclusionLedger, read_exclusions

pytest.importorskip("pyarrow")


def counter():
    state = {"n": 0}

    def clock() -> str:
        state["n"] += 1
        return f"2026-08-14T00:00:{state['n']:02d}+00:00"

    return clock


def make(stage="preprocess", codes=REASON_CODES) -> ExclusionLedger:
    return ExclusionLedger(stage=stage, allowed_codes=codes, clock=counter())


def test_columns_match_spec_section_3_6():
    assert COLUMNS == (
        "image_occurrence_id",
        "person_id",
        "stage",
        "reason_code",
        "detail",
        "timestamp",
    )


def test_records_a_row():
    ledger = make()
    ledger.record("img1", "p1", "unit_missing", "no sensitivity tag")
    assert len(ledger) == 1
    assert ledger.rows[0]["stage"] == "preprocess"
    assert ledger.rows[0]["reason_code"] == "unit_missing"


def test_rejects_a_code_outside_the_stage_vocabulary():
    # A free-text reason makes the ledger unqueryable, and an excluded record
    # nobody can count silently biases the cohort.
    ledger = make()
    with pytest.raises(ValueError, match="not a reason_code declared by stage"):
        ledger.record("img1", "p1", "looked_weird")


def test_each_stage_carries_its_own_vocabulary():
    from mival.stages.models import REASON_CODES as MODEL_CODES

    models = make(stage="models", codes=MODEL_CODES)
    models.record("img1", "p1", "tensor_missing")
    with pytest.raises(ValueError):
        models.record("img2", "p1", "unit_missing")  # a preprocess code


def test_counts_tally_by_reason_code():
    ledger = make()
    ledger.record("a", "p1", "unit_missing")
    ledger.record("b", "p1", "unit_missing")
    ledger.record("c", "p2", "lead_unavailable")
    assert ledger.counts() == {"unit_missing": 2, "lead_unavailable": 1}


def test_flush_writes_an_empty_part_when_nothing_was_excluded(tmp_path):
    # An absent part file is ambiguous — "nothing excluded" or "crashed before
    # writing" — and the STARD flow needs those distinguished.
    path = make().flush(tmp_path / "s1" / "exclusions" / "preprocess" / "h.parquet")
    assert path.is_file()
    frame = read_exclusions(tmp_path / "s1" / "exclusions")
    assert len(frame) == 0
    assert list(frame.columns) == list(COLUMNS)


def test_read_exclusions_reassembles_one_table_across_stages(tmp_path):
    from mival.stages.models import REASON_CODES as MODEL_CODES

    root = tmp_path / "s1" / "exclusions"
    pre = make()
    pre.record("a", "p1", "unit_missing")
    pre.record("b", "p2", "duration_short")
    pre.flush(root / "preprocess" / "h1.parquet")

    mod = make(stage="models", codes=MODEL_CODES)
    mod.record("c", "p3", "tensor_missing")
    mod.flush(root / "models" / "h2.parquet")

    frame = read_exclusions(root)
    assert len(frame) == 3
    assert set(frame["stage"]) == {"preprocess", "models"}
    assert list(frame.columns) == list(COLUMNS)


def test_read_exclusions_on_a_missing_root_returns_an_empty_table(tmp_path):
    frame = read_exclusions(tmp_path / "never-created")
    assert len(frame) == 0
    assert list(frame.columns) == list(COLUMNS)


def test_rerunning_a_config_hash_replaces_its_rows(tmp_path):
    root = tmp_path / "s1" / "exclusions"
    path = root / "preprocess" / "h1.parquet"
    first = make()
    first.record("a", "p1", "unit_missing")
    first.record("b", "p2", "unit_missing")
    first.flush(path)

    second = make()
    second.record("a", "p1", "unit_missing")
    second.flush(path)

    # Parts are replaced, not appended to, so a re-run cannot double-count.
    assert len(read_exclusions(root)) == 1


def test_person_id_may_be_absent(tmp_path):
    ledger = make()
    ledger.record("orphan-img", None, "unit_missing")
    assert ledger.rows[0]["person_id"] is None
