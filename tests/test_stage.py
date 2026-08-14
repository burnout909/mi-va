import json
import sys

import pytest

from mival.pipeline.layout import FAILED_MANIFEST_NAME
from mival.pipeline.manifest import RunManifest
from mival.pipeline.stage import (
    STAGE_ORDER,
    Stage,
    StageContext,
    StageResult,
    execute,
    get_stage,
    prepare,
    stage_names,
)

pytest.importorskip("pyarrow")

CODES = frozenset({"bad_record"})


class Recorder(Stage):
    """A stage that writes one artifact and drops one record."""

    name = "preprocess"
    reason_codes = CODES

    def __init__(self, boom: bool = False):
        self.boom = boom
        self.calls = 0

    def run(self, ctx: StageContext) -> StageResult:
        self.calls += 1
        if self.boom:
            raise RuntimeError("stage exploded")
        out = ctx.layout.artifact("out.txt")
        out.write_text("hello")
        ctx.ledger.record("img-9", "p-9", "bad_record", "synthetic")
        return StageResult(outputs=[out], counts={"in": 2, "out": 1})


def context(tmp_path, stage, spec=None, inputs=None):
    return prepare(
        stage=stage,
        study_id="s1",
        site="mimic",
        spec=spec or {"fs": 500},
        inputs=inputs or {},
        runs_root=tmp_path,
        seed=7,
    )


def test_registry_holds_strings_so_importing_it_loads_no_backend():
    # Same constraint as the Plan 1 adapter registry: the environment with
    # torch does not have TensorFlow, and mival.pipeline must import in both.
    import mival.pipeline.stage as module

    assert all(isinstance(value, str) for value in module._STAGES.values())
    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules


def test_stage_order_covers_all_six_stages_of_spec_section_3_3():
    assert STAGE_ORDER == (
        "retrieve",
        "profile",
        "preprocess",
        "models",
        "evaluate",
        "misclassify",
    )


def test_stage_names_lists_only_runnable_stages():
    assert stage_names() == ["preprocess", "models", "evaluate"]


def test_unbuilt_stage_reports_why_not_just_unknown():
    with pytest.raises(KeyError, match="Data4Life"):
        get_stage("retrieve")


def test_unknown_stage_is_distinguishable_from_an_unbuilt_one():
    with pytest.raises(KeyError, match="unknown stage"):
        get_stage("preprocesss")


def test_registered_stages_resolve_to_real_classes():
    for name in stage_names():
        stage = get_stage(name)
        assert stage.name == name


def test_execute_writes_artifact_manifest_and_ledger(tmp_path):
    stage = Recorder()
    ctx = context(tmp_path, stage)
    summary = execute(stage, ctx)

    assert summary["status"] == "ok"
    assert ctx.layout.artifact("out.txt").read_text() == "hello"
    assert ctx.layout.exclusions_path.is_file()

    manifest = RunManifest.read(ctx.layout.manifest_path)
    assert manifest.counts == {"in": 2, "out": 1, "excluded": 1}
    assert manifest.seed == 7
    assert manifest.output_artifacts[0].path == "artifacts/out.txt"
    assert manifest.compute["wall_time_s"] >= 0
    assert manifest.started_at and manifest.ended_at


def test_excluded_count_comes_from_the_ledger_not_the_stage(tmp_path):
    class Liar(Recorder):
        def run(self, ctx):
            result = super().run(ctx)
            result.counts["excluded"] = 999
            return result

    stage = Liar()
    ctx = context(tmp_path, stage)
    execute(stage, ctx)
    # The ledger is the authority: a stage that miscounts its own exclusions
    # would break the STARD participant flow.
    assert RunManifest.read(ctx.layout.manifest_path).counts["excluded"] == 1


def test_second_run_of_the_same_config_hash_is_skipped(tmp_path):
    stage = Recorder()
    execute(stage, context(tmp_path, stage))
    summary = execute(stage, context(tmp_path, stage))
    assert summary["status"] == "skipped"
    assert stage.calls == 1


def test_force_re_runs(tmp_path):
    stage = Recorder()
    execute(stage, context(tmp_path, stage))
    summary = execute(stage, context(tmp_path, stage), force=True)
    assert summary["status"] == "ok"
    assert stage.calls == 2


def test_a_changed_spec_gets_a_different_run_dir(tmp_path):
    stage = Recorder()
    first = context(tmp_path, stage, spec={"fs": 500})
    second = context(tmp_path, stage, spec={"fs": 250})
    assert first.layout.config_hash != second.layout.config_hash


def test_a_changed_input_artifact_gets_a_different_run_dir(tmp_path):
    stage = Recorder()
    source = tmp_path / "cohort.parquet"
    source.write_bytes(b"one")
    first = context(tmp_path, stage, inputs={"cohort_index": source})
    source.write_bytes(b"two")
    second = context(tmp_path, stage, inputs={"cohort_index": source})
    assert first.layout.config_hash != second.layout.config_hash


def test_a_failed_run_does_not_write_the_completion_marker(tmp_path):
    stage = Recorder(boom=True)
    ctx = context(tmp_path, stage)
    with pytest.raises(RuntimeError, match="stage exploded"):
        execute(stage, ctx)

    # Writing manifest.json here would make the crash look like a finished run
    # and skip it forever.
    assert not ctx.layout.manifest_path.exists()
    failed = ctx.layout.run_dir / FAILED_MANIFEST_NAME
    assert failed.is_file()
    assert "RuntimeError: stage exploded" in json.loads(failed.read_text())["errors"][0]


def test_a_failed_run_is_re_run_not_skipped(tmp_path):
    boom = Recorder(boom=True)
    ctx = context(tmp_path, boom)
    with pytest.raises(RuntimeError):
        execute(boom, ctx)

    good = Recorder()
    summary = execute(good, context(tmp_path, good))
    assert summary["status"] == "ok"
    assert good.calls == 1


def test_output_outside_the_run_dir_is_rejected(tmp_path):
    class Escapee(Recorder):
        def run(self, ctx):
            stray = tmp_path / "stray.txt"
            stray.write_text("x")
            return StageResult(outputs=[stray])

    stage = Escapee()
    with pytest.raises(ValueError, match="outside its run directory"):
        execute(stage, context(tmp_path, stage))


def test_missing_input_names_the_stage_and_the_slot(tmp_path):
    stage = Recorder()
    ctx = context(tmp_path, stage)
    with pytest.raises(KeyError, match="cohort_index"):
        ctx.input_path("cohort_index")
