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
    # misclassify is in STAGE_ORDER but unregistered (excluded 2026-09-22).
    assert stage_names() == ["retrieve", "profile", "preprocess", "models", "evaluate"]


def test_the_excluded_misclassify_stage_fails_loudly():
    with pytest.raises(KeyError, match="misclassify.*excluded"):
        get_stage("misclassify")


def test_unknown_stage_name_raises_a_clear_key_error():
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


def test_config_inputs_enter_the_config_hash(tmp_path):
    """Editing a file the stage discovers itself must invalidate the run.

    The ModelCard registry arrives as a directory rather than as --input, so
    without this hook a card edit left config_hash unchanged and the stale run
    was skipped forever.
    """
    registry = tmp_path / "registry"
    registry.mkdir()
    card = registry / "some-model.json"
    card.write_text('{"model_id": "a"}')

    class Registered(Recorder):
        def config_inputs(self, spec):
            return {f"modelcard:{p.stem}": p for p in sorted(registry.glob("*.json"))}

    stage = Registered()
    before = context(tmp_path, stage).layout.config_hash
    card.write_text('{"model_id": "a", "threshold": 0.5}')
    after = context(tmp_path, stage).layout.config_hash
    assert before != after


def test_config_inputs_are_recorded_in_the_manifest(tmp_path):
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "some-model.json").write_text('{"model_id": "a"}')

    class Registered(Recorder):
        def config_inputs(self, spec):
            return {"modelcard:some-model": registry / "some-model.json"}

    stage = Registered()
    ctx = context(tmp_path, stage)
    execute(stage, ctx)
    paths = [ref.path for ref in RunManifest.read(ctx.layout.manifest_path).input_artifacts]
    assert any(path.endswith("some-model.json") for path in paths)


def test_config_input_colliding_with_a_cli_input_is_rejected(tmp_path):
    source = tmp_path / "cohort.parquet"
    source.write_bytes(b"one")

    class Colliding(Recorder):
        def config_inputs(self, spec):
            return {"cohort_index": source}

    with pytest.raises(ValueError, match="collides"):
        context(tmp_path, Colliding(), inputs={"cohort_index": source})


# ---------------------------------------------------------------------------
# A finished run is skipped only if this code produced it
# ---------------------------------------------------------------------------


def test_a_run_produced_by_different_code_is_re_run_not_skipped(tmp_path):
    """config_hash covers the spec and the inputs, never mival's own source.

    Without this check, editing a default in this package would leave the hash
    unchanged, the finished run would be skipped, and the old numbers would be
    reported as new ones.
    """
    import json

    stage = Recorder()
    ctx = context(tmp_path, stage)
    assert execute(stage, ctx)["status"] == "ok"

    manifest_path = ctx.layout.manifest_path
    body = json.loads(manifest_path.read_text())
    body["code"]["source_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(body))

    again = execute(stage, context(tmp_path, stage))
    assert again["status"] == "ok"
    fresh = json.loads(manifest_path.read_text())
    assert any("different mival source" in warning for warning in fresh["warnings"])


def test_an_unreadable_manifest_re_runs_rather_than_trusts(tmp_path):
    stage = Recorder()
    ctx = context(tmp_path, stage)
    assert execute(stage, ctx)["status"] == "ok"
    ctx.layout.manifest_path.write_text("{not json")
    assert execute(stage, context(tmp_path, stage))["status"] == "ok"


def test_the_manifest_records_which_code_produced_it(tmp_path):
    from mival.pipeline.manifest import code_identity

    stage = Recorder()
    ctx = context(tmp_path, stage)
    execute(stage, ctx)
    manifest = RunManifest.read(ctx.layout.manifest_path)
    assert manifest.code["source_sha256"] == code_identity()["source_sha256"]
    assert manifest.code["mival_version"]
