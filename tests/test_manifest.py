import json

import pytest

from mival.pipeline.manifest import (
    SCHEMA_VERSION,
    ArtifactRef,
    RunManifest,
    capture_env,
    capture_git,
)


def make(**overrides) -> RunManifest:
    values = dict(
        run_id="s1.preprocess.abc",
        stage="preprocess",
        study_id="s1",
        site="mimic",
        config_hash="abc123",
    )
    values.update(overrides)
    return RunManifest(**values)


def test_carries_every_field_of_spec_section_3_7():
    body = make().to_dict()
    for key in (
        "run_id",
        "stage",
        "study_id",
        "site",
        "config_hash",
        "git",
        "input_artifacts",
        "output_artifacts",
        "env",
        "seed",
        "counts",
        "contamination",
        "compute",
        "warnings",
        "errors",
        "started_at",
        "ended_at",
    ):
        assert key in body, key


def test_round_trips_through_disk(tmp_path):
    manifest = make(seed=7, counts={"in": 10, "out": 8, "excluded": 2})
    path = manifest.write(tmp_path / "manifest.json")
    reloaded = RunManifest.read(path)
    assert reloaded.seed == 7
    assert reloaded.counts == {"in": 10, "out": 8, "excluded": 2}
    assert reloaded.stage == "preprocess"


def test_write_leaves_no_temp_file(tmp_path):
    # The manifest doubles as the completion marker, so a partial write would
    # make a crashed run look finished and permanently skippable.
    make().write(tmp_path / "manifest.json")
    assert [p.name for p in tmp_path.iterdir()] == ["manifest.json"]


def test_read_rejects_a_future_schema_version(tmp_path):
    path = tmp_path / "manifest.json"
    body = make().to_dict()
    body["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="schema_version"):
        RunManifest.read(path)


def test_artifact_ref_records_a_relative_path(tmp_path):
    # Absolute paths would differ between the laptop and DICOM-MIVA and make
    # otherwise identical manifests compare unequal.
    (tmp_path / "artifacts").mkdir()
    target = tmp_path / "artifacts" / "out.parquet"
    target.write_bytes(b"abc")
    ref = ArtifactRef.of(target, root=tmp_path)
    assert ref.path == "artifacts/out.parquet"
    assert ref.sha256 == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_artifact_refs_survive_a_round_trip(tmp_path):
    target = tmp_path / "out.parquet"
    target.write_bytes(b"abc")
    manifest = make(output_artifacts=[ArtifactRef.of(target, root=tmp_path)])
    reloaded = RunManifest.read(manifest.write(tmp_path / "manifest.json"))
    assert reloaded.output_artifacts[0].path == "out.parquet"
    assert isinstance(reloaded.output_artifacts[0], ArtifactRef)


def test_capture_git_reports_commit_and_dirty_flag():
    git = capture_git(".")
    assert set(git) == {"commit", "dirty"}
    if git["commit"] is not None:
        assert len(git["commit"]) == 40
        assert isinstance(git["dirty"], bool)


def test_capture_git_returns_nulls_when_git_cannot_run(tmp_path):
    # A stage must still produce a manifest on a machine without git, or in a
    # container where the source is copied rather than cloned.
    assert capture_git(tmp_path / "does-not-exist") == {"commit": None, "dirty": None}


def test_capture_env_does_not_import_a_backend():
    import sys

    capture_env()
    # Probing must not drag a 2 GB framework into a stage that does not use
    # one, and must not fail in the environment that lacks it.
    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules


def test_capture_env_reports_tracked_libraries():
    env = capture_env()
    assert env["python"]
    assert "numpy" in env["libraries"]
    assert "torch" in env["libraries"]  # recorded as absent, not omitted
