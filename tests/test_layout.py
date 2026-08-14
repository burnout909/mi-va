import pytest

from mival.pipeline.layout import DEFAULT_RUNS_ROOT, exclusions_root, run_layout


def test_path_shape_matches_spec_section_3_4(tmp_path):
    layout = run_layout("stemi-mimic-v1", "preprocess", "0123456789abcdef", tmp_path)
    assert layout.run_dir == tmp_path / "stemi-mimic-v1" / "preprocess" / "0123456789abcdef"
    assert layout.artifacts_dir == layout.run_dir / "artifacts"
    assert layout.logs_dir == layout.run_dir / "logs"
    assert layout.manifest_path == layout.run_dir / "manifest.json"


def test_default_runs_root_is_the_persistent_volume():
    # Spec §3.4: artifacts live on the 400 GB EBS volume, never on the
    # instance store, which is not assumed to survive a stop/start.
    assert str(DEFAULT_RUNS_ROOT).startswith("/data")


def test_exclusions_part_is_per_stage_and_per_config_hash(tmp_path):
    layout = run_layout("s1", "preprocess", "abc123", tmp_path)
    assert layout.exclusions_path == (
        tmp_path / "s1" / "exclusions" / "preprocess" / "abc123.parquet"
    )


def test_exclusions_root_spans_all_stages(tmp_path):
    layout = run_layout("s1", "models", "abc123", tmp_path)
    assert layout.exclusions_path.is_relative_to(exclusions_root(tmp_path, "s1"))


def test_incomplete_run_is_not_complete(tmp_path):
    layout = run_layout("s1", "preprocess", "abc123", tmp_path)
    layout.create_dirs()
    layout.artifact("partial.parquet").write_bytes(b"x")
    # Artifacts exist but the manifest does not: the run crashed and must be
    # re-run, not skipped.
    assert not layout.is_complete()
    layout.manifest_path.write_text("{}")
    assert layout.is_complete()


def test_create_dirs_is_idempotent(tmp_path):
    layout = run_layout("s1", "preprocess", "abc123", tmp_path)
    layout.create_dirs()
    layout.create_dirs()
    assert layout.artifacts_dir.is_dir()
    assert layout.exclusions_path.parent.is_dir()


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", ".hidden", "with space"])
def test_rejects_unsafe_path_components(tmp_path, bad):
    with pytest.raises(ValueError, match="not a valid path component"):
        run_layout(bad, "preprocess", "abc123", tmp_path)
