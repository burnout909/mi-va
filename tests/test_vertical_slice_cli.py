"""Directory inputs on the CLI (L1, plan 2026-10-07)."""
from mival.pipeline.cli import _parse_inputs
from mival.pipeline.hashing import sha256_path


def test_cli_accepts_a_directory_input(tmp_path):
    (tmp_path / "predictions").mkdir()
    assert _parse_inputs([f"predictions={tmp_path / 'predictions'}"])["predictions"] == tmp_path / "predictions"


def test_directory_checksum_tracks_names_and_content(tmp_path):
    folder = tmp_path / "d"
    folder.mkdir()
    (folder / "a.parquet").write_bytes(b"1")
    (folder / "b.parquet").write_bytes(b"2")
    first = sha256_path(folder)
    assert len(first) == 64 and sha256_path(folder) == first
    (folder / "b.parquet").write_bytes(b"3")
    assert sha256_path(folder) != first
    (folder / "b.parquet").write_bytes(b"2")
    (folder / "b.parquet").rename(folder / "c.parquet")
    assert sha256_path(folder) != first
    assert sha256_path(folder / "a.parquet") == sha256_path(folder / "a.parquet")


def test_manifest_records_a_directory_input(tmp_path):
    from mival.pipeline.manifest import ArtifactRef

    folder = tmp_path / "d"
    folder.mkdir()
    (folder / "a.parquet").write_bytes(b"1")
    assert ArtifactRef.of(folder).sha256 == sha256_path(folder)
