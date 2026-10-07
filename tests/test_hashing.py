import pytest

from mival.pipeline.hashing import (
    CONFIG_HASH_LENGTH,
    canonical_json,
    config_hash,
    sha256_file,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def test_sha256_file_matches_known_value(tmp_path):
    target = tmp_path / "x.bin"
    target.write_bytes(b"abc")
    assert sha256_file(target) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_sha256_file_streams_larger_than_one_chunk(tmp_path):
    import hashlib

    payload = bytes(range(256)) * 8192  # 2 MiB, larger than the 1 MiB chunk
    target = tmp_path / "big.bin"
    target.write_bytes(payload)
    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_config_hash_is_stable_across_dict_order():
    left = config_hash({"fs": 500, "leads": 12}, {"cohort": DIGEST_A})
    right = config_hash({"leads": 12, "fs": 500}, {"cohort": DIGEST_A})
    assert left == right


def test_config_hash_changes_with_spec():
    assert config_hash({"fs": 500}, {}) != config_hash({"fs": 250}, {})


def test_config_hash_changes_with_input_checksum():
    assert config_hash({}, {"cohort": DIGEST_A}) != config_hash({}, {"cohort": DIGEST_B})


def test_config_hash_changes_with_input_name():
    # Renaming an input is a real change: a different logical slot was filled.
    assert config_hash({}, {"cohort": DIGEST_A}) != config_hash({}, {"split": DIGEST_A})


def test_config_hash_length_and_alphabet():
    digest = config_hash({"a": 1}, {})
    assert len(digest) == CONFIG_HASH_LENGTH
    assert all(char in "0123456789abcdef" for char in digest)


def test_config_hash_rejects_a_path_passed_as_a_checksum():
    # The whole point of hashing checksums rather than paths is that results
    # stay reproducible across machines; a path slipping in defeats that.
    with pytest.raises(ValueError, match="64-character SHA-256"):
        config_hash({}, {"cohort": "/data/mi-val/cohort_index.parquet"})
