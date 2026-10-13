import json

import pytest

from mival.pipeline.cli import main
from mival.pipeline.study import load_study

pytest.importorskip("yaml")


def write_study(tmp_path, body: str):
    path = tmp_path / "study.yaml"
    path.write_text(body, encoding="utf-8")
    return path


MINIMAL = """
study_id: stemi-mimic-v1
site: mimic
seed: 20260814
stages:
  preprocess:
    duration_s: 10
"""


def test_loads_required_fields(tmp_path):
    study = load_study(write_study(tmp_path, MINIMAL))
    assert study.study_id == "stemi-mimic-v1"
    assert study.site == "mimic"
    assert study.seed == 20260814


def test_a_directory_resolves_to_its_study_yaml(tmp_path):
    write_study(tmp_path, MINIMAL)
    assert load_study(tmp_path).study_id == "stemi-mimic-v1"


def test_missing_required_field_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="site is required"):
        load_study(write_study(tmp_path, "study_id: s1\n"))


def test_stage_spec_is_isolated_from_other_stages(tmp_path):
    # Only the stage's own spec goes into its config_hash: editing the eval
    # grid must not invalidate preprocessing runs that never read it.
    study = load_study(
        write_study(
            tmp_path,
            "study_id: s1\nsite: mimic\nstages:\n  preprocess: {a: 1}\n  evaluate: {b: 2}\n",
        )
    )
    assert study.stage_spec("preprocess") == {"a": 1}
    assert study.stage_spec("evaluate") == {"b": 2}


def test_a_stage_with_no_spec_gets_an_empty_mapping(tmp_path):
    study = load_study(write_study(tmp_path, MINIMAL))
    assert study.stage_spec("models") == {}


def test_a_string_spec_is_loaded_from_its_file(tmp_path):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "eval.yaml").write_text("bootstrap: 2000\n", encoding="utf-8")
    study = load_study(
        write_study(tmp_path, "study_id: s1\nsite: mimic\nstages:\n  evaluate: eval/eval.yaml\n")
    )
    assert study.stage_spec("evaluate") == {"bootstrap": 2000}


def test_json_specs_are_accepted(tmp_path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"study_id": "s1", "site": "mimic"}), encoding="utf-8")
    assert load_study(path).site == "mimic"


def test_cli_lists_only_runnable_stages(capsys):
    assert main(["stages"]) == 0
    assert capsys.readouterr().out.split() == [
        "retrieve", "profile", "preprocess", "models", "evaluate",
    ]


def test_cli_rejects_an_unknown_stage_name(capsys):
    with pytest.raises(SystemExit):
        main(["run", "not-a-stage", "--study", "x"])


def test_cli_rejects_a_malformed_input_pair(tmp_path):
    write_study(tmp_path, MINIMAL)
    with pytest.raises(SystemExit, match="name=path"):
        main(["run", "preprocess", "--study", str(tmp_path), "--input", "justapath"])


def test_cli_rejects_a_missing_input_file(tmp_path):
    write_study(tmp_path, MINIMAL)
    with pytest.raises(SystemExit, match="is not a file"):
        main(
            [
                "run",
                "preprocess",
                "--study",
                str(tmp_path),
                "--input",
                f"cohort_index={tmp_path / 'nope.parquet'}",
            ]
        )


def test_cli_requires_the_declared_inputs(tmp_path):
    write_study(tmp_path, MINIMAL)
    with pytest.raises(SystemExit, match="requires --input for: cohort_index"):
        main(["run", "preprocess", "--study", str(tmp_path), "--runs-root", str(tmp_path / "runs")])


def _site_files(tmp_path, study_extra="", site_text=None):
    site = tmp_path / "config" / "site.yaml"
    site.parent.mkdir()
    site.write_text(site_text if site_text is not None else
                    "site: site-b\nretrieve:\n  dbms: sqlserver\n  schema: CDM.dbo\n"
                    "  dsn_env: ${MIVAL_HOME}/config/micdm.env\nmodels:\n  cohort_sources: [site-b-ecg]\n")
    study = tmp_path / "studies" / "s" / "study.yaml"
    study.parent.mkdir(parents=True)
    study.write_text("study_id: s\nsite_config: ${MIVAL_HOME}/config/site.yaml\nseed: 1\n"
                     "stages:\n  retrieve:\n    window_days: 7\n" + study_extra)
    return study


def test_site_config_is_merged_into_the_stage_specs(tmp_path, monkeypatch):
    from mival.pipeline.study import load_study

    monkeypatch.setenv("MIVAL_HOME", str(tmp_path))
    study = load_study(_site_files(tmp_path))
    assert study.site == "site-b"
    assert study.stage_spec("retrieve") == {
        "window_days": 7, "dbms": "sqlserver", "schema": "CDM.dbo",
        "dsn_env": f"{tmp_path}/config/micdm.env"}
    assert study.stage_spec("models") == {"cohort_sources": ["site-b-ecg"]}


def test_site_config_spec_hashes_like_the_inline_spec(tmp_path, monkeypatch):
    from mival.pipeline.hashing import config_hash
    from mival.pipeline.study import load_study

    monkeypatch.setenv("MIVAL_HOME", str(tmp_path))
    merged = load_study(_site_files(tmp_path)).stage_spec("retrieve")
    inline = {"window_days": 7, "dbms": "sqlserver", "schema": "CDM.dbo", "dsn_env": f"{tmp_path}/config/micdm.env"}
    assert config_hash(merged, {}) == config_hash(inline, {})


def test_a_key_in_both_places_is_rejected(tmp_path, monkeypatch):
    from mival.pipeline.study import load_study

    monkeypatch.setenv("MIVAL_HOME", str(tmp_path))
    with pytest.raises(ValueError, match="schema set both here and in"):
        load_study(_site_files(tmp_path, study_extra="    schema: cdm\n"))


def test_missing_site_config_file_is_reported(tmp_path, monkeypatch):
    from mival.pipeline.study import load_study

    monkeypatch.setenv("MIVAL_HOME", str(tmp_path / "elsewhere"))
    with pytest.raises(ValueError, match="does not exist"):
        load_study(_site_files(tmp_path))
