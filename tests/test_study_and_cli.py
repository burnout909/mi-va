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
    assert capsys.readouterr().out.split() == ["retrieve", "preprocess", "models", "evaluate", "misclassify"]


def test_cli_rejects_an_unbuilt_stage(capsys):
    with pytest.raises(SystemExit):
        main(["run", "profile", "--study", "x"])


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
