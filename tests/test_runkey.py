import pytest

from mival.pipeline.runkey import AXES, REPORT_AXES, RunKey


def make(**overrides) -> RunKey:
    values = dict(
        site="mimic",
        model_id="ecgfounder",
        training_mode="inference_only",
        recipe_id="r1a2b3c4",
        perturbation_id="baseline",
        label_def="primary",
        split="test",
        fold=None,
    )
    values.update(overrides)
    return RunKey(**values)


def test_axes_match_spec_section_3_4():
    assert AXES == (
        "site",
        "model_id",
        "training_mode",
        "recipe_id",
        "perturbation_id",
        "label_def",
        "split",
        "fold",
    )


def test_report_axes_are_not_run_key_axes():
    # Spec §3.4: subgroup and outcome re-cut existing predictions instead of
    # forcing another inference pass, so they must not multiply the run space.
    assert not set(REPORT_AXES) & set(AXES)


def test_string_round_trips_without_fold():
    key = make()
    assert RunKey.from_string(key.to_string()) == key


def test_string_round_trips_with_fold():
    key = make(fold=3, split="dev")
    assert RunKey.from_string(key.to_string()) == key


def test_fold_none_is_distinguishable_from_fold_zero():
    assert make(fold=None).to_string() != make(fold=0).to_string()
    assert RunKey.from_string(make(fold=0).to_string()).fold == 0


def test_canonical_string_is_a_safe_path_component():
    text = make(fold=2).to_string()
    assert "/" not in text and "\\" not in text and " " not in text


def test_rejects_value_containing_the_field_separator():
    with pytest.raises(ValueError, match="not a valid identifier"):
        make(model_id="ecg~founder")


def test_rejects_value_containing_the_key_separator():
    with pytest.raises(ValueError, match="not a valid identifier"):
        make(model_id="ecg=founder")


def test_underscores_in_a_value_round_trip():
    # The value charset permits '_', so the field separator must not be one.
    key = make(model_id="ecg__founder", training_mode="partial_unfreeze")
    assert RunKey.from_string(key.to_string()) == key


def test_rejects_value_with_path_separator():
    with pytest.raises(ValueError, match="not a valid identifier"):
        make(site="a/b")


def test_rejects_negative_fold():
    with pytest.raises(ValueError, match="non-negative"):
        make(fold=-1)


def test_rejects_bool_fold():
    # bool is an int subclass; a True fold would serialize as "True".
    with pytest.raises(TypeError):
        make(fold=True)


def test_from_dict_rejects_unknown_axis():
    values = make().to_dict()
    values["subgroup"] = "female"
    with pytest.raises(ValueError, match="unknown axes"):
        RunKey.from_dict(values)


def test_from_dict_rejects_missing_axis():
    values = make().to_dict()
    del values["split"]
    with pytest.raises(ValueError, match="missing axes"):
        RunKey.from_dict(values)


def test_from_string_rejects_wrong_field_count():
    with pytest.raises(ValueError, match="expected 8"):
        RunKey.from_string("site=mimic__model_id=ecgfounder")


def test_canonical_form_is_pinned():
    # Predictions are addressed by this string, so changing it silently
    # orphans every result file already on disk.
    assert make(fold=2).to_string() == (
        "site=mimic~model_id=ecgfounder~training_mode=inference_only~recipe_id=r1a2b3c4"
        "~perturbation_id=baseline~label_def=primary~split=test~fold=2"
    )


def test_from_string_rejects_reordered_fields():
    fields = make().to_string().split("~")
    fields[0], fields[1] = fields[1], fields[0]
    with pytest.raises(ValueError, match="expected 'site'"):
        RunKey.from_string("~".join(fields))


def test_is_hashable_so_it_can_key_a_dict():
    assert len({make(), make(), make(fold=1)}) == 2
