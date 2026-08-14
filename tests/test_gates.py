"""Gates 3 and 4 of spec §3.5.

The point of these two tests taken together is the asymmetry: leakage stops
the run, contamination only marks it.
"""

import pytest

from mival.gates import (
    ContaminationReport,
    LeakageError,
    check_contamination,
    check_leakage,
)


# --------------------------------------------------------------------------
# leakage — stops the run
# --------------------------------------------------------------------------


def test_disjoint_person_ids_pass():
    assert check_leakage(["p1", "p2"], ["p3", "p4"]) is None


def test_one_overlapping_person_id_stops_the_run_immediately():
    with pytest.raises(LeakageError) as excinfo:
        check_leakage(["p1", "p2", "p3"], ["p3", "p9"])
    message = str(excinfo.value)
    assert "1 person_id(s)" in message
    assert "p3" in message


def test_the_message_names_the_offenders_without_dumping_the_cohort():
    ids = [f"p{index}" for index in range(50)]
    with pytest.raises(LeakageError) as excinfo:
        check_leakage(ids, ids)
    message = str(excinfo.value)
    assert "50 in total" in message
    assert message.count("p") < 60  # a sample, not the whole cohort


def test_the_context_is_named_so_the_failing_arm_is_identifiable():
    with pytest.raises(LeakageError, match="for arm toy/linear_probe"):
        check_leakage(["p1"], ["p1"], "arm toy/linear_probe")


def test_identifier_types_do_not_hide_an_overlap():
    # A parquet round-trip can turn a person_id into an int on one side and a
    # str on the other; the gate must not be fooled by that.
    with pytest.raises(LeakageError):
        check_leakage([7], ["7"])


def test_an_empty_fitting_set_cannot_leak():
    assert check_leakage([], ["p1", "p2"]) is None


# --------------------------------------------------------------------------
# contamination — marks the run, never stops it
# --------------------------------------------------------------------------


def test_overlapping_corpora_raise_the_flag_without_raising_an_exception():
    report = check_contamination(["corpus-a", "corpus-b"], ["corpus-b"])
    assert report.flag is True
    assert report.overlapping_corpora == ("corpus-b",)
    assert report.checked is True


def test_disjoint_corpora_leave_the_flag_down():
    report = check_contamination(["corpus-a"], ["corpus-z"])
    assert report.flag is False
    assert report.overlapping_corpora == ()
    assert report.checked is True


def test_matching_ignores_case_and_separator_style():
    # A missed overlap turns a contaminated result into a clean-looking one,
    # so the comparison folds the spellings that mean the same corpus.
    report = check_contamination(["MIMIC_IV ECG"], ["mimic-iv-ecg"])
    assert report.flag is True
    # The card's own spelling is what gets reported, so the manifest stays
    # traceable back to the ModelCard.
    assert report.overlapping_corpora == ("MIMIC_IV ECG",)


def test_matching_is_not_fuzzy():
    assert check_contamination(["mimic-iv-ecg-subset"], ["mimic-iv-ecg"]).flag is False


def test_a_study_that_declares_no_sources_is_unchecked_not_clean():
    report = check_contamination(["corpus-a"], [])
    assert report.flag is False
    assert report.checked is False


def test_the_report_is_exactly_the_manifest_block_of_spec_3_7():
    report = check_contamination(["corpus-a"], ["corpus-a"])
    assert report.to_dict() == {"flag": True, "overlapping_corpora": ["corpus-a"]}


def test_merging_arms_keeps_the_run_marked_if_any_arm_is_contaminated():
    merged = ContaminationReport.merge(
        [
            check_contamination(["corpus-a"], ["corpus-z"]),
            check_contamination(["corpus-b", "corpus-z"], ["corpus-z"]),
        ]
    )
    assert merged.flag is True
    assert merged.overlapping_corpora == ("corpus-z",)


def test_merging_nothing_is_unchecked():
    assert ContaminationReport.merge([]).checked is False
