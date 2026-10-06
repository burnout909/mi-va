"""Shared output decoders (stage 2: one place for every non-classification output)."""
import numpy as np
import pytest

from mival.decode.outputs import decode_outputs, mtlr_survival, risk_from_conditional_survival


def test_conditional_survival_risk_with_partial_interval():
    p = np.array([[0.9, 0.8, 0.5, 0.5]])
    # bin 146 days, horizon 365 days = 2.5 bins: S = 0.9 * 0.8 * 0.5 ** 0.5
    np.testing.assert_allclose(risk_from_conditional_survival(p, 146, 365), [1 - 0.72 * 0.5 ** 0.5])
    np.testing.assert_allclose(risk_from_conditional_survival(p, 73, 365), [1 - 0.9 * 0.8 * 0.5 * 0.5])


def test_survival_curve_output_reads_its_first_units():
    output = {"type": "survival_curve", "intervals": 2, "bin_days": 365, "horizon_days": 365}
    raw = np.array([[0.75, 0.1, 9.0, 9.0]])
    np.testing.assert_allclose(decode_outputs(output, raw, fs=500), [0.25])


def test_mtlr_survival_matches_pycox_definition():
    phi = np.array([[0.0, 0.0, 0.0]])
    surv = mtlr_survival(phi)
    # cumsum_reverse(0) = 0 -> pmf uniform over 4 (3 bins + tail) -> S = 1 - [1/4, 2/4, 3/4]
    np.testing.assert_allclose(surv, [[0.75, 0.5, 0.25]])


def test_mtlr_output_interpolates_the_horizon():
    output = {"type": "mtlr", "n_bins": 3, "max_duration_days": 300, "horizon_days": 150}
    risk = decode_outputs(output, np.zeros((1, 3)), fs=500)
    # cuts 0, 150, 300 -> S(150) = 0.5
    np.testing.assert_allclose(risk, [0.5])


def test_segmentation_mask_with_time_major_logits():
    from test_vertical_slice_outputs import beat_mask

    mask = beat_mask()
    logits = np.eye(4)[mask][None]  # (1, T, 4)
    out = decode_outputs({"type": "segmentation_mask", "class_axis": 2}, logits, fs=250)
    np.testing.assert_allclose(out[0], [80.0, 100.0, 400.0])


def test_regression_with_transform():
    out = decode_outputs({"type": "regression", "value_index": 1, "transform": {"kind": "exp"}},
                         np.array([[9.0, 0.0], [9.0, 1.0]]), fs=500)
    np.testing.assert_allclose(out, [1.0, np.e])


def test_unknown_type_is_rejected():
    with pytest.raises(ValueError, match="output.type"):
        decode_outputs({"type": "histogram"}, np.zeros((1, 2)), fs=500)
