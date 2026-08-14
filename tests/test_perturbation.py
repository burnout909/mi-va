import numpy as np
import pytest

from mival.ops import Normalize
from mival.perturbation import (
    AXIS_ORDER,
    BASELINE_ID,
    DEFAULT_AXES,
    LEAD_DROPOUT_SETS,
    NoiseConfig,
    Perturbation,
    apply,
    baseline,
    build_grid,
    from_id,
    resolve_axes,
)
from mival.pipeline.runkey import RunKey
from mival.signal import LEADS_12, Signal
from mival.synthetic import synthetic_ecg

PROPHECG_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


def contract_signal(record_index=0, leads=LEADS_12, rate=500.0, n_samples=5000):
    """A tensor already compiled to an input contract, which is what stage 4 has."""
    return synthetic_ecg(record_index, leads=leads, sampling_rate_hz=rate, n_samples=n_samples)


# --------------------------------------------------------------------------
# The grid is the spec's grid
# --------------------------------------------------------------------------


def test_axes_and_levels_are_exactly_spec_section_4_3():
    assert dict(DEFAULT_AXES) == {
        "resample": (500, 250, 125, 100),
        "lead_dropout": ("none", "drop_V3V4", "precordial_only", "limb_only"),
        "duration": (10, 5, 2.5),
        "amplitude_scale": (1.0, 0.5, 2.0),
        "noise": ("none", "baseline_wander", "powerline_50hz", "emg"),
    }
    assert AXIS_ORDER == ("resample", "lead_dropout", "duration", "amplitude_scale", "noise")


def test_ofat_grid_moves_exactly_one_axis_off_baseline():
    grid = build_grid()
    base = baseline()
    assert grid[0].id == BASELINE_ID

    for point in grid[1:]:
        differing = [
            axis for axis in AXIS_ORDER if getattr(point, axis) != getattr(base, axis)
        ]
        assert len(differing) == 1, point.id


def test_ofat_grid_covers_every_non_baseline_level_and_nothing_else():
    produced = {axis: set() for axis in AXIS_ORDER}
    for point in build_grid():
        for axis in AXIS_ORDER:
            produced[axis].add(getattr(point, axis))
    for axis, levels in DEFAULT_AXES.items():
        assert produced[axis] == set(levels)


def test_ofat_condition_count_is_inside_the_spec_range():
    # Spec §4.3: "baseline 포함 약 13~16조건".
    grid = build_grid()
    assert len(grid) == 14
    assert 13 <= len(grid) <= 16


def test_cartesian_is_the_full_product_and_opt_in():
    assert build_grid("cartesian") != build_grid()
    assert len(build_grid("cartesian")) == 4 * 4 * 3 * 3 * 4


def test_ofat_ids_mean_the_same_thing_in_cartesian_mode():
    cartesian = {point.id: point.levels() for point in build_grid("cartesian")}
    for point in build_grid():
        assert cartesian[point.id] == point.levels()


def test_unknown_mode_and_unknown_axis_are_rejected():
    with pytest.raises(ValueError, match="unknown perturbation mode"):
        build_grid("shotgun")
    with pytest.raises(ValueError, match="unknown perturbation axes"):
        build_grid(axes={"jitter": [1, 2]})


def test_levels_may_be_overridden_from_the_study_spec():
    grid = build_grid(axes={"resample": [500, 200]})
    assert [point.id for point in grid if point.id.startswith("resample")] == ["resample-200"]
    assert resolve_axes({"resample": [500, 200]})["noise"] == DEFAULT_AXES["noise"]


# --------------------------------------------------------------------------
# perturbation_id
# --------------------------------------------------------------------------


def test_ids_are_readable_and_match_the_plan_examples():
    ids = [point.id for point in build_grid()]
    assert BASELINE_ID in ids
    assert "resample-125" in ids
    assert "leaddrop-precordial_only" in ids


@pytest.mark.parametrize("mode", ["ofat", "cartesian"])
def test_every_id_is_a_legal_run_key_value(mode):
    for point in build_grid(mode):
        key = RunKey(
            site="mimic",
            model_id="m",
            training_mode="inference_only",
            recipe_id="abc123",
            perturbation_id=point.id,
            label_def="primary",
            split="test",
        )
        assert RunKey.from_string(key.to_string()).perturbation_id == point.id


@pytest.mark.parametrize("mode", ["ofat", "cartesian"])
def test_ids_round_trip_through_from_id(mode):
    grid = build_grid(mode)
    assert len({point.id for point in grid}) == len(grid)
    for point in grid:
        assert from_id(point.id) == point


def test_from_id_rejects_unknown_tokens_and_levels():
    with pytest.raises(ValueError, match="unknown perturbation token"):
        from_id("jitter-3")
    with pytest.raises(ValueError, match="is not on axis"):
        from_id("resample-137")
    with pytest.raises(ValueError, match="twice"):
        from_id("resample-125__resample-100")


# --------------------------------------------------------------------------
# The rule: every perturbation restores input-contract form
# --------------------------------------------------------------------------


# The OFAT grid in full, plus a stride through the cartesian product so that
# combined axes are covered without paying for all 576 conditions.
SHAPE_CASES = build_grid() + tuple(build_grid("cartesian")[::53])


@pytest.mark.parametrize("point", SHAPE_CASES, ids=lambda p: p.id)
def test_every_perturbation_preserves_the_input_contract_shape(point):
    sig = contract_signal(1)
    out = apply(sig, point, seed=3)
    # 125 Hz in, still 500 Hz x 5000 samples out (spec §4.3).
    assert out.data.shape == (12, 5000)
    assert out.sampling_rate_hz == 500.0
    assert out.leads == LEADS_12
    assert out.unit == sig.unit
    assert out.data.dtype == np.float32
    assert np.isfinite(out.data).all()


def test_shape_is_preserved_for_a_shorter_eight_lead_contract():
    sig = contract_signal(2, leads=PROPHECG_LEADS, rate=250.0, n_samples=2500)
    for point in build_grid():
        out = apply(sig, point, seed=3)
        assert out.data.shape == (8, 2500)
        assert out.sampling_rate_hz == 250.0
        assert out.leads == PROPHECG_LEADS


def test_baseline_is_exactly_the_stored_tensor():
    sig = contract_signal(4)
    for scaling in ("none", "global_zscore", "per_lead_zscore"):
        out = apply(sig, BASELINE_ID, seed=1, scaling=scaling)
        assert out.data is sig.data


def test_lead_dropout_zeroes_leads_without_removing_them():
    sig = contract_signal(5)
    for level, dropped in LEAD_DROPOUT_SETS.items():
        out = apply(sig, from_id(f"leaddrop-{level}"), seed=1)
        assert out.leads == sig.leads
        assert out.n_leads == 12
        for position, lead in enumerate(sig.leads):
            if lead in dropped:
                assert not out.data[position].any(), lead
            else:
                assert np.array_equal(out.data[position], sig.data[position]), lead


def test_lead_dropout_ignores_leads_a_contract_does_not_carry():
    # PROPHECG has no III/aVR/aVL/aVF, so "precordial_only" drops only I and II.
    sig = contract_signal(6, leads=PROPHECG_LEADS, n_samples=5000)
    out = apply(sig, "leaddrop-precordial_only", seed=1)
    assert not out.data[0].any() and not out.data[1].any()
    assert out.data[2:].any()


def test_duration_keeps_the_leading_window_and_zero_pads_back():
    sig = contract_signal(7)
    out = apply(sig, "duration-2.5", seed=1)
    kept = int(2.5 * 500)
    assert np.array_equal(out.data[:, :kept], sig.data[:, :kept])
    assert not out.data[:, kept:].any()


def test_a_window_at_least_as_long_as_the_record_changes_nothing():
    # 10 s of a 10 s record removes no information.
    sig = contract_signal(8)
    axes = {"duration": [30, 10]}
    out = apply(sig, "duration-10", axes=axes, seed=1)
    assert np.array_equal(out.data, sig.data)
    assert baseline().duration == 10


def test_resample_restores_the_rate_and_loses_more_at_lower_rates():
    sig = contract_signal(9)
    errors = {}
    for level in (250, 125, 100):
        out = apply(sig, f"resample-{level}", seed=1)
        assert out.sampling_rate_hz == 500.0
        assert out.n_samples == 5000
        errors[level] = float(np.abs(out.data - sig.data).mean())
    assert errors[250] < errors[125] < errors[100]


def test_the_resample_round_trip_leaves_no_edge_transient():
    # A zero-padded polyphase filter puts a large step artifact in the last
    # samples; that would be misread as sampling-rate sensitivity.
    sig = contract_signal(10)
    out = apply(sig, "resample-100", seed=1)
    error = np.abs(out.data - sig.data)
    assert error[:, -20:].max() < 5.0 * error[:, 100:-100].max()


def test_an_upward_resample_level_is_an_identity():
    sig = contract_signal(11, rate=250.0, n_samples=2500)
    out = apply(sig, "resample-500", axes={"resample": [1000, 500]}, seed=1)
    assert np.array_equal(out.data, sig.data)


# --------------------------------------------------------------------------
# Amplitude and the contract's own normalization
# --------------------------------------------------------------------------


def test_amplitude_scale_is_a_real_gain_error_for_an_unnormalized_contract():
    sig = contract_signal(12)
    out = apply(sig, "amplitude-2", seed=1, scaling="none")
    assert np.allclose(out.data, sig.data * 2.0, atol=1e-6)


@pytest.mark.parametrize("scaling", ["global_zscore", "per_lead_zscore"])
def test_amplitude_scale_is_a_no_op_for_a_normalizing_contract(scaling):
    # A mis-calibrated ECG passes through the model's own normalization in
    # deployment, which removes the gain. Reporting a robustness failure here
    # would be reporting something that cannot happen.
    sig = Normalize(scaling).apply(contract_signal(13))
    out = apply(sig, "amplitude-2", seed=1, scaling=scaling)
    assert np.allclose(out.data, sig.data, atol=1e-5)


def test_apply_rejects_an_unknown_scaling():
    with pytest.raises(ValueError, match="unknown scaling"):
        apply(contract_signal(14), "amplitude-2", scaling="minmax")


# --------------------------------------------------------------------------
# Reproducibility: seed + perturbation_id (+ record)
# --------------------------------------------------------------------------

NOISE_IDS = ["noise-baseline_wander", "noise-powerline_50hz", "noise-emg"]


@pytest.mark.parametrize("point", build_grid(), ids=lambda p: p.id)
def test_the_same_seed_and_perturbation_id_give_a_bit_identical_result(point):
    sig = contract_signal(15)
    first = apply(sig, point, seed=42)
    second = apply(sig, point.id, seed=42)
    assert first.data.tobytes() == second.data.tobytes()


@pytest.mark.parametrize("perturbation_id", NOISE_IDS)
def test_a_different_seed_gives_a_different_noise_realization(perturbation_id):
    sig = contract_signal(16)
    first = apply(sig, perturbation_id, seed=1)
    second = apply(sig, perturbation_id, seed=2)
    assert first.data.tobytes() != second.data.tobytes()


@pytest.mark.parametrize("perturbation_id", NOISE_IDS)
def test_two_records_get_different_noise_under_one_seed(perturbation_id):
    # An identical noise realization across the cohort is a confound, not noise.
    first = apply(contract_signal(17), perturbation_id, seed=5)
    second = apply(contract_signal(18), perturbation_id, seed=5)
    assert first.data.tobytes() != second.data.tobytes()


def test_the_record_key_may_be_supplied_instead_of_derived():
    sig = contract_signal(19)
    a = apply(sig, "noise-emg", seed=5, record_key="img-1")
    b = apply(sig, "noise-emg", seed=5, record_key="img-1")
    c = apply(sig, "noise-emg", seed=5, record_key="img-2")
    assert a.data.tobytes() == b.data.tobytes()
    assert a.data.tobytes() != c.data.tobytes()


def test_seed_does_not_touch_the_deterministic_axes():
    sig = contract_signal(20)
    for perturbation_id in ("resample-125", "leaddrop-limb_only", "duration-5", "amplitude-0.5"):
        first = apply(sig, perturbation_id, seed=1)
        second = apply(sig, perturbation_id, seed=999)
        assert first.data.tobytes() == second.data.tobytes()


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------


@pytest.mark.parametrize("perturbation_id", NOISE_IDS)
def test_noise_lands_near_the_configured_snr(perturbation_id):
    sig = contract_signal(21)
    config = NoiseConfig(snr_db=20.0)
    out = apply(sig, perturbation_id, seed=7, noise=config)
    residual = out.data.astype(np.float64) - sig.data.astype(np.float64)
    achieved = 20.0 * np.log10(
        np.sqrt(np.mean(sig.data.astype(np.float64) ** 2)) / np.sqrt(np.mean(residual**2))
    )
    assert abs(achieved - 20.0) < 0.5


def test_powerline_energy_sits_at_the_frequency_named_by_the_level():
    sig = contract_signal(22)
    out = apply(sig, "noise-powerline_50hz", seed=7)
    residual = out.data[0].astype(np.float64) - sig.data[0].astype(np.float64)
    spectrum = np.abs(np.fft.rfft(residual))
    frequencies = np.fft.rfftfreq(residual.size, d=1.0 / 500.0)
    assert frequencies[int(spectrum.argmax())] == pytest.approx(50.0, abs=0.5)


def test_a_new_mains_frequency_is_a_level_change_not_a_code_change():
    sig = contract_signal(23)
    out = apply(
        sig,
        "noise-powerline_60hz",
        seed=7,
        axes={"noise": ["none", "powerline_50hz", "powerline_60hz"]},
    )
    residual = out.data[0].astype(np.float64) - sig.data[0].astype(np.float64)
    spectrum = np.abs(np.fft.rfft(residual))
    frequencies = np.fft.rfftfreq(residual.size, d=1.0 / 500.0)
    assert frequencies[int(spectrum.argmax())] == pytest.approx(60.0, abs=0.5)


def test_baseline_wander_energy_stays_low_frequency():
    sig = contract_signal(24)
    out = apply(sig, "noise-baseline_wander", seed=7)
    residual = out.data[0].astype(np.float64) - sig.data[0].astype(np.float64)
    spectrum = np.abs(np.fft.rfft(residual))
    frequencies = np.fft.rfftfreq(residual.size, d=1.0 / 500.0)
    assert frequencies[int(spectrum.argmax())] <= 0.6


def test_an_unknown_noise_level_is_rejected_at_grid_build_time():
    with pytest.raises(ValueError, match="unknown noise level"):
        build_grid(axes={"noise": ["none", "cosmic_rays"]})


def test_an_unknown_lead_dropout_level_is_rejected_at_grid_build_time():
    with pytest.raises(ValueError, match="unknown lead_dropout level"):
        build_grid(axes={"lead_dropout": ["none", "drop_everything"]})


def test_noise_on_a_silent_signal_is_a_no_op_rather_than_undefined():
    silent = Signal(
        data=np.zeros((12, 5000), dtype=np.float32),
        leads=LEADS_12,
        sampling_rate_hz=500.0,
        unit="mV",
    )
    out = apply(silent, "noise-emg", seed=1)
    assert not out.data.any()


def test_a_window_that_keeps_no_samples_is_rejected():
    broken = Perturbation(
        id="broken",
        resample=500,
        lead_dropout="none",
        duration=0.0,
        amplitude_scale=1.0,
        noise="none",
    )
    with pytest.raises(ValueError, match="keeps no samples"):
        apply(contract_signal(25), broken, seed=1)
