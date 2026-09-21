"""The backend-independent half of adapter.fit (spec §4.4).

Nothing here imports a backend: the hyperparameter contract, the logistic head
and the whole ``linear_probe`` path are NumPy, so they are testable in any
environment. A fake adapter stands in for torch/Keras, which is the point of
the design — if a test needs a real backend to exercise ``linear_probe``, the
logic ended up in the wrong file.
"""

from __future__ import annotations

import numpy as np
import pytest

from mival.adapters._training import (
    EarlyStopping,
    LinearHead,
    TrainingError,
    binary_cross_entropy,
    epoch_batches,
    fit_linear_head,
    parse_fit_data,
    parse_hparams,
    sigmoid,
    validation_score,
)
from mival.adapters.base import Adapter


# ---------------------------------------------------------------------------
# a backend-free adapter
# ---------------------------------------------------------------------------


class FakeAdapter(Adapter):
    """Representations are read straight out of the "tensor" the loader made.

    The first two columns of each synthetic record are its representation, so
    ``features`` is exact and the head has something learnable to find.
    """

    name = "fake"

    def __init__(self) -> None:
        self.feature_calls = 0

    def features(self, handle, batch, index=0):
        self.feature_calls += 1
        return np.asarray(batch)[:, :, 0]

    def with_head(self, handle, head, record):
        return {"handle": handle, "head": head, "fit_record": dict(record)}


def make_data(n=200, seed=0, n_validation=60, batch_size=16):
    """A separable-ish two-feature problem with a 20% event rate."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(np.int64)
    signal = np.stack([y * 1.5 + rng.normal(0, 1, n), rng.normal(0, 1, n)], axis=1)
    store = {f"rec-{i}": signal[i] for i in range(n)}

    def load_batch(paths):
        # (B, n_leads, n_samples): the adapter reads column 0 of each lead.
        return np.stack([np.repeat(store[p][:, None], 4, axis=1) for p in paths])

    split = n - n_validation
    data = {
        "tensor_paths": [f"rec-{i}" for i in range(split)],
        "y": y[:split],
        "person_id": [f"p-{i}" for i in range(split)],
        "fold": [0] * split,
        "label_column": "label_primary",
        "load_batch": load_batch,
        "batch_size": batch_size,
        "validation": {
            "tensor_paths": [f"rec-{i}" for i in range(split, n)],
            "y": y[split:],
            "person_id": [f"p-{i}" for i in range(split, n)],
        },
    }
    return data


# ---------------------------------------------------------------------------
# hyperparameters
# ---------------------------------------------------------------------------


def test_defaults_differ_by_mode():
    probe = parse_hparams({}, "linear_probe", 40, 160)
    finetune = parse_hparams({}, "full_finetune", 40, 160)
    # A frozen backbone tolerates a large step; a pretrained one being
    # fine-tuned does not.
    assert probe.lr > finetune.lr


def test_a_misspelled_hyperparameter_is_an_error_not_a_no_op():
    # Silently ignoring it would make two grid candidates identical while the
    # study recorded them as different settings.
    with pytest.raises(TrainingError, match="learning_rate"):
        parse_hparams({"learning_rate": 0.1}, "linear_probe", 40, 160)


def test_given_values_override_defaults_and_are_recorded():
    hp = parse_hparams({"lr": 0.5, "epochs": 3}, "linear_probe", 40, 160)
    assert (hp.lr, hp.epochs) == (0.5, 3)
    assert hp.to_record()["given"] == {"lr": 0.5, "epochs": 3}


def test_balanced_pos_weight_is_the_negative_to_positive_ratio():
    hp = parse_hparams({"pos_weight": "balanced"}, "linear_probe", 40, 160)
    assert hp.pos_weight == pytest.approx(4.0)


def test_pos_weight_defaults_to_one():
    # Reweighting inflates predicted risk, and calibration is its own reported
    # evaluation category; the operating point is the threshold policy's job.
    assert parse_hparams({}, "linear_probe", 40, 160).pos_weight == 1.0


def test_partial_unfreeze_without_groups_is_rejected():
    with pytest.raises(TrainingError, match="unfreeze_groups"):
        parse_hparams({}, "partial_unfreeze", 40, 160)


def test_inference_only_never_reaches_fit():
    with pytest.raises(TrainingError, match="unknown training mode"):
        parse_hparams({}, "inference_only", 40, 160)


@pytest.mark.parametrize(
    "hparams",
    [{"epochs": 0}, {"lr": 0}, {"lr": -1}, {"weight_decay": -0.1}, {"patience": 0}],
)
def test_out_of_range_values_are_rejected(hparams):
    with pytest.raises(TrainingError):
        parse_hparams(hparams, "linear_probe", 40, 160)


def test_unknown_early_stopping_metric_is_rejected():
    with pytest.raises(TrainingError, match="early_stopping_metric"):
        parse_hparams({"early_stopping_metric": "accuracy"}, "linear_probe", 40, 160)


# ---------------------------------------------------------------------------
# the data payload
# ---------------------------------------------------------------------------


def test_parse_fit_data_accepts_what_the_models_stage_builds():
    parsed = parse_fit_data(make_data())
    assert len(parsed.train) == 140
    assert parsed.validation is not None and len(parsed.validation) == 60
    assert parsed.label_column == "label_primary"


def test_single_class_training_data_is_rejected():
    data = make_data()
    data["y"] = np.zeros(len(data["tensor_paths"]), dtype=np.int64)
    with pytest.raises(TrainingError, match="single-class fit"):
        parse_fit_data(data)


def test_non_binary_labels_are_rejected():
    data = make_data()
    data["y"] = np.full(len(data["tensor_paths"]), 2, dtype=np.int64)
    with pytest.raises(TrainingError, match="0 or 1"):
        parse_fit_data(data)


def test_mismatched_lengths_are_rejected():
    data = make_data()
    data["y"] = data["y"][:-1]
    with pytest.raises(TrainingError, match="tensor paths"):
        parse_fit_data(data)


def test_absent_validation_is_allowed():
    data = make_data()
    data["validation"] = None
    assert parse_fit_data(data).validation is None


# ---------------------------------------------------------------------------
# batching, stopping, loss
# ---------------------------------------------------------------------------


def test_epoch_batches_cover_every_index_exactly_once():
    batches = epoch_batches(23, 8, np.random.default_rng(1))
    assert sorted(np.concatenate(batches).tolist()) == list(range(23))
    assert [len(batch) for batch in batches] == [8, 8, 7]


def test_unshuffled_batches_are_in_order():
    assert epoch_batches(5, 2)[0].tolist() == [0, 1]


def test_shuffling_leaves_the_global_numpy_generator_alone():
    # A fit that reseeded the global stream would change every later bootstrap
    # in the same process.
    np.random.seed(1234)
    before = np.random.random()
    np.random.seed(1234)
    epoch_batches(50, 8, np.random.default_rng(99))
    assert np.random.random() == before


def test_early_stopping_keeps_the_earliest_of_two_tying_epochs():
    stopper = EarlyStopping(patience=2)
    assert stopper.update(1, 0.80)
    assert not stopper.update(2, 0.80)
    assert stopper.best_epoch == 1


def test_early_stopping_fires_after_patience_misses():
    stopper = EarlyStopping(patience=2)
    stopper.update(1, 0.9)
    stopper.update(2, 0.5)
    assert not stopper.should_stop
    stopper.update(3, 0.5)
    assert stopper.should_stop


def test_a_score_prefers_better_calibration_when_auroc_ties():
    from mival.adapters._training import Score

    assert Score(1.0, -0.20) > Score(1.0, -0.60)
    # Discrimination still dominates: a better-calibrated worse ranker loses.
    assert Score(0.9, -0.01) < Score(0.95, -0.90)


def test_a_saturated_auroc_does_not_stop_the_fit_at_epoch_one():
    """On separable data AUROC ties at 1.0 from the first epoch.

    Without the cross-entropy tie-break, strict improvement never fires again
    and the fit keeps epoch 1 — a model that ranks perfectly while predicting
    close to the base rate for everyone. Calibration is a reported evaluation
    category (spec §4.5), so that is a real failure, not a cosmetic one.
    """
    rng = np.random.default_rng(23)
    y = np.repeat([0, 1], 150)
    X = np.column_stack([y * 8.0 + rng.normal(0, 0.1, 300), rng.normal(0, 1, 300)])
    hp = parse_hparams({"epochs": 200, "patience": 10}, "linear_probe", 150, 150)
    head, record = fit_linear_head(X, y, hp, (X[::3], y[::3]))
    assert record["best_epoch"] > 1
    assert record["best_validation_score"] == pytest.approx(1.0)
    probs = head.probabilities(X)
    assert probs[y == 1].mean() - probs[y == 0].mean() > 0.5


def test_early_stopping_can_minimise():
    stopper = EarlyStopping(patience=1, greater_is_better=False)
    stopper.update(1, 0.5)
    assert stopper.update(2, 0.4)


def test_binary_cross_entropy_matches_a_hand_computed_value():
    # p = 0.75 on a positive, p = 0.25 on a negative: both -log(0.75).
    loss = binary_cross_entropy(np.array([0.75, 0.25]), np.array([1, 0]))
    assert loss == pytest.approx(-np.log(0.75))


def test_binary_cross_entropy_is_finite_at_a_probability_of_one():
    assert np.isfinite(binary_cross_entropy(np.array([1.0]), np.array([0])))


def test_sigmoid_is_stable_at_large_magnitudes():
    values = sigmoid(np.array([-800.0, 0.0, 800.0]))
    assert np.all(np.isfinite(values))
    assert values.tolist() == [0.0, 0.5, 1.0]


# ---------------------------------------------------------------------------
# the logistic head
# ---------------------------------------------------------------------------


def test_the_head_recovers_a_known_separation():
    rng = np.random.default_rng(3)
    y = np.repeat([0, 1], 200)
    X = np.stack([y * 3.0 + rng.normal(0, 1, 400), rng.normal(0, 1, 400)], axis=1)
    hp = parse_hparams({"epochs": 400}, "linear_probe", 200, 200)
    head, record = fit_linear_head(X, y, hp)
    probs = head.probabilities(X)
    assert probs[y == 1].mean() > 0.8
    assert probs[y == 0].mean() < 0.2
    # The informative feature must dominate the noise one.
    assert abs(head.weights[0]) > 5 * abs(head.weights[1])
    assert record["n_features"] == 2


def test_the_head_is_exactly_reproducible_and_seed_independent():
    """A convex objective from a deterministic start consumes no randomness.

    That is a property worth pinning: it means a linear_probe arm reproduces
    bit-for-bit regardless of the run's seed.
    """
    X = np.random.default_rng(5).normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(np.int64)
    first, _ = fit_linear_head(X, y, parse_hparams({"seed": 1}, "linear_probe", 60, 60))
    second, _ = fit_linear_head(X, y, parse_hparams({"seed": 99}, "linear_probe", 60, 60))
    np.testing.assert_array_equal(first.weights, second.weights)


def test_a_constant_feature_does_not_produce_nans():
    X = np.column_stack([np.random.default_rng(7).normal(size=200), np.ones(200)])
    y = (X[:, 0] > 0).astype(np.int64)
    head, _ = fit_linear_head(X, y, parse_hparams({}, "linear_probe", 100, 100))
    assert np.all(np.isfinite(head.weights))
    assert np.all(np.isfinite(head.probabilities(X)))


def test_early_stopping_restores_the_best_epoch_not_the_last():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 3))
    y = (X[:, 0] + rng.normal(0, 0.5, 300) > 0).astype(np.int64)
    validation = (rng.normal(size=(100, 3)), (rng.normal(size=100) > 0).astype(np.int64))
    hp = parse_hparams({"epochs": 200, "patience": 3}, "linear_probe", 150, 150)
    _head, record = fit_linear_head(X, y, hp, validation)
    assert record["best_epoch"] is not None
    assert record["best_epoch"] <= record["epochs_run"]
    assert record["epochs_run"] < hp.epochs  # it stopped early


def test_a_single_class_validation_fold_says_what_to_do():
    X = np.random.default_rng(13).normal(size=(100, 2))
    y = (X[:, 0] > 0).astype(np.int64)
    validation = (np.random.default_rng(14).normal(size=(30, 2)), np.zeros(30, dtype=int))
    hp = parse_hparams({"epochs": 5}, "linear_probe", 50, 50)
    with pytest.raises(TrainingError, match="early_stopping_metric: loss"):
        fit_linear_head(X, y, hp, validation)


def test_loss_is_an_alternative_stopping_metric():
    X = np.random.default_rng(15).normal(size=(100, 2))
    y = (X[:, 0] > 0).astype(np.int64)
    validation = (np.random.default_rng(16).normal(size=(30, 2)), np.zeros(30, dtype=int))
    hp = parse_hparams(
        {"epochs": 5, "early_stopping_metric": "loss"}, "linear_probe", 50, 50
    )
    _head, record = fit_linear_head(X, y, hp, validation)
    assert record["validation_metric"] == "loss"


def test_weight_decay_shrinks_the_weights():
    X = np.random.default_rng(17).normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(np.int64)
    loose, _ = fit_linear_head(X, y, parse_hparams({"weight_decay": 0.0}, "linear_probe", 100, 100))
    tight, _ = fit_linear_head(X, y, parse_hparams({"weight_decay": 5.0}, "linear_probe", 100, 100))
    assert np.linalg.norm(tight.weights) < np.linalg.norm(loose.weights)


def test_the_head_applies_its_training_standardisation_at_inference():
    X = np.random.default_rng(19).normal(loc=50.0, scale=10.0, size=(200, 2))
    y = (X[:, 0] > 50).astype(np.int64)
    head, _ = fit_linear_head(X, y, parse_hparams({}, "linear_probe", 100, 100))
    assert head.mean == pytest.approx(X.mean(axis=0))
    manual = sigmoid(((X - head.mean) / head.scale) @ head.weights + head.bias)
    np.testing.assert_allclose(head.probabilities(X), manual)


def test_a_head_lifted_from_a_backend_module_applies_no_standardisation():
    head = LinearHead.of(np.array([[2.0, -1.0]]), 0.5)
    assert head.probabilities(np.array([[1.0, 1.0]]))[0] == pytest.approx(sigmoid(np.array([1.5]))[0])


def test_the_head_rejects_features_of_the_wrong_width():
    head = LinearHead.of(np.array([1.0, 2.0]), 0.0)
    with pytest.raises(TrainingError, match="2-dimensional features"):
        head.probabilities(np.zeros((3, 5)))


# ---------------------------------------------------------------------------
# Adapter.fit end to end, with no backend
# ---------------------------------------------------------------------------


def test_linear_probe_runs_through_the_generic_adapter_path():
    adapter = FakeAdapter()
    fitted = adapter.fit("handle", make_data(), "linear_probe", {"epochs": 100})
    assert isinstance(fitted["head"], LinearHead)
    assert fitted["fit_record"]["mode"] == "linear_probe"
    assert fitted["fit_record"]["hparams"]["epochs"] == 100


def test_linear_probe_reads_each_record_once_not_once_per_epoch():
    """The representation is cached because a frozen backbone cannot change it.

    With 140 training and 60 validation records at batch size 16, that is
    9 + 4 = 13 calls — not 13 x epochs.
    """
    adapter = FakeAdapter()
    adapter.fit("handle", make_data(), "linear_probe", {"epochs": 50})
    assert adapter.feature_calls == 13


def test_fit_does_not_mutate_the_handle_it_was_given():
    # models fits one handle per inner CV fold; a fit that wrote through to its
    # input would carry one fold's weights into the next fold's estimate.
    adapter = FakeAdapter()
    handle = {"weights": "original"}
    fitted = adapter.fit(handle, make_data(), "linear_probe", {"epochs": 20})
    assert handle == {"weights": "original"}
    assert fitted is not handle


def test_an_adapter_without_a_finetune_path_says_so():
    class ProbeOnly(FakeAdapter):
        pass

    with pytest.raises(NotImplementedError, match="full_finetune"):
        ProbeOnly().fit("handle", make_data(), "full_finetune", {"seed": 1})


def test_an_adapter_that_cannot_attach_a_head_says_so():
    class HeadlessAdapter(FakeAdapter):
        def with_head(self, handle, head, record):
            return Adapter.with_head(self, handle, head, record)

    with pytest.raises(NotImplementedError, match="inference_only"):
        HeadlessAdapter().fit("handle", make_data(), "linear_probe", {"epochs": 2})


def test_features_of_the_wrong_shape_are_caught_with_the_adapter_named():
    class Broken(FakeAdapter):
        def features(self, handle, batch, index=0):
            return np.zeros(len(batch))  # 1-D, not (B, D)

    with pytest.raises(TrainingError, match="features\\(\\) returned shape"):
        Broken().fit("handle", make_data(), "linear_probe", {"epochs": 2})


def test_attribute_is_optional_and_says_which_adapter_lacks_it():
    with pytest.raises(NotImplementedError, match="optional"):
        FakeAdapter().attribute("handle", np.zeros((1, 2, 3)))


# ---------------------------------------------------------------------------
# Several representations per handle (ensembles) — spec §2.5 symmetry
# ---------------------------------------------------------------------------


class EnsembleAdapter(Adapter):
    """Three "members" whose representations share no coordinate system.

    Member k sees the two real features permuted and offset by k, which is what
    independently trained members look like: each is informative, none is
    comparable to another coordinate for coordinate.
    """

    name = "fake_ensemble"

    def __init__(self, n_members=3):
        self.n_members = n_members
        self.calls = []

    def feature_set_names(self, handle):
        return [f"member{index}" for index in range(self.n_members)]

    def features(self, handle, batch, index=0):
        self.calls.append(index)
        block = np.asarray(batch)[:, :, 0]
        return np.roll(block, index, axis=1) + index

    def with_heads(self, handle, heads, record):
        return {"handle": handle, "heads": list(heads), "fit_record": dict(record)}


def test_one_head_is_fitted_per_representation(tmp_path):
    adapter = EnsembleAdapter()
    fitted = adapter.fit(object(), make_data(), "linear_probe", {"epochs": 30})
    assert len(fitted["heads"]) == 3
    assert fitted["fit_record"]["feature_sets"] == ["member0", "member1", "member2"]


def test_every_member_is_read_and_none_is_read_twice_per_epoch():
    adapter = EnsembleAdapter()
    adapter.fit(object(), make_data(n=32, n_validation=0, batch_size=32), "linear_probe",
                {"epochs": 40})
    # One features() call per member per split, not one per epoch.
    assert sorted(adapter.calls) == [0, 1, 2]


def test_the_per_member_fits_are_all_recorded_not_just_the_first():
    adapter = EnsembleAdapter()
    fitted = adapter.fit(object(), make_data(), "linear_probe", {"epochs": 30})
    members = fitted["fit_record"]["members"]
    assert [entry["feature_set"] for entry in members] == ["member0", "member1", "member2"]
    assert all("train_loss" in entry for entry in members)


def test_each_member_head_learns_its_own_representation():
    adapter = EnsembleAdapter()
    data = make_data(seed=3)
    fitted = adapter.fit(object(), data, "linear_probe", {"epochs": 200})
    # The members differ, so heads fitted on them must differ too; identical
    # heads would mean the permutation never reached the fit.
    weights = [head.weights.tolist() for head in fitted["heads"]]
    assert weights[0] != weights[1] != weights[2]


def test_a_single_representation_still_goes_through_with_head():
    adapter = FakeAdapter()
    fitted = adapter.fit(object(), make_data(), "linear_probe", {"epochs": 20})
    assert "head" in fitted and "heads" not in fitted


def test_an_adapter_with_several_representations_must_implement_with_heads():
    class Partial(EnsembleAdapter):
        with_heads = Adapter.with_heads

    with pytest.raises(NotImplementedError, match="with_heads"):
        Partial().fit(object(), make_data(), "linear_probe", {"epochs": 5})


def make_regression_data(n=200, seed=0, n_validation=60, batch_size=16):
    """y = 3*x0 - 2*x1 + 50 + noise, so a linear head must recover the coefficients."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, (n, 2))
    y = 3.0 * signal[:, 0] - 2.0 * signal[:, 1] + 50.0 + rng.normal(0, 0.1, n)
    store = {f"rec-{i}": signal[i] for i in range(n)}

    def load_batch(paths):
        return np.stack([np.repeat(store[p][:, None], 4, axis=1) for p in paths])

    split = n - n_validation
    return {
        "tensor_paths": [f"rec-{i}" for i in range(split)], "y": y[:split],
        "person_id": [f"p-{i}" for i in range(split)], "fold": [0] * split,
        "label_column": "label_value", "objective": "mse",
        "load_batch": load_batch, "batch_size": batch_size,
        "validation": {"tensor_paths": [f"rec-{i}" for i in range(split, n)], "y": y[split:],
                       "person_id": [f"p-{i}" for i in range(split, n)]},
    }


def test_mse_objective_accepts_continuous_labels():
    data = parse_fit_data(make_regression_data())
    assert data.objective == "mse"
    assert data.train.y.dtype == np.float64


def test_bce_objective_still_rejects_continuous_labels():
    body = make_regression_data()
    body["objective"] = "bce"
    with pytest.raises(TrainingError, match="labels must be 0 or 1"):
        parse_fit_data(body)


def test_linear_probe_with_mse_recovers_the_coefficients():
    adapter = FakeAdapter()
    fitted = adapter.fit("handle", make_regression_data(), "linear_probe", {"epochs": 2000, "lr": 0.05, "patience": 200})
    head = fitted["head"]
    assert head.link == "identity"
    x = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    predicted = head.predict(x)
    assert np.allclose(predicted, [53.0, 48.0, 50.0], atol=0.5)
    assert fitted["fit_record"]["objective"] == "mse"
    assert fitted["fit_record"]["validation_metric"] == "mae"


def test_validation_score_for_mse_is_negative_mae():
    score = validation_score(np.array([1.0, 3.0]), np.array([2.0, 2.0]), "auroc", "t", objective="mse")
    assert score.primary == -1.0
