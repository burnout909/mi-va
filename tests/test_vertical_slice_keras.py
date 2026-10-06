"""Keras adapter head selection, shared decoding and framed prediction (no TensorFlow needed)."""
import numpy as np

from mival.adapters.keras_adapter import KerasAdapter, KerasHandle
from mival.decode.frames import predict_in_frames


class FakeModel:
    """Two named heads; head 'curve' returns per-interval survival 0.5."""

    output_names = ["sex", "curve"]

    def __call__(self, x, training=False):
        n = x.shape[0]
        return [np.tile([[0.2, 0.8]], (n, 1)), np.full((n, 50), 0.5)]


class FakeCard:
    def __init__(self, output, layout="time_lead", fs=500):
        self.output = output
        self.ensemble = {"method": "none", "members": 1}
        self.model_id = "fake"
        self.runtime = {}

        class C:
            pass

        self.input_contract = C()
        self.input_contract.layout = layout
        self.input_contract.sampling_rate_hz = fs


def test_keras_selects_a_named_head_and_decodes_a_survival_curve():
    card = FakeCard({"type": "survival_curve", "head": "curve", "intervals": 25, "bin_days": 146, "horizon_days": 365})
    out = KerasAdapter().forward(KerasHandle(members=[FakeModel()], card=card), np.zeros((3, 12, 10), np.float32))
    np.testing.assert_allclose(out, 1 - 0.5 ** 2.5)


def test_keras_ensemble_of_decoded_regressions_is_averaged():
    class Reg:
        output_names = ["value"]

        def __init__(self, v):
            self.v = v

        def __call__(self, x, training=False):
            return np.full((x.shape[0], 1), self.v)

    card = FakeCard({"type": "regression", "value_index": 0})
    card.ensemble = {"method": "mean", "members": 2}
    out = KerasAdapter().forward(KerasHandle(members=[Reg(1.0), Reg(3.0)], card=card), np.zeros((2, 1, 4), np.float32))
    np.testing.assert_allclose(out, [2.0, 2.0])


def test_predict_in_frames_covers_the_tail_and_keeps_order():
    seen = []

    def predict(frames):
        seen.append(frames.shape)
        return frames.copy()  # identity "scores" (B, frame, C)

    x = np.arange(10, dtype=np.float32).reshape(1, 10, 1)
    out = predict_in_frames(predict, x, frame=4)
    assert out.shape == (1, 10, 1) and seen == [(3, 4, 1)]
    np.testing.assert_allclose(out[0, :, 0], np.arange(10))


def test_predict_in_frames_can_zscore_each_frame():
    x = np.concatenate([np.full((1, 4, 1), 5.0), np.arange(4, dtype=float).reshape(1, 4, 1)], axis=1)
    out = predict_in_frames(lambda f: f, x, frame=4, zscore_eps=0.01)
    np.testing.assert_allclose(out[0, :4, 0], 0.0)
    assert abs(out[0, 4:, 0].mean()) < 1e-9
