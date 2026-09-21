"""Keras 2.7 backend.

The archived PROPHECG runtime is intentionally CPU-only: it predates the CUDA
toolkit on this instance. TensorFlow is imported inside load() so that the
torch environment can import mival.adapters without TensorFlow installed.

This adapter fits a head (``linear_probe``) but does not backpropagate into the
backbone; see ``_finetune`` for why.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

import numpy as np

from mival.adapters._training import FitData, FitHParams, LinearHead, TrainingError
from mival.adapters.base import Adapter, to_model_layout
from mival.modelcard import ModelCard


@dataclass
class KerasHandle:
    members: List[Any]
    card: ModelCard
    #: One fitted head per ensemble member, in member order. Empty until a
    #: ``linear_probe`` has run; the card's own pooling rule combines them.
    heads: List[LinearHead] = field(default_factory=list)
    #: What the fit did, copied into ``train_log.jsonl`` by the models stage.
    fit_record: Optional[Mapping[str, Any]] = None


def _as_two_class(positive: np.ndarray) -> np.ndarray:
    """``(B,)`` positive-class probability as the ``(B, 2)`` form of the card.

    The card's ``ensemble.method`` is defined over the model's own output
    columns, so pooling reads the same shape whether the probabilities came
    from the published head or from a fitted one.
    """
    column = np.asarray(positive, dtype=np.float64).reshape(-1, 1)
    return np.hstack([1.0 - column, column])


def pool_ensemble(
    outputs: List[np.ndarray], method: str, positive_index: int
) -> np.ndarray:
    """Reduce per-member class probabilities to one positive-class score.

    Members are averaged first and the positive column is taken second. The
    reverse order gives a different answer for any non-degenerate ensemble.
    """
    if method == "mean_probability":
        pooled = np.mean(np.stack(outputs, axis=0), axis=0)
    elif method == "none":
        pooled = outputs[0]
    else:
        raise ValueError(f"unsupported ensemble method: {method}")
    return pooled[:, positive_index].astype(np.float64)


class KerasAdapter(Adapter):
    name = "keras"

    def load(self, card: ModelCard) -> KerasHandle:
        # The archived PROPHECG runtime predates this instance's CUDA toolkit,
        # so the CPU pin must win even if the surrounding process exported a
        # device. setdefault would silently leave an inherited value in place.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        members = [
            tf.keras.models.load_model(spec["uri"], compile=False)
            for spec in card.weights
        ]
        return KerasHandle(members=members, card=card)

    def forward(self, handle: KerasHandle, batch: np.ndarray) -> np.ndarray:
        if handle.heads:
            # A probed ensemble pools where the unprobed one pools: at the
            # probability level, by the card's own rule. Each member's head
            # emits a positive-class probability, so the two-column form
            # `pool_ensemble` expects is rebuilt from it.
            # `_as_two_class` also fits an identity head: column 1 is then the
            # predicted value rather than a probability, and `pool_ensemble`
            # still averages it across members the same way.
            outputs = [
                _as_two_class(head.predict(self.features(handle, batch, index)))
                for index, head in enumerate(handle.heads)
            ]
            return pool_ensemble(outputs, self._pooling(handle), 1)
        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        outputs = [
            np.asarray(member(arranged, training=False)) for member in handle.members
        ]
        column = (
            handle.card.output["value_index"]
            if handle.card.output.get("type") == "regression"
            else handle.card.output["positive_index"]
        )
        return pool_ensemble(outputs, self._pooling(handle), column)

    def _pooling(self, handle: KerasHandle) -> str:
        return handle.card.ensemble.get("method", "none")

    def feature_set_names(self, handle: KerasHandle) -> List[str]:
        """One representation per ensemble member.

        Independently trained members share no coordinate system, so there is
        no single representation for the ensemble — but there is one per
        member, and the card already says how member outputs combine. Probing
        every member and pooling by that rule keeps the ``linear_probe`` arm
        the same size and shape as the ``inference_only`` arm, which is what
        makes the spec §2.5 comparison a comparison.
        """
        if handle.card.feature_layer is None:
            raise NotImplementedError(
                f"{handle.card.model_id} declares no feature_layer; "
                "representation extraction is unavailable"
            )
        if len(handle.members) > 1 and self._pooling(handle) != "mean_probability":
            raise NotImplementedError(
                f"{handle.card.model_id} is a {len(handle.members)}-member ensemble whose "
                f"card declares ensemble.method={self._pooling(handle)!r}. Per-member heads "
                "can only be pooled by a rule the card states; declare "
                "'mean_probability' or reduce the card to a single member."
            )
        return [f"member{index}" for index in range(len(handle.members))]

    def features(self, handle: KerasHandle, batch: np.ndarray, index: int = 0) -> np.ndarray:
        names = self.feature_set_names(handle)
        if not 0 <= index < len(names):
            raise IndexError(
                f"{handle.card.model_id} has {len(names)} representations; feature set "
                f"{index} does not exist"
            )
        import tensorflow as tf

        member = handle.members[index]
        extractor = tf.keras.Model(
            inputs=member.input, outputs=member.get_layer(handle.card.feature_layer).output
        )
        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        return np.asarray(extractor(arranged, training=False))

    def trainable_groups(self, handle: KerasHandle) -> List[str]:
        return [layer.name for layer in handle.members[0].layers]

    # -- fitting -----------------------------------------------------------

    def with_heads(
        self, handle: KerasHandle, heads: List[LinearHead], record: Mapping[str, Any]
    ) -> KerasHandle:
        if len(heads) != len(handle.members):
            raise TrainingError(
                f"{handle.card.model_id}: {len(heads)} heads were fitted for "
                f"{len(handle.members)} ensemble members"
            )
        # Members are shared rather than copied: linear_probe never writes to
        # them, and the archived models are large.
        return KerasHandle(
            members=handle.members,
            card=handle.card,
            heads=list(heads),
            fit_record=dict(record),
        )

    def _finetune(
        self, handle: KerasHandle, data: FitData, mode: str, hp: FitHParams
    ) -> KerasHandle:
        """Not implemented, deliberately.

        No arm in spec §2.5 needs it: the study's trained arms are two linear
        probes and one torch full fine-tune. The archived runtime is also
        CPU-only by construction (the card's ``runtime.device`` is ``cpu``,
        because Keras 2.7 predates this instance's CUDA toolkit), so
        backpropagating through five ensemble members over a hospital-scale dev
        split is not something the available hardware can run in any case.

        A card that declares one of these modes is declaring something this
        adapter cannot deliver, and saying so here is better than discovering
        it after a long run.
        """
        raise TrainingError(
            f"the keras adapter does not implement training mode {mode!r} for "
            f"{handle.card.model_id!r}. Only 'inference_only' and 'linear_probe' are "
            "available: the archived Keras 2.7 runtime is CPU-only, so fine-tuning a "
            f"{len(handle.members)}-member ensemble is not runnable here. Remove "
            f"{mode!r} from the card's training_modes_supported, or move the model to a "
            "backend that can train it."
        )
