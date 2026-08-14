"""Keras 2.7 backend.

The archived PROPHECG runtime is intentionally CPU-only: it predates the CUDA
toolkit on this instance. TensorFlow is imported inside load() so that the
torch environment can import mival.adapters without TensorFlow installed.

This adapter fits a head (``linear_probe``) but does not backpropagate into the
backbone; see ``_finetune`` for why.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

import numpy as np

from mival.adapters._training import FitData, FitHParams, LinearHead, TrainingError
from mival.adapters.base import Adapter, to_model_layout
from mival.modelcard import ModelCard


@dataclass
class KerasHandle:
    members: List[Any]
    card: ModelCard
    #: Present once a head has been fitted on the frozen representation.
    head: Optional[LinearHead] = None
    #: What the fit did, copied into ``train_log.jsonl`` by the models stage.
    fit_record: Optional[Mapping[str, Any]] = None


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
        if handle.head is not None:
            return handle.head.probabilities(self.features(handle, batch))
        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        outputs = [
            np.asarray(member(arranged, training=False)) for member in handle.members
        ]
        return pool_ensemble(
            outputs,
            handle.card.ensemble.get("method", "none"),
            handle.card.output["positive_index"],
        )

    def features(self, handle: KerasHandle, batch: np.ndarray) -> np.ndarray:
        if handle.card.feature_layer is None:
            raise NotImplementedError(
                f"{handle.card.model_id} declares no feature_layer; "
                "representation extraction is unavailable"
            )
        if len(handle.members) > 1:
            # Averaging representations from independently trained members is
            # meaningless — they share no coordinate system — and silently
            # probing member 0 would make a "head-retrained" arm a
            # single-member arm while the inference_only arm keeps all five.
            # The comparison in spec §2.5 depends on those two being symmetric,
            # so this is refused rather than guessed.
            raise NotImplementedError(
                f"{handle.card.model_id} is a {len(handle.members)}-member ensemble "
                f"({handle.card.ensemble.get('method')!r}); a single representation is "
                "not defined for it. Probing one member would silently drop the ensemble "
                "that the inference_only arm uses, so the two arms would no longer be "
                "comparable."
            )
        import tensorflow as tf

        member = handle.members[0]
        extractor = tf.keras.Model(
            inputs=member.input, outputs=member.get_layer(handle.card.feature_layer).output
        )
        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        return np.asarray(extractor(arranged, training=False))

    def trainable_groups(self, handle: KerasHandle) -> List[str]:
        return [layer.name for layer in handle.members[0].layers]

    # -- fitting -----------------------------------------------------------

    def with_head(
        self, handle: KerasHandle, head: LinearHead, record: Mapping[str, Any]
    ) -> KerasHandle:
        # Members are shared rather than copied: linear_probe never writes to
        # them, and the archived models are large.
        return KerasHandle(
            members=handle.members,
            card=handle.card,
            head=head,
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
