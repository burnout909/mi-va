"""Keras 2.7 backend.

The archived PROPHECG runtime is intentionally CPU-only: it predates the CUDA
toolkit on this instance. TensorFlow is imported inside load() so that the
torch environment can import mival.adapters without TensorFlow installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from mival.adapters.base import Adapter, to_model_layout
from mival.modelcard import ModelCard


@dataclass
class KerasHandle:
    members: List[Any]
    card: ModelCard


class KerasAdapter(Adapter):
    name = "keras"

    def load(self, card: ModelCard) -> KerasHandle:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        members = [
            tf.keras.models.load_model(spec["uri"], compile=False)
            for spec in card.weights
        ]
        return KerasHandle(members=members, card=card)

    def forward(self, handle: KerasHandle, batch: np.ndarray) -> np.ndarray:
        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        outputs = [
            np.asarray(member(arranged, training=False)) for member in handle.members
        ]
        method = handle.card.ensemble.get("method", "none")
        if method == "mean_probability":
            pooled = np.mean(np.stack(outputs, axis=0), axis=0)
        elif method == "none":
            pooled = outputs[0]
        else:
            raise ValueError(f"unsupported ensemble method: {method}")
        return pooled[:, handle.card.output["positive_index"]].astype(np.float64)

    def features(self, handle: KerasHandle, batch: np.ndarray) -> np.ndarray:
        if handle.card.feature_layer is None:
            raise NotImplementedError(
                f"{handle.card.model_id} declares no feature_layer; "
                "representation extraction is unavailable"
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
