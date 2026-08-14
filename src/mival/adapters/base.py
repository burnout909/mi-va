"""Adapter interface.

Backends are imported lazily inside adapter methods so that `mival` core
imports cleanly in both the torch and the keras27 environment.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from mival.modelcard import ModelCard


def to_model_layout(batch: np.ndarray, layout: str) -> np.ndarray:
    """Convert canonical (B, n_leads, n_samples) into the model's layout."""
    if layout == "lead_time":
        return batch
    if layout == "time_lead":
        return np.ascontiguousarray(np.transpose(batch, (0, 2, 1)))
    raise ValueError(f"unknown layout: {layout}")


class Adapter:
    name: str = "adapter"

    def load(self, card: ModelCard) -> Any:
        raise NotImplementedError

    def forward(self, handle: Any, batch: np.ndarray) -> np.ndarray:
        """Return positive-class probability, shape (B,)."""
        raise NotImplementedError

    def features(self, handle: Any, batch: np.ndarray) -> np.ndarray:
        """Return representations, shape (B, D)."""
        raise NotImplementedError

    def trainable_groups(self, handle: Any) -> List[str]:
        raise NotImplementedError

    # fit() and attribute() are added in Plan 4 (Models stage).
