"""Adapter interface (spec §4.4).

Backends are imported lazily inside adapter methods so that `mival` core
imports cleanly in both the torch and the keras27 environment.

``fit`` is concrete here rather than abstract. ``linear_probe`` trains one
logistic layer on a frozen representation, and once ``features`` has produced
that representation nothing about the fit is backend-specific — so it is
implemented once, in NumPy, and both backends inherit exactly the same
optimiser, regularisation and stopping rule. Only the modes that actually
propagate gradients into the backbone are delegated to the backend, through
``_finetune``.
"""

from __future__ import annotations

from typing import Any, List, Mapping

import numpy as np

from mival.adapters._attribution import DEFAULT_STEPS
from mival.adapters._training import (
    FitData,
    FitHParams,
    FitSplit,
    LinearHead,
    TrainingError,
    fit_linear_head,
    parse_fit_data,
    parse_hparams,
)
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

    def attribute(
        self,
        handle: Any,
        batch: np.ndarray,
        baseline: Any = None,
        steps: int = DEFAULT_STEPS,
    ) -> np.ndarray:
        """Return per-sample saliency over the input, shape of ``batch``.

        Optional in spec §4.4. The method and — more importantly — the baseline
        are chosen in ``mival.adapters._attribution``; read that module before
        calling this, because ``baseline`` has no default and the reason it has
        none is the substance of the choice.
        """
        raise NotImplementedError(
            f"adapter {self.name!r} implements no attribute(); spec §4.4 marks it optional"
        )

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        handle: Any,
        data: Mapping[str, Any],
        mode: str,
        hparams: Mapping[str, Any],
    ) -> Any:
        """Fit ``handle`` for ``mode`` and return the handle to run inference with.

        The returned handle is a *new* object, not the argument mutated in
        place: ``models`` fits one handle per inner CV fold and compares their
        held-out predictions, so a fit that leaked into its input would carry
        one fold's weights into the next fold's estimate.
        """
        payload = parse_fit_data(data)
        hp = parse_hparams(hparams, mode, payload.train.n_pos, payload.train.n_neg)
        if mode == "linear_probe":
            return self._fit_linear_probe(handle, payload, hp)
        return self._finetune(handle, payload, mode, hp)

    def _fit_linear_probe(self, handle: Any, data: FitData, hp: FitHParams) -> Any:
        """Freeze the backbone, cache its representation, fit one logistic layer.

        Caching is what makes this cheap: with the backbone frozen the
        representation does not change between epochs, so the signals are read
        from disk once instead of once per epoch. It is exactly equivalent
        because ``features`` runs the module in inference mode — no dropout, no
        batch-norm update.
        """
        train_features = self._collect_features(handle, data.train, data)
        validation = None
        if data.validation is not None:
            validation = (
                self._collect_features(handle, data.validation, data),
                data.validation.y,
            )
        head, record = fit_linear_head(
            train_features, data.train.y, hp, validation, context="linear_probe"
        )
        record.update(
            {
                "mode": "linear_probe",
                "hparams": hp.to_record(),
                "n_validation": 0 if data.validation is None else len(data.validation),
                "frozen": "all",
            }
        )
        return self.with_head(handle, head, record)

    def _collect_features(
        self, handle: Any, split: FitSplit, data: FitData
    ) -> np.ndarray:
        """Representations for every record of ``split``, in order."""
        chunks: List[np.ndarray] = []
        for start in range(0, len(split), data.batch_size):
            paths = split.tensor_paths[start : start + data.batch_size]
            batch = data.load_batch(list(paths))
            block = np.asarray(self.features(handle, batch), dtype=np.float64)
            if block.ndim != 2 or block.shape[0] != len(paths):
                raise TrainingError(
                    f"adapter {self.name!r} features() returned shape {block.shape} for a "
                    f"batch of {len(paths)} records; (B, D) is required"
                )
            chunks.append(block)
        if not chunks:
            raise TrainingError(f"adapter {self.name!r}: no records to extract features from")
        return np.concatenate(chunks, axis=0)

    def with_head(self, handle: Any, head: LinearHead, record: Mapping[str, Any]) -> Any:
        """A copy of ``handle`` whose ``forward`` runs ``head`` over ``features``."""
        raise NotImplementedError(
            f"adapter {self.name!r} cannot attach a fitted head, so it supports no "
            "training mode beyond inference_only"
        )

    def _finetune(self, handle: Any, data: FitData, mode: str, hp: FitHParams) -> Any:
        """Fit with gradients reaching the backbone. Backend-specific."""
        raise NotImplementedError(
            f"adapter {self.name!r} does not implement training mode {mode!r}"
        )
