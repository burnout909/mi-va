"""PyTorch backend.

torch is imported inside methods so that importing `mival.adapters` stays
cheap and works in the keras27 environment where torch is absent.

Inference has one shape whatever produced the handle: run the module to get a
representation, then run a linear head over it. ``inference_only`` on a card
whose checkpoint already carries the right head is the case where the head
comes from the checkpoint; ``linear_probe`` and the fine-tuning modes fit one.
Keeping a single path means a fitted handle cannot accidentally be scored
through a different code path than an unfitted one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from mival.adapters._attribution import (
    DEFAULT_STEPS,
    integration_points,
    resolve_baseline,
)
from mival.adapters._training import (
    FitData,
    FitHParams,
    FitSplit,
    LinearHead,
    TrainingError,
    EarlyStopping,
    epoch_batches,
    validation_score,
)
from mival.adapters.base import Adapter, to_model_layout
from mival.modelcard import ModelCard


@dataclass
class TorchHandle:
    module: Any
    device: str
    card: ModelCard
    #: Present once the handle has been fitted. ``None`` means the module
    #: emits no positive-class probability yet.
    head: Optional[LinearHead] = None
    #: What the fit did, copied into ``train_log.jsonl`` by the models stage.
    fit_record: Optional[Mapping[str, Any]] = None


def _module_outputs(module: Any, tensor: Any) -> Tuple[Optional[Any], Any]:
    """``(logits, features)`` from a module that may return either or both.

    Modules in this project are constructed to return ``(logits, features)``.
    Accepting a bare tensor as well keeps the adapter from assuming one
    module's return signature.
    """
    output = module(tensor)
    if isinstance(output, tuple):
        if len(output) != 2:
            raise TrainingError(
                f"module returned a {len(output)}-tuple; expected (logits, features) or "
                "a features tensor"
            )
        return output[0], output[1]
    return None, output


def _module_features(module: Any, tensor: Any) -> Any:
    return _module_outputs(module, tensor)[1]


def _module_dtype(module: Any) -> Any:
    """The dtype the module's own parameters use.

    Attribution builds its input tensors rather than receiving them, so it has
    to match the module instead of hoping the caller's array already did.
    """
    import torch

    for parameter in module.parameters():
        return parameter.dtype
    return torch.float32


def _as_tensor(array: np.ndarray, device: str, dtype: Any) -> Any:
    import torch

    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=dtype)


class TorchAdapter(Adapter):
    name = "torch"

    def load(self, card: ModelCard) -> TorchHandle:
        import gc
        import sys

        import torch

        code_path = card.raw["x-mival"]["code_path"]
        if code_path not in sys.path:
            sys.path.insert(0, code_path)
        from net1d import Net1D  # official ECGFounder checkout

        # The checkpoint stores two NumPy scalar metadata objects. Only those
        # known globals are allow-listed; arbitrary pickle execution stays off.
        safe_globals = [
            (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            (np.dtype, "numpy.dtype"),
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(safe_globals):
            checkpoint = torch.load(
                card.weights[0]["uri"], map_location="cpu", weights_only=True
            )

        state_dict = checkpoint["state_dict"]
        n_classes = int(state_dict["dense.weight"].shape[0])
        declared = card.output.get("n_outputs")
        if declared is not None and int(declared) != n_classes:
            raise ValueError(
                f"card declares {declared} outputs but the checkpoint has {n_classes}"
            )
        module = Net1D(
            in_channels=12,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            n_classes=n_classes,
            use_bn=False,
            use_do=False,
            return_features=True,
            verbose=False,
        )
        module.load_state_dict(state_dict, strict=True)
        del checkpoint, state_dict
        gc.collect()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        module = module.to(device).eval()
        return TorchHandle(module=module, device=device, card=card)

    def features(self, handle: TorchHandle, batch: np.ndarray) -> np.ndarray:
        import torch

        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        tensor = torch.from_numpy(np.ascontiguousarray(arranged)).to(handle.device)
        with torch.inference_mode():
            features = _module_features(handle.module, tensor)
        return features.detach().cpu().numpy()

    def forward(self, handle: TorchHandle, batch: np.ndarray) -> np.ndarray:
        if handle.head is not None:
            return handle.head.probabilities(self.features(handle, batch))
        return self._published_probabilities(handle, batch)

    def _published_probabilities(
        self, handle: TorchHandle, batch: np.ndarray
    ) -> np.ndarray:
        """The checkpoint's own positive-class score, when the card declares one.

        Which column is the positive class, and whether the head is a softmax
        or per-class logits, are card facts. A checkpoint whose card names no
        ``positive_index`` publishes no probability for this outcome, and
        ``inference_only`` is genuinely unavailable for it — that is a property
        of the model, so the error says which model and what to do instead.
        """
        import torch

        index = handle.card.output.get("positive_index")
        if index is None:
            raise NotImplementedError(
                f"{handle.card.model_id!r} declares no output.positive_index, so its "
                "checkpoint emits no probability for this outcome. Fit a head with "
                "linear_probe, partial_unfreeze or full_finetune before calling forward."
            )
        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        tensor = torch.from_numpy(np.ascontiguousarray(arranged)).to(handle.device)
        with torch.inference_mode():
            logits, _features = _module_outputs(handle.module, tensor)
        if logits is None:
            raise NotImplementedError(
                f"{handle.card.model_id!r} returns a representation only; there is no "
                "published head to read a probability from"
            )
        return self._positive_probability(handle, logits).detach().cpu().numpy().astype(
            np.float64
        )

    def _positive_probability(self, handle: TorchHandle, logits: Any) -> Any:
        """Card-driven read of the positive class, kept as a tensor.

        ``forward`` detaches it; ``attribute`` differentiates through it. Both
        must read the same column of the same head, so the branch lives once.
        """
        import torch

        index = int(handle.card.output["positive_index"])
        kind = handle.card.output.get("type", "logits")
        if kind == "softmax":
            return torch.softmax(logits, dim=-1)[:, index]
        if kind in ("logits", "sigmoid"):
            return torch.sigmoid(logits[:, index])
        raise ValueError(
            f"{handle.card.model_id!r}: unsupported output.type {kind!r}; the torch "
            "adapter reads 'softmax', 'logits' or 'sigmoid'"
        )

    def trainable_groups(self, handle: TorchHandle) -> List[str]:
        return [name for name, _ in handle.module.named_children()]

    # -- attribution (spec §4.4, §4.6) --------------------------------------

    def attribute(
        self,
        handle: TorchHandle,
        batch: np.ndarray,
        baseline: Any = None,
        steps: int = DEFAULT_STEPS,
    ) -> np.ndarray:
        """Integrated Gradients of the positive-class probability.

        Read ``mival.adapters._attribution`` for why ``baseline`` is required.
        The returned array is in the canonical ``(B, n_leads, n_samples)``
        layout whatever the card's layout is, so a caller plotting a lead never
        has to know how the model wanted its input transposed.
        """
        import torch

        array = np.asarray(batch, dtype=np.float64)
        reference = resolve_baseline(baseline, array)
        layout = handle.card.input_contract.layout
        module = handle.module
        dtype = _module_dtype(module)

        x = _as_tensor(to_model_layout(array, layout), handle.device, dtype)
        base = _as_tensor(to_model_layout(reference, layout), handle.device, dtype)

        was_training = module.training
        module.eval()
        try:
            gradients = torch.zeros_like(x)
            for alpha in integration_points(steps):
                point = (base + float(alpha) * (x - base)).detach().requires_grad_(True)
                probability = self._probability_tensor(handle, point)
                (gradient,) = torch.autograd.grad(probability.sum(), point)
                gradients = gradients + gradient
            attribution = (x - base) * (gradients / float(steps))
        finally:
            if was_training:
                module.train()

        result = attribution.detach().cpu().numpy().astype(np.float64)
        if layout == "time_lead":
            result = np.transpose(result, (0, 2, 1))
        return np.ascontiguousarray(result)

    def _probability_tensor(self, handle: TorchHandle, tensor: Any) -> Any:
        """The positive-class probability as a differentiable tensor.

        The fitted-head branch re-expresses the NumPy ``LinearHead`` in torch
        rather than calling it, because attribution needs the gradient to reach
        the input through the head. It is the same arithmetic — standardise,
        one linear layer, sigmoid — so a probed handle and an ``inference_only``
        handle are attributed through paths that agree with what ``forward``
        returns.
        """
        import torch

        if handle.head is None:
            logits, _features = _module_outputs(handle.module, tensor)
            if logits is None:
                raise NotImplementedError(
                    f"{handle.card.model_id!r} returns a representation only; there is no "
                    "published head to attribute through"
                )
            if handle.card.output.get("positive_index") is None:
                raise NotImplementedError(
                    f"{handle.card.model_id!r} declares no output.positive_index, so there "
                    "is no probability to attribute. Fit a head first."
                )
            return self._positive_probability(handle, logits)

        head = handle.head
        features = _module_features(handle.module, tensor)
        as_tensor = lambda values: torch.as_tensor(  # noqa: E731 - three identical conversions
            np.asarray(values, dtype=np.float64), dtype=features.dtype, device=features.device
        )
        standardised = (features - as_tensor(head.mean)) / as_tensor(head.scale)
        return torch.sigmoid(standardised @ as_tensor(head.weights) + float(head.bias))

    # -- fitting -----------------------------------------------------------

    def with_head(
        self, handle: TorchHandle, head: LinearHead, record: Mapping[str, Any]
    ) -> TorchHandle:
        # The module is shared rather than copied: linear_probe never wrote to
        # it, and copying a backbone per inner CV fold would cost hundreds of
        # megabytes for nothing. The fine-tuning path, which does write, copies.
        return TorchHandle(
            module=handle.module,
            device=handle.device,
            card=handle.card,
            head=head,
            fit_record=dict(record),
        )

    def _finetune(
        self, handle: TorchHandle, data: FitData, mode: str, hp: FitHParams
    ) -> TorchHandle:
        """Backpropagate into the backbone: ``partial_unfreeze`` or ``full_finetune``.

        The fitted module is a deep copy. ``models`` fits a fresh handle per
        inner CV fold and compares their held-out predictions, so mutating the
        loaded module would carry one fold's weights into the next fold's
        estimate — the exact leak the internal CV exists to avoid.
        """
        import torch

        if hp.seed is None:
            raise TrainingError(
                f"training mode {mode!r} draws randomness (parameter initialisation, "
                "batch order, and any dropout in the backbone) and is not reproducible "
                "without a seed. Run the models stage with --seed."
            )

        module = copy.deepcopy(handle.module).to(handle.device)
        frozen = self._set_trainable(module, mode, hp)

        # torch's initialisers and dropout draw from the process-global
        # generator, so seeding it is the only way to make the fit
        # reproducible. The previous state is restored afterwards: a fit that
        # left the global stream reseeded would change the behaviour of
        # everything downstream in the same process.
        rng_state = torch.get_rng_state()
        try:
            torch.manual_seed(hp.seed)
            return self._run_finetune(handle, module, frozen, data, mode, hp)
        finally:
            torch.set_rng_state(rng_state)

    def _set_trainable(self, module: Any, mode: str, hp: FitHParams) -> Tuple[str, ...]:
        """Mark which parameters get gradients. Returns the frozen group names."""
        children = dict(module.named_children())
        if mode == "full_finetune":
            for parameter in module.parameters():
                parameter.requires_grad_(True)
            return ()
        unknown = [name for name in hp.unfreeze_groups if name not in children]
        if unknown:
            raise TrainingError(
                f"unfreeze_groups {unknown} are not trainable groups of this module; "
                f"it reports {sorted(children)}"
            )
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        for name in hp.unfreeze_groups:
            for parameter in children[name].parameters():
                parameter.requires_grad_(True)
        return tuple(name for name in children if name not in set(hp.unfreeze_groups))

    def _run_finetune(
        self,
        handle: TorchHandle,
        module: Any,
        frozen: Tuple[str, ...],
        data: FitData,
        mode: str,
        hp: FitHParams,
    ) -> TorchHandle:
        import torch

        device = handle.device
        layout = handle.card.input_contract.layout

        width = self._feature_width(module, data, layout, device)
        head = torch.nn.Linear(width, 1).to(device)
        # Starting the bias at the training log-odds means epoch 0 already
        # predicts the base rate, so the first gradients carry signal about the
        # features rather than about the prevalence.
        prevalence = float(data.train.y.mean())
        with torch.no_grad():
            head.bias.fill_(
                float(np.log(prevalence / (1.0 - prevalence)))
                if 0.0 < prevalence < 1.0
                else 0.0
            )

        parameters = [p for p in module.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            list(head.parameters()) + parameters, lr=hp.lr, weight_decay=hp.weight_decay
        )
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(hp.pos_weight, dtype=torch.float32, device=device)
        )

        rng = np.random.default_rng(hp.seed)
        stopper = EarlyStopping(hp.patience, greater_is_better=True)
        best: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
        trace: List[Dict[str, float]] = []
        epochs_run = 0

        for epoch in range(1, hp.epochs + 1):
            epochs_run = epoch
            module.train()
            # Frozen groups stay in eval mode so their batch-norm running
            # statistics are not updated by data they are not learning from.
            children = dict(module.named_children())
            for name in frozen:
                children[name].eval()
            head.train()

            total = 0.0
            for indices in epoch_batches(len(data.train), data.batch_size, rng):
                tensor = self._batch_tensor(data, data.train, indices, layout, device)
                targets = torch.from_numpy(
                    data.train.y[indices].astype(np.float32)
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = head(_module_features(module, tensor)).squeeze(-1)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu()) * len(indices)
            train_loss = total / len(data.train)

            if data.validation is None:
                best = (_cpu_state(module), _cpu_state(head))
                trace.append({"epoch": epoch, "train_loss": train_loss})
                continue

            probs = self._probabilities(module, head, data, data.validation, layout, device)
            score = validation_score(
                probs, data.validation.y, hp.early_stopping_metric, f"{mode} fit"
            )
            trace.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_score": score.primary,
                    "validation_loss": -score.tiebreak,
                }
            )
            if stopper.update(epoch, score):
                best = (_cpu_state(module), _cpu_state(head))
            if stopper.should_stop:
                break

        if best is not None:
            module.load_state_dict(best[0])
            head.load_state_dict(best[1])
        module = module.to(device).eval()

        # The fitted head is linear, so it is lifted out of torch and inference
        # runs the same features-then-head path as linear_probe. Nothing about
        # the fit is lost: the backbone's learned weights live in `module`.
        with torch.no_grad():
            lifted = LinearHead.of(
                head.weight.detach().cpu().numpy(), float(head.bias.detach().cpu())
            )
        record = {
            "mode": mode,
            "head": "logistic",
            "optimizer": "adamw",
            "hparams": hp.to_record(),
            "n_train": len(data.train),
            "n_validation": 0 if data.validation is None else len(data.validation),
            "n_features": int(lifted.weights.size),
            "frozen": list(frozen),
            "trainable_parameters": int(
                sum(p.numel() for p in module.parameters() if p.requires_grad)
            ),
            "epochs_run": epochs_run,
            "best_epoch": stopper.best_epoch,
            "best_validation_score": stopper.best_primary,
            "validation_metric": (
                hp.early_stopping_metric if data.validation is not None else None
            ),
            "trace": trace,
        }
        return TorchHandle(
            module=module,
            device=device,
            card=handle.card,
            head=lifted,
            fit_record=record,
        )

    # -- fine-tuning helpers ------------------------------------------------

    def _feature_width(self, module: Any, data: FitData, layout: str, device: str) -> int:
        """Probe the representation size with one batch.

        Read from the module rather than from the card: ``feature_width`` is
        not a required ModelCard field, and a card that disagreed with its own
        checkpoint would silently build the wrong head.
        """
        import torch

        indices = np.arange(min(data.batch_size, len(data.train)))
        tensor = self._batch_tensor(data, data.train, indices, layout, device)
        was_training = module.training
        module.eval()
        with torch.no_grad():
            features = _module_features(module, tensor)
        module.train(was_training)
        if features.dim() != 2:
            raise TrainingError(
                f"module produced features of shape {tuple(features.shape)}; a linear "
                "head needs (B, D)"
            )
        return int(features.shape[1])

    def _batch_tensor(
        self, data: FitData, split: FitSplit, indices: np.ndarray, layout: str, device: str
    ) -> Any:
        import torch

        paths = [split.tensor_paths[index] for index in indices]
        arranged = to_model_layout(data.load_batch(paths), layout)
        return torch.from_numpy(np.ascontiguousarray(arranged)).to(device)

    def _probabilities(
        self,
        module: Any,
        head: Any,
        data: FitData,
        split: FitSplit,
        layout: str,
        device: str,
    ) -> np.ndarray:
        import torch

        module.eval()
        head.eval()
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for indices in epoch_batches(len(split), data.batch_size):
                tensor = self._batch_tensor(data, split, indices, layout, device)
                logits = head(_module_features(module, tensor)).squeeze(-1)
                chunks.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)


def _cpu_state(module: Any) -> Dict[str, Any]:
    """A detached CPU copy of a module's parameters, for restoring the best epoch."""
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
