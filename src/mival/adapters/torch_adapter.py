"""PyTorch backend.

torch is imported inside methods so that importing `mival.adapters` stays
cheap and works in the keras27 environment where torch is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from mival.adapters.base import Adapter, to_model_layout
from mival.modelcard import ModelCard


@dataclass
class TorchHandle:
    module: Any
    device: str
    card: ModelCard


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
            _logits, features = handle.module(tensor)
        return features.detach().cpu().numpy()

    def forward(self, handle: TorchHandle, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "ECGFounder checkpoint has no STEMI head; fit one with linear_probe "
            "or full_finetune (Plan 4) before calling forward"
        )

    def trainable_groups(self, handle: TorchHandle) -> List[str]:
        return [name for name, _ in handle.module.named_children()]
