#!/usr/bin/env python3
"""Load the pinned ECGFounder checkpoint and run one GPU forward pass."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/mi-val/models/ecgfounder/12_lead_ECGFounder.pth"),
    )
    parser.add_argument("--source", type=Path, default=Path("/opt/ecgfounder"))
    parser.add_argument(
        "--source-commit",
        default="68d25f25e323a4a423b9d9e8ea2e0af3f234bf22",
    )
    parser.add_argument(
        "--hf-revision",
        default="d9b1793951b2342f5f7e84f1ac03cd37f8a08724",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.source))
    from net1d import Net1D  # noqa: PLC0415

    # The official checkpoint stores two NumPy scalar metadata objects. Only
    # those known globals are allow-listed; arbitrary pickle execution remains
    # disabled.
    safe_globals = [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        (np.dtype, "numpy.dtype"),
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(
            args.checkpoint,
            map_location="cpu",
            weights_only=True,
        )

    state_dict = checkpoint["state_dict"]
    n_classes = int(state_dict["dense.weight"].shape[0])
    model = Net1D(
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
    model.load_state_dict(state_dict, strict=True)
    step = int(checkpoint["step"])
    validation_auroc = float(checkpoint["val_auroc"])
    del checkpoint, state_dict
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    sample = torch.zeros((1, 12, 5000), dtype=torch.float32, device=device)
    with torch.inference_mode():
        logits, features = model(sample)

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "hf_revision": args.hf_revision,
        "source_commit": args.source_commit,
        "torch": torch.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "input_shape": list(sample.shape),
        "feature_shape": list(features.shape),
        "output_shape": list(logits.shape),
        "output_finite": bool(torch.isfinite(logits).all().item()),
        "pretraining_classes": n_classes,
        "checkpoint_step": step,
        "checkpoint_validation_auroc": validation_auroc,
        "strict_state_dict_load": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
