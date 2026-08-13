#!/usr/bin/env python3
"""Load all PROPHECG-STEMI members and verify mean-ensemble inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# The archived Keras 2.7 runtime is intentionally CPU-only. It avoids coupling
# this legacy model to a CUDA toolkit that predates the NVIDIA L4.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/data/mi-val/models/prophecg-stemi"),
    )
    args = parser.parse_args()

    model_paths = sorted(args.model_dir.glob("*_bestmodel.h5"))
    if len(model_paths) != 5:
        raise SystemExit(f"expected 5 ensemble members, found {len(model_paths)}")

    sample = np.zeros((1, 5000, 8), dtype=np.float32)
    predictions: list[np.ndarray] = []
    members = []
    for path in model_paths:
        model = tf.keras.models.load_model(path, compile=False)
        input_shape = tuple(model.input_shape)
        output_shape = tuple(model.output_shape)
        if input_shape != (None, 5000, 8) or output_shape != (None, 2):
            raise RuntimeError(
                f"unexpected model signature for {path.name}: "
                f"{input_shape} -> {output_shape}"
            )
        prediction = np.asarray(model(sample, training=False))
        predictions.append(prediction)
        members.append({"file": path.name, "sha256": sha256(path)})
        tf.keras.backend.clear_session()

    ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
    result = {
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__ if hasattr(tf.keras, "__version__") else "2.7.0",
        "device": "cpu",
        "input_shape": list(sample.shape),
        "member_output_shape": list(predictions[0].shape),
        "ensemble": "arithmetic mean of five softmax outputs",
        "ensemble_output_shape": list(ensemble.shape),
        "ensemble_output_finite": bool(np.isfinite(ensemble).all()),
        "ensemble_probability_sum": float(ensemble[0].sum()),
        "members": members,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
