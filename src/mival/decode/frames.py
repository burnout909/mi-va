"""Run a short-window model over a longer record (HeartKit: 2.56 s frames of a 10 s ECG)."""

from __future__ import annotations

from typing import Callable

import numpy as np


def predict_in_frames(predict: Callable[[np.ndarray], np.ndarray], x: np.ndarray, frame: int) -> np.ndarray:
    """``x`` (B, T, L) -> per-sample scores (B, T, C) from non-overlapping frames.

    The last frame is aligned to the end of the record so every sample is
    covered; where it overlaps the previous frame, the later frame wins.
    All frames of all records go through ``predict`` in one call.
    """
    batch, length = x.shape[0], x.shape[1]
    if length < frame:
        raise ValueError(f"record of {length} samples is shorter than the {frame}-sample frame")
    starts = list(range(0, length - frame + 1, frame))
    if starts[-1] + frame < length:
        starts.append(length - frame)
    frames = np.concatenate([x[:, s:s + frame] for s in starts], axis=0)
    scores = np.asarray(predict(frames))
    out = np.zeros((batch, length) + scores.shape[2:], dtype=scores.dtype)
    for k, s in enumerate(starts):
        out[:, s:s + frame] = scores[k * batch:(k + 1) * batch]
    return out
