"""Preprocessing primitives.

Ops are deterministic and side-effect free. The canonical application order is
fixed by the compiler, not by callers: scale_unit -> filters -> resample ->
lead selection -> crop/pad -> normalize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sps

from mival.signal import Signal


# resample_poly builds a filter proportional to max(up, down); an exact but
# enormous ratio is a data-quality signal, not something to silently compute.
_MAX_POLYPHASE_FACTOR = 10000


def _exact_rate(hz: float) -> Optional[Fraction]:
    """Return hz as an exact Fraction, or None if it cannot be represented."""
    frac = Fraction(hz).limit_denominator(10 ** 6)
    return frac if float(frac) == float(hz) else None


class Op:
    """Base class. Subclasses set `name` and implement `apply`."""

    name: str = "op"

    @property
    def params(self) -> Dict[str, Any]:
        raise NotImplementedError

    def apply(self, sig: Signal) -> Signal:
        raise NotImplementedError


@dataclass(frozen=True)
class Resample(Op):
    target_hz: float
    name: str = field(default="resample", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"target_hz": self.target_hz}

    def apply(self, sig: Signal) -> Signal:
        if sig.sampling_rate_hz == self.target_hz:
            return sig
        target = _exact_rate(self.target_hz)
        source = _exact_rate(sig.sampling_rate_hz)
        if target is None or source is None:
            raise ValueError(
                f"cannot resample {sig.sampling_rate_hz} Hz to {self.target_hz} Hz: "
                "one of the rates has no exact rational representation"
            )
        ratio = target / source
        up, down = ratio.numerator, ratio.denominator
        if max(up, down) > _MAX_POLYPHASE_FACTOR:
            raise ValueError(
                f"cannot resample {sig.sampling_rate_hz} Hz to {self.target_hz} Hz: "
                f"the exact ratio {up}/{down} exceeds the polyphase factor limit "
                f"{_MAX_POLYPHASE_FACTOR}"
            )
        resampled = sps.resample_poly(sig.data, up, down, axis=1)
        return Signal(
            data=np.ascontiguousarray(resampled, dtype=np.float32),
            leads=sig.leads,
            sampling_rate_hz=self.target_hz,
            unit=sig.unit,
        )


@dataclass(frozen=True)
class Crop(Op):
    n_samples: int
    anchor: str = "start"
    name: str = field(default="crop", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"n_samples": self.n_samples, "anchor": self.anchor}

    def apply(self, sig: Signal) -> Signal:
        if sig.n_samples < self.n_samples:
            raise ValueError(
                f"cannot crop to {self.n_samples}: source is shorter "
                f"({sig.n_samples} samples)"
            )
        if self.anchor == "center":
            start = (sig.n_samples - self.n_samples) // 2
        elif self.anchor == "start":
            start = 0
        else:
            raise ValueError(f"unknown crop anchor: {self.anchor}")
        return Signal(
            data=np.ascontiguousarray(
                sig.data[:, start : start + self.n_samples], dtype=np.float32
            ),
            leads=sig.leads,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )


@dataclass(frozen=True)
class Pad(Op):
    n_samples: int
    mode: str = "zero"
    anchor: str = "start"
    name: str = field(default="pad", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"n_samples": self.n_samples, "mode": self.mode, "anchor": self.anchor}

    def apply(self, sig: Signal) -> Signal:
        if sig.n_samples > self.n_samples:
            raise ValueError(
                f"cannot pad to {self.n_samples}: source is longer "
                f"({sig.n_samples} samples)"
            )
        if self.mode != "zero":
            raise ValueError(f"unknown pad mode: {self.mode}")
        deficit = self.n_samples - sig.n_samples
        before, after = (0, deficit) if self.anchor == "start" else (deficit, 0)
        padded = np.pad(sig.data, ((0, 0), (before, after)), mode="constant")
        return Signal(
            data=np.ascontiguousarray(padded, dtype=np.float32),
            leads=sig.leads,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )


@dataclass(frozen=True)
class OpChain:
    ops: Tuple[Op, ...]

    def apply(self, sig: Signal) -> Signal:
        for op in self.ops:
            sig = op.apply(sig)
        return sig

    def describe(self) -> List[Dict[str, Any]]:
        return [{"name": op.name, "params": op.params} for op in self.ops]
