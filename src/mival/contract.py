"""Model input contracts and the failure vocabulary of the contract gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Tuple

REASON_CODES: FrozenSet[str] = frozenset(
    {"lead_unavailable", "upsample_required", "duration_short", "unit_missing"}
)


class CompileError(Exception):
    """Raised when a recipe cannot be compiled for a record.

    Every instance carries a reason_code from the fixed vocabulary so the
    exclusion ledger stays queryable.
    """

    def __init__(self, reason_code: str, detail: str) -> None:
        if reason_code not in REASON_CODES:
            raise ValueError(f"unknown reason_code: {reason_code}")
        super().__init__(f"[{reason_code}] {detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class InputContract:
    leads: Tuple[str, ...]
    sampling_rate_hz: float
    duration_s: float
    unit: str
    scaling: str
    layout: str
    dtype: str
    filters: Tuple[Dict[str, Any], ...] = field(default=())

    @property
    def n_samples(self) -> int:
        return int(round(self.duration_s * self.sampling_rate_hz))

    @classmethod
    def from_dict(cls, body: Dict[str, Any]) -> "InputContract":
        return cls(
            leads=tuple(body["leads"]),
            sampling_rate_hz=float(body["sampling_rate_hz"]),
            duration_s=float(body["duration_s"]),
            unit=str(body["unit"]),
            scaling=str(body["scaling"]),
            layout=str(body["layout"]),
            dtype=str(body["dtype"]),
            filters=tuple(body.get("filters", ())),
        )
