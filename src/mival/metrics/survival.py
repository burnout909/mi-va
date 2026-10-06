"""Survival discrimination: Harrell's C and AUROC at a fixed horizon.

A higher score means higher risk. Inputs are follow-up time, a 0/1 event flag
and the score, one row per person.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

METRICS = ("c_index",)
METRIC_STEMS = ("auroc_horizon",)


def harrell_c(time: Sequence, event: Sequence, risk: Sequence) -> float:
    """Concordance over pairs where the earlier time is an event; risk ties count 1/2.

    Pairs tied on time are not comparable. Exact, and O(levels x distinct risks)
    so it stays fast on a large cohort with day-resolution times.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64) > 0
    risk = np.asarray(risk, dtype=np.float64)
    if time.size == 0 or not event.any():
        return float("nan")
    values, rank = np.unique(risk, return_inverse=True)
    later = np.zeros(values.size, dtype=np.float64)  # risk-rank histogram of records with a later time
    n_later = 0
    concordant = tied = comparable = 0.0
    order = np.argsort(-time, kind="stable")
    levels, starts = np.unique(-time[order], return_index=True)
    bounds = list(starts) + [order.size]
    for k in range(len(levels)):
        group = order[bounds[k]:bounds[k + 1]]
        events = group[event[group]]
        if events.size and n_later:
            below = np.cumsum(later) - later
            ranks = rank[events]
            concordant += below[ranks].sum()
            tied += later[ranks].sum()
            comparable += n_later * events.size
        np.add.at(later, rank[group], 1.0)
        n_later += group.size
    return float((concordant + 0.5 * tied) / comparable) if comparable else float("nan")


def auroc_at_horizon(time: Sequence, event: Sequence, risk: Sequence, horizon: float) -> float:
    """AUROC of the score for an event by ``horizon``; records censored earlier are left out."""
    from .discrimination import auroc

    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64) > 0
    risk = np.asarray(risk, dtype=np.float64)
    case = event & (time <= horizon)
    control = ~case & (time >= horizon)
    keep = case | control
    if case[keep].all() or not case[keep].any():
        return float("nan")
    score = risk[keep]
    span = float(score.max() - score.min())
    scaled = np.full(score.shape, 0.5) if span == 0.0 else (score - score.min()) / span
    return float(auroc(case[keep].astype(int), scaled))


def survival_metrics(time: Sequence, event: Sequence, risk: Sequence, horizons: Sequence[float]) -> Dict[str, float]:
    out = {"c_index": harrell_c(time, event, risk)}
    for horizon in horizons:
        out[f"auroc_horizon@{int(horizon) if float(horizon).is_integer() else horizon}"] = auroc_at_horizon(
            time, event, risk, horizon)
    return out
