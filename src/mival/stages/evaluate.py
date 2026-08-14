"""Evaluation stage (spec §4.5). Implemented by Plan 5.

Contract pinned by Plan 2. The long format is what keeps the schema stable as
axes are added (spec §4.5), so ``METRICS_LONG_COLUMNS`` is the run_key axes
plus a fixed tail — never one column per metric.
"""

from __future__ import annotations

from mival.pipeline.runkey import AXES
from mival.pipeline.stage import Stage, StageContext, StageResult

#: Output artifacts, relative to ``ctx.layout.artifacts_dir``.
METRICS_LONG = "metrics_long.parquet"
FIGURE_DIR = "figures"

#: Spec §4.5: run_key columns, then the report-only axes, then the value block.
METRICS_LONG_COLUMNS = AXES + (
    "subgroup",
    "outcome",
    "category",
    "metric",
    "value",
    "ci_lo",
    "ci_hi",
    "n",
    "n_events",
    "suppressed",
    "contaminated",
)

#: Spec §4.5 categories. "Overall performance" is deliberately absent: Brier is
#: a composite, and by the Murphy decomposition its reliability term belongs to
#: calibration and its resolution term to discrimination.
CATEGORIES = ("discrimination", "calibration", "clinical_utility", "interpretation")

#: Spec §4.5. Resampling unit is person, not ECG — a patient may contribute
#: several recordings, and resampling by recording understates the CI.
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_UNIT = "person_id"

#: Spec §4.5: computed but marked, never used in conclusions.
SUBGROUP_SUPPRESS_MIN_EVENTS = 10


class EvaluateStage(Stage):
    name = "evaluate"
    # Evaluation aggregates; it does not drop records. Records absent from a
    # prediction file were already excluded upstream and are already in the
    # ledger, so re-recording them here would double-count the STARD flow.
    reason_codes = frozenset()

    def required_inputs(self) -> tuple:
        return ("predictions", "cohort_split")

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError("Plan 5 implements the evaluation stage")
