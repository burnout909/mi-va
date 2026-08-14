"""Preprocess stage (spec §4.3). Implemented by Plan 3.

This file currently pins the contract only. The surface below — class name,
stage name, reason_code vocabulary, required inputs, output artifact names —
is fixed by Plan 2 because ``models`` and ``evaluate`` are written against it
in parallel. Changing any of it changes another stage's inputs.
"""

from __future__ import annotations

from mival.contract import REASON_CODES
from mival.pipeline.stage import Stage, StageContext, StageResult

#: Output artifact names, relative to ``ctx.layout.artifacts_dir``.
PREPROCESS_INDEX = "preprocess_index.parquet"
TENSOR_DIR = "tensors"

#: Columns of ``preprocess_index.parquet``. Stage 4 joins on these.
PREPROCESS_INDEX_COLUMNS = (
    "image_occurrence_id",
    "person_id",
    "model_id",
    "recipe_id",
    "perturbation_id",
    "tensor_path",
    "n_leads",
    "n_samples",
    "sampling_rate_hz",
)


class PreprocessStage(Stage):
    name = "preprocess"
    # The compile failures of spec §4.3 are exactly this stage's exclusions.
    reason_codes = REASON_CODES

    def required_inputs(self) -> tuple:
        return ("cohort_index",)

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError("Plan 3 implements the preprocess stage")
