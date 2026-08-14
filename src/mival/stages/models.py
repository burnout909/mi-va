"""Models stage (spec §4.4). Implemented by Plan 4.

Contract pinned by Plan 2. ``evaluate`` reads ``PREDICTION_COLUMNS`` and is
being written in parallel against it.

This is the one stage that imports a backend, and it must do so lazily —
through ``mival.adapters.get_adapter`` — because the environment holding torch
does not hold TensorFlow.
"""

from __future__ import annotations

from mival.pipeline.stage import Stage, StageContext, StageResult

#: Output artifacts, relative to ``ctx.layout.artifacts_dir``.
PREDICTIONS_DIR = "predictions"
TRAIN_LOG = "train_log.jsonl"

#: Spec §4.4. One parquet per run_key at ``predictions/<run_key>.parquet``.
PREDICTION_COLUMNS = (
    "image_occurrence_id",
    "person_id",
    "split",
    "fold",
    "label_primary",
    "label_sens1",
    "label_sens2",
    "label_sens3",
    "prob",
    "logit",
    "model_id",
    "training_mode",
    "recipe_id",
    "perturbation_id",
    "seed",
    "site",
)

#: Spec §4.4. Primary is `refit_sens95`: STEMI miss cost is asymmetric, so
#: Youden — which weighs sensitivity and specificity equally — is clinically
#: inappropriate. Thresholds are never selected on the test set.
THRESHOLD_POLICIES = ("legacy", "refit_youden", "refit_sens95")
PRIMARY_THRESHOLD_POLICY = "refit_sens95"

#: Spec §4.4 training modes.
TRAINING_MODES = ("inference_only", "linear_probe", "partial_unfreeze", "full_finetune")

#: This stage's closed exclusion vocabulary (spec §3.6). Distinct from the
#: preprocess codes: a record can reach stage 4 and still be droppable.
REASON_CODES = frozenset({"tensor_missing", "label_missing", "split_unassigned"})


class ModelsStage(Stage):
    name = "models"
    reason_codes = REASON_CODES

    def required_inputs(self) -> tuple:
        return ("preprocess_index", "cohort_split")

    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError("Plan 4 implements the models stage")
