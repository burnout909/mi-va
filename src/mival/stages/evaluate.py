"""Evaluation stage (spec §4.5). Implemented by Plan 5.

Contract pinned by Plan 2. The long format is what keeps the schema stable as
axes are added (spec §4.5), so ``METRICS_LONG_COLUMNS`` is the run_key axes
plus a fixed tail — never one column per metric.

The stage is deliberately thin. All arithmetic lives in ``mival.metrics``,
which is axis-free; everything here is loading, grouping and labelling. That
is the concrete form of the spec's "evaluation is one function that computes
four categories, and the rest is a groupby": adding an axis changes the
grouping key and nothing else.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mival.pipeline.layout import exclusions_root
from mival.pipeline.ledger import read_exclusions
from mival.pipeline.runkey import AXES, REPORT_AXES, RunKey
from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.tables import read_table, write_table
from mival.stages.models import PREDICTION_COLUMNS, PRIMARY_THRESHOLD_POLICY, THRESHOLD_POLICIES
from mival.threshold import SENS95_TARGET_SENSITIVITY

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

# Category 4 (interpretation) holds a schema slot and emits no rows in this
# plan: attribution maps and lead/time-region agreement are a qualitative
# reading against clinical criteria (spec §4.5, FUTURE-AI Explainability), not
# a number this stage can compute. The category constant and
# ``mival.metrics.INTERPRETATION_METRICS`` exist so that adding it later is an
# addition, not a schema change.

#: Spec §4.5. Resampling unit is person, not ECG — a patient may contribute
#: several recordings, and resampling by recording understates the CI.
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_UNIT = "person_id"

#: Spec §4.5: computed but marked, never used in conclusions.
SUBGROUP_SUPPRESS_MIN_EVENTS = 10

# ---------------------------------------------------------------------------
# Added by Plan 5, below the pinned contract above.
# ---------------------------------------------------------------------------

#: Paired model comparisons (spec §4.5) go in their own table. A difference
#: between two arms is not a property of one point in run_key space, so writing
#: it into ``metrics_long.parquet`` would mean either inventing a run_key that
#: no run produced, or silently attributing the difference to one of the arms.
COMPARISONS = "comparisons.parquet"
COMPARISON_COLUMNS = (
    "comparison",
    "confirmatory",
    "arm_a",
    "arm_b",
    "subgroup",
    "outcome",
    "category",
    "metric",
    "value",
    "ci_lo",
    "ci_hi",
    "p_value",
    "n",
    "n_events",
)

#: The slice label for "no subgroup restriction". Report axes always carry a
#: value so that a filter on them never has to special-case a missing row.
SUBGROUP_ALL = "all"
#: Default value of the ``outcome`` report axis: the label the run was scored
#: against. Other outcomes (30-day mortality, PCI within 24 h) are additional
#: values of the same axis and join onto the same predictions (spec §3.4).
OUTCOME_DIAGNOSTIC = "stemi"

#: Spec §2.4: the two splits of the frozen cohort table. Only dev may carry a
#: threshold fit (spec §4.4); ``mival.threshold`` refuses anything else, so
#: this stage names the dev split rather than excluding the test one.
DEV_SPLIT = "dev"
TEST_SPLIT = "test"

#: Fallback when no threshold can be fitted. 0.5 is not a clinical choice; it
#: is the neutral value used only when the run carries no development split,
#: and it is always accompanied by a warning in the manifest.
FALLBACK_THRESHOLD = 0.5

#: Optional input: the models stage's ``train_log.jsonl``. When supplied it is
#: the authority on both the operating threshold and the contamination flag,
#: because that stage fitted the one and ran the gate for the other. Without
#: it the threshold is refitted here from dev predictions, using the same
#: ``mival.threshold`` policies, and contamination falls back to the study
#: spec's restatement.
TRAIN_LOG_INPUT = "train_log"

#: Spec §4.5: the clinically plausible decision band for STEMI.
DEFAULT_DECISION_BAND = (0.01, 0.10)


class EvaluateStage(Stage):
    name = "evaluate"
    # Evaluation aggregates; it does not drop records. Records absent from a
    # prediction file were already excluded upstream and are already in the
    # ledger, so re-recording them here would double-count the STARD flow.
    reason_codes = frozenset()

    def required_inputs(self) -> tuple:
        """The two inputs Plan 2 pinned.

        Three optional inputs are also read when supplied, all of them joins
        onto predictions that already exist rather than reasons to run
        inference again (spec §3.4):

        ``train_log``
            The models stage's ``train_log.jsonl``; the authority on the
            operating threshold (spec §4.4) and the contamination flag
            (spec §3.5).
        ``outcomes``
            Extra label columns — 30-day mortality, PCI within 24 h — which
            become further values of the ``outcome`` report axis.
        ``attributes``
            Extra person- or record-level columns to slice the ``subgroup``
            report axis by.
        """
        return ("predictions", "cohort_split")

    def run(self, ctx: StageContext) -> StageResult:
        from mival.metrics import DEFAULT_DECISION_THRESHOLDS, DEFAULT_MURPHY_BINS

        spec = dict(ctx.spec or {})
        warnings: List[str] = []

        settings = _Settings.from_spec(spec, DEFAULT_DECISION_THRESHOLDS, DEFAULT_MURPHY_BINS)
        if settings.seed is None:
            # The bootstrap seed is the run's seed unless the evaluation spec
            # overrides it, so the manifest's `seed` field is the one that
            # actually reproduces the intervals.
            settings.seed = int(ctx.seed) if ctx.seed is not None else 0
        predictions = _load_predictions(ctx.input_path("predictions"), settings.label_def)
        cohort = read_table(ctx.input_path("cohort_split"))
        predictions = _attach_cohort_columns(predictions, cohort)
        for name in ("outcomes", "attributes"):
            if name in ctx.inputs:
                predictions = _attach_table(predictions, read_table(ctx.inputs[name]))

        recorded: Dict[Tuple[str, ...], float] = {}
        if TRAIN_LOG_INPUT in ctx.inputs:
            recorded, settings.recorded_contamination = _recorded_thresholds_and_contamination(
                ctx.inputs[TRAIN_LOG_INPUT], settings, warnings
            )
        thresholds = _fit_thresholds(predictions, settings, warnings, recorded)

        rows: List[Dict[str, Any]] = []
        run_keys: List[RunKey] = []
        curves: Dict[str, List[Tuple[str, Any, Any]]] = {}
        for key, group in _group_by_run_key(predictions):
            run_keys.append(key)
            threshold = thresholds[_arm_of(key)]
            for outcome, label_column in settings.outcome_columns(key, group, warnings):
                for subgroup, slice_frame in _subgroup_slices(group, settings.subgroups):
                    rows.extend(
                        _rows_for_slice(
                            key=key,
                            frame=slice_frame,
                            label_column=label_column,
                            outcome=outcome,
                            subgroup=subgroup,
                            threshold=threshold,
                            settings=settings,
                            contaminated=settings.is_contaminated(key),
                        )
                    )
                    if subgroup == SUBGROUP_ALL and outcome == settings.primary_outcome:
                        # Same restriction the metrics use: a record without a
                        # label for this outcome is not part of this curve.
                        labelled = slice_frame[slice_frame[label_column].notna()]
                        if len(labelled):
                            curves.setdefault(outcome, []).append(
                                (
                                    _curve_label(key),
                                    labelled[label_column].to_numpy(dtype=float),
                                    labelled["prob"].to_numpy(dtype=float),
                                )
                            )

        metrics_path = write_table(rows, ctx.layout.artifact(METRICS_LONG), METRICS_LONG_COLUMNS)
        outputs = [metrics_path]

        comparison_rows = _comparison_rows(predictions, settings, thresholds, warnings)
        outputs.append(
            write_table(
                comparison_rows, ctx.layout.artifact(COMPARISONS), COMPARISON_COLUMNS
            )
        )

        if settings.draw_figures:
            n_analysed = int(predictions["image_occurrence_id"].nunique())
            outputs.extend(_draw_figures(ctx, rows, curves, settings, n_analysed, warnings))

        return StageResult(
            outputs=outputs,
            counts={"in": int(len(predictions)), "out": len(rows)},
            warnings=warnings,
            run_keys=run_keys,
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_predictions(path: Path, label_def: Optional[str] = None):
    """Read the models stage output, whether a directory or a single file.

    Prediction files are named by their run_key (spec §3.4), which is the only
    place ``label_def`` is recorded — it is not a column of
    ``PREDICTION_COLUMNS``. Parsing the name back into a ``RunKey`` recovers
    every axis, so the axes never have to be re-derived from the contents.

    A single file is also accepted, because the CLI addresses inputs as files
    (an input is hashed into the config_hash, and a directory has no hash).
    Such a file must carry the axes as columns; ``label_def`` may instead come
    from the evaluation spec, and is never guessed.
    """
    import pandas

    target = Path(path)
    if target.is_dir():
        frames = []
        for part in sorted(target.glob("*.parquet")):
            frame = read_table(part)
            try:
                key = RunKey.from_string(part.stem)
            except ValueError as exc:
                raise ValueError(
                    f"prediction file {part.name} is not named by a run_key: {exc}"
                ) from exc
            _check_axes_agree(frame, key, part.name)
            for axis, value in key.to_dict().items():
                frame[axis] = value
            frames.append(frame)
        if not frames:
            raise ValueError(f"no prediction parquet files under {target}")
        predictions = pandas.concat(frames, ignore_index=True)
    else:
        predictions = read_table(target)
        if "label_def" not in predictions.columns and label_def is not None:
            predictions["label_def"] = label_def

    missing = [column for column in ("image_occurrence_id", "person_id", "prob") if column not in predictions.columns]
    if missing:
        raise ValueError(
            f"predictions are missing required columns {missing}; expected the schema "
            f"{list(PREDICTION_COLUMNS)}"
        )
    absent_axes = [axis for axis in AXES if axis not in predictions.columns]
    if absent_axes:
        raise ValueError(
            f"predictions are missing run_key axes {absent_axes}. A single prediction file "
            "must carry every axis as a column (set 'label_def' in the evaluation spec if it "
            "is not one of them); a directory of files takes the axes from the run_key in "
            "each file name instead."
        )
    return predictions


def _check_axes_agree(frame, key: RunKey, filename: str) -> None:
    """A file's contents must match the run_key in its name.

    The name is what the axes are taken from, so a mismatch would silently
    attribute one run's predictions to another coordinate. Rather than pick a
    winner, refuse.
    """
    import pandas

    for axis, expected in key.to_dict().items():
        if axis not in frame.columns or expected is None:
            continue
        present = frame[axis].dropna()
        if present.empty:
            continue
        found = {str(value) for value in pandas.unique(present)}
        if found != {str(expected)}:
            raise ValueError(
                f"prediction file {filename} is named {axis}={expected!r} but its rows carry "
                f"{sorted(found)}"
            )


def _attach_cohort_columns(predictions, cohort):
    """Join the frozen split table (spec §2.4) by person.

    ``cohort_split.parquet`` is the authority on split and fold, and is also
    where person-level attributes such as sex or age band travel. Columns
    already present in the predictions win, so a join can add report axes but
    can never quietly rewrite what the model actually scored.
    """
    if "person_id" not in cohort.columns:
        raise ValueError("cohort_split must have a person_id column (spec §2.4)")
    new_columns = [
        column
        for column in cohort.columns
        if column == "person_id" or column not in predictions.columns
    ]
    if len(new_columns) == 1:
        return predictions
    right = cohort[new_columns].drop_duplicates(subset=["person_id"])
    merged = predictions.merge(right, on="person_id", how="left")
    return merged


def _attach_table(predictions, extra):
    """Join an optional report-axis table on whichever key it carries.

    Report axes are cheap precisely because they need no new inference (spec
    §3.4): a downstream outcome or a hidden stratum is a join onto predictions
    that already exist.
    """
    for key in ("image_occurrence_id", "person_id"):
        if key in extra.columns and key in predictions.columns:
            new_columns = [
                column
                for column in extra.columns
                if column == key or column not in predictions.columns
            ]
            if len(new_columns) == 1:
                return predictions
            return predictions.merge(
                extra[new_columns].drop_duplicates(subset=[key]), on=key, how="left"
            )
    raise ValueError(
        "an optional evaluation input must join on image_occurrence_id or person_id"
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class _Settings:
    """The evaluation spec, with every default stated once."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        default_decision_thresholds: Sequence[float],
        default_murphy_bins: Optional[int],
    ):
        self.subgroups: Tuple[str, ...] = tuple(spec.get("subgroups", ()) or ())
        self.outcomes: Mapping[str, Any] = dict(
            spec.get("outcomes", {OUTCOME_DIAGNOSTIC: None}) or {OUTCOME_DIAGNOSTIC: None}
        )
        self.primary_outcome = next(iter(self.outcomes), OUTCOME_DIAGNOSTIC)
        self.label_columns: Mapping[str, str] = dict(spec.get("label_columns", {}) or {})
        #: Only used when predictions arrive as one file without the axis.
        self.label_def: Optional[str] = (
            None if spec.get("label_def") is None else str(spec["label_def"])
        )
        self.decision_thresholds: Tuple[float, ...] = tuple(
            spec.get("decision_thresholds", default_decision_thresholds)
        )
        self.decision_band = tuple(spec.get("decision_band", DEFAULT_DECISION_BAND))
        # An explicit ``murphy_bins: null`` asks for the exact decomposition on
        # unique forecast values; omitting the key gets the binned one, which
        # is the only meaningful decomposition for a continuous score.
        self.murphy_bins = spec["murphy_bins"] if "murphy_bins" in spec else default_murphy_bins
        self.calibration_knots = int(spec.get("calibration_knots", 4))
        self.dev_split = str(spec.get("dev_split", DEV_SPLIT))
        self.threshold: Optional[float] = (
            None if spec.get("threshold") is None else float(spec["threshold"])
        )
        self.threshold_policy = str(spec.get("threshold_policy", PRIMARY_THRESHOLD_POLICY))
        if self.threshold_policy not in THRESHOLD_POLICIES:
            raise ValueError(
                f"threshold_policy {self.threshold_policy!r} is not one of {list(THRESHOLD_POLICIES)}"
            )
        self.sensitivity_target = float(
            spec.get("sensitivity_target", SENS95_TARGET_SENSITIVITY)
        )
        self.legacy_threshold = float(spec.get("legacy_threshold", FALLBACK_THRESHOLD))
        self.bootstrap = bool(spec.get("bootstrap", True))
        self.bootstrap_replicates = int(spec.get("bootstrap_replicates", BOOTSTRAP_REPLICATES))
        self.bootstrap_alpha = float(spec.get("bootstrap_alpha", 0.05))
        self.bootstrap_metrics: Optional[Tuple[str, ...]] = (
            tuple(spec["bootstrap_metrics"]) if spec.get("bootstrap_metrics") else None
        )
        # None means "take the study seed", which is what the manifest records.
        self.seed: Optional[int] = None if spec.get("seed") is None else int(spec["seed"])
        self.contamination: Mapping[str, bool] = dict(spec.get("contamination", {}) or {})
        #: Filled from the models stage's train_log when that input is supplied.
        self.recorded_contamination: Dict[Tuple[str, ...], bool] = {}
        self.comparisons: Sequence[Mapping[str, Any]] = tuple(spec.get("comparisons", ()) or ())
        self.comparison_metrics: Tuple[str, ...] = tuple(
            spec.get("comparison_metrics", ("auroc", "auprc"))
        )
        self.draw_figures = bool(spec.get("figures", True))
        self.min_events = int(spec.get("min_events", SUBGROUP_SUPPRESS_MIN_EVENTS))

    @classmethod
    def from_spec(cls, spec, default_decision_thresholds, default_murphy_bins) -> "_Settings":
        return cls(spec, default_decision_thresholds, default_murphy_bins)

    def is_contaminated(self, key: RunKey) -> bool:
        """Spec §3.5: contamination is flagged, not blocked.

        The models stage is the authority — it ran the gate against the
        ModelCard's ``pretraining_corpora`` — so its train_log wins when it is
        available. The study spec's per-model restatement is the fallback for
        evaluating a prediction table on its own.
        """
        arm = _arm_of(key)
        if arm in self.recorded_contamination:
            return bool(self.recorded_contamination[arm])
        return bool(self.contamination.get(key.model_id, False))

    def label_column_for(self, label_def: str) -> str:
        return self.label_columns.get(label_def, f"label_{label_def}")

    def outcome_columns(self, key: RunKey, frame, warnings: List[str]):
        """``(outcome, column)`` pairs for one run_key group.

        ``None`` as a configured column means "the label this run was scored
        against", which is the ``label_def`` axis of its own run_key.
        """
        pairs = []
        for outcome, column in self.outcomes.items():
            resolved = self.label_column_for(key.label_def) if column is None else str(column)
            if resolved not in frame.columns:
                message = (
                    f"outcome {outcome!r} needs column {resolved!r}, which is absent for "
                    f"run_key {key.to_string()}; the outcome is skipped"
                )
                if message not in warnings:
                    warnings.append(message)
                continue
            pairs.append((outcome, resolved))
        return pairs


# ---------------------------------------------------------------------------
# Grouping — the only place that knows the axes exist
# ---------------------------------------------------------------------------


def _group_by_run_key(predictions):
    """Yield ``(RunKey, frame)`` for every distinct coordinate present."""
    grouped = predictions.groupby(list(AXES), dropna=False, sort=True)
    for values, frame in grouped:
        if not isinstance(values, tuple):
            values = (values,)
        fields = dict(zip(AXES, values))
        fields["fold"] = _optional_int(fields.get("fold"))
        yield RunKey.from_dict({axis: fields[axis] for axis in AXES}), frame


def _optional_int(value) -> Optional[int]:
    if value is None:
        return None
    text = str(value)
    if text in ("", "nan", "None", "NaT", "na", "<NA>"):
        return None
    return int(float(text))


def _arm_of(key: RunKey) -> Tuple[str, ...]:
    """The identity of a model configuration, ignoring split and fold.

    Thresholds are a property of the arm, not of the split it is measured on:
    refitting one per split would mean the test estimate used an operating
    point chosen on the test set (spec §4.4).
    """
    return tuple(getattr(key, axis) for axis in AXES if axis not in ("split", "fold"))


def _subgroup_slices(frame, subgroups: Sequence[str]):
    """The whole slice, then one slice per observed value of each subgroup column."""
    yield SUBGROUP_ALL, frame
    for column in subgroups:
        if column not in frame.columns:
            continue
        for value in sorted(frame[column].dropna().unique(), key=str):
            selected = frame[frame[column] == value]
            if len(selected):
                yield f"{column}={value}", selected


def _curve_label(key: RunKey) -> str:
    return f"{key.model_id}/{key.training_mode}/{key.split}"


# ---------------------------------------------------------------------------
# The one function that computes the four categories, per slice
# ---------------------------------------------------------------------------


def _rows_for_slice(
    key: RunKey,
    frame,
    label_column: str,
    outcome: str,
    subgroup: str,
    threshold: float,
    settings: _Settings,
    contaminated: bool,
) -> List[Dict[str, Any]]:
    import numpy as np

    from mival.metrics import category_of, compute_metrics
    from mival.metrics.bootstrap import bootstrap_ci, event_strata

    usable = frame[frame[label_column].notna()]
    y = usable[label_column].to_numpy(dtype=float)
    p = usable["prob"].to_numpy(dtype=float)
    n = int(y.size)
    n_events = int(np.sum(y == 1.0))
    if n == 0:
        return []

    def statistic(indices) -> Dict[str, float]:
        return compute_metrics(
            y[indices],
            p[indices],
            threshold=threshold,
            decision_thresholds=settings.decision_thresholds,
            murphy_bins=settings.murphy_bins,
            n_knots=settings.calibration_knots,
        )

    point = statistic(np.arange(n))
    intervals: Dict[str, Any] = {}
    if settings.bootstrap and settings.bootstrap_replicates > 0 and n_events > 0:
        wanted = settings.bootstrap_metrics
        selected = point if wanted is None else {k: v for k, v in point.items() if k in wanted}

        def restricted(indices) -> Dict[str, float]:
            values = statistic(indices)
            return values if wanted is None else {k: values[k] for k in selected}

        groups = (
            usable[BOOTSTRAP_UNIT].to_numpy() if BOOTSTRAP_UNIT in usable.columns else None
        )
        intervals = bootstrap_ci(
            restricted,
            n_rows=n,
            n_replicates=settings.bootstrap_replicates,
            seed=settings.seed,
            groups=groups,
            strata=event_strata(y, groups),
            alpha=settings.bootstrap_alpha,
            point=selected,
        )

    # Spec §4.5: computed, marked, and not used in conclusions.
    suppressed = n_events < settings.min_events

    rows: List[Dict[str, Any]] = []
    base = key.to_dict()
    for metric, value in point.items():
        interval = intervals.get(metric)
        row = dict(base)
        # The report-only axes (spec §3.4) are written from REPORT_AXES rather
        # than by name, so they stay in step with the run_key module.
        row.update(dict(zip(REPORT_AXES, (subgroup, outcome))))
        row.update(
            {
                "category": category_of(metric),
                "metric": metric,
                "value": _finite(value),
                "ci_lo": _finite(interval.ci_lo) if interval is not None else None,
                "ci_hi": _finite(interval.ci_hi) if interval is not None else None,
                "n": n,
                "n_events": n_events,
                "suppressed": bool(suppressed),
                "contaminated": bool(contaminated),
            }
        )
        rows.append(row)
    return rows


def _finite(value) -> Optional[float]:
    """NaN becomes NULL in the table: an undefined metric is not a number."""
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) or math.isinf(number) else number


# ---------------------------------------------------------------------------
# Operating thresholds (spec §4.4) — fitted off the test split, never on it
# ---------------------------------------------------------------------------


def _fit_thresholds(
    predictions,
    settings: _Settings,
    warnings: List[str],
    recorded: Optional[Mapping[Tuple[str, ...], float]] = None,
) -> Dict[Tuple[str, ...], float]:
    """One operating threshold per arm.

    Preference order: an explicit value in the spec, then the value the models
    stage recorded when it fitted the policy, then a refit here on dev rows.
    The models stage is preferred because it is the stage that held the
    ModelCard and the out-of-fold dev probabilities; refitting is the fallback
    for the case where only a prediction table survived.
    """
    import numpy as np

    thresholds: Dict[Tuple[str, ...], float] = {}
    for key, _ in _group_by_run_key(predictions):
        arm = _arm_of(key)
        if arm in thresholds:
            continue
        if settings.threshold is not None:
            thresholds[arm] = settings.threshold
            continue
        if recorded and arm in recorded:
            thresholds[arm] = float(recorded[arm])
            continue
        development = predictions
        for axis, value in zip([axis for axis in AXES if axis not in ("split", "fold")], arm):
            development = development[development[axis] == value]
        development = development[development["split"] == settings.dev_split]
        label_column = settings.label_column_for(key.label_def)
        if label_column in development.columns:
            development = development[development[label_column].notna()]
        else:
            development = development.iloc[0:0]
        y = development[label_column].to_numpy(dtype=float) if len(development) else np.array([])
        if y.size == 0 or len(np.unique(y)) < 2:
            thresholds[arm] = settings.legacy_threshold
            warnings.append(
                f"no usable development split for arm {arm}; falling back to the neutral "
                f"threshold {settings.legacy_threshold} — threshold-dependent metrics for "
                "this arm are not a refitted operating point"
            )
            continue
        thresholds[arm] = _refit(
            development["prob"].to_numpy(dtype=float), y, settings
        )
    return thresholds


def _refit(p, y, settings: _Settings) -> float:
    """Delegate to the policies of ``mival.threshold`` (spec §4.4).

    The policy implementation is not repeated here. Beyond avoiding a second
    copy, ``SplitScores`` carries the split its probabilities came from and the
    refit functions refuse anything but dev, so the "no threshold is ever
    chosen on test" rule is enforced by the type rather than by this call site
    remembering to filter.
    """
    from mival.threshold import REFIT_SENS95, REFIT_YOUDEN, SplitScores, refit_sens95, refit_youden

    scores = SplitScores.of(settings.dev_split, p, y)
    if settings.threshold_policy == REFIT_YOUDEN:
        return float(refit_youden(scores).value)
    if settings.threshold_policy == REFIT_SENS95:
        return float(refit_sens95(scores, settings.sensitivity_target).value)
    return settings.legacy_threshold


def _recorded_thresholds_and_contamination(
    path: Path, settings: _Settings, warnings: List[str]
):
    """Read the models stage's ``train_log.jsonl`` (spec §4.4, §3.5).

    That stage fitted every threshold policy on out-of-fold dev probabilities
    and ran the contamination gate against each ModelCard. Both belong to the
    run that produced the predictions, so when the log is available it is read
    rather than reconstructed.
    """
    import json

    thresholds: Dict[Tuple[str, ...], float] = {}
    contaminated: Dict[Tuple[str, ...], bool] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        arms = set()
        for text in entry.get("run_keys", ()):
            try:
                arms.add(_arm_of(RunKey.from_string(text)))
            except ValueError:  # pragma: no cover - a malformed log line
                warnings.append(f"train_log holds an unparseable run_key: {text!r}")
        policy = settings.threshold_policy or entry.get(
            "primary_threshold_policy", PRIMARY_THRESHOLD_POLICY
        )
        fit = (entry.get("thresholds") or {}).get(policy)
        flag = bool((entry.get("contamination") or {}).get("flag", False))
        for arm in arms:
            contaminated[arm] = flag
            if fit is not None and fit.get("value") is not None:
                thresholds[arm] = float(fit["value"])
            else:
                warnings.append(
                    f"train_log has no {policy!r} threshold for arm {arm}; it will be refitted "
                    "from the dev predictions"
                )
    return thresholds, contaminated


# ---------------------------------------------------------------------------
# Paired model comparison (spec §4.5) — bootstrap of the difference, not DeLong
# ---------------------------------------------------------------------------


def _comparison_rows(
    predictions, settings: _Settings, thresholds, warnings: List[str]
) -> List[Dict[str, Any]]:
    import numpy as np

    from mival.metrics import category_of, compute_metrics
    from mival.metrics.bootstrap import event_strata, paired_bootstrap_difference

    rows: List[Dict[str, Any]] = []
    for comparison in settings.comparisons:
        name = str(comparison.get("name", "comparison"))
        key_a, frame_a = _select_arm(predictions, comparison["a"], name)
        key_b, frame_b = _select_arm(predictions, comparison["b"], name)
        label_a = settings.label_column_for(key_a.label_def)
        label_b = settings.label_column_for(key_b.label_def)
        # Renamed before the join rather than relying on merge suffixes: when
        # both arms share a label_def the two label columns have the same name
        # and pandas would suffix them too, silently changing which column the
        # outcome is read from.
        left = frame_a[["image_occurrence_id", "person_id", "prob", label_a]].rename(
            columns={"prob": "prob_a", label_a: "y_a"}
        )
        right = frame_b[["image_occurrence_id", "prob", label_b]].rename(
            columns={"prob": "prob_b", label_b: "y_b"}
        )
        merged = left.merge(right, on="image_occurrence_id", how="inner")
        merged = merged[merged["y_a"].notna() & merged["y_b"].notna()]
        if merged.empty:
            warnings.append(f"comparison {name!r} has no records in common; skipped")
            continue
        if label_a == label_b and not merged["y_a"].equals(merged["y_b"]):
            raise ValueError(
                f"comparison {name!r}: the two arms disagree on {label_a!r} for records they "
                "share; a paired comparison requires the same outcome on both sides"
            )
        y = merged["y_a"].to_numpy(dtype=float)
        prob_a = merged["prob_a"].to_numpy(dtype=float)
        prob_b = merged["prob_b"].to_numpy(dtype=float)
        groups = merged["person_id"].to_numpy()

        def statistic(y_values, p_values) -> Dict[str, float]:
            values = compute_metrics(
                y_values,
                p_values,
                threshold=thresholds[_arm_of(key_a)],
                decision_thresholds=settings.decision_thresholds,
                murphy_bins=settings.murphy_bins,
                n_knots=settings.calibration_knots,
            )
            return {metric: values[metric] for metric in settings.comparison_metrics}

        differences = paired_bootstrap_difference(
            statistic,
            y,
            prob_a,
            prob_b,
            n_replicates=settings.bootstrap_replicates if settings.bootstrap else 0,
            seed=settings.seed,
            groups=groups,
            strata=event_strata(y, groups),
            alpha=settings.bootstrap_alpha,
        )
        for metric, result in sorted(differences.items()):
            rows.append(
                {
                    "comparison": name,
                    "confirmatory": bool(comparison.get("confirmatory", False)),
                    "arm_a": key_a.to_string(),
                    "arm_b": key_b.to_string(),
                    "subgroup": str(comparison.get("subgroup", SUBGROUP_ALL)),
                    "outcome": str(comparison.get("outcome", settings.primary_outcome)),
                    "category": category_of(metric),
                    "metric": f"delta_{metric}",
                    "value": _finite(result.value),
                    "ci_lo": _finite(result.ci_lo),
                    "ci_hi": _finite(result.ci_hi),
                    "p_value": _finite(result.p_value),
                    "n": int(y.size),
                    "n_events": int(np.sum(y == 1.0)),
                }
            )
    return rows


def _select_arm(predictions, filters: Mapping[str, Any], name: str):
    """Resolve a partial axis filter to exactly one run_key.

    Ambiguity is an error rather than a silent pool: quietly concatenating two
    folds into "the model" is how a comparison ends up measuring something
    nobody specified.
    """
    frame = predictions
    for axis, value in filters.items():
        if axis not in AXES:
            raise ValueError(f"comparison {name!r} filters on {axis!r}, which is not a run_key axis")
        frame = frame[frame[axis].astype(str) == str(value)]
    matches = list(_group_by_run_key(frame)) if len(frame) else []
    if len(matches) != 1:
        raise ValueError(
            f"comparison {name!r} arm {dict(filters)} selects {len(matches)} run_keys; "
            "it must select exactly one"
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _draw_figures(
    ctx: StageContext, rows, curves, settings: _Settings, n_analysed: int, warnings: List[str]
):
    if importlib.util.find_spec("matplotlib") is None:
        warnings.append("matplotlib is not installed; figures were skipped")
        return []

    from mival import figures as figure_module
    from mival.metrics import decision_curve

    directory = ctx.layout.artifact(FIGURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    written = []

    stages, exclusions = _stard_counts(ctx, n_analysed)
    if stages:
        written.append(figure_module.stard_flow(stages, exclusions, directory / "stard_flow.png"))

    primary = curves.get(settings.primary_outcome, [])
    if primary:
        written.append(figure_module.roc_pr_curves(primary, directory / "roc_pr.png"))
        written.append(
            figure_module.calibration_curves(
                primary, directory / "calibration.png", n_knots=settings.calibration_knots
            )
        )
        grid = _decision_grid(settings)
        written.append(
            figure_module.decision_curves(
                [(label, decision_curve(y, p, grid)) for label, y, p in primary],
                directory / "decision_curve.png",
                band=tuple(settings.decision_band),
            )
        )

    written.extend(_slice_figures(rows, directory, figure_module))
    return written


def _decision_grid(settings: _Settings):
    import numpy as np

    lo, hi = float(settings.decision_band[0]), float(settings.decision_band[1])
    return [float(value) for value in np.linspace(max(lo / 2.0, 0.001), max(hi * 3.0, 0.3), 60)]


def _slice_figures(rows, directory: Path, figure_module):
    """Degradation, forest and rank-reversal figures, all read off the long table.

    They are built from ``metrics_long`` rather than from the raw predictions
    so that what a figure shows is exactly what the table reports.
    """
    written = []
    auroc_rows = [row for row in rows if row["metric"] == "auroc"]
    f1_rows = [row for row in rows if row["metric"] == "f1"]

    degradation: Dict[str, List[Tuple[str, float, float, float]]] = {}
    for row in auroc_rows:
        if row["subgroup"] != SUBGROUP_ALL or row["value"] is None:
            continue
        degradation.setdefault(f"{row['model_id']}/{row['training_mode']}", []).append(
            (
                str(row["perturbation_id"]),
                row["value"],
                _or_nan(row["ci_lo"]),
                _or_nan(row["ci_hi"]),
            )
        )
    degradation = {name: sorted(points, key=lambda item: item[0]) for name, points in degradation.items()}
    if any(len(points) > 1 for points in degradation.values()):
        written.append(
            figure_module.degradation_curve(
                degradation, directory / "perturbation_degradation.png", metric_label="AUROC"
            )
        )

    forest = [
        (
            f"{row['model_id']}/{row['subgroup']}",
            row["value"],
            _or_nan(row["ci_lo"]),
            _or_nan(row["ci_hi"]),
        )
        for row in auroc_rows
        if row["subgroup"] != SUBGROUP_ALL and row["value"] is not None
    ]
    if forest:
        flags = [
            row["suppressed"]
            for row in auroc_rows
            if row["subgroup"] != SUBGROUP_ALL and row["value"] is not None
        ]
        written.append(
            figure_module.forest_plot(
                forest,
                directory / "subgroup_forest.png",
                reference=0.5,
                metric_label="AUROC",
                suppressed=flags,
            )
        )

    conditions: Dict[str, List[Tuple[str, float]]] = {}
    for row in f1_rows:
        if row["value"] is None:
            continue
        conditions.setdefault(f"{row['perturbation_id']}|{row['subgroup']}", []).append(
            (f"{row['model_id']}/{row['training_mode']}", row["value"])
        )
    conditions = {name: values for name, values in sorted(conditions.items()) if len(values) > 1}
    if len(conditions) > 1:
        written.append(
            figure_module.rank_reversal(
                conditions, directory / "rank_reversal_f1.png", metric_label="F1"
            )
        )
    return written


def _or_nan(value) -> float:
    return float("nan") if value is None else float(value)


def _stard_counts(
    ctx: StageContext, n_analysed: int
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, str, int]]]:
    """Build the STARD flow from the exclusion ledger (spec §3.6).

    The diagram is a query, never a hand-kept count. The ledger records what
    was dropped and this stage knows what survived, so the top box — the
    number assessed for eligibility — is their sum and cannot drift from the
    per-stage boxes below it.
    """
    from mival.pipeline.stage import STAGE_ORDER

    root = exclusions_root(ctx.layout.runs_root, ctx.study_id)
    ledger = read_exclusions(root)
    tallies = ledger.groupby(["stage", "reason_code"]).size().reset_index(name="n")
    exclusions = [
        (str(row.stage), str(row.reason_code), int(row.n)) for row in tallies.itertuples()
    ]
    per_stage = {str(stage): int(count) for stage, count in ledger.groupby("stage").size().items()}
    ordered = [stage for stage in STAGE_ORDER if stage in per_stage]
    ordered.extend(sorted(stage for stage in per_stage if stage not in STAGE_ORDER))
    entered = int(n_analysed) + int(sum(per_stage.values()))
    stages: List[Tuple[str, int]] = [("assessed for eligibility", entered)]
    remaining = entered
    for stage in ordered:
        remaining -= per_stage[stage]
        stages.append((stage, remaining))
    stages.append(("analysed", int(n_analysed)))
    return stages, exclusions
