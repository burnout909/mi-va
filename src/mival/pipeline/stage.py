"""Stage protocol, registry, and the shared execution wrapper (spec §3.1, §3.3).

Stages are coupled to each other only through files and schemas — never
through imports (spec §3.1). The registry therefore holds *strings*, and the
target module is imported only when that stage is actually run. This is the
same constraint Plan 1 solved for adapters: ``models`` imports torch or
TensorFlow, and the environment that has one does not have the other, so
``import mival.pipeline.stage`` must not drag either in.

The wrapper owns everything identical across stages — skip/force, directory
creation, timing, ledger flushing, manifest writing — so that a stage
implementation contains only its own logic.
"""

from __future__ import annotations

import importlib
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional

from .hashing import config_hash
from .layout import FAILED_MANIFEST_NAME, RunLayout, run_layout
from .ledger import ExclusionLedger
from .manifest import ArtifactRef, RunManifest
from .runkey import RunKey

# Spec §3.3, in pipeline order. Stage 6 (misclassification) is registered when
# it is implemented; an unregistered name fails loudly rather than silently
# doing nothing.
_STAGES: Dict[str, str] = {
    "preprocess": "mival.stages.preprocess:PreprocessStage",
    "models": "mival.stages.models:ModelsStage",
    "evaluate": "mival.stages.evaluate:EvaluateStage",
}

# Stages 1, 2 and 6 of spec §3.3, kept here so that `mival stages` shows the
# real pipeline rather than only the parts that happen to be built, and so a
# typo is distinguishable from a stage that is deliberately not built yet.
_UNBUILT: Dict[str, str] = {
    "retrieve": "blocked on Data4Life installation and MI-CDM access (Plan 6)",
    "profile": "blocked on Data4Life installation and MI-CDM access (Plan 6)",
    "misclassify": "not yet implemented (Plan 7)",
}

# Spec §3.3 pipeline order, including the parts not yet built.
STAGE_ORDER = ("retrieve", "profile", "preprocess", "models", "evaluate", "misclassify")


def stage_names() -> List[str]:
    """Only the stages that can actually run — the CLI's choices."""
    return [name for name in STAGE_ORDER if name in _STAGES]


def get_stage(name: str):
    """Instantiate a stage by name, importing its module only now."""
    if name in _UNBUILT:
        raise KeyError(f"stage {name!r} is {_UNBUILT[name]}")
    if name not in _STAGES:
        raise KeyError(f"unknown stage {name!r}; runnable stages are {', '.join(stage_names())}")
    module_path, _, attr = _STAGES[name].partition(":")
    module = importlib.import_module(module_path)
    return getattr(module, attr)()


@dataclass
class StageContext:
    """Everything a stage may read. Stages write only through ``layout``."""

    study_id: str
    site: str
    spec: Mapping[str, Any]
    layout: RunLayout
    ledger: ExclusionLedger
    inputs: Mapping[str, Path] = field(default_factory=dict)
    seed: Optional[int] = None

    def input_path(self, name: str) -> Path:
        try:
            return self.inputs[name]
        except KeyError as exc:
            raise KeyError(
                f"stage {self.layout.stage!r} requires input artifact {name!r}, which was "
                f"not supplied; supplied inputs are {sorted(self.inputs)}"
            ) from exc


@dataclass
class StageResult:
    """What a stage hands back to the wrapper."""

    outputs: List[Path] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    contamination: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)
    run_keys: List[RunKey] = field(default_factory=list)


class Stage:
    """Base class carrying the two attributes the wrapper needs.

    Subclasses set ``name`` and ``reason_codes`` and implement ``run``.
    ``reason_codes`` is the stage's closed exclusion vocabulary; the ledger
    rejects anything outside it.
    """

    name: str = ""
    reason_codes: FrozenSet[str] = frozenset()

    def spec_key(self) -> str:
        """Which key of study.yaml holds this stage's spec."""
        return self.name

    def required_inputs(self) -> tuple:
        return ()

    def run(self, ctx: StageContext) -> StageResult:  # pragma: no cover - abstract
        raise NotImplementedError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _peak_memory_bytes() -> Optional[int]:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is kilobytes on Linux and bytes on macOS. Normalizing here
    # keeps DICOM-MIVA numbers comparable with numbers measured on a laptop.
    return int(peak) if platform.system() == "Darwin" else int(peak) * 1024


def prepare(
    stage: Stage,
    study_id: str,
    site: str,
    spec: Mapping[str, Any],
    inputs: Mapping[str, Path],
    runs_root,
    seed: Optional[int] = None,
) -> StageContext:
    """Build the context: hash the config, resolve paths, arm the ledger."""
    from .hashing import sha256_file

    checksums = {name: sha256_file(path) for name, path in sorted(inputs.items())}
    digest = config_hash(spec, checksums)
    layout = run_layout(study_id, stage.name, digest, runs_root)
    return StageContext(
        study_id=study_id,
        site=site,
        spec=spec,
        layout=layout,
        ledger=ExclusionLedger(stage=stage.name, allowed_codes=stage.reason_codes),
        inputs=dict(inputs),
        seed=seed,
    )


def execute(stage: Stage, ctx: StageContext, force: bool = False) -> Dict[str, Any]:
    """Run a stage, or skip it when its output already exists.

    Returns a small summary for the CLI. The manifest on disk is the record;
    this return value is for the terminal.
    """
    layout = ctx.layout
    if layout.is_complete() and not force:
        return {"status": "skipped", "config_hash": layout.config_hash, "run_dir": str(layout.run_dir)}

    layout.create_dirs()
    started_at = _now()
    clock = time.monotonic()
    errors: List[str] = []
    result = StageResult()
    try:
        result = stage.run(ctx)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        wall_time = time.monotonic() - clock
        ctx.ledger.flush(layout.exclusions_path)

        counts = {"in": 0, "out": 0}
        counts.update(result.counts)
        # The ledger is the authority on how many records were dropped: a
        # stage that miscounts its own exclusions would break the STARD flow,
        # so its own "excluded" count is overwritten rather than trusted.
        counts["excluded"] = len(ctx.ledger)

        manifest = RunManifest(
            run_id=f"{ctx.study_id}.{stage.name}.{layout.config_hash}",
            stage=stage.name,
            study_id=ctx.study_id,
            site=ctx.site,
            config_hash=layout.config_hash,
            seed=ctx.seed,
            input_artifacts=[ArtifactRef.of(path) for _, path in sorted(ctx.inputs.items())],
            output_artifacts=[_output_ref(path, layout) for path in result.outputs],
            counts=counts,
            contamination=result.contamination or {"flag": False, "overlapping_corpora": []},
            compute={
                "wall_time_s": round(wall_time, 3),
                "peak_memory_bytes": _peak_memory_bytes(),
                "throughput_records_per_s": (
                    round(counts["out"] / wall_time, 3)
                    if wall_time > 0 and counts.get("out")
                    else None
                ),
                "gpu_hours": None,
            },
            warnings=list(result.warnings),
            errors=errors,
            started_at=started_at,
            ended_at=_now(),
        )
        # The manifest doubles as the completion marker, so a failed run must
        # not write to manifest.json — doing so would make the crash look like
        # a finished run and skip it forever. The failure is still recorded,
        # under a name the skip check does not look at.
        manifest.write(layout.run_dir / FAILED_MANIFEST_NAME if errors else layout.manifest_path)

    return {
        "status": "ok",
        "config_hash": layout.config_hash,
        "run_dir": str(layout.run_dir),
        "counts": manifest.counts,
        "exclusions": ctx.ledger.counts(),
    }


def _output_ref(path: Path, layout: RunLayout) -> ArtifactRef:
    try:
        return ArtifactRef.of(path, root=layout.run_dir)
    except ValueError as exc:
        raise ValueError(
            f"stage {layout.stage!r} returned output artifact {path} which is outside its "
            f"run directory {layout.run_dir}; stages must write only through ctx.layout"
        ) from exc
