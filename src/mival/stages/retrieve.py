"""Retrieve stage (spec §4.1): every ECG in the CDM with its nearest label."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.tables import write_table

COHORT_INDEX = "cohort_index.parquet"
SUMMARY = "retrieve_summary.json"
SQL_DIR = "sql"
SQL_FILE = Path(__file__).with_name("sql") / "retrieve_ecg_label.sql"

COHORT_INDEX_COLUMNS = (
    "image_occurrence_id", "person_id", "local_path", "index_datetime",
    "label_datetime", "label_delta_days", "label_value",
    "label_primary", "label_sens1", "label_sens2", "label_sens3",
)
REASON_CODES = frozenset({"label_missing", "label_implausible", "path_unresolvable", "file_missing"})

Query = Callable[[str, Mapping[str, Any]], Any]
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class RetrieveSpec:
    dsn_env: str
    schema: str
    label_concept_id: int
    window_days: int
    window_days_sens2: int
    implausible_below: float
    local_path_root: str
    require_local_file: bool
    primary_cutoff: float = 40.0   # label_primary = value <= primary_cutoff
    sens1_cutoff: float = 50.0     # label_sens1 = value < sens1_cutoff

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "RetrieveSpec":
        for key in ("dsn_env", "local_path_root"):
            if not body.get(key):
                raise ValueError(f"retrieve.{key} is required")
        schema = str(body.get("schema", "cdm"))
        if not _IDENTIFIER.match(schema):
            raise ValueError(f"retrieve.schema {schema!r} is not a plain identifier")
        return cls(
            dsn_env=str(body["dsn_env"]),
            schema=schema,
            label_concept_id=int(body.get("label_concept_id", 3027172)),
            window_days=int(body.get("window_days", 7)),
            window_days_sens2=int(body.get("window_days_sens2", 30)),
            implausible_below=float(body.get("implausible_below", 5)),
            local_path_root=str(body["local_path_root"]),
            require_local_file=bool(body.get("require_local_file", True)),
            primary_cutoff=float(body.get("primary_cutoff", 40)),
            sens1_cutoff=float(body.get("sens1_cutoff", 50)),
        )

    def params(self, window_days: int) -> dict:
        return {"label_concept_id": self.label_concept_id, "window_days": window_days}


class RetrieveStage(Stage):
    name = "retrieve"
    reason_codes = REASON_CODES

    def __init__(self, query: Optional[Query] = None) -> None:
        self._query = query  # injected in tests; None means PostgreSQL via dsn_env

    def run(self, ctx: StageContext) -> StageResult:
        spec = RetrieveSpec.from_mapping(ctx.spec or {})
        query = self._query or (lambda sql, params: query_postgres(sql, params, spec.dsn_env))
        sql = SQL_FILE.read_text(encoding="utf-8").format(schema=spec.schema)

        narrow = query(sql, spec.params(spec.window_days))
        wide = query(sql, spec.params(spec.window_days_sens2))
        if narrow["label_value"].notna().sum() == 0:
            raise ValueError(
                f"no label within {spec.window_days} days for any ECG; check retrieve.label_concept_id"
            )
        frame = narrow.merge(
            wide[["image_occurrence_id", "label_value"]].rename(columns={"label_value": "label_value_sens2"}),
            on="image_occurrence_id", how="left",
        )
        frame["local_path"] = frame["local_path"].map(lambda p: resolve_local_path(p, spec.local_path_root))

        kept = exclude(frame, spec, ctx.ledger)
        kept = attach_labels(kept, spec)
        index_path = write_table(kept[list(COHORT_INDEX_COLUMNS)].to_dict("records"),
                                 ctx.layout.artifact(COHORT_INDEX), COHORT_INDEX_COLUMNS)
        sql_paths = [write_sql(ctx, sql, spec.params(days), days) for days in (spec.window_days, spec.window_days_sens2)]
        summary_path = write_summary(ctx, frame, kept)
        return StageResult(outputs=[index_path, summary_path, *sql_paths],
                           counts={"in": len(frame), "out": len(kept)})


def resolve_local_path(cdm_path: Any, root: str) -> Optional[str]:
    """Re-root the part after 'files/' under this site's DICOM root (ledger A-1)."""
    _, sep, tail = str(cdm_path).partition("/files/")
    # A bare filename with nothing after 'files/' is not a real CDM shard path
    # (those always nest by patient), so it is left unresolved rather than
    # rerooted into a path that does not exist.
    if not sep or "/" not in tail:
        return None
    return str(Path(root) / "files" / tail)


def exclude(frame, spec: RetrieveSpec, ledger):
    """Drop rows in the spec's order, so each record is excluded for one reason only."""
    def drop(frame, mask, code, detail):
        for row in frame[mask].itertuples(index=False):
            ledger.record(row.image_occurrence_id, row.person_id, code, detail(row))
        return frame[~mask]

    frame = drop(frame, frame["label_value"].isna(), "label_missing",
                 lambda r: f"no label within {spec.window_days} days")
    frame = drop(frame, frame["label_value"] <= spec.implausible_below, "label_implausible",
                 lambda r: f"label_value={r.label_value}")
    frame = drop(frame, frame["local_path"].isna(), "path_unresolvable",
                 lambda r: "no 'files/' segment in the CDM local_path")
    if spec.require_local_file:
        frame = drop(frame, ~frame["local_path"].map(lambda p: Path(p).is_file()), "file_missing",
                     lambda r: r.local_path)
    return frame


def attach_labels(frame, spec: RetrieveSpec):
    import numpy as np

    value, wide = frame["label_value"], frame["label_value_sens2"]
    # Nullable labels are float with NaN: parquet round-trips that, and the
    # models stage already reads NaN as "no label".
    return frame.assign(
        label_primary=(value <= spec.primary_cutoff).astype("int64"),
        label_sens1=(value < spec.sens1_cutoff).astype("int64"),
        label_sens2=(wide <= spec.primary_cutoff).astype("float64").where(wide.notna()),
        label_sens3=np.nan,
    )


def write_sql(ctx: StageContext, sql: str, params: Mapping[str, Any], days: int) -> Path:
    filled = sql
    for key, value in params.items():
        filled = filled.replace(f"%({key})s", str(value))
    path = ctx.layout.artifact(SQL_DIR, f"window_{days}.sql")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(filled, encoding="utf-8")
    return path


def write_summary(ctx: StageContext, frame, kept) -> Path:
    quantiles = kept["label_value"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    payload = {
        "n_ecg": int(len(frame)),
        "n_labelled": int(frame["label_value"].notna().sum()),
        "n_out": int(len(kept)),
        "n_persons_out": int(kept["person_id"].nunique()),
        "excluded": ctx.ledger.counts(),
        "positives": {name: int(kept[name].sum()) for name in ("label_primary", "label_sens1", "label_sens2")},
        "label_value_quantiles": {f"p{int(q * 100)}": float(v) for q, v in quantiles.items()},
        "label_delta_days": {str(k): int(v) for k, v in kept["label_delta_days"].value_counts().sort_index().items()},
    }
    path = ctx.layout.artifact(SUMMARY)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def query_postgres(sql: str, params: Mapping[str, Any], dsn_env: str):
    """Run one query with libpq settings read from an env file; nothing is logged."""
    import pandas
    import psycopg

    for line in Path(dsn_env).read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip().startswith("PG"):
            os.environ[key.strip()] = value.strip().strip("'\"")
    with psycopg.connect() as connection, connection.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [column.name for column in cursor.description]
        return pandas.DataFrame(cursor.fetchall(), columns=columns)
