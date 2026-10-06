"""Retrieve stage (spec §4.1): every ECG in the CDM with its nearest label."""

from __future__ import annotations

import decimal
import json
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
SNAPSHOT_SQL_FILE = Path(__file__).with_name("sql") / "label_snapshot.sql"
PERSON_SQL_FILE = Path(__file__).with_name("sql") / "retrieve_ecg_person.sql"

#: ``label_source.kind``. ``measurement`` is the original nearest-measurement
#: label; the others are derived from the person, death and visit tables or
#: from a per-ECG file, and leave the measurement cutoff columns empty.
LABEL_KINDS = ("measurement", "age_at_ecg", "death_within", "machine_measurement")
#: Extra cohort_index columns written by each derived kind.
DERIVED_COLUMNS = {
    "age_at_ecg": (),
    "death_within": ("label_event", "label_time_days"),
    "machine_measurement": ("label_pr", "label_qrs", "label_qt"),
}
#: Plausible machine-measured intervals in ms; outside is a device artefact or
#: a missing-value code (29999, 65535, ...).
INTERVAL_BOUNDS = {"label_pr": (60.0, 400.0), "label_qrs": (40.0, 250.0), "label_qt": (200.0, 700.0)}
#: Out-of-hospital deaths in MIMIC-IV are recorded up to one year after the
#: last hospital contact; follow-up past that is not observed.
DEATH_FOLLOW_UP_DAYS = 365

COHORT_INDEX_COLUMNS = (
    "image_occurrence_id", "person_id", "local_path", "index_datetime",
    "label_datetime", "label_delta_days", "label_value",
    "label_primary", "label_sens1", "label_sens2", "label_sens3",
)
REASON_CODES = frozenset({"label_missing", "label_implausible", "path_unresolvable", "file_missing", "not_selected"})

Query = Callable[[str, Mapping[str, Any]], Any]
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class RetrieveSpec:
    dsn_env: str
    schema: str
    label_concept_id: int
    modality_concept_id: int
    window_days: int
    window_days_sens2: int
    implausible_below: float
    local_path_root: str
    require_local_file: bool
    primary_cutoff: float = 40.0   # label_primary = value <= primary_cutoff
    sens1_cutoff: float = 50.0     # label_sens1 = value < sens1_cutoff
    label_kind: str = "measurement"
    horizon_days: int = 365
    measurements_path: Optional[str] = None
    one_per_person: bool = False

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "RetrieveSpec":
        for key in ("dsn_env", "local_path_root"):
            if not body.get(key):
                raise ValueError(f"retrieve.{key} is required")
        if body.get("modality_concept_id") is None:
            raise ValueError("retrieve.modality_concept_id is required")
        schema = str(body.get("schema", "cdm"))
        if not _IDENTIFIER.match(schema):
            raise ValueError(f"retrieve.schema {schema!r} is not a plain identifier")
        source = body.get("label_source") or {}
        kind = str(source.get("kind", "measurement"))
        if kind not in LABEL_KINDS:
            raise ValueError(f"retrieve.label_source.kind {kind!r} is not one of {list(LABEL_KINDS)}")
        if kind == "machine_measurement" and not source.get("path"):
            raise ValueError("retrieve.label_source.path is required for kind 'machine_measurement'")
        return cls(
            label_kind=kind,
            horizon_days=int(source.get("horizon_days", 365)),
            measurements_path=None if source.get("path") is None else str(source["path"]),
            one_per_person=bool(body.get("one_per_person", False)),
            dsn_env=str(body["dsn_env"]),
            schema=schema,
            label_concept_id=int(body.get("label_concept_id", 3027172)),
            modality_concept_id=int(body["modality_concept_id"]),
            window_days=int(body.get("window_days", 7)),
            window_days_sens2=int(body.get("window_days_sens2", 30)),
            implausible_below=float(body.get("implausible_below", 5)),
            local_path_root=str(body["local_path_root"]),
            require_local_file=bool(body.get("require_local_file", True)),
            primary_cutoff=float(body.get("primary_cutoff", 40)),
            sens1_cutoff=float(body.get("sens1_cutoff", 50)),
        )

    def params(self, window_days: int) -> dict:
        return {
            "label_concept_id": self.label_concept_id,
            "window_days": window_days,
            "modality_concept_id": self.modality_concept_id,
        }


class RetrieveStage(Stage):
    name = "retrieve"
    reason_codes = REASON_CODES

    def __init__(self, query: Optional[Query] = None) -> None:
        self._query = query  # injected in tests; None means PostgreSQL via dsn_env

    def config_inputs(self, spec: Mapping[str, Any]) -> dict:
        """The per-ECG label file of ``machine_measurement``, so its content is hashed."""
        source = (spec or {}).get("label_source") or {}
        if source.get("kind") == "machine_measurement" and source.get("path"):
            return {"machine_measurements": Path(str(source["path"]))}
        return {}

    def run(self, ctx: StageContext) -> StageResult:
        spec = RetrieveSpec.from_mapping(ctx.spec or {})
        query = self._query or (lambda sql, params: query_postgres(sql, params, spec.dsn_env))
        if spec.label_kind != "measurement":
            return self._run_derived(ctx, spec, query)
        sql = SQL_FILE.read_text(encoding="utf-8").format(schema=spec.schema)

        snapshot_sql = SNAPSHOT_SQL_FILE.read_text(encoding="utf-8").format(schema=spec.schema)
        snapshot_params = {"label_concept_id": spec.label_concept_id}

        narrow = query(sql, spec.params(spec.window_days))
        wide = query(sql, spec.params(spec.window_days_sens2))
        snapshot = read_snapshot(query(snapshot_sql, snapshot_params))
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
        sql_paths = [write_sql(ctx, sql, spec.params(days), f"window_{days}")
                     for days in (spec.window_days, spec.window_days_sens2)]
        sql_paths.append(write_sql(ctx, snapshot_sql, snapshot_params, "label_snapshot"))
        summary_path = write_summary(ctx, frame, kept, snapshot)
        return StageResult(outputs=[index_path, summary_path, *sql_paths],
                           counts={"in": len(frame), "out": len(kept)})


    def _run_derived(self, ctx: StageContext, spec: RetrieveSpec, query: Query) -> StageResult:
        """Labels computed per ECG rather than read from a nearby measurement."""
        import numpy as np

        sql = PERSON_SQL_FILE.read_text(encoding="utf-8").format(schema=spec.schema)
        params = {"modality_concept_id": spec.modality_concept_id}
        frame = query(sql, params)
        frame["local_path"] = frame["local_path"].map(lambda p: resolve_local_path(p, spec.local_path_root))
        frame = frame.assign(label_datetime=None, label_delta_days=np.nan, label_primary=np.nan,
                             label_sens1=np.nan, label_sens2=np.nan, label_sens3=np.nan)
        frame = derive_labels(frame, spec)
        kept = exclude_derived(frame, spec, ctx.ledger)
        if spec.one_per_person:
            kept = one_per_person(kept, ctx.seed, ctx.ledger)
        columns = COHORT_INDEX_COLUMNS + DERIVED_COLUMNS[spec.label_kind]
        index_path = write_table(kept[list(columns)].to_dict("records"),
                                 ctx.layout.artifact(COHORT_INDEX), columns)
        sql_path = write_sql(ctx, sql, params, "ecg_person")
        payload = {
            "label_kind": spec.label_kind,
            "n_ecg": int(len(frame)),
            "n_out": int(len(kept)),
            "n_persons_out": int(kept["person_id"].nunique()),
            "excluded": ctx.ledger.counts(),
        }
        for column in ("label_value",) + DERIVED_COLUMNS[spec.label_kind]:
            values = kept[column].dropna()
            if len(values):
                payload[f"{column}_quantiles"] = {
                    f"p{int(q * 100)}": float(v) for q, v in values.quantile([0.05, 0.5, 0.95]).items()}
        if spec.label_kind == "death_within":
            payload["events"] = int(kept["label_event"].sum())
        summary_path = ctx.layout.artifact(SUMMARY)
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return StageResult(outputs=[index_path, summary_path, sql_path],
                           counts={"in": len(frame), "out": len(kept)})


def derive_labels(frame, spec: RetrieveSpec):
    """Fill ``label_value`` or the kind's extra columns; NaN where none can be derived."""
    import numpy as np
    import pandas

    index = pandas.to_datetime(frame["index_datetime"])
    if spec.label_kind == "age_at_ecg":
        frame["label_value"] = (index.dt.year - frame["year_of_birth"]).astype(float)
        return frame
    frame["label_value"] = np.nan
    if spec.label_kind == "death_within":
        death = pandas.to_datetime(frame["death_date"])
        last = pandas.to_datetime(frame["last_visit_end"])
        end = np.minimum(index + pandas.Timedelta(days=spec.horizon_days),
                         last + pandas.Timedelta(days=DEATH_FOLLOW_UP_DAYS))
        event = death.notna() & (death <= end)
        stop = death.where(event, end)
        frame["label_event"] = event.astype(float).where(last.notna())
        frame["label_time_days"] = (stop - index).dt.days.astype(float).where(last.notna())
        return frame
    measurements = pandas.read_csv(spec.measurements_path,
                                   usecols=["study_id", "p_onset", "qrs_onset", "qrs_end", "t_end"])
    intervals = pandas.DataFrame({
        "study_id": measurements["study_id"].astype("int64"),
        "label_pr": measurements["qrs_onset"] - measurements["p_onset"],
        "label_qrs": measurements["qrs_end"] - measurements["qrs_onset"],
        "label_qt": measurements["t_end"] - measurements["qrs_onset"],
    }).astype({"label_pr": float, "label_qrs": float, "label_qt": float})
    for column, (low, high) in INTERVAL_BOUNDS.items():
        raw = intervals[column]
        intervals[column] = raw.where(raw.between(low, high), -1.0)  # -1: present but implausible
    frame["study_id"] = frame["local_path"].map(_study_id)
    merged = frame.merge(intervals.drop_duplicates("study_id"), on="study_id", how="left")
    return merged


def _study_id(path: Any) -> Optional[int]:
    """The ``s<digits>`` directory of a MIMIC-IV-ECG path."""
    match = re.search(r"/s(\d+)/", str(path))
    return int(match.group(1)) if match else None


def exclude_derived(frame, spec: RetrieveSpec, ledger):
    def drop(frame, mask, code, detail):
        for row in frame[mask].itertuples(index=False):
            ledger.record(row.image_occurrence_id, row.person_id, code, detail(row))
        return frame[~mask]

    if spec.label_kind == "age_at_ecg":
        frame = drop(frame, frame["label_value"].isna(), "label_missing", lambda r: "no year_of_birth")
        frame = drop(frame, frame["label_value"] < 18, "label_implausible", lambda r: f"age={r.label_value}")
    elif spec.label_kind == "death_within":
        frame = drop(frame, frame["label_time_days"].isna(), "label_implausible",
                     lambda r: "no visit to bound follow-up")
        frame = drop(frame, frame["label_time_days"] <= 0, "label_implausible",
                     lambda r: f"follow-up {r.label_time_days} days")
    else:
        columns = list(DERIVED_COLUMNS["machine_measurement"])
        frame = drop(frame, frame[columns].isna().any(axis=1), "label_missing",
                     lambda r: "no machine measurement for this study_id")
        frame = drop(frame, (frame[columns] < 0).any(axis=1), "label_implausible",
                     lambda r: "interval outside plausible bounds or a missing-value code")
    frame = drop(frame, frame["local_path"].isna(), "path_unresolvable",
                 lambda r: "no 'files/' segment in the CDM local_path")
    if spec.require_local_file:
        frame = drop(frame, ~frame["local_path"].map(lambda p: Path(p).is_file()), "file_missing",
                     lambda r: r.local_path)
    return frame


def one_per_person(frame, seed, ledger):
    """One seeded random ECG per person; the rest are ledgered as not selected."""
    import numpy as np

    rng = np.random.default_rng(seed)
    ordered = frame.sort_values("image_occurrence_id").reset_index(drop=True)
    keys = rng.random(len(ordered))
    chosen = ordered.assign(_key=keys).sort_values(["person_id", "_key"]).groupby("person_id").head(1).index
    mask = ordered.index.isin(chosen)
    for row in ordered[~mask].itertuples(index=False):
        ledger.record(row.image_occurrence_id, row.person_id, "not_selected", "one ECG per person")
    return ordered[mask]


def resolve_local_path(cdm_path: Any, root: str) -> Optional[str]:
    """Re-root the part after 'files/' under this site's DICOM root (ledger A-1)."""
    _, sep, tail = str(cdm_path).partition("/files/")
    return str(Path(root) / "files" / tail) if sep else None


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


def read_snapshot(frame) -> dict:
    """The label table's row count and highest measurement_id when this run read it."""
    if len(frame) != 1:
        raise ValueError(
            f"the label snapshot query returned {len(frame)} rows; it aggregates and "
            "must return exactly one"
        )
    row = frame.iloc[0]
    # Postgres returns count as bigint and max(measurement_id) as whatever the
    # column is, which psycopg may hand back as Decimal; JSON wants neither.
    return {
        "label_rows": _as_int(row["n_rows"]),
        "max_measurement_id": _as_int(row["max_measurement_id"]),
    }


def _as_int(value: Any) -> Optional[int]:
    return None if value is None or value != value else int(value)


def write_sql(ctx: StageContext, sql: str, params: Mapping[str, Any], name: str) -> Path:
    filled = sql
    for key, value in params.items():
        filled = filled.replace(f"%({key})s", str(value))
    path = ctx.layout.artifact(SQL_DIR, f"{name}.sql")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(filled, encoding="utf-8")
    return path


def write_summary(ctx: StageContext, frame, kept, snapshot: Mapping[str, Any]) -> Path:
    quantiles = kept["label_value"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    payload = {
        "snapshot": dict(snapshot),
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


def _decimals_to_float(frame):
    """Cast every column whose non-null values are ``decimal.Decimal`` to float64."""
    # psycopg maps Postgres numeric to decimal.Decimal, and numpy cannot mix
    # Decimal with float (quantile's linear interpolation, say), so every such
    # column is normalized once, right where a query result enters this
    # process. Nothing here is specific to any one column or label.
    for column in frame.columns:
        present = frame[column].dropna()
        if not present.empty and isinstance(present.iloc[0], decimal.Decimal):
            frame[column] = frame[column].astype(float)
    return frame


def read_connection_settings(dsn_env: str) -> dict:
    """The connection keywords held in an env file; the values are never logged."""
    # Kept out of os.environ: exporting them would leak the credentials into
    # every subprocess of this run, and into anything that dumps the
    # environment. Only the path of this file ever enters a hash.
    keywords = {"PGHOST": "host", "PGPORT": "port", "PGDATABASE": "dbname",
                "PGUSER": "user", "PGPASSWORD": "password"}
    settings = {}
    for line in Path(dsn_env).read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if sep and key in keywords:
            settings[keywords[key]] = value.strip().strip("'\"")
    return settings


def query_postgres(sql: str, params: Mapping[str, Any], dsn_env: str):
    """Run one query with libpq settings read from an env file; nothing is logged."""
    import pandas
    import psycopg

    settings = read_connection_settings(dsn_env)
    with psycopg.connect(**settings) as connection, connection.cursor() as cursor:
        cursor.execute(sql, params)
        columns = [column.name for column in cursor.description]
        frame = pandas.DataFrame(cursor.fetchall(), columns=columns)
        return _decimals_to_float(frame)
