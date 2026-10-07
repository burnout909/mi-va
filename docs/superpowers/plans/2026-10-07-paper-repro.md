# 논문 방식 재현과 자동 추출 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 모델 13개(16건)를 논문 방식 study로 옮겨 서버에서 돌리고 논문 보고값과 같은 조건으로 비교한다. 이어서 논문과 repo에서 같은 형식의 study를 자동으로 뽑는 절차를 만들고 채점한다.

**Architecture:**
- 논문 하나가 `studies/repro/<paper_id>/` 하나다. 안에 실행용 `study.yaml`, 보고값 `claims.yaml`, 근거 `evidence.yaml`을 둔다.
- 프레임워크에는 opt-in 필드만 더한다. 지표, cut 방향, 분 단위 간격, 외부 라벨과 고정 split, 사망 추적 규칙, Antolini C, DeepHit 출력이 이에 해당한다.
- 비교와 채점은 새 패키지 `mival.repro`에 둔다. evaluate stage는 지표를 넓히는 것 말고는 바꾸지 않는다.
- 차수별로 구현하고, 각 차수의 전체 실행은 서버 queue에 넣은 뒤 다음 차수 구현으로 넘어간다.

**Tech Stack:** Python 3.9+ (`mival` env 3.10), numpy, scipy, pandas/pyarrow, PyYAML, pytest. 서버 env에는 lifelines, statsmodels, pycox가 없어 Cox와 Antolini를 numpy로 구현한다. 서버는 bash, tmux, PostgreSQL MI-CDM.

**Spec:** `docs/superpowers/specs/2026-10-07-paper-repro-design.md` ("구현 전 확인으로 바뀐 것" 절 포함)

## Global Constraints

- 기존 공통 study 6개(`lvef`, `ecg-age`, `mortality`, `delineation`, `ntprobnp`, `potassium`)의 stage별 `config_hash`와 기존 테스트 결과가 바뀌지 않는다. 새 필드는 모두 기본값이 지금 동작과 같다.
- 정답은 Claude가 만들고 사람이 검토하지 않는다. 모든 정답 필드에 `evidence.yaml` 항목(`value, quote, where, status, reviewed_by: null`)을 쓴다.
- `evidence.status` 값: `stated | inferred | not_stated | our_choice`
- `deviations.kind` 값: `our_choice | data_limit | different_reference | different_model | paper_vs_code`
- 판정 허용 범위 기본값: AUROC/C-index ±0.02, MAE·시점 오차 ±10%, 상관계수 ±0.05. study에서 `compare.tolerance`로 바꾼다.
- 원문 PDF와 텍스트는 `docs/repro/sources/<paper_id>/`에 두고 git에 넣지 않는다 (`.gitignore`).
- 서버 실행 규칙:
  - push는 서버 repo에만 한다: `GIT_SSH_COMMAND="ssh -i $K -p 2022" git push -q -f ssh://ubuntu@<server-ip>/data/mi-val/mi-va HEAD:refs/heads/vs-incoming` 후 `/data/mi-val/mi-va-vs`에서 `git reset -q --hard vs-incoming`. GitHub에는 push하지 않는다.
  - 데이터 삭제, 새 인증, 스펙 밖 L2, 같은 오류 3회면 멈춘다.
  - 인스턴스는 끄지 않는다.
  - `/data/mi-val/mi-va`(LVEF 체인)는 건드리지 않는다.
- 서버 접속: `K=/Users/minseongkim/Desktop/youlab/<key>.pem; ssh -i $K -p 2022 ubuntu@<server-ip>`. DB env: `set -a; . /data/mi-val/secrets/micdm.env; set +a`.
- 테스트: `python3 -m pytest -q` (현재 722 passed, 13 skipped).
- 커밋 메시지 끝:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Peg6jKchfYcH9sDY8zwmy7
  ```

## Review Focus

1. **cut 문자열의 경계값.** `">6.5"`에서 라벨이 정확히 6.5면 음성이고, `">=6.5"`면 양성이어야 한다. → Task 4 `test_auroc_cut_boundary`
2. **log 변환할 값에 0이나 음수가 있는 경우.** 예측이 0 이하면 그 행만 빠지고 경고가 남아야 하고, 지표 전체가 NaN이 되면 안 된다. → Task 3 `test_log10_drops_nonpositive_with_warning`
3. **claim이 metrics_long에서 여러 행과 맞는 경우.** perturbation이나 training_mode 축 때문에 생긴다. 조용히 첫 행을 고르지 말고, 후보를 보여주며 실패해야 한다. → Task 5 `test_ambiguous_match_raises`
4. **ECG 시각이 없는 ECG.** `machine_measurements.csv`에 study_id가 없는 경우다. `label_missing`으로 ledger에 남고 날짜 기준으로 대신 짝지으면 안 된다. → Task 7 `test_minutes_window_needs_ecg_time`
5. **Antolini에서 같은 시각의 사건과 censoring.** brute-force 기준 구현과 정확히 같아야 한다. → Task 12 `test_antolini_matches_bruteforce_with_ties`

---

## File Structure

| 파일 | 역할 |
|---|---|
| `src/mival/repro/__init__.py` | 패키지 |
| `src/mival/repro/claims.py` | claims.yaml과 evidence.yaml을 읽고 검증 |
| `src/mival/repro/review.py` | evidence → `docs/repro/gold-review.md`, 검토 수 세기 |
| `src/mival/repro/compare.py` | claims × metrics_long → comparison.csv/md, CLI |
| `src/mival/repro/cox.py` | numpy Cox(Breslow) 적합, HR과 CI |
| `src/mival/repro/derived_survival.py` | 다른 study 예측으로 파생 변수를 만들고 Cox → metrics_long 형식 행 (Lima 생존) |
| `src/mival/repro/score.py` | B: 자동 추출과 정답을 필드 단위로 채점 |
| `src/mival/metrics/regression.py` | 수정: pearson, spearman, `auroc_cut` |
| `src/mival/metrics/survival.py` | 수정: `antolini_c` |
| `src/mival/stages/evaluate.py` | 수정: `regression_metrics_extra`, `label_transform`, 문자열 cut, `curves` 입력 |
| `src/mival/stages/retrieve.py` | 수정: `window_minutes`·`ecg_time_path`, `one_per_person: first`, `ecg_label_file` 종류, `split_source`, `death_followup` 종류 |
| `src/mival/stages/profile.py` | 수정: `use_fixed_split` |
| `src/mival/decode/outputs.py` | 수정: `deephit` 출력, `curve_grid_days` |
| `src/mival/stages/models.py` | 수정: 생존곡선 sidecar 저장 |
| `registry/repro/*.json`, `registry/wrappers/mival_wrap_cavalab.py` | cavalab MIMIC DeepHit 카드 2개, wrapper 확장 |
| `studies/repro/<paper_id>/{study.yaml,claims.yaml,evidence.yaml,notes.md}` | 정답 12개 |
| `scripts/repro/*.py`, `scripts/repro/*.sh` | 라벨 파일, split 파일, ESRD 표, queue, 실행 |
| `docs/repro/extraction-guide.md`, `docs/repro/extraction-schema.json` | B 지침서와 출력 스키마 |
| `docs/repro/gold-review.md`, `docs/repro/sources.md`, `docs/repro/summary.xlsx` | 산출 문서 |

---

## 0차: 정답 형식과 내용

### Task 1: claims/evidence 스키마와 검토표 생성기

**Files:**
- Create: `src/mival/repro/__init__.py`, `src/mival/repro/claims.py`, `src/mival/repro/review.py`
- Modify: `.gitignore`
- Test: `tests/test_repro_claims.py`

**Interfaces:**
- Produces:
  - `load_claims(path: Path) -> Claims`
    - `Claims` (dataclass): `paper: dict`, `claims: list[Claim]`, `deviations: list[Deviation]`, `tolerance: dict[str, float]`
    - `Claim`: `id, model_id, label_def, metric, value, subgroup="all", outcome=None, ci=None, aggregate="single", cohort="", source="", comparable=True, reason=""`
    - `Deviation`: `field, kind, why, claims: tuple[str, ...]` (빈 tuple은 모든 claim)
  - `load_evidence(path: Path) -> dict[str, EvidenceItem]`
    - `EvidenceItem`: `value, quote, where, status, reviewed_by`
  - `review_counts(evidence: dict) -> tuple[int, int]` → (검토 수, 전체 수)
  - `write_gold_review(studies_root: Path, out: Path) -> int` → 쓴 행 수

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repro_claims.py
from pathlib import Path

import pytest

from mival.repro.claims import load_claims, load_evidence
from mival.repro.review import review_counts, write_gold_review

CLAIMS = """
paper: {id: chiu2024-kardionet, doi: 10.1016/j.jacep.2024.07.023, version_read: medRxiv v1}
claims:
  - {id: c1, model_id: kardionet-k-12lead, label_def: value, metric: "auroc_cut@>6.5",
     subgroup: "esrd=1", value: 0.852, ci: [0.745, 0.956], source: "medRxiv v1 Table 2"}
  - {id: c2, model_id: kardionet-k-12lead, label_def: value, metric: mae, value: 0.527,
     subgroup: "esrd=1", source: "medRxiv v1 Table 2"}
deviations:
  - {field: attributes.esrd, kind: our_choice, why: "ESRD 정의가 논문에 없음", claims: [c1, c2]}
"""

EVIDENCE = """
label_source.window_minutes:
  value: 60
  quote: "within 1 hour"
  where: "medRxiv v1 Methods"
  status: stated
  reviewed_by: null
claims.c1.value:
  value: 0.852
  quote: "AUC 0.852"
  where: "Table 2"
  status: stated
  reviewed_by: "MK 2026-10-08"
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_load_claims(tmp_path):
    claims = load_claims(write(tmp_path, "claims.yaml", CLAIMS))
    assert [c.id for c in claims.claims] == ["c1", "c2"]
    assert claims.claims[0].ci == (0.745, 0.956)
    assert claims.claims[1].subgroup == "esrd=1"
    assert claims.deviations[0].claims == ("c1", "c2")


@pytest.mark.parametrize("bad, message", [
    ("kind: our_choice", None),
    ("kind: guess", "deviations\\[0\\].kind"),
])
def test_deviation_kind_is_closed(tmp_path, bad, message):
    text = CLAIMS.replace("kind: our_choice", bad)
    if message is None:
        load_claims(write(tmp_path, "c.yaml", text))
    else:
        with pytest.raises(ValueError, match=message):
            load_claims(write(tmp_path, "c.yaml", text))


def test_duplicate_claim_id_rejected(tmp_path):
    text = CLAIMS.replace("id: c2", "id: c1")
    with pytest.raises(ValueError, match="duplicate claim id 'c1'"):
        load_claims(write(tmp_path, "c.yaml", text))


def test_deviation_names_unknown_claim(tmp_path):
    text = CLAIMS.replace("claims: [c1, c2]", "claims: [c9]")
    with pytest.raises(ValueError, match="unknown claim 'c9'"):
        load_claims(write(tmp_path, "c.yaml", text))


def test_evidence_status_closed_and_counts(tmp_path):
    evidence = load_evidence(write(tmp_path, "e.yaml", EVIDENCE))
    assert review_counts(evidence) == (1, 2)
    with pytest.raises(ValueError, match="status"):
        load_evidence(write(tmp_path, "e2.yaml", EVIDENCE.replace("status: stated", "status: maybe", 1)))


def test_gold_review_table(tmp_path):
    study = tmp_path / "studies" / "kardionet-k"
    study.mkdir(parents=True)
    write(study, "claims.yaml", CLAIMS)
    write(study, "evidence.yaml", EVIDENCE)
    out = tmp_path / "gold-review.md"
    assert write_gold_review(tmp_path / "studies", out) == 2
    text = out.read_text(encoding="utf-8")
    assert "| kardionet-k | label_source.window_minutes | 60 |" in text
    assert "검토 1 / 2" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_repro_claims.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mival.repro'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mival/repro/__init__.py
"""Paper-protocol reproduction: claims, comparison, extraction scoring (spec 2026-10-07)."""
```

```python
# src/mival/repro/claims.py
"""claims.yaml (what a paper reported) and evidence.yaml (where each value came from)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

DEVIATION_KINDS = ("our_choice", "data_limit", "different_reference", "different_model", "paper_vs_code")
EVIDENCE_STATUSES = ("stated", "inferred", "not_stated", "our_choice")
AGGREGATES = ("single", "mean_over_seeds", "median_over_seeds", "best_of_seeds")


@dataclass(frozen=True)
class Claim:
    id: str
    model_id: str
    label_def: str
    metric: str
    value: float
    subgroup: str = "all"
    outcome: Optional[str] = None
    ci: Optional[Tuple[float, float]] = None
    aggregate: str = "single"
    cohort: str = ""
    source: str = ""
    comparable: bool = True
    reason: str = ""


@dataclass(frozen=True)
class Deviation:
    field: str
    kind: str
    why: str
    claims: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Claims:
    paper: Dict[str, Any]
    claims: List[Claim]
    deviations: List[Deviation]
    tolerance: Dict[str, float] = field(default_factory=dict)

    def deviations_for(self, claim_id: str) -> List[Deviation]:
        return [d for d in self.deviations if not d.claims or claim_id in d.claims]


@dataclass(frozen=True)
class EvidenceItem:
    value: Any
    quote: str
    where: str
    status: str
    reviewed_by: Optional[str]


def _read(path: Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_claims(path: Path) -> Claims:
    body = _read(path)
    claims: List[Claim] = []
    seen = set()
    for i, raw in enumerate(body.get("claims") or []):
        for key in ("id", "model_id", "label_def", "metric", "value"):
            if raw.get(key) is None:
                raise ValueError(f"{path}: claims[{i}].{key} is required")
        if raw["id"] in seen:
            raise ValueError(f"{path}: duplicate claim id {raw['id']!r}")
        seen.add(raw["id"])
        aggregate = str(raw.get("aggregate", "single"))
        if aggregate not in AGGREGATES:
            raise ValueError(f"{path}: claims[{i}].aggregate {aggregate!r} is not one of {list(AGGREGATES)}")
        ci = raw.get("ci")
        claims.append(Claim(
            id=str(raw["id"]), model_id=str(raw["model_id"]), label_def=str(raw["label_def"]),
            metric=str(raw["metric"]), value=float(raw["value"]),
            subgroup=str(raw.get("subgroup", "all")),
            outcome=None if raw.get("outcome") is None else str(raw["outcome"]),
            ci=None if ci is None else (float(ci[0]), float(ci[1])),
            aggregate=aggregate, cohort=str(raw.get("cohort", "")), source=str(raw.get("source", "")),
            comparable=bool(raw.get("comparable", True)), reason=str(raw.get("reason", "")),
        ))
    deviations: List[Deviation] = []
    for i, raw in enumerate(body.get("deviations") or []):
        kind = str(raw.get("kind"))
        if kind not in DEVIATION_KINDS:
            raise ValueError(f"{path}: deviations[{i}].kind {kind!r} is not one of {list(DEVIATION_KINDS)}")
        names = tuple(str(c) for c in (raw.get("claims") or ()))
        for name in names:
            if name not in seen:
                raise ValueError(f"{path}: deviations[{i}] names unknown claim {name!r}")
        deviations.append(Deviation(field=str(raw.get("field", "")), kind=kind, why=str(raw.get("why", "")), claims=names))
    tolerance = {str(k): float(v) for k, v in ((body.get("compare") or {}).get("tolerance") or {}).items()}
    return Claims(paper=dict(body.get("paper") or {}), claims=claims, deviations=deviations, tolerance=tolerance)


def load_evidence(path: Path) -> Dict[str, EvidenceItem]:
    body = _read(path)
    out: Dict[str, EvidenceItem] = {}
    for key, raw in body.items():
        status = str(raw.get("status"))
        if status not in EVIDENCE_STATUSES:
            raise ValueError(f"{path}: {key}.status {status!r} is not one of {list(EVIDENCE_STATUSES)}")
        out[str(key)] = EvidenceItem(value=raw.get("value"), quote=str(raw.get("quote") or ""),
                                     where=str(raw.get("where") or ""), status=status,
                                     reviewed_by=raw.get("reviewed_by"))
    return out
```

```python
# src/mival/repro/review.py
"""The human review sheet for the gold studies, and how much of it is done."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from .claims import EvidenceItem, load_evidence

#: Fields that change a result go first in the sheet (spec ① 정답 검토).
PRIORITY = ("label_source", "window", "cut", "claims.", "split", "one_per_person", "subgroup")


def review_counts(evidence: Dict[str, EvidenceItem]) -> Tuple[int, int]:
    return sum(1 for item in evidence.values() if item.reviewed_by), len(evidence)


def _rank(key: str) -> int:
    for i, stem in enumerate(PRIORITY):
        if stem in key:
            return i
    return len(PRIORITY)


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_gold_review(studies_root: Path, out: Path) -> int:
    rows = []
    reviewed = total = 0
    for evidence_path in sorted(Path(studies_root).glob("*/evidence.yaml")):
        paper_id = evidence_path.parent.name
        evidence = load_evidence(evidence_path)
        done, n = review_counts(evidence)
        reviewed += done
        total += n
        for key, item in sorted(evidence.items(), key=lambda kv: (_rank(kv[0]), kv[0])):
            rows.append(f"| {paper_id} | {key} | {_cell(item.value)} | {_cell(item.quote)} | "
                        f"{_cell(item.where)} | {item.status} | {_cell(item.reviewed_by or '')} |")
    head = [
        "# 정답 검토표",
        "",
        f"검토 {reviewed} / {total} 필드. 원문은 `docs/repro/sources/<paper_id>/`, 받은 위치는 `docs/repro/sources.md`.",
        "확인한 행은 `evidence.yaml`의 `reviewed_by`에 이름과 날짜를 적고 이 표를 다시 만든다:",
        "`python3 -c \"from mival.repro.review import write_gold_review as w; w('studies/repro', 'docs/repro/gold-review.md')\"`",
        "",
        "| paper_id | 필드 | 값 | 원문 인용 | 위치 | status | 확인 |",
        "|---|---|---|---|---|---|---|",
    ]
    Path(out).write_text("\n".join(head + rows) + "\n", encoding="utf-8")
    return len(rows)
```

`.gitignore`에 한 줄 추가: `docs/repro/sources/`

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_repro_claims.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/mival/repro tests/test_repro_claims.py .gitignore
git commit -m "feat: claims and evidence files for paper-protocol studies, with the gold review sheet"
```

### Task 2: 정답 12개 작성 (원문 수집, claims, evidence, notes, study 초안)

조사 작업이다. 코드는 Task 1의 loader로 검증한다. study.yaml 본문은 각 차수 Task에서 완성하고, 여기서는 claims, evidence, notes만 확정한다.

**Files:**
- Create: `studies/repro/<paper_id>/{claims.yaml,evidence.yaml,notes.md}` × 12
  - paper_id: `heartwise, ecgfounder, lima-age, singstad-age, ai-ntprobnp, vonbachmann-k, kardionet-k, cavalab-code15, cavalab-mimic, lima-survival, semiseg, openecg`
- Create: `docs/repro/sources.md`, `docs/repro/gold-review.md`
- Test: `tests/test_repro_gold.py`

**Interfaces:**
- Consumes: `load_claims`, `load_evidence`, `write_gold_review` (Task 1)
- Produces:
  - claim의 `metric` 이름은 metrics_long 이름 그대로 쓴다: `auroc`, `auprc`, `mae`, `r2`, `pearson`, `spearman`, `auroc_cut@>6.5`, `auroc_cut@<3.5`, `auroc_cut@>5.5`, `c_index`, `antolini_c`, `auroc_horizon@365`, `hr`, `auc_cox@365`
  - Lima 생존의 `hr`, `auc_cox@365`는 Task 15가 같은 이름으로 만든다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repro_gold.py
from pathlib import Path

import pytest

from mival.repro.claims import load_claims, load_evidence

ROOT = Path(__file__).resolve().parents[1] / "studies" / "repro"
PAPERS = ("heartwise", "ecgfounder", "lima-age", "singstad-age", "ai-ntprobnp", "vonbachmann-k",
          "kardionet-k", "cavalab-code15", "cavalab-mimic", "lima-survival", "semiseg", "openecg")


@pytest.mark.parametrize("paper_id", PAPERS)
def test_gold_files_load_and_every_claim_has_evidence(paper_id):
    folder = ROOT / paper_id
    claims = load_claims(folder / "claims.yaml")
    evidence = load_evidence(folder / "evidence.yaml")
    assert claims.claims, paper_id
    for claim in claims.claims:
        assert f"claims.{claim.id}.value" in evidence, (paper_id, claim.id)
    for key, item in evidence.items():
        if item.status in ("stated", "inferred"):
            assert item.quote and item.where, (paper_id, key)
        if item.status in ("inferred", "our_choice"):
            assert "why" in (folder / "notes.md").read_text(encoding="utf-8") or item.quote, (paper_id, key)
    assert (folder / "notes.md").is_file()


def test_openecg_claims_are_marked_not_comparable():
    claims = load_claims(ROOT / "openecg" / "claims.yaml")
    assert all(not c.comparable for c in claims.claims if c.metric.startswith("boundary"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_repro_gold.py -q`
Expected: FAIL with `FileNotFoundError` for `studies/repro/heartwise/claims.yaml`

- [ ] **Step 3: 원문을 모으고 sources.md를 쓴다**

| paper_id | 받을 것 |
|---|---|
| ecgfounder | arXiv 2410.04133 v4 PDF, PMC12327759 본문, repo `PKUDigitalHealth/ECGFounder` (`csv/LVEF.csv`, `finetune_model.py`, `dataset.py`) |
| cavalab-code15, cavalab-mimic | BioData Mining 19:6 (PMC12821881), arXiv 2406.17002 v3·v4, repo `cavalab/ecg-survival-benchmark` (`MIMIC_IV_PreProcess_Jan2025.py`, 학습·평가 코드) |
| kardionet-k | medRxiv 10.1101/2024.05.08.24307064 v1 |
| vonbachmann-k | Sci Rep PMC11222546 |
| heartwise | Eur Heart J doi:10.1093/eurheartj/ehaf1119, PMC11908279 |
| lima-age, lima-survival | Nat Commun PMC8387361 |
| singstad-age | NLDL 2023 / medRxiv 22280640 |
| ai-ntprobnp | Clin Chem Lab Med doi:10.1515/cclm-2023-0743 |
| semiseg | arXiv 2507.18323 |
| openecg | PyPI `openecg` 문서 (codec v6) |

- 각 원문을 `docs/repro/sources/<paper_id>/`에 두고, 텍스트를 `pdftotext -layout`으로 같은 폴더에 만든다 (B의 근거 유효성 검사가 이 텍스트를 쓴다).
- `docs/repro/sources.md`에 paper_id, URL, 버전, 받은 날짜를 적는다.

- [ ] **Step 4: 정답을 쓴다**

**claim 목록 (최소)**

| paper_id | claim |
|---|---|
| heartwise | `heartwise-lvef-binary/primary auroc 0.900` (MHI 내부). 외부 pooled 0.917도 넣는다 (`comparable: true`, cohort 표시) |
| ecgfounder | `ecgfounder/primary auroc 0.8674`, `ecgfounder/value mae 7.0117` |
| lima-age | `mae 8.38`, `r2 0.71`. 외부 ELSA-Brasil `mae 8.44`, SaMi-Trop `mae 10.04` |
| singstad-age | `mae 8.3` |
| ai-ntprobnp | `pearson 0.566` (HCHS 내부). SHIP 외부 0.642, 0.655. log 척도 여부를 원문으로 확인해 evidence에 적는다 |
| vonbachmann-k | `mae 0.285`, `pearson 0.582`, `spearman 0.561`, `auroc_cut@<3.5 0.809`, `auroc_cut@>5.5 0.892`. 모두 `aggregate: mean_over_seeds` |
| kardionet-k | 12-lead: `mae 0.527`, `auroc_cut@>6.5 0.852 [0.745, 0.956]`. `subgroup: "esrd=1"` |
| cavalab-code15 | DeepSurv `antolini_c 0.80`, MTLR `antolini_c 0.83`. `aggregate: median_over_seeds` (inferred) |
| cavalab-mimic | DeepHit ResNet `antolini_c 0.77`, DeepHit InceptionTime `antolini_c 0.78` |
| lima-survival | `hr 1.79` (gap > 8년), `auc_cox@365 0.80` |
| semiseg | 모델 2개 × `mae` PR/QRS/QT (`label_def: pr/qrs/qt`) |
| openecg | `boundary_f1 0.855`, `boundary_error_ms 11.1`. 둘 다 `comparable: false`, `reason: "MIMIC 기계판독 시점은 대표 박동 기준 ms라 원신호 경계를 비교할 수 없음"` |

**deviation (최소)**

| paper_id | deviation |
|---|---|
| ecgfounder | `different_model` (linear probe vs full fine-tune)<br>`paper_vs_code` (필터 1–30 Hz vs 0.67–40 Hz)<br>`our_choice` (LVEF < 50을 양성으로) |
| cavalab-code15 | `data_limit` (1년 추적)<br>`paper_vs_code` (7초 vs 10초) |
| cavalab-mimic | `paper_vs_code` (InceptionTime 파일명 "no Dem" vs Table 3 "+age+sex")<br>`our_choice` (마지막 퇴원 = `visit_occurrence` 마지막 종료일) |
| kardionet-k | `our_choice` (ESRD 정의) |
| vonbachmann-k | `different_model` (공개 weight 하나 vs seed 5개 평균) |
| semiseg, openecg | `different_reference` (사람 주석 vs 기계판독) |
| 인구집단만 다른 것 | `data_limit` |

**evidence와 notes**
- 실행 조건 필드를 key로 쓴다: `label_source.kind`, `label_source.window_minutes`, `one_per_person`, `split`, cut, 전처리 필드.
- 추론했거나 우리가 정한 것은 `notes.md`에 `why:` 줄로 이유를 적는다.

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_repro_gold.py -q`
Expected: `13 passed`

- [ ] **Step 6: 검토표 생성 후 Commit**

```bash
python3 -c "from mival.repro.review import write_gold_review as w; print(w('studies/repro', 'docs/repro/gold-review.md'))"
git add studies/repro docs/repro/sources.md docs/repro/gold-review.md tests/test_repro_gold.py
git commit -m "docs: gold claims and evidence for twelve paper-protocol studies (unreviewed)"
```

---

## 1차: 지표만 바꾸는 것 (M1, M2, 비교) → 실행

### Task 3: pearson, spearman, log 변환 (M1)

**Files:**
- Modify: `src/mival/metrics/regression.py`, `src/mival/stages/evaluate.py` (`_Settings.__init__`, `_regression_rows`)
- Test: `tests/test_repro_metrics.py`

**Interfaces:**
- Produces:
  - `regression.pearson(y, p) -> float`, `regression.spearman(y, p) -> float`
  - `regression_metrics(y, p, cuts, extra: Sequence[str] = ()) -> dict`. `extra`에 든 이름만 더한다.
  - evaluate spec:
    - `regression_metrics_extra: [pearson, spearman]` (기본 `[]`)
    - `label_transform: log10` (기본 없음). 라벨과 예측 둘 다 변환하고, 0 이하 행은 빼고 경고를 남긴다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repro_metrics.py
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("sklearn")

from mival.metrics import category_of
from mival.metrics.regression import pearson, regression_metrics, spearman
from mival.pipeline.runkey import RunKey
from mival.stages.evaluate import _regression_rows, _Settings


def test_pearson_spearman_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 3.0, 2.0, 5.0])
    assert pearson(y, p) == pytest.approx(np.corrcoef(y, p)[0, 1])
    assert spearman(y, p) == pytest.approx(0.8)
    assert category_of("pearson") == category_of("spearman") == "regression"


def test_extra_metrics_are_opt_in():
    y = np.arange(10.0)
    assert "pearson" not in regression_metrics(y, y, (5.0,))
    assert set(regression_metrics(y, y, (5.0,), extra=("pearson", "spearman"))) >= {"pearson", "spearman"}


def _frame(y, p):
    return pd.DataFrame({"label_value": y, "pred_value": p, "person_id": range(len(y))})


def _key():
    return RunKey.from_dict({"site": "s", "model_id": "m", "training_mode": "inference_only", "recipe_id": "r",
                             "perturbation_id": "p", "label_def": "value", "split": "test", "fold": None})


def test_log10_drops_nonpositive_with_warning():
    settings = _Settings({"regression_cuts": [125], "label_transform": "log10",
                          "regression_metrics_extra": ["pearson"], "bootstrap": False}, (0.05,), 10)
    warnings = []
    rows = _regression_rows(_key(), _frame([10.0, 100.0, 1000.0, 50.0], [20.0, 0.0, 900.0, 40.0]),
                            "label_value", "o", "all", settings, False, warnings=warnings)
    values = {r["metric"]: r for r in rows}
    assert values["mae"]["n"] == 3
    assert values["pearson"]["value"] is not None
    assert any("label_transform" in w for w in warnings)
```

`RunKey.from_dict`가 없으면, 기존 테스트(`tests/test_runkey.py`)가 RunKey를 만드는 방식을 그대로 쓴다.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_repro_metrics.py -q`
Expected: FAIL with `ImportError: cannot import name 'pearson'`

- [ ] **Step 3: Write minimal implementation**

`src/mival/metrics/regression.py`:

```python
METRICS = ("mae", "rmse", "r2", "bias", "loa_lo", "loa_hi", "pearson", "spearman")


def pearson(y_true: Sequence, y_pred: Sequence) -> float:
    y, p = _arrays(y_true, y_pred)
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def spearman(y_true: Sequence, y_pred: Sequence) -> float:
    from scipy.stats import rankdata

    y, p = _arrays(y_true, y_pred)
    return pearson(rankdata(y), rankdata(p))


EXTRA = {"pearson": pearson, "spearman": spearman}


def regression_metrics(y_true, y_pred, cuts, extra: Sequence[str] = ()) -> Dict[str, float]:
    low, high = limits_of_agreement(y_true, y_pred)
    values = {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred),
              "bias": bias(y_true, y_pred), "loa_lo": low, "loa_hi": high}
    for name in extra:
        values[name] = EXTRA[name](y_true, y_pred)
    for cut in cuts:
        values.update(cut_metric(y_true, y_pred, cut))   # Task 4. Until then: f"auroc_below@{cut:g}"
    return values
```

`src/mival/stages/evaluate.py`:

- `_Settings.__init__`에 다음을 추가한다.

  ```python
  self.regression_extra = tuple(str(m) for m in (spec.get("regression_metrics_extra") or ()))
  unknown = [m for m in self.regression_extra if m not in ("pearson", "spearman")]
  if unknown:
      raise ValueError(f"evaluate.regression_metrics_extra has unknown metrics {unknown}")
  self.label_transform = spec.get("label_transform")
  if self.label_transform not in (None, "log10"):
      raise ValueError(f"evaluate.label_transform {self.label_transform!r} is not one of [None, 'log10']")
  ```

- `_regression_rows`에 키워드 인자 `warnings: Optional[List[str]] = None`을 더한다. `usable`을 만든 뒤 다음을 넣는다.

  ```python
  if settings.label_transform == "log10":
      positive = (usable[label_column] > 0) & (usable["pred_value"] > 0)
      dropped = int((~positive).sum())
      if dropped and warnings is not None:
          warnings.append(f"label_transform log10 dropped {dropped} rows with a value <= 0 "
                          f"for {key.to_string()} ({subgroup})")
      usable = usable[positive].assign(**{label_column: np.log10(usable[label_column]),
                                          "pred_value": np.log10(usable["pred_value"])})
  ```

  `regression_metrics(..., extra=settings.regression_extra)`로 부른다.
- `_rows_for_slice`도 `warnings`를 받아 넘기고, `run()`에서 `warnings=warnings`를 넘긴다.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_repro_metrics.py tests/test_metrics_regression.py tests/test_vertical_slice_evaluate.py tests/test_stage_evaluate.py -q`
Expected: 모두 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mival/metrics/regression.py src/mival/stages/evaluate.py tests/test_repro_metrics.py
git commit -m "feat: opt-in Pearson and Spearman, and a log10 label transform for regression arms"
```

### Task 4: 방향 있는 cut (M2)

**Files:**
- Modify: `src/mival/metrics/regression.py`, `src/mival/stages/evaluate.py` (`_Settings._cuts`, `_regression_rows`의 events)
- Test: `tests/test_repro_metrics.py` (추가)

**Interfaces:**
- Produces:
  - `parse_cut(cut) -> tuple[str, float]`: 숫자 `40` → `("<=", 40.0)` (기존 `auroc_below`). 문자열 `">6.5"` → `(">", 6.5)`
  - `cut_metric(y, p, cut) -> dict`: 숫자면 `{"auroc_below@40": ...}`, 문자열이면 `{"auroc_cut@>6.5": ...}`
  - `cut_events(y, cut) -> np.ndarray[float]`: 양성 0/1
  - 지표 stem `auroc_cut` (category regression)

- [ ] **Step 1: Write the failing test** (`tests/test_repro_metrics.py`에 추가)

```python
from mival.metrics.regression import auroc_below, cut_events, cut_metric, parse_cut


def test_parse_cut():
    assert parse_cut(40) == ("<=", 40.0)
    assert parse_cut(">6.5") == (">", 6.5)
    assert parse_cut(" <= 40 ") == ("<=", 40.0)
    with pytest.raises(ValueError, match="cut"):
        parse_cut("~6")


def test_numeric_cut_keeps_old_name_and_value():
    y = np.array([30.0, 45.0, 60.0, 35.0])
    p = np.array([32.0, 50.0, 55.0, 48.0])
    assert cut_metric(y, p, 40) == {"auroc_below@40": auroc_below(y, p, 40.0)}


def test_auroc_cut_boundary():
    y = np.array([6.5, 7.0, 4.0, 5.0])
    p = np.array([6.0, 7.5, 4.2, 4.8])
    assert cut_events(y, ">6.5").tolist() == [0.0, 1.0, 0.0, 0.0]
    assert cut_events(y, ">=6.5").tolist() == [1.0, 1.0, 0.0, 0.0]
    assert cut_metric(y, p, ">6.5") == {"auroc_cut@>6.5": 1.0}
    assert cut_metric(y, p, "<5")["auroc_cut@<5"] == 1.0
    assert category_of("auroc_cut@>6.5") == "regression"


def test_settings_accept_string_cuts():
    settings = _Settings({"regression_cuts": [">6.5", "<3.5"]}, (0.05,), 10)
    assert settings.cuts_for("value") == (">6.5", "<3.5")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_repro_metrics.py -q`
Expected: FAIL with `ImportError: cannot import name 'cut_events'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mival/metrics/regression.py
import operator
import re

METRIC_STEMS = ("auroc_below", "auroc_cut")
_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
_CUT = re.compile(r"^\s*(<=|>=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_cut(cut) -> Tuple[str, float]:
    """A number is the legacy 'value <= cut'; a string names its direction, e.g. '>6.5'."""
    if isinstance(cut, (int, float)):
        return "<=", float(cut)
    match = _CUT.match(str(cut))
    if not match:
        raise ValueError(f"regression cut {cut!r} is neither a number nor '<op><number>' with op in < <= > >=")
    return match.group(1), float(match.group(2))


def cut_events(y_true: Sequence, cut) -> np.ndarray:
    op, value = parse_cut(cut)
    return _OPS[op](np.asarray(y_true, dtype=float), value).astype(float)


def cut_metric(y_true: Sequence, y_pred: Sequence, cut) -> Dict[str, float]:
    if isinstance(cut, (int, float)):
        return {f"auroc_below@{float(cut):g}": auroc_below(y_true, y_pred, float(cut))}
    from .survival import rank_auroc

    op, value = parse_cut(cut)
    p = np.asarray(y_pred, dtype=float)
    score = p - value if op in (">", ">=") else value - p
    return {f"auroc_cut@{op}{value:g}": rank_auroc(cut_events(y_true, cut), score)}
```

`evaluate.py` `_Settings._cuts`: 숫자는 `float`, 문자열은 `parse_cut`으로 검사한 뒤 원래 문자열을 공백 없이 보관한다.

```python
def _cuts(value, where):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(...)  # 기존 메시지 그대로
    from mival.metrics.regression import parse_cut
    out = []
    for cut in value:
        if isinstance(cut, str):
            parse_cut(cut)
            out.append(cut.replace(" ", ""))
        else:
            out.append(float(cut))
    return tuple(out)
```

`_regression_rows`의 events는 `events = cut_events(y, cuts[0])`이다. 숫자 cut이면 `<=`라서 기존 값과 같다.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest -q`
Expected: 전체 PASS (기존 722 + 새 테스트)

- [ ] **Step 5: Commit**

```bash
git add src/mival/metrics/regression.py src/mival/stages/evaluate.py tests/test_repro_metrics.py
git commit -m "feat: regression cuts can name their direction ('>6.5'); numeric cuts keep auroc_below"
```

### Task 5: 비교 리포트 (C1)

**Files:**
- Create: `src/mival/repro/compare.py`
- Test: `tests/test_repro_compare.py`

**Interfaces:**
- Consumes: `load_claims`, `load_evidence`, `review_counts` (Task 1)
- Produces:
  - `compare(claims: Claims, metrics: pd.DataFrame) -> pd.DataFrame`
    - 열: `claim_id, model_id, label_def, metric, subgroup, paper, value, ci_lo, ci_hi, diff, verdict, deviations`
    - verdict: `재현됨 | 차이 있음 | 비교 불가 | 계산 못 함`
  - CLI `python -m mival.repro.compare <study_dir> <metrics_long.parquet> [--out DIR]` → `comparison.csv`, `comparison.md`
  - `tolerance_for(metric, overrides) -> tuple[str, float]`: `("abs", 0.02)` 또는 `("rel", 0.10)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repro_compare.py
import pandas as pd
import pytest

from mival.repro.claims import Claim, Claims, Deviation
from mival.repro.compare import compare, tolerance_for


def metrics(rows):
    base = {"site": "s", "training_mode": "inference_only", "recipe_id": "r", "perturbation_id": "p0",
            "split": "test", "fold": None, "outcome": "o", "subgroup": "all"}
    return pd.DataFrame([{**base, **r} for r in rows])


def claims(*items, deviations=()):
    return Claims(paper={"id": "x"}, claims=list(items), deviations=list(deviations))


def test_tolerance_families():
    assert tolerance_for("auroc_cut@>6.5", {}) == ("abs", 0.02)
    assert tolerance_for("mae", {}) == ("rel", 0.10)
    assert tolerance_for("pearson", {}) == ("abs", 0.05)
    assert tolerance_for("mae", {"mae": 0.2}) == ("rel", 0.2)


def test_verdicts():
    m = metrics([
        {"model_id": "a", "label_def": "value", "metric": "mae", "value": 0.55, "ci_lo": 0.50, "ci_hi": 0.60},
        {"model_id": "a", "label_def": "value", "metric": "auroc_cut@>6.5", "value": 0.70, "ci_lo": 0.65, "ci_hi": 0.75},
    ])
    c = claims(
        Claim(id="c1", model_id="a", label_def="value", metric="mae", value=0.527),
        Claim(id="c2", model_id="a", label_def="value", metric="auroc_cut@>6.5", value=0.852),
        Claim(id="c3", model_id="a", label_def="value", metric="boundary_f1", value=0.855,
              comparable=False, reason="기준이 다름"),
        Claim(id="c4", model_id="a", label_def="value", metric="r2", value=0.7),
        deviations=[Deviation(field="x", kind="our_choice", why="ESRD 정의", claims=("c2",))],
    )
    out = compare(c, m).set_index("claim_id")
    assert out.loc["c1", "verdict"] == "재현됨"          # 0.527 is inside [0.50, 0.60]
    assert out.loc["c2", "verdict"] == "차이 있음"
    assert out.loc["c2", "deviations"] == "our_choice: ESRD 정의"
    assert out.loc["c3", "verdict"] == "비교 불가"
    assert out.loc["c4", "verdict"] == "계산 못 함"
    assert out.loc["c2", "diff"] == pytest.approx(0.70 - 0.852)


def test_ambiguous_match_raises():
    m = metrics([
        {"model_id": "a", "label_def": "value", "metric": "mae", "value": 1.0, "perturbation_id": "p0"},
        {"model_id": "a", "label_def": "value", "metric": "mae", "value": 2.0, "perturbation_id": "p1"},
    ])
    c = claims(Claim(id="c1", model_id="a", label_def="value", metric="mae", value=1.0))
    with pytest.raises(ValueError, match="c1.*2 rows.*perturbation_id"):
        compare(c, m)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_repro_compare.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mival.repro.compare'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mival/repro/compare.py
"""Paper claims against this framework's metrics_long, one verdict per claim (spec ③)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .claims import Claims, load_claims, load_evidence
from .review import review_counts

ABSOLUTE = {"auroc": 0.02, "auprc": 0.02, "auroc_cut": 0.02, "auroc_below": 0.02, "c_index": 0.02,
            "antolini_c": 0.02, "auroc_horizon": 0.02, "auc_cox": 0.02,
            "pearson": 0.05, "spearman": 0.05, "r2": 0.05}
RELATIVE = {"mae": 0.10, "rmse": 0.10, "hr": 0.10, "boundary_error_ms": 0.10}
#: Axes that can make one claim match several rows; the error names those that differ.
DISAMBIGUATING = ("training_mode", "recipe_id", "perturbation_id", "outcome", "site")
COLUMNS = ("claim_id", "model_id", "label_def", "metric", "subgroup", "paper", "value",
           "ci_lo", "ci_hi", "diff", "verdict", "deviations")


def tolerance_for(metric: str, overrides: Dict[str, float]) -> Tuple[str, float]:
    stem = metric.split("@", 1)[0]
    if stem in RELATIVE:
        return "rel", float(overrides.get(stem, RELATIVE[stem]))
    return "abs", float(overrides.get(stem, ABSOLUTE.get(stem, 0.02)))


def _num(value):
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def compare(claims: Claims, metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"] == "test"]
    rows: List[dict] = []
    for claim in claims.claims:
        devs = "; ".join(f"{d.kind}: {d.why}" for d in claims.deviations_for(claim.id))
        row = {"claim_id": claim.id, "model_id": claim.model_id, "label_def": claim.label_def,
               "metric": claim.metric, "subgroup": claim.subgroup, "paper": claim.value,
               "value": None, "ci_lo": None, "ci_hi": None, "diff": None, "deviations": devs}
        if not claim.comparable:
            rows.append({**row, "verdict": "비교 불가", "deviations": "; ".join(x for x in (devs, claim.reason) if x)})
            continue
        hit = test[(test["model_id"] == claim.model_id) & (test["label_def"] == claim.label_def)
                   & (test["metric"] == claim.metric) & (test["subgroup"] == claim.subgroup)]
        if claim.outcome is not None:
            hit = hit[hit["outcome"] == claim.outcome]
        if len(hit) == 0:
            rows.append({**row, "verdict": "계산 못 함"})
            continue
        if len(hit) > 1:
            differ = [a for a in DISAMBIGUATING if a in hit.columns and hit[a].nunique(dropna=False) > 1]
            raise ValueError(f"claim {claim.id} matches {len(hit)} rows of metrics_long; they differ in "
                             f"{differ}. Give the claim an outcome or run evaluate on one setting.")
        found = hit.iloc[0]
        value, lo, hi = _num(found["value"]), _num(found.get("ci_lo")), _num(found.get("ci_hi"))
        if value is None:
            rows.append({**row, "verdict": "계산 못 함"})
            continue
        kind, tol = tolerance_for(claim.metric, claims.tolerance)
        diff = value - claim.value
        within_tol = abs(diff) <= (tol if kind == "abs" else tol * abs(claim.value))
        within_ci = lo is not None and hi is not None and lo <= claim.value <= hi
        rows.append({**row, "value": value, "ci_lo": lo, "ci_hi": hi, "diff": diff,
                     "verdict": "재현됨" if (within_ci or within_tol) else "차이 있음"})
    return pd.DataFrame(rows, columns=list(COLUMNS))


def _fmt(value) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def to_markdown(table: pd.DataFrame, paper_id: str, reviewed: Tuple[int, int]) -> str:
    lines = [f"# {paper_id} 논문 대비 재현", "",
             f"정답은 사람이 검토하지 않음 (검토 {reviewed[0]} / 전체 {reviewed[1]} 필드).", "",
             "| claim | 모델 | 지표 | subgroup | 논문 | 재현 (95% CI) | 차이 | 판정 | 조건 차이 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in table.to_dict("records"):
        ci = f" ({_fmt(r['ci_lo'])}–{_fmt(r['ci_hi'])})" if r["ci_lo"] is not None else ""
        lines.append(f"| {r['claim_id']} | {r['model_id']}/{r['label_def']} | {r['metric']} | {r['subgroup']} | "
                     f"{r['paper']:.3f} | {_fmt(r['value'])}{ci} | {_fmt(r['diff'])} | {r['verdict']} | {r['deviations']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mival.repro.compare")
    parser.add_argument("study_dir")
    parser.add_argument("metrics_long", nargs="+", help="one or more metrics_long.parquet files")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    folder = Path(args.study_dir)
    claims = load_claims(folder / "claims.yaml")
    evidence_path = folder / "evidence.yaml"
    reviewed = review_counts(load_evidence(evidence_path)) if evidence_path.is_file() else (0, 0)
    metrics = pd.concat([pd.read_parquet(p) for p in args.metrics_long], ignore_index=True)
    table = compare(claims, metrics)
    out = Path(args.out) if args.out else folder / "results"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison.csv", index=False)
    (out / "comparison.md").write_text(to_markdown(table, folder.name, reviewed), encoding="utf-8")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

여러 metrics_long을 받는 이유: Lima 생존(Task 15)은 별도 스크립트가 같은 형식의 parquet을 만든다.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_repro_compare.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/mival/repro/compare.py tests/test_repro_compare.py
git commit -m "feat: compare paper claims with metrics_long, one verdict and its deviations per claim"
```

### Task 6: 서버 queue와 1차 study 4개 실행

1차 study 4개는 코호트가 기존 공통 study와 같다. 그래서 기존 예측(`/data/mi-val/runs/<src>/_full/predictions_all`)과 기존 profile의 `cohort_split`으로 evaluate만 다시 돈다.

**Files:**
- Create: `scripts/repro/rq_worker.sh`, `scripts/repro/enqueue.sh`, `scripts/repro/run_repro.sh`
- Create: `studies/repro/{heartwise,lima-age,singstad-age,ai-ntprobnp}/study.yaml`
- Test: `tests/test_repro_studies.py`

**Interfaces:**
- Produces:
  - `run_repro.sh <paper_id> eval-only <source_study> [attributes.parquet]`
  - `run_repro.sh <paper_id> full [N|all]`: 기존 `vs_sample.sh` 흐름에 `studies/repro/<id>`를 넣는다. Task 10, 14에서 쓴다.
  - 결과는 `/data/mi-val/runs/repro-<paper_id>/`에 남고, 비교는 `studies/repro/<id>/results/`(서버 worktree)에 남는다.
  - `enqueue.sh <name> -- <command...>`는 `/data/mi-val/queue/NNN-<name>.sh`를 만든다.
  - `rq_worker.sh`는 tmux `rq`에서 돈다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repro_studies.py
from pathlib import Path

import pytest

from mival.pipeline.study import load_study
from mival.stages.evaluate import _Settings

ROOT = Path(__file__).resolve().parents[1] / "studies" / "repro"
TIER1 = ("heartwise", "lima-age", "singstad-age", "ai-ntprobnp")


@pytest.mark.parametrize("paper_id", TIER1)
def test_tier1_study_loads_and_evaluate_settings_parse(paper_id):
    study = load_study(ROOT / paper_id)
    assert study.study_id == f"repro-{paper_id}"
    _Settings(study.stage_spec("evaluate"), (0.05,), 10)


def test_ntprobnp_is_log_scale_with_pearson():
    spec = load_study(ROOT / "ai-ntprobnp").stage_spec("evaluate")
    assert spec["label_transform"] == "log10"
    assert "pearson" in spec["regression_metrics_extra"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_repro_studies.py -q`
Expected: FAIL with `FileNotFoundError` for `studies/repro/heartwise/study.yaml`

- [ ] **Step 3: study.yaml 4개를 쓴다**

- 각 파일은 원본 study(`studies/lvef`, `studies/ecg-age`, `studies/ntprobnp`)를 복사하고 두 군데만 바꾼다: `study_id: repro-<paper_id>`, `evaluate`.
- 원본의 retrieve, profile, preprocess, models 블록은 그대로 둔다. 기록용이며 eval-only에서는 실행하지 않는다.

`studies/repro/ai-ntprobnp/study.yaml`의 evaluate:

```yaml
  evaluate:
    outcomes: {ntprobnp: null}
    regression_cuts: [125, 300]
    regression_metrics_extra: [pearson, spearman]
    label_transform: log10       # evidence: 논문 R이 log 척도인지 Task 2에서 확인한 값으로 둔다
    bootstrap_replicates: 2000
    figures: false
```

나머지 3개의 evaluate:

| study | 원본 | evaluate |
|---|---|---|
| lima-age | `ecg-age` | Lima arm만 남기고 `regression_metrics_extra: [pearson]` 추가. 논문 보고가 MAE·R²라서 cut은 원본 그대로 |
| singstad-age | `ecg-age` | Singstad arm만, 나머지는 lima-age와 같음 |
| heartwise | `lvef` | `outcomes: {primary: null}` 그대로, `figures: false` |

eval-only에서 다른 모델의 예측을 걸러야 하므로 `run_repro.sh`는 `models.arms`의 model_id만 predictions에서 골라 evaluate에 넣는다.

- [ ] **Step 4: 스크립트 3개를 쓴다**

```bash
# scripts/repro/enqueue.sh
#!/bin/bash
# usage: enqueue.sh <name> -- <command...>   -> /data/mi-val/queue/NNN-<name>.sh
set -e
Q=/data/mi-val/queue; mkdir -p $Q/log
name=$1; shift; [ "$1" = "--" ] && shift
n=$(ls $Q | grep -E '^[0-9]{3}-' | sed 's/-.*//' | sort -n | tail -1); n=$(printf '%03d' $(( 10#${n:-0} + 1 )))
printf '#!/bin/bash\nset -o pipefail\n%s\n' "$*" > $Q/$n-$name.sh; chmod +x $Q/$n-$name.sh; echo $Q/$n-$name.sh
```

```bash
# scripts/repro/rq_worker.sh  (tmux new -d -s rq 'bash /data/mi-val/mi-va-vs/scripts/repro/rq_worker.sh')
#!/bin/bash
Q=/data/mi-val/queue; mkdir -p $Q/log
while true; do
  job=$(ls $Q | grep -E '^[0-9]{3}-.*\.sh$' | sort | head -1)
  if [ -z "$job" ]; then sleep 30; continue; fi
  base=${job%.sh}; mv $Q/$job $Q/$base.running
  echo "$(date -u +%FT%TZ) start $base" >> $Q/log/worker.log
  if bash $Q/$base.running > $Q/log/$base.log 2>&1; then mv $Q/$base.running $Q/$base.done; s=done
  else mv $Q/$base.running $Q/$base.failed; s=failed; fi
  echo "$(date -u +%FT%TZ) $s $base" >> $Q/log/worker.log
done
```

```bash
# scripts/repro/run_repro.sh
#!/bin/bash
# usage: run_repro.sh <paper_id> eval-only <source_study> [attributes.parquet]
#        run_repro.sh <paper_id> full [N|all]
set -o pipefail
P=$1; MODE=$2; W=/data/mi-val/mi-va-vs; R=/data/mi-val/runs; S=repro-$P; ST=$W/studies/repro/$P
py=/data/mi-val/envs/mival/bin/python
cli() { (cd $W && PYTHONPATH=src $py -c "import sys; from mival.pipeline.cli import main; sys.exit(main())" "$@"); }
latest() { ls -td $1/*/ 2>/dev/null | while read d; do [ -f $d/manifest.json ] && echo $d && break; done; }
mkdir -p $R/$S/_eval
if [ "$MODE" = eval-only ]; then
  SRC=$3; ATTR=$4
  SPL=$(latest $R/$SRC/profile)artifacts/cohort_split.parquet
  $py - <<PY || exit 1
import yaml, pandas as pd, pathlib
st = yaml.safe_load(open("$ST/study.yaml")); keep = {a["model_id"] for a in st["stages"]["models"]["arms"]}
out = pathlib.Path("$R/$S/_eval/predictions"); out.mkdir(parents=True, exist_ok=True)
for f in pathlib.Path("$R/$SRC/_full/predictions_all").glob("*.parquet"):
    t = pd.read_parquet(f)
    if t["model_id"].iloc[0] in keep: t.to_parquet(out / f.name, index=False)
print(sorted(p.name for p in out.glob("*.parquet")))
PY
  EXTRA=""; [ -n "$ATTR" ] && EXTRA="--input attributes=$ATTR"
  cli run evaluate --study $ST --runs-root $R --input predictions=$R/$S/_eval/predictions --input cohort_split=$SPL $EXTRA || exit 1
else
  N=${3:-1000}; ln -sfn $ST $W/studies/$S
  bash $W/scripts/repro/vs_flow.sh $S $N || exit 1   # Task 10 Step 3에서 만든다
fi
E=$(latest $R/$S/evaluate)
(cd $W && PYTHONPATH=src $py -m mival.repro.compare $ST ${E}artifacts/metrics_long.parquet --out $ST/results)
```

- [ ] **Step 5: Run test, commit, push to server, enqueue**

```bash
python3 -m pytest tests/test_repro_studies.py -q          # Expected: 5 passed
python3 -m pytest -q                                       # Expected: all pass
git add scripts/repro studies/repro/*/study.yaml tests/test_repro_studies.py
git commit -m "feat: server queue and evaluate-only runs for the four tier-1 paper studies"
K=/Users/minseongkim/Desktop/youlab/<key>.pem
GIT_SSH_COMMAND="ssh -i $K -p 2022" git push -q -f ssh://ubuntu@<server-ip>/data/mi-val/mi-va HEAD:refs/heads/vs-incoming
ssh -i $K -p 2022 ubuntu@<server-ip> 'cd /data/mi-val/mi-va-vs && git reset -q --hard vs-incoming && \
  (tmux has-session -t rq 2>/dev/null || tmux new -d -s rq "bash scripts/repro/rq_worker.sh") && \
  for p in heartwise:lvef lima-age:ecg-age singstad-age:ecg-age ai-ntprobnp:ntprobnp; do \
    bash scripts/repro/enqueue.sh repro-${p%%:*} -- bash /data/mi-val/mi-va-vs/scripts/repro/run_repro.sh ${p%%:*} eval-only ${p##*:}; done; ls /data/mi-val/queue'
```

Expected: `001-repro-heartwise.sh` … `004-repro-ai-ntprobnp.sh`가 보인다. 몇 분 안에 `.done`이 되고 `studies/repro/<id>/results/comparison.md`가 생긴다. 서버 worktree의 결과는 Task 18에서 로컬로 가져온다.

---

## 2차: 코호트 규칙 (R1, R2, R3) → 실행

### Task 7: 분 단위 간격과 사람당 첫 ECG (R1)

**Files:**
- Modify: `src/mival/stages/retrieve.py` (`RetrieveSpec`, `_nearest_file_measurement`, `one_per_person`, `exclude_derived`, `config_inputs`)
- Create: `scripts/repro/export_potassium_labels.py`
- Test: `tests/test_repro_retrieve.py`

**Interfaces:**
- Produces (study에서 쓰는 필드):
  - `label_source: {kind: measurement_file, path, window_minutes: 60, ecg_time_path: <machine_measurements.csv>}`
    - `window_minutes`가 있으면 `window_days`를 무시한다. ECG 시각은 `ecg_time_path`의 `ecg_time`(study_id로 연결)이다.
    - ECG 시각이 없으면 `label_missing`, 상세 "no ecg_time for study_id".
  - `one_per_person: first`: person마다 `(ECG 시각 또는 index_datetime, image_occurrence_id)`가 가장 이른 것. `true`는 기존 무작위 선택이다.
  - `cohort_index`에 `label_delta_minutes`(float)를 더한다. 분 단위 모드일 때만 생긴다.
- `export_potassium_labels.py <out.csv>`는 `cdm.measurement` concept 3023103에서 `person_id, measurement_datetime, value, source`를 뽑는다. `source`는 `measurement_source_value`이고, itemid 50971(검사실)과 227442(ICU)를 구분하는 데 쓴다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repro_retrieve.py
from datetime import date

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from tests.test_vertical_slice_retrieve import ecg, run, spec  # noqa: E402


def files(tmp_path, labs, times):
    lab_path = tmp_path / "k.csv"
    pd.DataFrame(labs, columns=["person_id", "measurement_datetime", "value"]).to_csv(lab_path, index=False)
    mm = tmp_path / "mm.csv"
    pd.DataFrame(times, columns=["study_id", "ecg_time"]).to_csv(mm, index=False)
    return str(lab_path), str(mm)


def test_minutes_window_nearest(tmp_path):
    lab, mm = files(tmp_path,
                    [(10, "2180-01-01 08:00:00", 4.0), (10, "2180-01-01 09:20:00", 5.5), (11, "2180-01-01 12:00:00", 3.9)],
                    [(40000001, "2180-01-01 09:00:00"), (40000002, "2180-01-01 10:30:00")])
    body = spec(tmp_path, "measurement_file", source={"path": lab, "window_minutes": 60, "ecg_time_path": mm})
    index, excluded = run(tmp_path, body, [ecg(1, 10, date(2180, 1, 1)), ecg(2, 11, date(2180, 1, 1))])
    assert index["label_value"].tolist() == [5.5]              # 20 min beats 60 min
    assert index["label_delta_minutes"].tolist() == [20.0]
    assert excluded == {"2": "label_missing"}                   # 90 min away


def test_minutes_window_needs_ecg_time(tmp_path):
    lab, mm = files(tmp_path, [(10, "2180-01-01 09:00:00", 4.0)], [(49999999, "2180-01-01 09:00:00")])
    body = spec(tmp_path, "measurement_file", source={"path": lab, "window_minutes": 60, "ecg_time_path": mm})
    index, excluded = run(tmp_path, body, [ecg(1, 10, date(2180, 1, 1))])
    assert len(index) == 0 and excluded == {"1": "label_missing"}


def test_one_per_person_first(tmp_path):
    lab, mm = files(tmp_path,
                    [(10, "2180-01-01 09:00:00", 4.0), (10, "2180-01-01 15:00:00", 4.5)],
                    [(40000001, "2180-01-01 15:10:00"), (40000002, "2180-01-01 09:05:00")])
    body = spec(tmp_path, "measurement_file", source={"path": lab, "window_minutes": 60, "ecg_time_path": mm},
                one_per_person="first")
    index, excluded = run(tmp_path, body, [ecg(1, 10, date(2180, 1, 1)), ecg(2, 10, date(2180, 1, 1))])
    assert index["image_occurrence_id"].tolist() == [2]
    assert excluded == {"1": "not_selected"}


def test_window_days_path_unchanged(tmp_path):
    lab, _ = files(tmp_path, [(10, "2180-01-02 09:00:00", 4.0)], [])
    index, _ = run(tmp_path, spec(tmp_path, "measurement_file", source={"path": lab}, window_days=1),
                   [ecg(1, 10, date(2180, 1, 1))])
    assert index["label_value"].tolist() == [4.0] and "label_delta_minutes" not in index.columns
```

`tests/__init__.py`가 없으면 공용 헬퍼(`ecg`, `run`, `spec`)를 `tests/_retrieve_helpers.py`로 옮기고 두 테스트 파일이 함께 import한다.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_repro_retrieve.py -q`
Expected: FAIL. `window_minutes`가 무시되어 `label_value`가 `[4.0]`이 된다 (같은 날 가장 이른 값).

- [ ] **Step 3: Write minimal implementation**

`RetrieveSpec`에 필드를 더한다:

```python
window_minutes: Optional[float] = None
ecg_time_path: Optional[str] = None
one_per_person: Any = False   # False | True | "first"
```

`from_mapping`:

```python
window_minutes=None if source.get("window_minutes") is None else float(source["window_minutes"]),
ecg_time_path=None if source.get("ecg_time_path") is None else str(source["ecg_time_path"]),
one_per_person=("first" if body.get("one_per_person") == "first" else bool(body.get("one_per_person", False))),
```

`window_minutes`가 있는데 `ecg_time_path`가 없으면 `ValueError("retrieve.label_source.ecg_time_path is required with window_minutes")`.

`config_inputs`는 `ecg_time_path`가 있을 때 `{"ecg_time": Path(...)}`를 더한다. 없으면 반환값이 이전과 같아서 config_hash가 그대로다.

`_nearest_file_measurement`의 분기:

```python
if spec.window_minutes is not None:
    times = pandas.read_csv(spec.ecg_time_path, usecols=["study_id", "ecg_time"])
    times = times.drop_duplicates("study_id").assign(ecg_time=lambda t: pandas.to_datetime(t["ecg_time"]))
    ecg = frame[["image_occurrence_id", "person_id", "local_path"]].assign(study_id=frame["local_path"].map(_study_id))
    ecg = ecg.merge(times, on="study_id", how="left")
    pairs = ecg.dropna(subset=["ecg_time"]).merge(labs, on="person_id", how="inner")
    pairs["_delta"] = (pairs["measurement_datetime"] - pairs["ecg_time"]).dt.total_seconds() / 60.0
    pairs = pairs[pairs["_delta"].abs() <= spec.window_minutes]
    pairs = pairs.assign(_abs=pairs["_delta"].abs()).sort_values(["image_occurrence_id", "_abs", "measurement_datetime"])
    best = pairs.drop_duplicates("image_occurrence_id")[["image_occurrence_id", "value", "measurement_datetime", "_delta"]]
    best = best.rename(columns={"value": "label_value", "measurement_datetime": "label_datetime",
                                "_delta": "label_delta_minutes"})
    out = frame.drop(columns=["label_value", "label_datetime", "label_delta_days"], errors="ignore")
    out = out.merge(best, on="image_occurrence_id", how="left").assign(label_delta_days=np.nan)
    return out.merge(ecg[["image_occurrence_id", "ecg_time"]], on="image_occurrence_id", how="left")
```

나머지 연결:
- `_run_derived`의 columns: `label_delta_minutes`와 `ecg_time`이 frame에 있으면 cohort_index 열에 더한다. 열이 없으면 지금과 같다.
- `exclude_derived`의 `measurement_file` 상세 문구: `window_minutes`가 있으면 `f"no value within {spec.window_minutes:g} minutes (or no ecg_time)"`.
- `one_per_person(frame, seed, ledger, rule=True)`: `rule == "first"`이면 다음으로 고른다.

  ```python
  order_key = frame["ecg_time"] if "ecg_time" in frame else pandas.to_datetime(frame["index_datetime"])
  chosen = frame.assign(_t=order_key).sort_values(["person_id", "_t", "image_occurrence_id"]).groupby("person_id").head(1).index
  ```

  ledger 상세는 "first ECG per person". 호출부 두 곳은 `one_per_person(kept, ctx.seed, ctx.ledger, spec.one_per_person)`로 바꾼다.

`scripts/repro/export_potassium_labels.py`:

```python
"""Serum potassium with timestamps from MI-CDM, for minute-level pairing (spec R1)."""
import sys
from mival.stages.retrieve import query_postgres

SQL = """select person_id, measurement_datetime, value_as_number as value, measurement_source_value as source
from cdm.measurement where measurement_concept_id = 3023103 and value_as_number is not null
and measurement_datetime is not null"""

if __name__ == "__main__":
    frame = query_postgres(SQL, {}, "/data/mi-val/secrets/micdm.env")
    frame.to_csv(sys.argv[1], index=False)
    print(len(frame), frame["source"].value_counts().head().to_dict())
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_repro_retrieve.py tests/test_vertical_slice_retrieve.py tests/test_stage_retrieve.py -q && python3 -m pytest -q`
Expected: 모두 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mival/stages/retrieve.py scripts/repro/export_potassium_labels.py tests/test_repro_retrieve.py
git commit -m "feat: minute-level label windows from the ECG time, and the first ECG per person"
```

### Task 8: 외부 라벨 파일과 고정 split (R2)

**Files:**
- Modify: `src/mival/stages/retrieve.py` (새 종류 `ecg_label_file`, 공통 `split_source`)
- Modify: `src/mival/stages/profile.py` (`use_fixed_split`)
- Create: `scripts/repro/make_split_ecgfounder.py`
- Test: `tests/test_repro_retrieve.py`, `tests/test_repro_profile.py`

**Interfaces:**
- Produces:
  - `label_source: {kind: ecg_label_file, path, key: study_id, value_column: LVEF, primary: "<50"}`
    - `path`는 `study_id`와 `value_column` 열이 있는 CSV다.
    - `label_value` = 값. `label_primary` = `cut_events(value, primary)` (Task 4의 함수).
    - 파일에 없는 ECG는 `label_missing`이다.
  - `split_source: {path, key: study_id|subject_id, column: split}` (retrieve 최상위, 모든 종류 공통)
    - cohort_index에 `fixed_split` 열을 더한다. 값은 `test | dev`이고, 파일의 `train`과 `val`은 `dev`로 쓴다.
    - `subject_id`는 local_path의 `/p<digits>/` 두 번째 단계에서 뽑는다 (`p1000/p10000032` → 10000032).
  - profile `use_fixed_split: true`:
    - person의 split은 그 사람의 ECG 중 하나라도 `test`면 `test`, 아니면 `dev`다.
    - fold는 dev person에 `seed`로 0..n_folds-1을 나눈다.
    - `fixed_split`이 없는 ECG가 있으면 `ValueError`.
  - `make_split_ecgfounder.py <LVEF.csv> <out.csv>`는 저자 코드를 그대로 옮긴다. `train_test_split(test_size=0.2, shuffle=False)` 후 남은 20%를 다시 50/50으로 나눈다. 출력 열은 `study_id, split, LVEF`이고, 건수를 출력해 논문(60,333 / 7,539 / 7,541)과 대조한다.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_repro_retrieve.py에 추가
def test_ecg_label_file_with_fixed_split(tmp_path):
    lab = tmp_path / "lvef.csv"
    pd.DataFrame({"study_id": [40000001, 40000002], "LVEF": [35.0, 60.0]}).to_csv(lab, index=False)
    split = tmp_path / "split.csv"
    pd.DataFrame({"study_id": [40000001, 40000002], "split": ["test", "train"]}).to_csv(split, index=False)
    body = spec(tmp_path, "ecg_label_file",
                source={"path": str(lab), "key": "study_id", "value_column": "LVEF", "primary": "<50"},
                split_source={"path": str(split), "key": "study_id", "column": "split"})
    index, excluded = run(tmp_path, body, [ecg(1, 10, date(2180, 1, 1)), ecg(2, 11, date(2180, 1, 1)),
                                           ecg(3, 12, date(2180, 1, 1))])
    assert index["label_value"].tolist() == [35.0, 60.0]
    assert index["label_primary"].tolist() == [1.0, 0.0]
    assert index["fixed_split"].tolist() == ["test", "dev"]
    assert excluded == {"3": "label_missing"}


def test_split_by_subject_id(tmp_path):
    split = tmp_path / "split.csv"
    pd.DataFrame({"subject_id": [10000032], "split": ["test"]}).to_csv(split, index=False)
    body = spec(tmp_path, "age_at_ecg", split_source={"path": str(split), "key": "subject_id", "column": "split"})
    index, _ = run(tmp_path, body, [ecg(1, 10, date(2180, 1, 1), birth=2120)])
    assert index["fixed_split"].tolist() == ["test"]
```

```python
# tests/test_repro_profile.py
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from mival.pipeline.stage import execute, prepare
from mival.pipeline.tables import read_table, write_table
from mival.stages.profile import COHORT_SPLIT, ProfileStage


def test_use_fixed_split(tmp_path):
    cohort = pd.DataFrame({"image_occurrence_id": [1, 2, 3, 4], "person_id": [10, 10, 11, 12],
                           "fixed_split": ["dev", "test", "dev", "dev"], "label_primary": [0, 1, 0, 1]})
    path = write_table(cohort.to_dict("records"), tmp_path / "ci.parquet", tuple(cohort.columns))
    stage = ProfileStage()
    ctx = prepare(stage, "s", "site", {"use_fixed_split": True, "stratify_on": "label_primary",
                                       "min_test_positives": 0, "n_folds": 2},
                  {"cohort_index": path}, tmp_path / "runs", seed=1)
    execute(stage, ctx)
    split = read_table(ctx.layout.artifact(COHORT_SPLIT)).set_index("person_id")
    assert split.loc[10, "split"] == "test"
    assert split.loc[11, "split"] == "dev" and split.loc[12, "split"] == "dev"
    assert set(split.loc[[11, 12], "fold"]) <= {0.0, 1.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_repro_retrieve.py tests/test_repro_profile.py -q`
Expected: FAIL with `retrieve.label_source.kind 'ecg_label_file' is not one of [...]`

- [ ] **Step 3: Write minimal implementation**

retrieve의 상수 변경:
- `LABEL_KINDS`에 `"ecg_label_file"`을 더한다.
- `DERIVED_COLUMNS["ecg_label_file"] = ()`
- `FILE_KINDS`에 `"ecg_label_file"`을 더한다.

`derive_labels`:

```python
if spec.label_kind == "ecg_label_file":
    from mival.metrics.regression import cut_events
    table = pandas.read_csv(spec.measurements_path, usecols=[spec.file_key, spec.value_column])
    table = table.drop_duplicates(spec.file_key).rename(columns={spec.value_column: "label_value"})
    frame = frame.drop(columns=["label_value"], errors="ignore")
    frame[spec.file_key] = frame["local_path"].map(_study_id if spec.file_key == "study_id" else _subject_id)
    frame = frame.merge(table, on=spec.file_key, how="left")
    if spec.primary is not None:
        frame["label_primary"] = pandas.Series(cut_events(frame["label_value"], spec.primary), index=frame.index).where(frame["label_value"].notna())
    return frame
```

나머지 retrieve 변경:
- `RetrieveSpec`에 `file_key="study_id"`, `value_column="value"`, `primary=None`, `split_source=None`(dict)를 더한다.
- `_subject_id(path)`는 `re.search(r"/p\d+/p(\d+)/", str(path))`로 뽑는다.
- `exclude_derived`: `ecg_label_file`은 `label_value.isna()`이면 `label_missing`이다. implausible 검사는 하지 않는다. 저자 라벨을 그대로 쓰기 위해서다.
- split: `_run_derived`(와 measurement 경로)가 kept를 만든 뒤 `attach_fixed_split(kept, spec.split_source)`를 부른다. 파일을 읽어 key로 merge하고, `train`과 `val`을 `dev`로 매핑해 `fixed_split` 열을 만든다. 파일에 없는 ECG는 NaN으로 남기고, profile이 이를 거부한다.
- `config_inputs`: `split_source.path`가 있으면 `{"split_file": Path(...)}`를 더한다.

profile 변경:
- `ProfileSpec`에 `use_fixed_split: bool = False`를 더한다.
- `run()`에서 `use_fixed_split`이면 `assign_splits` 대신 다음을 쓴다.

```python
if cohort["fixed_split"].isna().any():
    raise ValueError(f"{int(cohort['fixed_split'].isna().sum())} ECGs have no fixed_split")
persons = cohort.groupby("person_id")["fixed_split"].agg(lambda s: "test" if (s == "test").any() else "dev").reset_index(name="split")
rng = np.random.default_rng(seed)
dev = persons["split"] == "dev"
fold = np.full(len(persons), np.nan)
fold[np.flatnonzero(dev)] = (rng.permutation(int(dev.sum())) % spec.n_folds).astype(float)
split = persons.assign(fold=fold)[list(COHORT_SPLIT_COLUMNS)]
```

`scripts/repro/make_split_ecgfounder.py`:

```python
"""ECGFounder's LVEF split, as the authors' code makes it (finetune_model.py: shuffle=False)."""
import re
import sys

import pandas as pd

from sklearn.model_selection import train_test_split

lvef = pd.read_csv(sys.argv[1])            # 저자 코드와 같은 순서 (파일 순서, subject_id 정렬)
train, rest = train_test_split(lvef, test_size=0.2, shuffle=False)
val, test = train_test_split(rest, test_size=0.5, shuffle=False)
lvef["split"] = ["train"] * len(train) + ["val"] * len(val) + ["test"] * len(test)
lvef["study_id"] = lvef["waveform_path"].map(lambda p: int(re.search(r"/s(\d+)/", "/" + str(p)).group(1)))
lvef[["study_id", "subject_id", "split", "LVEF"]].to_csv(sys.argv[2], index=False)
print(lvef["split"].value_counts().to_dict())   # 조사에서 재현한 값: train 60,329 / val 7,541 / test 7,542
```

저자 코드와 같은 `train_test_split`을 그대로 부른다. 저자 코드가 split 전에 정렬하거나 거르는 단계가 있으면 Task 2에서 받은 `finetune_model.py`·`dataset.py`를 보고 그대로 옮긴다.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_repro_retrieve.py tests/test_repro_profile.py -q && python3 -m pytest -q`
Expected: 모두 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mival/stages/retrieve.py src/mival/stages/profile.py scripts/repro/make_split_ecgfounder.py tests/test_repro_retrieve.py tests/test_repro_profile.py
git commit -m "feat: per-ECG label files and fixed splits, so a paper's own labels and test set can be used"
```

### Task 9: ESRD 부분집단 표 (R3)

**Files:**
- Create: `scripts/repro/build_esrd_attributes.py`, `src/mival/repro/esrd.py`
- Test: `tests/test_repro_esrd.py`

**Interfaces:**
- Produces:
  - `esrd_flags(conditions: pd.DataFrame, procedures: pd.DataFrame, persons: Sequence[int]) -> pd.DataFrame`
    - 입력 열: `person_id, source_value`
    - 출력 열: `person_id, esrd`(0/1)
  - 스크립트는 `attributes_esrd.parquet`를 쓴다. evaluate에는 `--input attributes=...`와 `subgroups: [esrd]`로 넣고, subgroup 이름은 `esrd=1`이 된다.
  - 정의 (our_choice, notes에 적음):
    - 진단 ICD-10 `N186`, `Z992` / ICD-9 `5856`, `V4511`
    - 또는 투석 시술 ICD-9-PCS `3995`, `5498` / ICD-10-PCS `5A1D*`
    - 투석만 있고 N18.6/Z99.2/585.6 진단이 없으면 AKI 투석으로 보고 제외한다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repro_esrd.py
import pandas as pd

from mival.repro.esrd import esrd_flags


def test_esrd_flags():
    conditions = pd.DataFrame({"person_id": [1, 2, 3], "source_value": ["N186", "I10", "5856"]})
    procedures = pd.DataFrame({"person_id": [2, 4, 3], "source_value": ["5A1D70Z", "3995", "3995"]})
    out = esrd_flags(conditions, procedures, [1, 2, 3, 4, 5]).set_index("person_id")["esrd"].to_dict()
    # 2: 투석만, ESRD 진단 없음 -> AKI 투석으로 보고 0
    # 4: 같은 이유로 0
    assert out == {1: 1, 2: 0, 3: 1, 4: 0, 5: 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_repro_esrd.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/mival/repro/esrd.py
"""ESRD/dialysis-dependent persons, our definition for Kardio-Net's cohort (spec R3, our_choice)."""
from typing import Sequence

import pandas as pd

ESRD_DIAGNOSES = ("N186", "Z992", "5856", "V4511")


def _clean(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(".", "", regex=False).str.upper().str.strip()


def esrd_flags(conditions: pd.DataFrame, procedures: pd.DataFrame, persons: Sequence[int]) -> pd.DataFrame:
    diagnosed = set(conditions.loc[_clean(conditions["source_value"]).isin(ESRD_DIAGNOSES), "person_id"])
    # Dialysis alone may be for acute kidney injury; it counts only with an ESRD diagnosis,
    # so the procedure list documents the definition but never adds a person by itself.
    flags = pd.DataFrame({"person_id": list(persons)})
    flags["esrd"] = flags["person_id"].isin(diagnosed).astype(int)
    return flags
```

투석 시술만으로는 사람이 추가되지 않으므로 정의는 진단 기준이다. 이것은 스펙의 "합집합" 제안보다 좁다. ledger에 `Ruling`으로 남기고, notes에도 "투석 시술 단독은 AKI와 구분 불가라 제외"로 적는다.

스크립트(`scripts/repro/build_esrd_attributes.py <cohort_index.parquet> <out.parquet>`):
- `query_postgres`로 `cdm.condition_occurrence`의 `condition_source_value`와 `cdm.procedure_occurrence`의 `procedure_source_value`를 cohort의 person_id에 대해 읽는다. `where person_id = any(%(ids)s)`
- `esrd_flags`로 표를 만들어 parquet으로 쓴다.
- ESRD 사람 수와 ECG 수를 출력한다.

- [ ] **Step 4: Run test**

Run: `python3 -m pytest tests/test_repro_esrd.py -q`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/mival/repro/esrd.py scripts/repro/build_esrd_attributes.py tests/test_repro_esrd.py
git commit -m "feat: ESRD person flags from diagnosis codes, joined at evaluate as a subgroup"
```

### Task 10: 2차 study 3개, 전체 흐름 스크립트, 실행

**Files:**
- Create: `studies/repro/{vonbachmann-k,kardionet-k,ecgfounder}/study.yaml`
- Create: `scripts/repro/vs_flow.sh` (서버 `/data/mi-val/runs/vs_sample.sh`를 repo로 옮기고 `studies/$S`를 그대로 씀)
- Modify: `tests/test_repro_studies.py`

**Interfaces:**
- Consumes: Task 7, 8, 9의 필드, `run_repro.sh full`

- [ ] **Step 1: 테스트 추가 (실패 확인)**

```python
TIER2 = ("vonbachmann-k", "kardionet-k", "ecgfounder")


@pytest.mark.parametrize("paper_id", TIER2)
def test_tier2_retrieve_spec_parses(paper_id):
    from mival.stages.retrieve import RetrieveSpec
    study = load_study(ROOT / paper_id)
    RetrieveSpec.from_mapping(study.stage_spec("retrieve"))
    _Settings(study.stage_spec("evaluate"), (0.05,), 10)
```

Run: `python3 -m pytest tests/test_repro_studies.py -q` → Expected: FAIL (파일 없음)

- [ ] **Step 2: study.yaml 3개를 쓴다**

`vonbachmann-k`:

```yaml
study_id: repro-vonbachmann-k
site: dicom-miva
seed: 20261007
stages:
  retrieve:
    dsn_env: /data/mi-val/secrets/micdm.env
    schema: cdm
    modality_concept_id: 4145308
    label_source:
      kind: measurement_file
      path: /data/mi-val/datasets/mimic-iv/potassium_labels_dt.csv   # Task 7 스크립트 출력
      window_minutes: 60
      ecg_time_path: /data/mi-val/datasets/mimic-iv-ecg/machine_measurements.csv
    implausible_below: 1.5
    local_path_root: /scratch/mi-val/dicom
    require_local_file: true
    one_per_person: first
  profile: {test_fraction: 0.2, n_folds: 5, stratify_on: none, min_test_positives: 0}
  preprocess:   # studies/potassium/study.yaml와 같음
    loader: dicom
    registry: registry/potassium
    allow_upsample: false
    pad_policy: zero
    perturbation: {axes: {resample: [500], duration: [10.24]}}
  models:
    registry: registry/potassium
    cohort_sources: [mimic-iv-ecg]
    batch_size: 64
    arms:
      - {model_id: vonbachmann-k, training_mode: inference_only, label_def: value}
  evaluate:
    outcomes: {potassium: null}
    regression_cuts: ["<3.5", ">5.5"]
    regression_metrics_extra: [pearson, spearman]
    bootstrap_replicates: 2000
    figures: false
```

나머지 2개:

| study | vonbachmann-k와 다른 점 |
|---|---|
| `kardionet-k` | `one_per_person: false`<br>arm `kardionet-k-12lead`<br>`regression_cuts: [">6.5", ">=5.5"]`<br>`subgroups: [esrd]` |
| `ecgfounder` | `studies/lvef`의 preprocess와 models를 쓰되 ecgfounder arm 2개만 (`primary`, `value`)<br>retrieve는 `label_source: {kind: ecg_label_file, path: /data/mi-val/datasets/ecgfounder/LVEF_split.csv, key: study_id, value_column: LVEF, primary: "<50"}`, `split_source: {path: 같은 파일, key: study_id, column: split}`, `one_per_person: false`<br>profile `use_fixed_split: true`, `stratify_on: label_primary`<br>evaluate `outcomes: {primary: null}`, `regression_cuts: ["<50"]` |

- [ ] **Step 3: `scripts/repro/vs_flow.sh`**

서버의 `/data/mi-val/runs/vs_sample.sh` 본문을 복사한다 (위 Task 6 Step 4 이전 조사에 원문이 있다). 바꾸는 곳은 둘이다.
- `W=/data/mi-val/mi-va-vs`는 그대로 두고, `studies/$S` 대신 `$STUDY_DIR`(기본 `studies/$S`)를 쓴다. `run_repro.sh`는 `STUDY_DIR=studies/repro/$P`로 부른다. 이렇게 하면 Task 6의 `ln -sfn` 줄이 필요 없으니 지운다.
- kardionet-k는 evaluate에 `--input attributes=$ATTR`을 넣어야 한다. `vs_flow.sh`가 `$ATTR` 환경변수가 있으면 evaluate에 붙이도록 한다.

- [ ] **Step 4: 서버 준비와 샘플 실행 (각 study 1,000건)**

```bash
# 로컬: 테스트, 커밋, push
python3 -m pytest -q
git add studies/repro scripts/repro tests/test_repro_studies.py
git commit -m "feat: tier-2 paper studies (potassium x2, ECGFounder) and the repo copy of the run flow"
GIT_SSH_COMMAND="ssh -i $K -p 2022" git push -q -f ssh://ubuntu@<server-ip>/data/mi-val/mi-va HEAD:refs/heads/vs-incoming
# 서버
cd /data/mi-val/mi-va-vs && git reset -q --hard vs-incoming
PYTHONPATH=src /data/mi-val/envs/mival/bin/python scripts/repro/export_potassium_labels.py /data/mi-val/datasets/mimic-iv/potassium_labels_dt.csv
mkdir -p /data/mi-val/datasets/ecgfounder && curl -sL -o /data/mi-val/datasets/ecgfounder/LVEF.csv https://raw.githubusercontent.com/PKUDigitalHealth/ECGFounder/master/csv/LVEF.csv
/data/mi-val/envs/mival/bin/python scripts/repro/make_split_ecgfounder.py /data/mi-val/datasets/ecgfounder/LVEF.csv /data/mi-val/datasets/ecgfounder/LVEF_split.csv
for p in vonbachmann-k kardionet-k ecgfounder; do bash scripts/repro/run_repro.sh $p full 1000 || echo "$p sample FAILED"; done
```

Expected:
- 세 샘플 모두 `comparison.md`까지 생긴다.
- `make_split_ecgfounder` 출력이 조사 때 재현한 건수(train 60,329 / val 7,541 / test 7,542)와 같다.
- kardionet-k 샘플은 ESRD가 거의 없어서 그 claim이 `계산 못 함`이어도 정상이다.

샘플 전에 kardionet-k의 ESRD 표를 만든다: retrieve가 끝난 뒤 `build_esrd_attributes.py <cohort_index> /data/mi-val/datasets/mimic-iv/attributes_esrd.parquet`을 실행하고, `ATTR` 환경변수로 넘긴다.

- [ ] **Step 5: 전체 실행을 queue에 넣는다**

```bash
for p in vonbachmann-k kardionet-k ecgfounder; do
  bash scripts/repro/enqueue.sh repro-$p -- "ATTR=/data/mi-val/datasets/mimic-iv/attributes_esrd.parquet bash /data/mi-val/mi-va-vs/scripts/repro/run_repro.sh $p full all"
done
```

---

## 3차: 생존 (R4, M3, D1) + 분할 → 실행

### Task 11: 사망 추적 규칙 `death_followup` (R4)

**Files:**
- Modify: `src/mival/stages/retrieve.py`
- Test: `tests/test_repro_retrieve.py`

**Interfaces:**
- Produces: `label_source: {kind: death_followup, censor_rule: cavalab}`
  - `label_event`, `label_time_days`
  - 사건: `death_date`가 있으면 사건이다. 시간 = death − ECG 날짜, 최소 1일.
  - censoring: 방문이 없으면 그 사람의 마지막 ECG 날짜까지. 있으면 max(last_visit_end + 365일, 마지막 ECG 날짜)까지.
  - 시간이 0 이하인 censoring 행은 `label_implausible`이다.

- [ ] **Step 1: Write the failing test**

```python
def test_death_followup_cavalab_rule(tmp_path):
    d0 = date(2180, 1, 1)
    rows = [
        ecg(1, 10, d0, death=date(2185, 1, 1), last_visit=date(2181, 1, 1)),   # 5년 뒤 사망: 제한 없음
        ecg(2, 11, d0, death=d0, last_visit=d0),                                 # 같은 날 사망 -> 1일
        ecg(3, 12, d0, death=None, last_visit=date(2180, 6, 1)),                 # 마지막 퇴원 + 365
        ecg(4, 13, d0, death=None, last_visit=None),                             # 방문 없음, ECG 하나 -> 0일, 제외
        ecg(5, 14, d0, death=None, last_visit=None), ecg(6, 14, date(2180, 3, 1), death=None, last_visit=None),
    ]
    index, excluded = run(tmp_path, spec(tmp_path, "death_followup", source={"censor_rule": "cavalab"}), rows)
    got = {int(r.image_occurrence_id): (int(r.label_event), int(r.label_time_days)) for r in index.itertuples()}
    assert got[1] == (1, 1827)
    assert got[2] == (1, 1)
    assert got[3] == (0, 517)
    assert got[5] == (0, 60)                 # 같은 사람의 마지막 ECG까지
    assert excluded["4"] == "label_implausible" and excluded["6"] == "label_implausible"
```

- [ ] **Step 2: Run** → Expected: FAIL `kind 'death_followup' is not one of`

- [ ] **Step 3: Implement**

`LABEL_KINDS`에 `"death_followup"`을 더하고, `DERIVED_COLUMNS["death_followup"] = ("label_event", "label_time_days")`로 둔다.

`derive_labels`:

```python
if spec.label_kind == "death_followup":
    death = pandas.to_datetime(frame["death_date"])
    last_visit = pandas.to_datetime(frame["last_visit_end"])
    last_ecg = index.groupby(frame["person_id"]).transform("max")
    censor = last_ecg.where(last_visit.isna(), np.maximum(last_visit + pandas.Timedelta(days=365), last_ecg))
    event = death.notna()
    days = (death.where(event, censor) - index).dt.days.astype(float)
    frame["label_event"] = event.astype(float)
    frame["label_time_days"] = days.where(~event, np.maximum(days, 1.0))
    return frame
```

`exclude_derived`의 `death_followup`: `(label_time_days <= 0) & (label_event == 0)`이면 `label_implausible`, 상세 "no follow-up after the ECG".

`censor_rule`은 `cavalab`만 받고, 다른 값이면 `ValueError`를 낸다.

- [ ] **Step 4: Run** `python3 -m pytest tests/test_repro_retrieve.py -q && python3 -m pytest -q` → PASS

- [ ] **Step 5: Commit**

```bash
git commit -am "feat: death_followup label with the cavalab censoring rule (no horizon cap)"
```

### Task 12: Antolini C와 생존곡선 sidecar (M3)

**Files:**
- Modify: `src/mival/metrics/survival.py`, `src/mival/decode/outputs.py`, `src/mival/stages/models.py`, `src/mival/stages/evaluate.py`
- Test: `tests/test_repro_survival.py`

**Interfaces:**
- Produces:
  - `antolini_c(time, event, surv: np.ndarray (N, G), grid: np.ndarray (G,)) -> float`
    - S_i(t)는 grid 위 계단함수다 (t 이하 마지막 grid의 값, t < grid[0]이면 1).
    - 비교 가능 쌍: δ_i = 1이고 (T_j > T_i 또는 (T_j == T_i이고 δ_j == 0)).
    - 일치: S_i(T_i) < S_j(T_i). 동점은 0.5.
    - Task 단계에서 pycox `concordance_td(method='antolini')` 원문을 받아 동점 규칙을 확인하고, 다르면 원문을 따른다. 그 경우 `Ruling`을 남긴다.
  - decode `deephit` 출력: `{type: deephit, n_bins, max_duration_days, horizon_days, curve_grid: true}`
    - pmf = softmax(pad(φ, 0))[:, :-1], S = 1 − cumsum(pmf)
    - 컷 = linspace(0, max, n_bins)
    - 위험값(1 − S(horizon))은 `mtlr`과 같은 보간이다.
  - `curve_grid: true`이면 decode가 (N, 1 + n_bins)를 돌려준다. 0열은 위험값, 나머지는 S다. `mtlr`도 같은 옵션을 받는다.
  - models stage: 그런 카드의 survival arm은 `pred_value`에 0열을 쓴다. 나머지는 `artifacts/curves/<model_id>.parquet`(`image_occurrence_id`, `s_0..s_{G-1}`)와 `curves/<model_id>.grid.json`(컷 일수)으로 저장한다.
  - evaluate 옵션 입력 `curves`(디렉터리): 있으면 survival arm의 `antolini_c`를 계산한다. 곡선이 없는 arm은 계산하지 않고 경고를 남긴다.
  - `bootstrap_metrics`로 Antolini의 bootstrap을 끌 수 있다 (기존 설정 사용).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repro_survival.py
import numpy as np
import pytest

from mival.decode.outputs import decode_outputs
from mival.metrics.survival import antolini_c


def _step(surv_row, grid, t):
    k = np.searchsorted(grid, t, side="right") - 1
    return 1.0 if k < 0 else surv_row[k]


def brute(time, event, surv, grid):
    num = den = 0.0
    for i in range(len(time)):
        if not event[i]:
            continue
        for j in range(len(time)):
            if j == i:
                continue
            if time[j] > time[i] or (time[j] == time[i] and not event[j]):
                den += 1
                si, sj = _step(surv[i], grid, time[i]), _step(surv[j], grid, time[i])
                num += 1.0 if si < sj else 0.5 if si == sj else 0.0
    return num / den


def test_antolini_matches_bruteforce_with_ties():
    rng = np.random.default_rng(3)
    n, g = 200, 12
    grid = np.linspace(0, 1100, g)
    surv = np.sort(rng.uniform(0, 1, (n, g)), axis=1)[:, ::-1]
    surv = np.round(surv, 1)                                  # 동점을 일부러 만든다
    time = rng.choice(np.array([50, 100, 100, 300, 365, 700, 1000]), n).astype(float)
    event = rng.integers(0, 2, n)
    assert antolini_c(time, event, surv, grid) == pytest.approx(brute(time, event, surv, grid))


def test_antolini_equals_harrell_for_proportional_curves():
    from mival.metrics.survival import harrell_c
    rng = np.random.default_rng(1)
    n = 300
    risk = rng.normal(size=n)
    grid = np.linspace(0, 1000, 50)
    surv = np.exp(-np.outer(np.exp(risk), grid / 1000.0))      # 곡선이 교차하지 않음
    time = rng.integers(1, 1000, n).astype(float)
    event = rng.integers(0, 2, n)
    assert antolini_c(time, event, surv, grid) == pytest.approx(harrell_c(time, event, risk), abs=0.01)


def test_deephit_decode_with_curve():
    phi = np.array([[0.0, 0.0, 0.0], [5.0, -5.0, -5.0]])
    out = decode_outputs({"type": "deephit", "n_bins": 3, "max_duration_days": 200, "horizon_days": 100,
                          "curve_grid": True}, phi, 400)
    assert out.shape == (2, 4)
    assert np.all(np.diff(out[:, 1:], axis=1) <= 1e-12)        # S는 감소
    assert out[1, 0] > out[0, 0]                                # 첫 칸에 몰린 쪽이 위험이 큼
```

- [ ] **Step 2: Run** → Expected: FAIL `cannot import name 'antolini_c'`

- [ ] **Step 3: Implement**

```python
# src/mival/metrics/survival.py
METRICS = ("c_index", "antolini_c")


def antolini_c(time, event, surv, grid) -> float:
    """Antolini's time-dependent concordance on step survival curves over ``grid`` (pycox 'antolini').

    Events are grouped by the grid cell of their time. Partners with a time at or past the next
    cut are ranked once per cell with ``searchsorted``. Partners inside the same cell are compared
    pairwise, which keeps the exact tie rules.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64) > 0
    surv = np.asarray(surv, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    cell = np.searchsorted(grid, time, side="right") - 1          # -1: before the first cut, S = 1
    conc = comp = 0.0
    for k in np.unique(cell[event]):
        s_col = np.ones(time.size) if k < 0 else surv[:, k]
        upper = grid[k + 1] if k + 1 < grid.size else np.inf
        ev = np.flatnonzero(event & (cell == k))
        far = np.flatnonzero(time >= upper)
        far_s = np.sort(s_col[far])
        si = s_col[ev]
        greater = far_s.size - np.searchsorted(far_s, si, side="right")
        equal = np.searchsorted(far_s, si, side="right") - np.searchsorted(far_s, si, side="left")
        conc += greater.sum() + 0.5 * equal.sum()
        comp += far_s.size * ev.size
        near = np.flatnonzero(cell == k)
        ti, tj = time[ev][:, None], time[near][None, :]
        ok = (tj > ti) | ((tj == ti) & ~event[near][None, :])
        a, b = s_col[ev][:, None], s_col[near][None, :]
        conc += ((a < b) & ok).sum() + 0.5 * ((a == b) & ok).sum()
        comp += ok.sum()
    return float(conc / comp) if comp else float("nan")
```

`time >= upper`인 상대는 모두 T_j > T_i이므로 비교 가능하다. 같은 칸 안에서 `ok`는 자기 자신을 뺀다: `tj > ti`가 거짓이고, 자기 자신은 사건이라 `~event`도 거짓이다.

```python
# src/mival/decode/outputs.py
DECODED_TYPES = (..., "deephit")


def deephit_survival(phi):
    phi = np.asarray(phi, dtype=np.float64)
    padded = np.concatenate([phi, np.zeros((phi.shape[0], 1))], axis=1)
    padded = padded - padded.max(axis=1, keepdims=True)
    pmf = np.exp(padded)
    pmf = pmf / pmf.sum(axis=1, keepdims=True)
    return 1.0 - np.cumsum(pmf[:, :-1], axis=1)
```

`decode_outputs`의 `mtlr`/`deephit` 분기:
- `surv = (mtlr_survival if kind == "mtlr" else deephit_survival)(flat[:, :n_bins])`
- `cuts = linspace(0, max, n_bins)`, `values = 1 - interp(horizon)`
- `output.get("curve_grid")`이면 `transform`을 적용한 values와 surv를 `np.column_stack([values, surv])`로 붙여 반환한다.

models stage:
- `_predict` 호출 전에 `curve = card.output.get("curve_grid")`를 본다.
- 참이면 `column=None`으로 받은 (N, 1+G)에서 0열을 예측값으로 쓴다.
- 나머지 열은 `_write_curves(ctx, card, records, matrix[:, 1:])`로 `artifacts/curves/`에 저장한다. 컷은 `np.linspace(0, max_duration_days, n_bins)`이고 grid.json에 쓴다.
- `_predict` 안의 크기 검사(`probs.size != len(window)`)는 curve 모드에서 `probs.shape[0]`으로 비교한다.

evaluate:
- `run()`에서 `ctx.inputs.get("curves")`가 있으면 model_id별 곡선을 읽어 `settings.curves = {model_id: (ids, surv, grid)}`로 둔다.
- `_survival_rows`에서 그 arm의 곡선이 있으면 `usable`의 image_occurrence_id 순서로 곡선을 맞춘다. statistic에 `antolini_c(time[idx], event[idx], surv[idx], grid)`를 더한다.

- [ ] **Step 4: Run** `python3 -m pytest tests/test_repro_survival.py -q && python3 -m pytest -q` → PASS

- [ ] **Step 5: Commit**

```bash
git commit -am "feat: Antolini concordance from stored survival curves; DeepHit decoding"
```

### Task 13: cavalab MIMIC DeepHit 카드와 wrapper (D1)

**Files:**
- Modify: `registry/wrappers/mival_wrap_cavalab.py`
- Create: `registry/repro/cavalab-deephit-mimic-resnet.json`, `registry/repro/cavalab-deephit-mimic-inception.json`
- Create: `scripts/repro/make_split_cavalab_mimic.py`
- Test: `tests/test_repro_cavalab.py`

**Interfaces:**
- Produces:
  - wrapper `DeepSurvCode15(..., pad=648)`에 `pad` kwarg를 더한다. 기본은 지금 값이다.
  - `InceptionCode15`는 cavalab `InceptionTime_Support`를 쓰는 builder이고, 구조 kwargs는 checkpoint에서 읽는다.
  - 카드 입력: 400 Hz, `duration_s: 10`, `crop_anchor: center`. 10초 5,000 → 4,000 샘플은 preprocess가 400 Hz 리샘플로 만든다. pad 48.
  - 카드 출력: `{type: deephit, n_bins: <ckpt>, max_duration_days: <ckpt>, horizon_days: 365, curve_grid: true}`
  - `make_split_cavalab_mimic.py <out.csv>`: 저자 `MIMIC_IV_PreProcess_Jan2025.py`와 학습 코드의 split(사람 단위 64/16/20, test seed 12345)을 그대로 옮긴다. 출력 열은 `subject_id, split`이다.

- [ ] **Step 1: 서버에서 weight와 저자 코드 확인 (조사 단계, 결과를 카드에 기록)**

```bash
ssh ... 'cd /opt/ecg-survival-benchmark && git log --oneline -1; ls; \
  mkdir -p /data/mi-val/models/cavalab-deephit-mimic && cd /data/mi-val/models/cavalab-deephit-mimic && \
  ls; /data/mi-val/envs/ecgfounder/bin/python -c "
import torch,sys
for f in sys.argv[1:]:
    c=torch.load(f,map_location=\"cpu\",weights_only=False); print(f, list(c.keys()))
    print({k:v for k,v in c.items() if k not in (\"model_state_dict\",\"optimizer_state_dict\")})
" *.pt'
```

- Zenodo 16877773의 MIMIC 파일 2개가 서버에 없으면 받는다. 새 인증이 필요 없는 공개 레코드다.
- checkpoint의 `NM`, `NS`, `NT`, cuts(또는 max_duration), 구조 하이퍼파라미터를 기록한다.
- `max_duration_days`와 `n_bins`를 카드에 적는다.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_repro_cavalab.py
import json
from pathlib import Path

import pytest

from mival.modelcard import load_card

REG = Path(__file__).resolve().parents[1] / "registry" / "repro"


@pytest.mark.parametrize("name", ["cavalab-deephit-mimic-resnet", "cavalab-deephit-mimic-inception"])
def test_card_loads_and_declares_curves(name):
    card = load_card(REG / f"{name}.json")
    assert card.output["type"] == "deephit" and card.output["curve_grid"] is True
    assert card.input_contract.sampling_rate_hz == 400 and card.input_contract.duration_s == 10
    assert "mimic-iv-ecg" in json.loads((REG / f"{name}.json").read_text())["x-mival"]["pretraining_corpora"]


def test_wrapper_pad_default_unchanged():
    torch = pytest.importorskip("torch")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "registry" / "wrappers"))
    import mival_wrap_cavalab as w
    assert w.PAD == 648
```

`pretraining_corpora`에 `mimic-iv-ecg`가 있으면 models stage의 오염 검사가 이 arm을 contaminated로 표시한다. 같은 데이터로 학습한 모델이라 맞는 표시이고, 막지는 않는다 (스펙 §3.5).

- [ ] **Step 3: Run** → Expected: FAIL (카드 없음)

- [ ] **Step 4: Implement**

- wrapper:
  - `__init__`에 `pad=PAD`를 받아 `self.pad`로 두고, `forward`에서 `F.pad(x, (self.pad, self.pad))`를 쓴다.
  - `InceptionCode15`는 저자 `InceptionTime_Support`의 모델 생성 함수를 같은 방식(`sys.path`)으로 불러 만든다. 정규화와 pad 처리는 같다.
  - `convert`가 InceptionTime checkpoint도 받도록 키를 확인한다.
- 카드 2개: `cavalab-mtlr-code15.json`을 본떠 쓴다. `weights`의 `derived_from` sha256은 Step 1에서 계산하고, `notes`에 Step 1의 확인 사항을 적는다.
- `make_split_cavalab_mimic.py`: 저자 코드의 split 함수를 그대로 옮긴다 (`numpy.random.default_rng(12345)` 또는 저자가 쓴 RNG, 순서까지 동일). 출력 test 사람 수를 논문 Table과 대조해 출력한다.

- [ ] **Step 5: Run tests, commit**

```bash
python3 -m pytest tests/test_repro_cavalab.py -q && python3 -m pytest -q
git add registry/repro registry/wrappers/mival_wrap_cavalab.py scripts/repro/make_split_cavalab_mimic.py tests/test_repro_cavalab.py
git commit -m "feat: cavalab MIMIC-trained DeepHit cards and the authors' MIMIC split"
```

서버 `/data/mi-val/models/_wrappers`에 wrapper 사본을 갱신한다 (README 규칙).

### Task 14: 3차 study 4개와 실행

**Files:**
- Create: `studies/repro/{cavalab-code15,cavalab-mimic,semiseg,openecg}/study.yaml`
- Modify: `tests/test_repro_studies.py`, `scripts/repro/vs_flow.sh` (models가 만든 `curves`를 evaluate에 `--input curves=`로 넘김)

- [ ] **Step 1: 테스트 추가 (실패 확인)**: TIER3 4개가 로드되는지 확인하고 retrieve·evaluate 설정을 파싱한다. Expected: FAIL (파일 없음)

- [ ] **Step 2: study.yaml**

| study | 설정 |
|---|---|
| `cavalab-code15` | 원본 `studies/mortality`에서 cavalab arm 2개만<br>evaluate `outcomes: {death_1y: null}`, `horizons_days: [365]`<br>MTLR 카드에 `curve_grid: true`를 켠 사본을 `registry/repro/`에 둔다 (원본 카드는 그대로)<br>DeepSurv는 곡선이 없으니 `antolini_c` claim에 `different_model` deviation과 "PH 모델이라 Harrell C로 대신"을 적고, claim metric을 `c_index`로 둔다 |
| `cavalab-mimic` | retrieve `label_source: {kind: death_followup, censor_rule: cavalab}`, `one_per_person: false`, `split_source: {path: /data/mi-val/datasets/cavalab/mimic_split.csv, key: subject_id, column: split}`<br>profile `use_fixed_split: true`, `stratify_on: label_event`<br>preprocess 400 Hz, 10 s<br>models `registry/repro` (DeepHit 2개)<br>evaluate `horizons_days: [365, 1825]`, `bootstrap_replicates: 200` (Antolini 비용) |
| `semiseg` | 원본 `studies/delineation`에서 SemiSeg arm 6개만, evaluate는 그대로 |
| `openecg` | 원본에서 OpenECG arm 3개만. claim은 전부 `comparable: false`이고, 간격 MAE는 참고값으로만 나온다 |

semiseg와 openecg는 코호트가 원본과 같다. 그래서 `run_repro.sh <id> eval-only delineation`으로 돈다.

- [ ] **Step 3: 샘플 실행 (cavalab 2개)**: Task 10 Step 4와 같은 방식으로 1,000건을 돌린다.
  - Expected: `curves/*.parquet`이 생긴다.
  - Expected: metrics_long에 `antolini_c`가 있다.
  - Expected: cavalab-mimic의 샘플 test 건수는 split 파일 비율과 맞는다.

- [ ] **Step 4: Commit, push, enqueue (4개)**

```bash
git add studies/repro registry/repro scripts/repro tests/test_repro_studies.py
git commit -m "feat: tier-3 paper studies (cavalab x2, SemiSeg, OpenECG)"
# push 후 서버에서:
bash scripts/repro/enqueue.sh repro-cavalab-code15 -- bash /data/mi-val/mi-va-vs/scripts/repro/run_repro.sh cavalab-code15 full all
bash scripts/repro/enqueue.sh repro-cavalab-mimic -- bash /data/mi-val/mi-va-vs/scripts/repro/run_repro.sh cavalab-mimic full all
bash scripts/repro/enqueue.sh repro-semiseg -- bash /data/mi-val/mi-va-vs/scripts/repro/run_repro.sh semiseg eval-only delineation
bash scripts/repro/enqueue.sh repro-openecg -- bash /data/mi-val/mi-va-vs/scripts/repro/run_repro.sh openecg eval-only delineation
```

---

## 4차: Lima 생존 (M5) → 실행

### Task 15: Cox와 파생 생존 분석

**Files:**
- Create: `src/mival/repro/cox.py`, `src/mival/repro/derived_survival.py`
- Create: `studies/repro/lima-survival/study.yaml`
- Test: `tests/test_repro_cox.py`

**Interfaces:**
- Produces:
  - `fit_cox(time, event, X) -> CoxFit`
    - `CoxFit`: `beta, se, hr, hr_ci (k, 2), loglik`
    - Breslow 동점, Newton-Raphson, 수렴 1e-9 또는 50회
  - `derived_survival.main(study_dir, age_predictions, mortality_cohort, out)`
    - Lima 나이 예측(ecg-age models predictions)과 사망 라벨(`death_followup` 또는 `death_within` cohort_index)을 image_occurrence_id로 붙인다.
    - `gap8 = pred − true > 8`
    - 행 1: Cox(gap8 + 나이 + 성별)로 `hr`(gap8)과 CI
    - 행 2: Cox(나이 + 성별 + ECG-age)를 dev에서 적합하고, test에서 위험 점수의 `auroc_horizon@365`를 계산한다. 이름은 `auc_cox@365`.
    - metrics_long과 같은 열로 parquet을 쓴다.
  - study.yaml은 실행 정의가 아니라 입력 경로와 공변량을 적은 `derived:` 블록이다:

    ```yaml
    derived:
      predictions_from: ecg-age
      labels_from: mortality
      model_id: lima-ecg-age
      gap_years: 8
    ```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repro_cox.py
import numpy as np
import pytest

from mival.repro.cox import fit_cox


def test_cox_recovers_known_hazard_ratio():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.integers(0, 2, n).astype(float)
    z = rng.normal(size=n)
    hazard = np.exp(np.log(1.8) * x + 0.3 * z)
    t = rng.exponential(1.0 / hazard)
    c = rng.exponential(2.0, n)
    time, event = np.minimum(t, c), (t <= c).astype(float)
    fit = fit_cox(time, event, np.column_stack([x, z]))
    assert fit.hr[0] == pytest.approx(1.8, rel=0.1)
    assert fit.hr_ci[0, 0] < 1.8 < fit.hr_ci[0, 1]
    assert fit.beta[1] == pytest.approx(0.3, abs=0.06)


def test_cox_handles_tied_times():
    time = np.array([1, 1, 2, 2, 3, 3, 4, 5], dtype=float)
    event = np.array([1, 1, 1, 0, 1, 0, 1, 0], dtype=float)
    x = np.array([[1], [0], [1], [0], [1], [0], [0], [0]], dtype=float)
    fit = fit_cox(time, event, x)
    assert np.isfinite(fit.beta).all() and fit.beta[0] > 0
```

- [ ] **Step 2: Run** → Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement** `fit_cox` (Breslow partial likelihood):
  - 시간 내림차순으로 정렬한다.
  - 누적 위험집합 합 `S0 = cumsum(exp(Xβ))`, `S1 = cumsum(exp(Xβ) X)`, `S2 = cumsum(exp(Xβ) X Xᵀ)`을 구한다. 동점은 같은 시간 그룹의 마지막 누적값을 쓴다.
  - gradient = Σ_events (x_i − S1/S0)
  - Hessian = −Σ_events (S2/S0 − (S1/S0)(S1/S0)ᵀ)
  - Newton step을 하되, step-halving으로 loglik이 줄지 않게 한다.
  - se = sqrt(diag(inv(−H))), hr = exp β, CI = exp(β ± 1.96 se)

  `derived_survival.py`는 위 Interfaces대로 쓴다. 분할은 mortality cohort_split을 쓴다. 출력 parquet의 열은 `METRICS_LONG_COLUMNS`이고, `label_def`는 `survival`, `metric`은 `hr`과 `auc_cox@365`다. `hr`의 CI는 Cox CI이고, `auc_cox@365`의 CI는 person bootstrap 2,000회다.

  `category_of`가 `hr`, `auc_cox`를 알아야 비교 표가 만들어지는 것은 아니다. compare는 category를 쓰지 않는다. 그래서 metrics 레지스트리는 바꾸지 않는다.

- [ ] **Step 4: Run** `python3 -m pytest tests/test_repro_cox.py -q && python3 -m pytest -q` → PASS

- [ ] **Step 5: Commit, push, enqueue**

```bash
git add src/mival/repro/cox.py src/mival/repro/derived_survival.py studies/repro/lima-survival tests/test_repro_cox.py
git commit -m "feat: numpy Cox and the Lima ECG-age mortality analysis from existing predictions"
# 서버:
bash scripts/repro/enqueue.sh repro-lima-survival -- "cd /data/mi-val/mi-va-vs && PYTHONPATH=src /data/mi-val/envs/mival/bin/python -m mival.repro.derived_survival studies/repro/lima-survival && PYTHONPATH=src /data/mi-val/envs/mival/bin/python -m mival.repro.compare studies/repro/lima-survival studies/repro/lima-survival/results/derived_metrics.parquet --out studies/repro/lima-survival/results"
```

라벨: 사망 추적은 Lima 논문 방식(전체 추적)에 맞춘다. 그래서 `labels_from`은 `repro-cavalab-mimic`의 retrieve cohort_index(`death_followup`, 사람당 ECG 전부)를 쓰고, ecg-age 예측이 있는 ECG와 교집합만 쓴다. 이 선택은 Task 2의 lima-survival notes에 `our_choice`로 적는다.

---

## B: 자동 추출과 채점

### Task 16: 지침서, 스키마, 채점기

**Files:**
- Create: `docs/repro/extraction-guide.md`, `docs/repro/extraction-schema.json`, `src/mival/repro/score.py`
- Test: `tests/test_repro_score.py`

**Interfaces:**
- Produces:
  - `score(auto_dir: Path, gold_dir: Path, source_text: Path) -> dict`
    - 키: `fields_correct, fields_wrong, fields_missing, hallucinated, claims_exact, claims_total, quotes_valid, quotes_total, deviations_found, deviations_total`
  - 필드 비교 규칙:
    - study.yaml은 평탄화한 경로(`stages.retrieve.label_source.window_minutes`)로 비교한다.
    - 숫자는 같으면 일치, 문자열은 공백과 대소문자를 무시하고 일치를 본다.
    - 정답에 있고 자동에 없으면 누락이다.
    - 정답 evidence가 `not_stated`인데 자동이 값을 채우면 지어낸 정보다.
  - claims 비교: (model_id, label_def, metric, subgroup)로 짝짓고 value가 정확히 같으면 일치로 센다.
  - 근거 유효성: 공백을 정규화한 뒤 자동 evidence의 `quote`가 `source_text`(Task 2의 pdftotext 결과)의 부분 문자열인지 본다.
  - deviation 찾기: 정답 deviation의 (field, kind)와 같은 것이 자동에 있으면 찾은 것으로 센다.
  - CLI: `python -m mival.repro.score <auto_dir> <gold_dir> <source.txt>` → JSON 출력

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repro_score.py
from pathlib import Path

from mival.repro.score import score


def make(dir_, study, claims, evidence):
    dir_.mkdir(parents=True)
    (dir_ / "study.yaml").write_text(study, encoding="utf-8")
    (dir_ / "claims.yaml").write_text(claims, encoding="utf-8")
    (dir_ / "evidence.yaml").write_text(evidence, encoding="utf-8")


STUDY = "study_id: x\nsite: s\nstages:\n  retrieve:\n    label_source: {kind: measurement_file, window_minutes: %s}\n    one_per_person: %s\n"
CLAIMS = ("paper: {id: p}\nclaims:\n  - {id: c1, model_id: m, label_def: value, metric: mae, value: %s}\n"
          "deviations:\n  - {field: f, kind: our_choice, why: w}\n")
EVID = ("stages.retrieve.one_per_person: {value: x, quote: \"%s\", where: p1, status: %s, reviewed_by: null}\n"
        "claims.c1.value: {value: 0.5, quote: \"MAE 0.527\", where: t2, status: stated, reviewed_by: null}\n")


def test_score_counts(tmp_path):
    make(tmp_path / "gold", STUDY % (60, "first"), CLAIMS % 0.527, EVID % ("", "not_stated"))
    make(tmp_path / "auto", STUDY % (60, "true"), CLAIMS % 0.527, EVID % ("first ECG only", "stated"))
    src = tmp_path / "src.txt"
    src.write_text("... MAE  0.527 in the test set ...", encoding="utf-8")
    out = score(tmp_path / "auto", tmp_path / "gold", src)
    assert out["fields_correct"] >= 2 and out["fields_wrong"] == 1      # one_per_person 다름
    assert out["hallucinated"] == 1                                      # 정답 not_stated에 값을 채움
    assert (out["claims_exact"], out["claims_total"]) == (1, 1)
    assert (out["quotes_valid"], out["quotes_total"]) == (1, 2)          # 두 번째 인용은 원문에 없음
    assert (out["deviations_found"], out["deviations_total"]) == (1, 1)
```

- [ ] **Step 2: Run** → Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`score.py`는 위 규칙대로 쓴다. 평탄화는 다음과 같다.

```python
def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        out.update(flatten(v, key + ".") if isinstance(v, dict) else {key: v})
    return out
```

study 비교에서 `study_id`와 `site`는 뺀다.

`docs/repro/extraction-guide.md`에 들어갈 내용 (정답 파일을 인용하지 않는다):
1. 입력과 출력 파일 4개
2. study.yaml에서 쓸 수 있는 필드 목록 (이 계획에서 만든 필드 포함)과 각 필드를 논문의 무엇에서 찾는지
3. claim metric 이름 규칙
4. deviation 5종의 뜻과 판단 기준
5. "논문에 없으면 `not_stated`, 추측 금지" 규칙과 그 이유
6. 인용은 원문 그대로 짧게 쓰고, 위치(쪽, 표, 절, 코드 파일:줄)를 적는다
7. 같은 논문에 study가 여러 개일 수 있다 (예: 회귀와 생존)

`extraction-schema.json`은 claims.yaml과 evidence.yaml의 JSON Schema다. Task 1 loader 규칙을 그대로 옮긴다.

- [ ] **Step 4: Run** `python3 -m pytest tests/test_repro_score.py -q && python3 -m pytest -q` → PASS

- [ ] **Step 5: Commit**

```bash
git add docs/repro/extraction-guide.md docs/repro/extraction-schema.json src/mival/repro/score.py tests/test_repro_score.py
git commit -m "feat: extraction guide, schema and field-level scorer for automated study extraction"
```

### Task 17: 개발용 튜닝과 평가용 추출

**Files:**
- Create: `docs/repro/extraction-runs/<set>/<paper_id>/run{1,2,3}/...` (자동 추출 결과), `docs/repro/extraction-report.md`

- [ ] **Step 1: 개발용 7건 추출**
  - 대상: heartwise, lima-age(lima-survival 포함, 같은 논문), ai-ntprobnp, vonbachmann-k, cavalab-code15, semiseg
  - 논문마다 서브에이전트 하나를 띄운다.
  - 프롬프트에는 `extraction-guide.md`, `extraction-schema.json`, 원문 경로(`docs/repro/sources/<id>/`), repo URL만 준다.
  - 작업 디렉터리는 `studies/repro`가 없는 git worktree(`git worktree add ../mi-va-extract HEAD` 후 `rm -rf studies/repro docs/repro/gold-review.md`)로 한다.
  - 출력은 `docs/repro/extraction-runs/dev/<id>/run1/`에 둔다.

- [ ] **Step 2: 채점과 지침서 수정**
  - `python -m mival.repro.score`로 채점한다.
  - 틀린 필드마다 지침서의 어느 규칙이 부족했는지 고친다. 정답 값을 지침서에 쓰지 않는다.
  - 지침서를 고칠 때마다 커밋하고, 개발용을 다시 돌린다 (최대 3회).

- [ ] **Step 3: 평가용 추출 (지침서 고정)**
  - 대상: ecgfounder, singstad-age, kardionet-k, cavalab-mimic, openecg, 그리고 정답 없는 HeartWise <50 head와 xECG
  - 논문마다 3회 돌린다.
  - 정답 없는 2건은 채점하지 않고 `gold-review.md`의 "사람 판정" 절에 결과를 붙인다.

- [ ] **Step 4: 리포트**
  - `extraction-report.md`에 넣을 것:
    - 세트별 필드 일치율, 지어낸 정보 수, 근거 유효율, deviation 찾은 비율, 3회 일관성(3회 모두 같은 값인 필드 비율)
    - 맨 위에 "정답 미검토 (검토 N / 전체 M)" 문구와, 같은 모델이 정답과 추출을 함께 해서 생기는 위험을 적는다.
  - (선택) 평가용 2건(kardionet-k, ecgfounder)은 자동 study를 `studies/repro-auto/<id>`로 서버에서 돌려 판정이 정답 study와 같은지 본다.

- [ ] **Step 5: Commit**

```bash
git add docs/repro/extraction-guide.md docs/repro/extraction-runs docs/repro/extraction-report.md docs/repro/gold-review.md
git commit -m "docs: automated extraction runs and scores (dev 7, eval 5 + 2 unscored)"
```

### Task 18: 결과 모으기

**Files:**
- Create: `scripts/repro/make_summary.py`, `docs/repro/summary.xlsx`, `docs/repro/results.md`
- Modify: `docs/mival-model-results.html` (논문 방식 재현 열 추가)

- [ ] **Step 1**: 서버 worktree의 `studies/repro/*/results/comparison.csv`를 로컬 `studies/repro/*/results/`로 가져온다 (`scp -P 2022 -i $K -r`).

- [ ] **Step 2**: `make_summary.py`가 모든 comparison.csv를 합쳐 `summary.xlsx`를 만든다.
  - 서식: Pretendard 11, 검은 얇은 테두리, 머리행 F4F6F7. `scratchpad/make_xlsx.py`와 같은 서식이다.
  - 첫 행에 "정답 미검토" 문구를 둔다.
  - `results.md`는 판정별 개수와 "차이 있음" 항목의 조건 차이를 정리한다.

- [ ] **Step 3**: `docs/mival-model-results.html`의 표에 "논문 방식 재현 (판정)" 열을 더하고 같은 artifact URL로 다시 publish한다.

- [ ] **Step 4: Commit**

```bash
git add scripts/repro/make_summary.py docs/repro/summary.xlsx docs/repro/results.md docs/mival-model-results.html studies/repro/*/results
git commit -m "docs: paper-protocol reproduction summary across the twelve gold studies"
```

---

## 실행 순서 요약

| 순서 | Task | 끝나면 |
|---|---|---|
| 0 | 1 → 2 | 정답 12개 커밋 |
| 1 | 3 → 4 → 5 → 6 | queue에 1차 4건 (evaluate만, 수 분) |
| 2 | 7 → 8 → 9 → 10 | queue에 2차 3건 (ECGFounder probe 포함, 수 시간) |
| 3 | 11 → 12 → 13 → 14 | queue에 3차 4건 |
| 4 | 15 | queue에 Lima 생존 |
| B | 16 → 17 | 서버 실행과 무관하게 진행 |
| 끝 | 18 | 모든 queue가 `.done`인 뒤 |

queue가 도는 동안 다음 차수 구현을 한다. 실패한 job(`.failed`)은 로그를 보고 고친 뒤 다시 enqueue한다. 같은 오류가 3회 나면 멈추고 보고한다.
