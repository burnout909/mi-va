# Plan 5 — Evaluation stage

**Spec:** `docs/spec/2026-08-14-research-design.md` §4.5 (읽을 것). §3.4도 해당.

**목표:** prediction을 읽어 범주 4개 × 축 6개로 지표를 산출하고 `metrics_long.parquet`과 figure를 낸다.

## 소유 파일 — 이것만 만들거나 고친다

- Modify: `src/mival/stages/evaluate.py` (현재 계약만 박힌 stub)
- Create: `src/mival/metrics/__init__.py`, `src/mival/metrics/{discrimination,calibration,utility}.py`, `src/mival/metrics/bootstrap.py`, `src/mival/figures.py`
- Create: `tests/test_metrics_*.py`, `tests/test_bootstrap.py`, `tests/test_stage_evaluate.py`

다른 파일은 건드리지 않는다. `src/mival/pipeline/`은 완성된 읽기 전용 코드다. `src/mival/stages/{preprocess,models}.py`는 **다른 에이전트가 지금 동시에 작성 중**이다 — 읽되 절대 수정하지 않는다.

## 이미 있는 것

```python
from mival.pipeline.runkey import AXES, REPORT_AXES, RunKey
from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.tables import read_table, write_table
from mival.stages.models import PREDICTION_COLUMNS   # 읽기 전용 import
```

`ctx.layout.artifact("...")`로만 경로를 만든다.

## 고정된 계약 (`src/mival/stages/evaluate.py` 상단에 이미 있음 — 값 바꾸지 말 것)

`METRICS_LONG_COLUMNS`, `CATEGORIES`, `BOOTSTRAP_REPLICATES = 2000`, `BOOTSTRAP_UNIT = "person_id"`, `SUBGROUP_SUPPRESS_MIN_EVENTS = 10`, 산출물 이름.

## 범주 4개 (spec §4.5 표 그대로)

| # | 범주 | 지표 |
|---|---|---|
| 1 | `discrimination` | AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, MCC, balanced accuracy, discrimination slope |
| 2 | `calibration` | CITL(mean) → intercept/slope(weak) → flexible curve(moderate), Brier + scaled Brier + Murphy 분해 |
| 3 | `clinical_utility` | decision curve, net benefit vs treat-all / treat-none |
| 4 | `interpretation` | attribution, lead/time region (정성) |

**"Overall performance"를 별도 층으로 두지 않는다.** 근거: Brier는 독립 성질이 아니라 합성량이며 Murphy 분해에 따라 `Brier = reliability − resolution + uncertainty`다. reliability는 calibration에, resolution은 discrimination에 대응하므로 Brier는 calibration 층에 두고 분해값을 함께 보고한다. STEMI처럼 유병률이 낮으면 Brier가 uncertainty 항에 지배되므로 **scaled Brier(IPA)를 반드시 병기**한다.

clinical utility를 별도 범주로 두는 근거: net benefit은 순위 기반이 아니어서 discrimination이 아니고, 예측–관측 일치가 아니어서 calibration이 아니다. "이 threshold에서 이 모델을 쓰는 것이 안 쓰는 것보다 나은가"라는 결정이론적 질문이다. STEMI의 임상적으로 타당한 threshold 대역은 낮은 쪽(약 1–10%)이다.

범주 4(interpretation)는 이번 범위에서 **스키마 자리만 확보**하고 계산은 구현하지 않아도 된다. 그렇게 했다면 보고서에 명시할 것.

## 불확실성과 비교 (spec §4.5)

- **Bootstrap**: person-level **stratified** bootstrap 2,000회, percentile CI. 환자당 ECG가 복수일 수 있으므로 **재표집 단위는 반드시 person이다.** ECG 단위 재표집은 CI를 과소추정한다 — 이 성질을 테스트로 고정한다(클러스터가 있는 합성 데이터에서 person 단위 CI가 record 단위 CI보다 넓어야 한다).
- **모델 간 비교**: **DeLong은 clustered data에 부적절하므로 쓰지 않는다.** primary는 **paired bootstrap of difference**(person 단위 재표집). net benefit 차이도 동일.
- **다중비교**: §2.5의 사전 지정 2쌍만 confirmatory, 나머지는 exploratory로 명시. Bonferroni 남발 대신 사전 지정 + CI 보고.
- **Subgroup 억제**: `n_events < 10`이면 계산은 하되 `suppressed=true`로 표시하고 결론에 사용하지 않는다.

bootstrap은 seed로 재현 가능해야 한다. `numpy.random.default_rng(seed)`를 쓰고 전역 RNG는 건드리지 않는다.

## 축 (spec §4.5)

`site`, `subgroup`, `perturbation`, `label 정의`, `outcome`, `split`.

**평가 코드는 "범주 4개를 계산하는 함수" 하나이며 나머지는 groupby다.** 축이 늘어도 지표 코드가 바뀌지 않아야 한다 — 축 이름이 지표 함수 안에 등장하면 실패다. `subgroup`과 `outcome`은 `REPORT_AXES`이지 `AXES`가 아니라는 점에 주의(§3.4): 예측을 다시 돌리지 않고 기존 prediction을 다시 나누거나 다른 label을 붙이면 된다.

## Figure (전부 코드 산출물)

STARD flow(exclusion ledger에서 자동 생성 — `mival.pipeline.ledger.read_exclusions`), ROC/PR, flexible calibration curve, decision curve, perturbation 열화 곡선, subgroup forest plot, 순위 역전 도표(F1).

matplotlib은 lazy import하고 `Agg` 백엔드를 명시적으로 쓴다(헤드리스 인스턴스).

## 하지 말 것

- 지표 함수 안에 축 이름·모델 이름·subgroup 이름 하드코딩
- long format을 wide로 바꾸기. 축이 늘어도 스키마가 불변인 것이 long format의 존재 이유다
- scikit-learn이 이미 정확히 제공하는 것을 재구현하기 (`roc_auc_score` 등은 써도 된다). 반대로 sklearn에 없는 것(Murphy 분해, net benefit, flexible calibration, paired bootstrap)은 직접 짜고 **알려진 값으로 검증**한다
- DeLong 검정
- `tensorflow`를 import하는 코드 실행 — **이 머신에서 프로세스를 hard-abort시킨다.**

## 검증

`python3 -m pytest tests/ -q` 전부 통과. 기존 162 passed / 11 skipped를 깨뜨리지 않는다.

지표는 **손으로 계산 가능한 작은 입력에 대한 기대값**으로 테스트한다. "결과가 0과 1 사이다" 같은 테스트는 아무것도 검증하지 않는다. 최소한:
- AUROC가 알려진 작은 예제에서 정확한 값을 낸다 (완전 분리 → 1.0, 무작위 → 0.5 포함)
- `Brier = reliability − resolution + uncertainty`가 수치적으로 성립한다
- scaled Brier가 baseline 모델에 대해 0이다
- net benefit이 treat-all/treat-none과 알려진 지점에서 일치한다
- person 단위 bootstrap CI가 클러스터 데이터에서 record 단위보다 넓다
- 같은 seed면 bootstrap 결과가 재현된다
- `n_events < 10`인 subgroup이 `suppressed=true`로 표시되되 값은 계산된다
- 출력 컬럼이 `METRICS_LONG_COLUMNS`와 정확히 일치한다

## 보고

끝나면 `.superpowers/notes/plan5-report.md`에 다음을 쓴다: 만든 파일, 테스트 결과 한 줄, 내린 설계 결정과 근거, 구현하지 않고 남긴 것, 확신이 없는 부분. **커밋하지 말 것** — 세 stage가 같은 브랜치에서 동시에 작업 중이라 동시 커밋은 git index.lock에서 깨진다. 파일만 남기고 끝낸다. 커밋은 컨트롤러가 검토 후 한다.
