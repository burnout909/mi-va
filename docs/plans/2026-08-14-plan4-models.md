# Plan 4 — Models stage

**Spec:** `docs/spec/2026-08-14-research-design.md` §4.4 (읽을 것). §3.4, §3.5, §3.6도 해당.

**목표:** preprocess tensor에 모델을 돌려 `predictions/<run_key>.parquet`을 낸다. training mode 4종을 지원하고, threshold를 정책 3종으로 산출하며, leakage/contamination gate를 건다.

## 소유 파일 — 이것만 만들거나 고친다

- Modify: `src/mival/stages/models.py` (현재 계약만 박힌 stub)
- Create: `src/mival/threshold.py`, `src/mival/gates.py`
- Create: `tests/test_threshold.py`, `tests/test_gates.py`, `tests/test_stage_models.py`

다른 파일은 건드리지 않는다. `src/mival/{adapters,compiler,modelcard,ops,signal}.py`와 `src/mival/pipeline/`은 완성된 읽기 전용 코드다. `src/mival/stages/{preprocess,evaluate}.py`는 **다른 에이전트가 지금 동시에 작성 중**이다 — 읽되 절대 수정하지 않는다.

## 이미 있는 것 (읽고 쓸 것, 다시 만들지 말 것)

```python
from mival.adapters import get_adapter           # 문자열 lazy lookup. torch/keras를 직접 import하지 말 것
from mival.modelcard import load_card, load_registry
from mival.pipeline.runkey import RunKey         # 8축, to_string()이 파일명
from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.tables import read_table, write_table
from mival.stages.preprocess import PREPROCESS_INDEX_COLUMNS   # 읽기 전용 import
```

`ctx.layout.artifact("...")`로만 경로를 만든다.

## 고정된 계약 (`src/mival/stages/models.py` 상단에 이미 있음 — 값 바꾸지 말 것)

- `PREDICTION_COLUMNS` — Plan 5가 이 스키마를 읽는다
- 산출물: `artifacts/predictions/<run_key>.parquet` (`RunKey.to_string()`), `artifacts/train_log.jsonl`
- `TRAINING_MODES`, `THRESHOLD_POLICIES`, `PRIMARY_THRESHOLD_POLICY`, `REASON_CODES`
- `required_inputs() == ("preprocess_index", "cohort_split")`

## Training mode (spec §4.4 표 그대로)

| mode | 학습 대상 | dev 사용 | 필수 기록 |
|---|---|---|---|
| `inference_only` | 없음 | 미사용 | weights checksum |
| `linear_probe` | 최종 head | dev + 내부 CV | feature_layer, head hparams |
| `partial_unfreeze` | 지정 group | dev + 내부 CV | unfreeze group 목록 |
| `full_finetune` | 전체 | dev + 내부 CV | 전체 hparams |

**하이퍼파라미터와 early stopping은 dev 내부 CV로만 결정한다. test는 마지막 한 번만 접촉하며 접촉 사실을 manifest에 기록한다.** test set에 두 번 접촉하는 코드 경로가 존재하면 안 된다.

adapter interface(§4.4)는 `load / forward / features / trainable_groups / fit / attribute`. `mival/adapters/base.py`를 읽고 이미 있는 것과 없는 것을 확인할 것 — 없는 메서드가 있으면 base에 추가하지 말고 보고서에 적어 escalate한다 (base.py는 공용 파일이다).

## Threshold 정책 (spec §4.4)

| 정책 | 내용 |
|---|---|
| `legacy` | 원 논문 값 그대로 (ModelCard `x-mival.threshold.value`에서 읽는다. 값을 코드에 쓰지 말 것) |
| `refit_youden` | dev에서 Youden J 재추정 |
| `refit_sens95` | dev에서 sensitivity 95% 지점 |

**셋 다 보고하되 primary는 `refit_sens95`.** 근거: STEMI는 miss 비용이 압도적이라 민감도·특이도를 대칭 취급하는 Youden은 임상적으로 부적절하다. **test set에서는 threshold를 선택하지 않는다** — 재추정은 dev에서만 일어나야 하고, 이를 테스트로 고정한다.

`sensitivity 95% 지점`은 유일하지 않을 수 있다(동률 확률값). 어느 쪽을 고르는지 결정론적으로 정하고 그 이유를 주석에 남긴다.

## Gate (spec §3.5)

| gate | 조건 | 실패 시 |
|---|---|---|
| **Leakage** | 학습에 쓰인 `person_id`가 test split에 부재 | **즉시 중단** (예외) |
| **Contamination** | ModelCard `pretraining_corpora` ∩ 평가 cohort 출처 = ∅ | **중단하지 않고** manifest·결과표에 `contaminated=true` 표시 |

contamination이 차단이 아니라 표시인 이유: 오염된 조건의 성능도 보고 가치가 있고, 오염 여부가 결과 해석의 축이다. MIMIC-IV-ECG는 검토된 12개 FM 중 9개의 사전학습에 포함되었으므로 이 gate는 MIMIC 실행에서 실제로 켜진다.

`StageResult.contamination`에 `{"flag": bool, "overlapping_corpora": [...]}`를 넣으면 wrapper가 manifest에 기록한다.

주의: `registry/models/*.json`의 `pretraining_corpora` 값은 **잠정값**이다 (`docs/decisions/README.md` Models Open Question 5). gate 로직은 정확히 짜되, 카드 값이 확정 전이라는 사실을 보고서에 적는다.

## 하지 말 것

- 모델별 분기 (`if model_id == ...`). ECGFounder·PROPHECG 이름이 이 stage 코드에 등장하면 실패다. 모든 모델 특성은 ModelCard에서 온다.
- `import torch` / `import tensorflow`를 모듈 최상단에서. 반드시 `get_adapter`를 통해서만. `import mival.stages.models`가 어느 환경에서도 성공해야 한다.
- **`tensorflow`를 import하는 코드를 실행하지 말 것 — 이 머신에서 프로세스를 hard-abort시킨다.** torch도 이 머신에 없다. 테스트는 fake adapter로 짠다.
- threshold 값·정규화 상수·클래스 인덱스를 코드에 하드코딩. 전부 카드에서 읽는다.

## 검증

`python3 -m pytest tests/ -q` 전부 통과. 기존 162 passed / 11 skipped를 깨뜨리지 않는다. 테스트는 weights·torch·tensorflow 없이 로컬에서 돌아야 한다 — adapter를 fake로 주입할 수 있게 설계할 것.

최소한 다음을 테스트한다:
- 4개 training mode 각각이 무엇을 학습 가능 상태로 두는지 (fake adapter로)
- leakage gate가 겹치는 `person_id`에 대해 즉시 예외를 던진다
- contamination gate가 겹칠 때 flag를 세우되 **중단하지 않는다**
- 세 threshold 정책이 알려진 확률/라벨 벡터에 대해 정확한 값을 낸다 (직접 계산한 기대값으로)
- threshold 재추정이 dev만 보고 test는 보지 않는다
- 출력 parquet 컬럼이 `PREDICTION_COLUMNS`와 정확히 일치하고 파일명이 `RunKey.to_string()`이다

## 보고

끝나면 `.superpowers/notes/plan4-report.md`에 다음을 쓴다: 만든 파일, 테스트 결과 한 줄, 내린 설계 결정과 근거, escalate할 항목, 확신이 없는 부분. **커밋하지 말 것** — 세 stage가 같은 브랜치에서 동시에 작업 중이라 동시 커밋은 git index.lock에서 깨진다. 파일만 남기고 끝낸다. 커밋은 컨트롤러가 검토 후 한다.
