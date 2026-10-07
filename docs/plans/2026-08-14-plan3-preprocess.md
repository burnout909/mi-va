# Plan 3 — Preprocess stage

**Spec:** `docs/spec/2026-08-14-research-design.md` §4.3 (읽을 것). §3.1, §3.6도 해당.

**목표:** cohort의 각 ECG를 모델별 입력계약에 맞춰 컴파일·저장하고, 컴파일 실패를 exclusion ledger에 기록한다. perturbation grid를 정의하고 on-the-fly로 적용 가능한 형태로 노출한다.

## 소유 파일 — 이 세 개만 만들거나 고친다

- Create: `src/mival/perturbation.py`
- Modify: `src/mival/stages/preprocess.py` (현재 계약만 박힌 stub)
- Create: `tests/test_perturbation.py`, `tests/test_stage_preprocess.py`

다른 파일은 건드리지 않는다. `src/mival/{ops,compiler,contract,signal,modelcard}.py`와 `src/mival/pipeline/`은 완성된 코드이며 **읽기 전용**이다. 이 stage와 병렬로 다른 두 stage가 작성 중이므로 공용 파일 수정은 곧 충돌이다.

## 이미 있는 것 (읽고 쓸 것, 다시 만들지 말 것)

```python
from mival.compiler import compile_recipe        # (card, source_metadata) -> OpChain, CompileError 발생
from mival.contract import CompileError, REASON_CODES
from mival.signal import Signal, SourceMetadata
from mival.ops import OpChain                    # 고정 순서 op 라이브러리
from mival.modelcard import load_card, load_registry
from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.tables import read_table, write_table   # parquet, lazy import
```

`compile_recipe`가 이미 input contract gate 전부를 구현한다. **재구현 금지.** `CompileError.reason_code`를 그대로 `ctx.ledger.record(...)`에 넘긴다.

`ctx.layout.artifact("...")`로만 경로를 만든다. 문자열로 경로를 조립하지 않는다.

## 고정된 계약 (`src/mival/stages/preprocess.py` 상단에 이미 있음 — 값 바꾸지 말 것)

- 산출물: `artifacts/preprocess_index.parquet`, `artifacts/tensors/`
- `PREPROCESS_INDEX_COLUMNS` — Plan 4가 이 컬럼으로 join한다
- `reason_codes = REASON_CODES` (5종)
- `required_inputs() == ("cohort_index",)`

## Perturbation (spec §4.3)

축과 수준은 **spec에 적힌 값 그대로**:

```
resample:        [500, 250, 125, 100]
lead_dropout:    [none, drop_V3V4, precordial_only, limb_only]
duration:        [10, 5, 2.5]
amplitude_scale: [1.0, 0.5, 2.0]
noise:           [none, baseline_wander, powerline_50hz, emg]
```

기본 정책은 **OFAT(한 번에 한 축)** — baseline 포함 13~16조건. full cartesian은 opt-in(`mode: cartesian`).

**절대 놓치면 안 되는 규칙:** 변형 후 반드시 입력계약 형태로 되돌린다. 500 Hz → 125 Hz로 낮춘 뒤 **다시 500 Hz로 복원**해 모델에 넣는다. 그래야 측정 대상이 정보 손실뿐이고 shape 불일치가 섞이지 않는다. duration, lead_dropout도 같다 — dropout된 lead는 0으로 채워 lead 수를 유지한다. 이 규칙을 어기면 이 stage의 존재 이유가 사라진다.

**perturbation tensor는 디스크에 저장하지 않는다.** `mival/perturbation.py`는 순수 함수 라이브러리로, `apply(signal, perturbation_id, seed) -> Signal` 형태를 제공한다. Plan 4의 dataloader가 이걸 호출한다. 재현성은 `seed + perturbation_id`로 보장 — 같은 (seed, perturbation_id, record)는 항상 같은 출력이어야 하고, 이걸 테스트로 고정한다.

`perturbation_id`는 사람이 읽을 수 있고 `RunKey` 값 문자 집합 `[A-Za-z0-9][A-Za-z0-9._-]*`을 만족해야 한다(예: `baseline`, `resample-125`, `leaddrop-precordial_only`).

## 이 stage가 하는 일

1. `cohort_index.parquet`을 읽는다 (컬럼: spec §4.1)
2. `registry/models/*.json`의 각 카드에 대해 `compile_recipe(card, source_metadata)` 호출
3. 성공 → recipe를 적용해 tensor를 `tensors/<recipe_id>/`에 저장, `preprocess_index`에 행 추가
4. `CompileError` → `ctx.ledger.record(image_occurrence_id, person_id, exc.reason_code, str(exc))`, tensor 없음
5. `StageResult(outputs=[...], counts={"in": N, "out": M})` 반환 — `excluded`는 wrapper가 ledger에서 채우므로 넣지 않는다

`recipe_id`는 컴파일된 op chain에서 결정론적으로 유도한다(같은 chain → 같은 id). 모델 id를 그대로 쓰지 말 것 — 두 모델이 같은 입력계약을 가지면 tensor를 공유해야 한다.

## 하지 말 것

- 모델별 분기 (`if model_id == ...`). 새 모델 등록은 ModelCard 하나 추가로 끝나야 한다 (C1 주장). 이 stage에 모델 이름이 등장하면 실패다.
- 근거 없는 상수. 필터 차단주파수, 노이즈 진폭 같은 값은 spec이나 인용 가능한 출처에서 오거나, 없으면 spec에 없다고 명시하고 설정값으로 노출한다.
- DICOM 실제 읽기 — 이번 범위 밖이다. tensor 소스는 `local_path`에서 온다고 가정하고, 테스트는 합성 신호(`mival.synthetic.synthetic_ecg`)로 fixture를 만든다.
- `tensorflow`를 import하는 코드 실행. **이 머신에서 tensorflow import는 프로세스를 hard-abort시킨다.**

## 검증

`python3 -m pytest tests/ -q` 전부 통과. 기존 162 passed / 11 skipped를 깨뜨리지 않는다.
테스트는 weights도 torch도 tensorflow도 없이 로컬에서 돌아야 한다.

최소한 다음을 테스트한다:
- OFAT grid가 정확히 spec의 축·수준을 낸다
- 각 perturbation이 입력계약 shape을 **보존**한다 (125 Hz로 낮춰도 출력은 500 Hz × 5000 샘플)
- `(seed, perturbation_id)`가 같으면 출력이 비트 단위로 같다; seed가 다르면 다르다 (noise 축)
- `CompileError` 5종이 각각 ledger에 정확한 reason_code로 남는다
- 입력계약이 같은 두 카드가 같은 `recipe_id`를 받는다
- stage가 만든 `preprocess_index`가 `PREPROCESS_INDEX_COLUMNS`와 정확히 일치한다

## 보고

끝나면 `.superpowers/notes/plan3-report.md`에 다음을 쓴다: 만든 파일, 테스트 결과 한 줄, 내린 설계 결정과 근거, 확신이 없는 부분. **커밋하지 말 것** — 세 stage가 같은 브랜치에서 동시에 작업 중이라 동시 커밋은 git index.lock에서 깨진다. 파일만 남기고 끝낸다. 커밋은 컨트롤러가 검토 후 한다.
