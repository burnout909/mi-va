# Plan 7 — Misclassification stage

**Spec:** `docs/spec/2026-08-14-research-design.md` §4.6 (읽을 것). §3.4(run_key)도 해당.

**목표:** predictions를 읽어 selector 4종으로 사례를 뽑고, 사례별 동반 정보와 그림을 붙여
`cases.parquet`을 낸다. 사람이 채우는 `review.csv`를 `case_id`로 병합한다.

## 이 단계가 무엇인가

spec §4.6: **"특정 비교를 수행하는 모듈이 아니라 run_key 공간에서 흥미로운 지점을 뽑는
질의 엔진"**. 이것이 설계 전체를 결정한다.

- selector는 `study.yaml`에 **선언**한다. 코드에 모델 이름·축 이름을 쓰지 않는다.
- `contrast`가 축을 특정하지 않으므로 `model_id` / `training_mode` / `perturbation_id` /
  `label_def` / `site` 축의 불일치가 **전부 같은 코드**로 나온다. 축마다 분기가 생기면 실패다.
- 5단계(confirmatory)와 파일·단계로 분리한다. 6단계는 성능표를 다시 계산하지 않는다.

## 소유 파일

- Create: `src/mival/stages/_predictions.py` (5·6단계 공용 로더 + operating threshold)
- Create: `src/mival/stages/misclassify.py`
- Modify: `src/mival/stages/evaluate.py` (공용 코드로 위임만 — 동작 변경 없음)
- Modify: `src/mival/pipeline/stage.py` (`_UNBUILT` → `_STAGES`)
- Modify: `src/mival/figures.py` (`case_waveform` 추가)
- Modify: `src/mival/adapters/{base,torch_adapter}.py` (`attribute()`)
- Create: `tests/test_stage_misclassify.py`, `tests/test_attribution.py`

## 공용 계층을 먼저 만드는 이유

`boundary` selector는 threshold를 필요로 한다. **6단계가 쓰는 threshold는 5단계가 보고한
바로 그 숫자여야 한다** — 아니면 "임계값 바로 아래 사례"가 보고된 결정 규칙과 다른
규칙의 경계가 된다. 그래서 threshold 산출과 prediction 로딩을 `_predictions.py`로 옮기고
두 stage가 같은 함수를 호출한다. 복제 후 "테스트로 일치를 고정"하는 것보다 낫다.

## selector 4종 (spec §4.6 표)

pattern = run_key 축의 부분집합 → 값. 지정하지 않은 축은 자유.

| selector | 선언 | 정의 |
|---|---|---|
| `error` | `reference`, `k` | FP 상위 k(prob 내림차순) / FN 하위 k(prob 오름차순) |
| `boundary` | `reference`, `epsilon` | `abs(prob − threshold) < ε` |
| `contrast` | `a`, `b`, `criterion`, `tau` | `label_flip` 또는 `abs(Δprob) > τ` |
| `instability` | `pattern`, `over`, `tau` | 같은 record에 대해 `over` 축만 다른 행들의 `prob_std > τ` |

**pattern은 record당 정확히 한 행으로 해석되어야 한다.** 두 행 이상이면 거부하고 어떤 축이
갈렸는지 말한다(`instability`만 예외 — 그 경우 `over`에 선언된 축만 갈릴 수 있다).
`perturbation_id`를 고정하는 것을 잊은 study가 perturbation 민감도를 fold 불안정으로
보고하는 사고를 이 검사가 막는다.

`instability`가 현재 산출물에서 비는 경우: 한 번의 models 실행에서 dev record는 자기
fold에 정확히 한 번만 나타난다(out-of-fold 예측). seed 간 비교는 서로 다른 run 디렉터리의
predictions를 함께 넘겨야 한다. **빈 결과는 경고로 보고하고 조용히 넘어가지 않는다.**

## 출력

- `cases.parquet` — `CASE_COLUMNS`. `case_id`는 `"<selector>::<image_occurrence_id>"`로
  **run에 무관하게 안정**해야 한다(config_hash·k·tau가 바뀌어도 살아남은 사례의 id는 불변).
  해시가 아니라 읽을 수 있는 문자열인 이유는 사람이 `review.csv`에서 이 값을 본다는 것이다.
- `review_template.csv` — 사람이 채울 파일. 기존 verdict가 있으면 채워서 내보낸다.
- `review_unmatched.csv` — 넘겨받은 review 중 이번 사례 집합에 없는 행. **버리지 않는다.**
- `figures/cases/<case_id>.png` — waveform + 동반 정보.
- `selectors.json` — selector별 후보 수·선정 수·경고.

`verdict` 어휘는 spec §4.6 FMEA 표에서 온 기본값을 갖되 `study.yaml`에서 선언으로
바꿀 수 있다. 자유 텍스트면 집계가 불가능하고, 코드에 박으면 사전 지정이 아니라 하드코딩이다.

## 사례별 동반 정보 (spec §4.6)

waveform plot · acquisition metadata · preprocess warning · attribution overlay ·
**label 정의 4종의 값** · cohort/index context.

label 4종은 predictions에 이미 컬럼으로 있다. 나머지는 optional input의 join이다
(없으면 그 칸이 비고, 없다고 실패하지 않는다).

## attribution

`Adapter.attribute()`를 여기서 정한다. **Integrated Gradients**(Sundararajan, Taly, Yan
2017)를 torch에 구현하고 **baseline을 선언 필수로 만든다**. 기본값을 두지 않는 이유:
ECG에서 0 baseline은 "중립 입력"이 아니라 asystole이며, IG의 귀속값은 baseline 선택에
전적으로 의존한다. 조용한 기본값은 해석 가능한 그림처럼 보이는 임의의 결과를 만든다.

검증은 IG의 정의 성질인 **completeness**로 한다: `sum(attribution) ≈ F(x) − F(baseline)`.
"값이 0과 1 사이다" 류의 테스트는 아무것도 검증하지 않는다.

attribution overlay는 **선택 사항**이다. adapter가 지원하지 않으면(keras) 그림에서 그
칸이 빠지고 경고가 남는다. 단계 전체가 실패하지 않는다.

## 하지 말 것

- selector 코드 안에 축 이름·모델 이름 분기
- 5단계 지표 재계산
- `tensorflow`를 import하는 코드 실행 — 이 머신에서 프로세스를 hard-abort시킨다
- review 결과를 덮어쓰거나 조용히 버리기

## 검증

`PYTHONPATH=src python3 -m pytest tests/ -q` 전부 통과. 기존 517 passed / 11 skipped를
깨뜨리지 않는다. 최소한 다음을 고정한다.

- pattern이 record당 두 행에 걸리면 거부하고 갈린 축을 이름으로 말한다
- `contrast`가 `model_id` 축과 `label_def` 축에 대해 **같은 코드 경로**로 동작한다
- `case_id`가 k·tau·config_hash를 바꿔도 불변이다
- review 병합이 재실행에서 verdict를 보존하고, 매칭되지 않은 review를 잃지 않는다
- 6단계 threshold가 5단계 threshold와 같은 값이다
- IG completeness가 합성 모듈에서 수치적으로 성립한다
