# 회귀·생존·분할 대표 모델 3개를 끝까지 돌리기 (1단계)

2026-10-07. [스키마 gap 판정](../../model-search/schema-gaps.md) 다음 단계다. task마다 대표 모델
하나를 retrieve → evaluate까지 서버에서 실제로 돌리고, 그 과정에서 필요한 최소 확장만 만든다.
공통 모듈 정리(2단계)와 나머지 12개 모델(3단계)은 이 결과를 보고 따로 설계한다.

사용자가 이 스펙 이후 계획 승인과 서버 실행까지 위임했다 (2026-10-07 대화). 멈추는 조건은 아래
"실행 규칙"에 있다.

## 목표와 성공 기준

- 세 study가 서버에서 전체 코호트로 retrieve, profile, preprocess, models, evaluate를 끝내고
  manifest와 지표 표를 남긴다.
- `studies/lvef`의 stage별 `config_hash`와 기존 테스트 결과가 바뀌지 않는다.
- 디버깅 중 고친 것은 [`adaptations.md`](../../decisions/adaptations.md)에 L0/L1로 남는다.

## 범위

| task | study | 모델 (카드) | 라벨 | 지표 |
|---|---|---|---|---|
| 회귀 | `studies/ecg-age` | `lima-ecg-age` | ECG 시점 나이 | MAE, bias 등 기존 회귀 지표 |
| 생존 | `studies/mortality` | `cavalab-deepsurv-code15` | 1년 사망 (사건, 시간) | Harrell C-index, 1년 AUROC |
| 분할 | `studies/delineation` | `semisegecg-resnet18` | 기계 측정 PR, QRS, QT | 간격별 회귀 지표 |

결정 (2026-10-07 대화):

- 생존 기준 시점은 **1년**이다. MIMIC 병원 밖 사망은 마지막 퇴원 후 1년까지만 기록된다.
- 분할 모델에는 **lead II 하나**만 넣는다. 카드 `leads: ["II"]`로 표현되어 fan-out이 필요 없다.
- 코호트는 **사람당 ECG 1건**을 seed로 고정해 무작위로 고른다.

## 카드와 registry

- 카드는 task별 디렉터리에 둔다: `registry/regression/`, `registry/survival/`, `registry/segmentation/`.
  preprocess가 registry 디렉터리의 모든 카드를 컴파일하므로(L2-13), 디렉터리를 나누면 코드 변경
  없이 study끼리 섞이지 않는다. `registry/models/`와 `registry/drafts/`는 건드리지 않는다.
- 카드는 `registry/drafts/`의 초안에서 시작해 서버에서 확인한 값(sha256, uri, builder)으로 채운다.
- 모델별로 특이한 전처리(cavalab의 checkpoint 저장 NM/NS z-score, SemiSegECG의 필터 등)는 서버의
  weight 옆 wrapper 모듈에 넣고 카드 `builder`가 가리킨다. 카드에 이미 있는 기능만 쓴다(L0).

## retrieve: 라벨 종류

`label_source.kind`를 추가한다. 없으면 `measurement`(지금 동작)라 LVEF는 바뀌지 않는다.

| kind | 채우는 열 | 정의 |
|---|---|---|
| `measurement` | `label_value`, `label_primary` 등 | 지금 그대로 |
| `age_at_ecg` | `label_value` | ECG 연도 − `person.year_of_birth` (±1년) |
| `death_within` | `label_event`, `label_time_days` | 종료일 = min(사망일, ECG + `horizon_days`, 마지막 방문 종료 + 365일). 사건 = 사망일 ≤ 종료일 |
| `machine_measurement` | `label_pr`, `label_qrs`, `label_qt` | MIMIC-IV-ECG `machine_measurements.csv`, 파일 경로의 study id로 연결 |

- 새 kind에서는 LVEF cutoff 열(`label_primary`, `label_sens*`)을 비운다(NaN). `label_concept_id`,
  `primary_cutoff` 등 measurement 전용 키는 읽지 않는다.
- `one_per_person: true`면 seed로 사람당 1건을 고르고 나머지는 제외 사유 `not_selected`로 ledger에 남긴다.
- 기계 측정값이 없거나 생리적으로 불가능한 값(0 이하, 장비 결측 코드)은 `label_implausible`로 제외한다.
- `machine_measurements.csv`의 경로와 sha256은 retrieve 입력으로 `config_hash`에 들어간다.

## models: 라벨 정의와 출력 종류

- arm `label_def`:
  - `value`: 기존 회귀.
  - `pr`, `qrs`, `qt`: 연속값. 회귀와 같이 처리하고 `label_<def>` 열과 `pred_value`를 비교한다.
  - `survival`: 새 갈래. threshold를 맞추지 않는다.
- predictions에 `label_event`, `label_time_days`, `label_pr`, `label_qrs`, `label_qt` 열을 더한다.
  한 행 = ECG 한 건은 유지한다.
- 카드 `output.type` (torch adapter):
  - `risk_score`: 출력 값 하나를 `pred_value`로 쓴다.
  - `segmentation_mask`: (B, C, T) 출력을 argmax한 뒤 `src/mival/decode/intervals.py`로 PR, QRS, QT(ms)를
    계산하고 arm의 `label_def`에 해당하는 값을 `pred_value`로 쓴다. 규칙은 SemiSegECG 저자 코드
    (`compute_numerics`, 박동별 간격의 중앙값)를 따른다. 마스크 자체는 저장하지 않는다.
- inference_only 짝 검사: `survival` ↔ `risk_score`, `pr`/`qrs`/`qt` ↔ `segmentation_mask` 또는
  `regression`, `value` ↔ `regression`, 이진 ↔ 지금 그대로.
- 입력 계약에 `crop_anchor`, `pad_anchor` (`start` | `center`, 기본 `start`)를 더하고 Pad에 `center`를
  구현한다. 기본값이 지금 동작과 같으므로 LVEF 레시피는 바뀌지 않는다.

## evaluate

- 회귀 범주는 arm의 `label_def`가 고른 라벨 열을 쓴다 (`value` → `label_value`, `pr` → `label_pr`).
- `regression_cuts`는 리스트(지금 형식) 또는 `label_def`별 매핑(`{value: [65], pr: [200], qrs: [120],
  qt: [450]}`)을 받는다.
- 생존 범주: Harrell C-index(1년 censoring), 1년 AUROC(1년 전 censored 제외), 사람 단위 bootstrap
  95% CI. calibration은 내지 않는다 (baseline hazard 없음).
- 그림은 기존 것만. 생존 그림은 2단계.

## 테스트

- 기존 테스트 전부 통과.
- LVEF 불변: `studies/lvef`의 retrieve, profile, preprocess, models, evaluate spec hash가 변경 전과 같다.
- 단위 테스트(합성 데이터): 나이·1년 사망 라벨과 censoring 경계, 사람당 1건 선택의 seed 재현성,
  마스크 → 간격, Pad center, C-index 알려진 예.

## 실행 규칙 (서버)

1. 사전 준비: `machine_measurements.csv` 반입, 마지막 방문 종료일 SQL과 필요한 index.
2. 3개 study를 1,000건 샘플로 끝까지 돌리며 디버깅한다.
3. LVEF 체인이 끝난 뒤 GPU를 쓴다. 전체 코호트로 실행한다.
4. 커밋은 `feat/framework-core`에, 코드 동기화는 서버 저장소 push로만 한다. GitHub에 push하지 않는다.
5. 멈추는 조건: 데이터 삭제가 필요할 때, 새 인증이 필요할 때, L2급 구조 변경이 필요할 때, 같은
   오류가 세 번 고쳐도 남을 때. 그 task만 멈추고 나머지는 계속한다.
6. 인스턴스는 끄지 않는다.

## 하지 않는 것

- 나머지 12개 모델, keras 3, ONNX, 앙상블, 출력 역변환, lead fan-out, 생존곡선형 출력
- 공통 모듈 정리 (2단계)
- `registry/models/`, `studies/lvef/` 수정
