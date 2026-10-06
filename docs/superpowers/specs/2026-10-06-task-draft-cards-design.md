# task별 카드·study 초안으로 스키마 gap 드러내기

2026-10-06. LVEF의 L1 수정과 전체 실행 **전에**, task마다 대표 모델을 카드와 study.yaml로
적어 보고 지금 프레임워크가 표현하지 못하는 것을 목록으로 만든다.

## 목적

지금 프레임워크는 STEMI와 LVEF 모델로만 만들어졌다. 다른 출력 형태(생존곡선, 샘플별 마스크,
여러 head)나 다른 라벨(사망까지의 시간, 생년에서 계산한 나이)을 넣으면 어디가 안 맞는지 아직
모른다. LVEF 코드를 고치기 전에 이것을 먼저 드러낸다.

- 산출물은 **"프레임워크가 지원하지 못하는 것 목록"**이다. 실행 가능한 카드가 아니다.
- 이 목록을 보고 LVEF L1 수정 범위와, 어떤 L2를 언제 만들지 정한다.
- 코드는 바꾸지 않는다. L2는 보고만 한다 (수정 분류 L0/L1/L2 원칙).
- weight를 새로 받거나 서버에 반입하지 않는다. 서버를 켜지 않는다.

## 범위

task 네 개에 대표 모델만 쓴다. task마다 5개를 정식으로 고르는 것은 문헌 검색으로 나중에 한다.

| task | 무엇으로 | 이번에 쓰는 카드 |
|---|---|---|
| classification | **LVEF** (2026-10-06 결정) | 카드가 없는 LVEF 분류 모델 2개 |
| regression | 나이 | 1개 |
| survival | 전체 사망 | 2개 |
| segmentation | P/QRS/T 구간 분할 | 1개 |

classification을 AF가 아니라 LVEF로 하는 이유: LVEF가 원래 진행 중인 task다. LVEF는 이미
study와 카드 4개(HeartWise binary ≤40, HeartWise 회귀, ECGFounder, xECG)가 있고 2026-09-22
샘플 실행으로 확인했다. 남은 것은 서버에 반입했지만 카드가 없는 두 모델이다. AF 검색 결과
(잠정 5개)는 [모델 탐색 장부](../../model-search/af-classification-ledger.csv)에 두고 이번 범위에서 뺀다.

regression을 LVEF 회귀가 아니라 나이로 하는 이유: LVEF 회귀는 이미 돌아가는 형태라 새 gap이
나오지 않는다. 나이 모델은 입력 단위가 모호한 경우와 라벨이 측정값이 아니라 생년에서 계산되는
경우를 시험한다.

### 카드 6개

| task | model_id | 출처 | 이 모델로 시험하는 것 |
|---|---|---|---|
| classification | `heartwise-lvef-under50` | HF heartwise/EfficientNetV2_LVEF_under_50, 서버 반입 2026-09-21 | cutoff <50. 기존 HeartWise ≤40과 입력 계약이 같음 |
| classification | `echonext-mini` | Columbia, 서버 반입 2026-09-21 | 12개 label 중 index 0 `lvef_lte_45`, tabular 입력 7개, cutoff ≤45 |
| regression | `lima-ecg-age` | Lima et al. Nat Commun 2021, Zenodo 4892365 | 출력 나이(년), 입력 단위 모호 (mV vs mV×10) |
| survival | `ml4h-ecg2af` | Khurshid et al. Circulation 2022, broadinstitute/ml4h | head 5개, 25구간 조건부 생존확률, keras 3 |
| survival | `cavalab-deepsurv-code15` | Lukyanenko et al. BioData Mining 2025, Zenodo 16877773 | Cox log-risk 하나, baseline hazard 없음 |
| segmentation | `semisegecg-resnet18` | Park et al. CIKM 2025, vuno/semi-seg-ecg | `(B,4,2500)` 샘플별 마스크, single-lead, 라이선스 표기 충돌 |

모두 학습 데이터에 MIMIC-IV-ECG가 없다. 같은 저장소의 MIMIC 학습 변형(cavalab MIMICIV 모델 등)은
쓰지 않는다.

### study 초안 4개

| study | 라벨 | 대상 |
|---|---|---|
| `lvef` | 기존 LVEF 라벨 + 모델별 cutoff (<50, ≤45) | 기존 arm 6개 + 새 두 모델 |
| `ecg-age` | person 생년으로 계산한 ECG 시점 나이 | `lima-ecg-age` |
| `mortality` | 전체 사망까지의 시간과 censoring (CDM death) | `ml4h-ecg2af` 사망 head, `cavalab-deepsurv-code15` |
| `delineation` | MIMIC-IV-ECG 기계 측정 PR/QRS/QT 간격 | `semisegecg-resnet18` |

- `lvef` 초안은 기존 `studies/lvef/study.yaml`을 고치지 않고 복사본에 arm을 더한다.
- survival endpoint를 전체 사망으로 맞춘 이유: 두 survival 모델이 함께 평가될 수 있는 유일한
  endpoint다. ECG2AF에는 사망 head가 있고, cavalab은 사망으로 학습했다.
- segmentation은 MIMIC에 사람이 그은 P/QRS/T 구간이 없어서, 마스크에서 계산한 간격을 기계 측정
  간격과 비교하는 방식으로 적는다.

## 산출물과 위치

```
registry/drafts/<model_id>.json        카드 초안 6개
studies/drafts/<study>/study.yaml      study 초안 4개
docs/decisions/adaptations.md          새 L2 항목
docs/model-search/schema-gaps.md       카드·study별 판정 한 장 요약
```

`registry/models/`에 두지 않는 이유: preprocess가 그 디렉터리의 **모든** 카드를 컴파일한다
(`src/mival/stages/preprocess.py:250`). 새 카드를 넣으면 LVEF의 레시피와 `config_hash`가 바뀐다.
이 사실 자체도 gap으로 기록한다. `heartwise-lvef-under50`을 `registry/models/`로 옮겨 LVEF 전체
실행에 넣을지는 이 작업 다음, LVEF 실행을 준비할 때 정한다.

## 판정 방법

지금 코드에 직접 넣어 본다. 검사는 scratchpad의 일회용 스크립트로 하고 repo에 넣지 않는다.

- 카드: `mival.modelcard.load_card`
- study: `mival.pipeline.study.load_study`, 그리고 stage별 `*Spec.from_mapping`

결과는 셋으로 나눈다.

1. **통과**: 지금 스키마로 뜻대로 실행된다.
2. **로드되지만 뜻이 틀림**: 예를 들어 ECG2AF를 `logits`로 적으면 로드되지만 head 4개를 버리고
   생존곡선을 확률로 읽는다. 카드 `notes`와 원장에 적는다.
3. **표현 불가**: 필요한 필드를 카드의 `x-mival-proposed` 블록(study는 `x-proposed`)에 제안으로
   적는다. 이 모델에서 본 사실만 적고, 일반화된 스키마는 설계하지 않는다.

loader는 모르는 키를 무시하므로 `x-mival-proposed`가 있어도 `load_card`는 통과할 수 있다. 그래서
2와 3은 로드 결과만으로 정하지 않는다. 카드에 적은 내용을 기존 adapter·stage가 어떻게 읽는지 코드
위치와 함께 근거를 남긴다.

## 카드에 채우는 값

- **LVEF 두 모델**: 서버 반입 기록([`docs/models/heartwise-lvef.md`](../../models/heartwise-lvef.md),
  [`docs/models/echonext-mini.md`](../../models/echonext-mini.md))의 실측 값을 쓴다. 문서에 잘려 있는
  값(sha256 앞뒤만 적힌 것 등)은 서버를 켤 때 manifest에서 채우고, 그 전까지 `null`로 둔다.
- **나머지 네 모델**: weight를 받지 않으므로 sha256, 서버 경로(`uri`), `code_commit`, runtime 버전은
  `null`로 두고 `notes`에 "unverified"라고 쓴다. 대표 모델 검색 때 실제로 열어 확인한 값(출력 shape,
  클래스 순서, 입력 단위 실험)은 출처와 함께 `notes`에 적는다.
- 문서끼리 모순되는 값(Lima 입력 단위, SemiSegECG 라이선스, cavalab MIMIC crop)은 모순 그대로 적는다.

## 이미 보이는 gap

초안을 쓰면서 확인할 시작점이다. 최종 목록은 판정 결과로 정한다.

| gap | 근거 모델·study | 위치 |
|---|---|---|
| `output.type`이 softmax/logits/sigmoid/regression 넷뿐 | ECG2AF, cavalab, SemiSegECG | `src/mival/adapters/torch_adapter.py:334` |
| 출력 head가 여러 개인 모델 | ECG2AF (5개) | 카드 `output` |
| 생존곡선을 기준 시점 위험 하나로 줄이는 규칙 | ECG2AF | — |
| baseline hazard가 없어 절대 위험을 못 냄 | cavalab | — |
| ECG 한 건에 값 하나가 아닌 샘플별 출력 | SemiSegECG | models·evaluate 예측 형식 |
| lead를 하나씩 넣는 모델 | SemiSegECG | `input_contract` |
| waveform 외 tabular 입력 (기존 L2-3) | EchoNext | `input_contract` |
| 여러 label 출력 중 하나를 고르는 inference_only | EchoNext (12개 중 index 0) | 카드 `output.positive_index` |
| cutoff가 40·50 두 개로 고정, ≤45 표현 불가 | EchoNext, `lvef` | `src/mival/stages/retrieve.py:43` |
| retrieve가 측정값 하나를 cutoff로 자르는 형태만 받음 | `ecg-age`, `mortality`, `delineation` | `src/mival/stages/retrieve.py:150` |
| 분류가 아니면 regression으로 갈라지는 models 분기 | survival, segmentation | `src/mival/stages/models.py:771` |
| study가 모델을 registry 디렉터리 단위로 고름 | 전부 | `src/mival/stages/preprocess.py:250` |
| keras 3 `.keras` 포맷 | ECG2AF | keras env는 2.7 |

## 원장 기록

판정 2와 3은 `docs/decisions/adaptations.md`의 L2 목록에 기존 형식(제목, 설명, 발견, 상태 "보고됨.
만들지 않음", 착수 조건)으로 올린다. 원인이 같은 gap은 한 항목으로 묶고 근거 모델을 나열한다.
이미 있는 항목(L2-3 tabular 등)은 새로 만들지 않고 근거 모델만 더한다.

## 하지 않는 것

- 스키마, loader, adapter, stage 코드 변경
- weight 다운로드, 서버 반입, 추론
- task별 정식 5개 선정
- 기존 `studies/lvef/study.yaml`, `registry/models/` 수정
- LVEF L1 수정과 전체 실행 (이 작업 다음)
