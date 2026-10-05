# task별 카드·study 초안으로 스키마 gap 드러내기

2026-10-06. LVEF의 L1 수정과 전체 실행 **전에**, 다른 task의 대표 모델을 카드와
study.yaml로 적어 보고 지금 프레임워크가 표현하지 못하는 것을 목록으로 만든다.

## 목적

- 산출물은 "프레임워크가 지원하지 못하는 것 목록"이다. 실행 가능한 카드가 아니다.
- 이 목록을 보고 LVEF L1 수정 범위와, 어떤 L2를 언제 만들지 정한다.
- 코드는 바꾸지 않는다. L2는 보고만 한다 (수정 분류 L0/L1/L2 원칙).
- weight는 받거나 서버에 반입하지 않는다. 서버를 켜지 않는다.

## 범위

task마다 대표 모델만 쓴다. 정식 5개 선정은 AF 때처럼 문헌 검색으로 나중에 한다.
AF는 [모델 탐색 장부](../../model-search/af-classification-ledger.csv)의 잠정 5개를 쓴다.

| task | model_id (초안) | 출처 | 왜 이 모델 |
|---|---|---|---|
| classification (AF) | `ribeiro-2020` | Ribeiro Nat Commun 2020, Zenodo 3765717 | 6개 출력 중 AF 하나, 400 Hz·4096 샘플, 1e-4 V 단위 |
| classification (AF) | `ecgfounder-150head` | NEJM AI 2025 | 기존 `ecgfounder` 카드는 backbone용. 150개 출력 중 AF를 inference_only로 쓰는 별도 카드 |
| classification (AF) | `deepecg-sl-77` | EHJ 2026, HF heartwise/EfficientNetV2_77_Classes | 77개 중 Afib, threshold 일부 누락 |
| classification (AF) | `ecglib-afib` | BSPC 2024, EcgLib v1.1.0 | AF 전용 이진 모델, 구조 4종 중 하나 |
| classification (AF) | `ptbxl-xresnet1d101` | IEEE JBHI 2021 | 71개 statement 중 AFIB, 100 Hz, fastai v1 |
| regression | `lima-ecg-age` | Nat Commun 2021, Zenodo 4892365 | 나이(년), 입력 단위 모호 |
| survival | `ml4h-ecg2af` | Circulation 2022, ml4h model_zoo | head 5개, 25구간 조건부 생존확률 |
| survival | `cavalab-deepsurv-code15` | BioData Mining 2025, Zenodo 16877773 | Cox log-risk 하나, baseline hazard 없음 |
| segmentation | `semisegecg-resnet18` | CIKM 2025, vuno/semi-seg-ecg | `(B,4,2500)` 샘플별 마스크, single-lead |

모두 학습 데이터에 MIMIC-IV-ECG가 없다. 같은 저장소의 MIMIC 학습 변형은 쓰지 않는다.

study 초안은 4개다.

| study | 라벨 | 대상 모델 |
|---|---|---|
| `af` | ECG 판독 문구의 AF | AF 5개 |
| `ecg-age` | 생년으로 계산한 ECG 시점 나이 | `lima-ecg-age` |
| `mortality` | 전체 사망까지의 시간과 censoring | `ml4h-ecg2af`의 사망 head, `cavalab-deepsurv-code15` |
| `delineation` | 기계 측정 PR/QRS/QT 간격 | `semisegecg-resnet18` |

survival endpoint를 전체 사망으로 맞춘 이유: 두 survival 모델이 공통으로 평가될 수 있는
유일한 endpoint다 (ECG2AF는 사망 head가 있고, cavalab은 사망으로 학습했다).

## 산출물과 위치

```
registry/drafts/<model_id>.json        카드 초안 9개
studies/drafts/<study>/study.yaml      study 초안 4개
docs/decisions/adaptations.md          새 L2 항목
docs/model-search/schema-gaps.md       카드·study별 판정 한 장 요약
```

`registry/models/`에 두지 않는 이유: preprocess가 그 디렉터리의 **모든** 카드를
컴파일한다 (`src/mival/stages/preprocess.py:250`). 새 카드를 넣으면 LVEF의 레시피와
`config_hash`가 바뀐다. 이 사실 자체도 gap으로 기록한다 (study가 모델을 디렉터리 단위로 고른다).

## 판정 방법

느낌이 아니라 지금 코드를 통과시켜 본다. 검사는 scratchpad의 일회용 스크립트로 하고
repo에 넣지 않는다.

- 카드: `mival.modelcard.load_card`
- study: `mival.pipeline.study.load_study`, 그리고 stage별 `*Spec.from_mapping`

판정은 셋이다.

1. **통과**: 지금 스키마로 뜻대로 실행된다.
2. **로드되지만 뜻이 틀림**: 예를 들어 ECG2AF를 `logits`로 적으면 로드되지만 head 4개를
   버리고 생존곡선을 확률로 읽는다. 카드 `notes`와 원장에 적는다.
3. **표현 불가**: 필요한 필드를 카드의 `x-mival-proposed` 블록(study는 `x-proposed`)에
   제안 형태로 적는다. 제안은 이 모델에서 본 사실만 적고, 일반화된 스키마를 설계하지 않는다.

loader가 모르는 키는 무시되므로 `x-mival-proposed`가 있어도 `load_card`는 통과할 수 있다.
그래서 판정 2와 3은 로드 결과만으로 정하지 않고, 카드에 적은 내용을 기존 adapter·stage가
어떻게 읽는지 코드 위치와 함께 근거를 남긴다.

## 카드에 채우지 못하는 값

weight를 받지 않으므로 sha256, 서버 경로(`uri`), `code_commit`, runtime 버전은 `null`로 두고
`notes`에 "unverified"라고 쓴다. 대표 모델 검색 때 실제로 열어 확인한 값(출력 shape, 클래스
순서, 입력 단위 실험 등)은 출처를 `notes`에 적는다. 문서끼리 모순되는 값(Lima 입력 단위,
SemiSegECG 라이선스, cavalab MIMIC crop)은 모순 그대로 적는다.

## 이미 보이는 gap (초안 작성 중 확인할 것)

| gap | 근거 모델 | 위치 |
|---|---|---|
| `output.type`이 softmax/logits/sigmoid/regression 넷뿐 | ECG2AF, cavalab, SemiSegECG | `src/mival/adapters/torch_adapter.py:334` |
| 출력 head가 여러 개인 모델 | ECG2AF (5개) | 카드 `output` |
| 생존곡선을 기준 시점 위험 하나로 줄이는 규칙 | ECG2AF | — |
| baseline hazard가 없어 절대 위험을 못 냄 | cavalab | — |
| ECG 한 건에 값 하나가 아닌 샘플별 출력 | SemiSegECG | models/evaluate 예측 형식 |
| lead를 하나씩 넣는 모델 | SemiSegECG | `input_contract` |
| retrieve가 측정값 하나를 cutoff로 자르는 형태만 받음 | af, ecg-age, mortality, delineation | `src/mival/stages/retrieve.py:150` |
| 분류가 아니면 regression으로 갈라지는 models 분기 | survival, segmentation | `src/mival/stages/models.py:771` |
| study가 모델을 registry 디렉터리 단위로 고름 | 전부 | `src/mival/stages/preprocess.py:250` |
| keras 3 `.keras` 포맷 | ECG2AF | keras env는 2.7 |

이 표는 시작점이다. 최종 목록은 초안을 쓰고 판정한 결과로 정한다.

## 원장 기록

표현 불가(판정 3)와 뜻이 틀린 로드(판정 2)는 `docs/decisions/adaptations.md`의 L2 목록에
기존 형식(제목, 설명, 발견, 상태 "보고됨. 만들지 않음", 착수 조건)으로 하나씩 올린다.
같은 원인의 gap은 한 항목으로 묶고 근거 모델을 나열한다.

## 하지 않는 것

- 스키마, loader, adapter, stage 코드 변경
- weight 다운로드, 서버 반입, 추론
- segmentation·survival·regression의 정식 5개 선정
- LVEF L1 수정과 전체 실행 (이 작업 다음)
