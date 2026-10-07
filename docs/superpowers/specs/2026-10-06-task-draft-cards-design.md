# task별 카드·study 초안으로 스키마 gap 드러내기

2026-10-06. LVEF의 L1 수정과 전체 실행 **전에**, task마다 모델 5개를 카드와 study.yaml로
적어 보고 지금 프레임워크가 표현하지 못하는 것을 목록으로 만든다.

## 목적

지금 프레임워크는 STEMI와 LVEF 모델로만 만들어졌다. 다른 출력 형태(생존곡선, 샘플별 마스크,
여러 head)나 다른 라벨(사망까지의 시간, 혈액검사 값, 생년에서 계산한 나이)을 넣으면 어디가
안 맞는지 아직 모른다. LVEF 코드를 고치기 전에 이것을 먼저 드러낸다.

- 산출물은 **"프레임워크가 지원하지 못하는 것 목록"**이다. 실행 가능한 카드가 아니다.
- 이 목록을 보고 LVEF L1 수정 범위와, 어떤 L2를 언제 만들지 정한다.
- 코드는 바꾸지 않는다. L2는 보고만 한다 (수정 분류 L0/L1/L2 원칙).
- weight를 새로 받거나 서버에 반입하지 않는다. 서버를 켜지 않는다.

## 범위

task 네 개에 모델 5개씩. 후보와 선정 근거는
[task별 후보 모델](../../model-search/task-candidates-2026-10-06.md)에 있다.

| task | 무엇으로 | 5개 | 새로 쓰는 카드 |
|---|---|---|---|
| classification | **LVEF** | HeartWise ≤40, HeartWise <50, EchoNext-Mini, ECGFounder, xECG | 2 (카드 없는 HeartWise <50, EchoNext) |
| regression | 나이·칼륨·NT-proBNP | Lima age, Kardio-Net K⁺, AI-NT-proBNP, von Bachmann K⁺, Singstad age | 5 |
| survival | 전체 사망 | ECG2AF, cavalab DeepSurv, cavalab MTLR, ECG2HF, ECG2Stroke | 5 |
| segmentation | P/QRS/T 구간 | OpenECG codec_v6, HRNetV2, SemiSegECG ResNet-18, HeartKit TCN, SemiSegECG ViT-Tiny | 5 |

새 카드는 **17개**다. LVEF의 나머지 세 모델(HeartWise ≤40, ECGFounder, xECG)은 카드가 이미 있다.

- classification을 AF가 아니라 LVEF로 하는 이유: LVEF가 원래 진행 중인 task다. AF 검색 결과는
  [모델 탐색 장부](../../model-search/af-classification-ledger.csv)에 두고 이번 범위에서 뺀다.
- regression 대상이 셋으로 갈린 이유: 나이만으로는 공개 모델 5개가 안 나온다. 셋 다 MIMIC-IV에
  정답이 있는 값이다 (나이는 person 생년, 칼륨과 NT-proBNP는 검사 결과).
- survival endpoint를 전체 사망으로 맞춘 이유: 다섯 모델 모두 사망 head가 있거나 사망으로 학습했다.
  ml4h 세 모델의 주 endpoint(AF, HF, 뇌졸중)는 이번 범위에서 다루지 않는다.
- segmentation은 MIMIC에 사람이 그은 P/QRS/T 구간이 없어서, 마스크에서 계산한 간격을 기계 측정
  PR/QRS/QT와 비교하는 방식으로 적는다.

모든 모델이 학습에 MIMIC-IV-ECG를 쓰지 않았다. 문서상 약점은 그대로 둔다: 라이선스 없음
(AI-NT-proBNP, Singstad), 논문 없음(OpenECG, HRNetV2, HeartKit), 같은 계열(ml4h 셋, cavalab 둘,
SemiSegECG 둘).

### study 초안

| study | 라벨 | 대상 |
|---|---|---|
| `lvef` | 기존 LVEF 라벨 + 모델별 cutoff (<50, ≤45) | 기존 arm 6개 + HeartWise <50, EchoNext |
| `ecg-age` | person 생년으로 계산한 ECG 시점 나이 | Lima, Singstad |
| `potassium` | ECG 전후 혈청 칼륨 | Kardio-Net, von Bachmann |
| `ntprobnp` | ECG 전후 NT-proBNP (log) | AI-NT-proBNP |
| `mortality` | 전체 사망까지의 시간과 censoring (CDM death) | survival 5개 |
| `delineation` | MIMIC-IV-ECG 기계 측정 PR/QRS/QT | segmentation 5개 |

`lvef` 초안은 기존 `studies/lvef/study.yaml`을 고치지 않고 복사본에 arm을 더한다.

## 산출물과 위치

```
registry/drafts/<model_id>.json        카드 초안 17개
studies/drafts/<study>/study.yaml      study 초안 6개
docs/decisions/adaptations.md          새 L2 항목
docs/model-search/schema-gaps.md       카드·study별 판정 한 장 요약
```

`registry/models/`에 두지 않는 이유: preprocess가 그 디렉터리의 **모든** 카드를 컴파일한다
(`src/mival/stages/preprocess.py:250`). 새 카드를 넣으면 LVEF의 레시피와 `config_hash`가 바뀐다.
이 사실 자체도 gap으로 기록한다. HeartWise <50을 `registry/models/`로 옮겨 LVEF 전체 실행에
넣을지는 이 작업 다음, LVEF 실행을 준비할 때 정한다.

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
- **나머지 15개**: weight를 받지 않으므로 sha256, 서버 경로(`uri`), `code_commit`, runtime 버전은
  `null`로 두고 `notes`에 "unverified"라고 쓴다. 후보 검색 때 실제로 열어 확인한 값(출력 shape,
  클래스 순서, 입력 단위 실험)은 출처와 함께 `notes`에 적는다.
- 문서끼리 모순되는 값(Lima 입력 단위, SemiSegECG 라이선스, ECG2Stroke 정규화·bin 폭)은 모순
  그대로 적는다.

## 이미 보이는 gap

초안을 쓰면서 확인할 시작점이다. 최종 목록은 판정 결과로 정한다.

| gap | 근거 모델·study | 위치 |
|---|---|---|
| `output.type`이 softmax/logits/sigmoid/regression 넷뿐 | survival 5개, segmentation 5개 | `src/mival/adapters/torch_adapter.py:334` |
| 출력 head가 여러 개인 모델 | ml4h 셋, OpenECG | 카드 `output` |
| survival 출력 인코딩이 셋 (구간 조건부 생존확률, Cox log-risk, MTLR) | survival 5개 | — |
| 생존곡선을 기준 시점 위험 하나로 줄이는 규칙 | ml4h 셋, cavalab MTLR | — |
| baseline hazard가 없어 절대 위험을 못 냄 | cavalab DeepSurv | — |
| ECG 한 건에 값 하나가 아닌 샘플별 출력 | segmentation 5개 | models·evaluate 예측 형식 |
| lead를 하나씩 넣는 모델 | segmentation 5개 | `input_contract` |
| 출력을 원래 단위로 되돌리는 변환 (z-score 역변환, log, 곱셈) | von Bachmann, AI-NT-proBNP | 카드 `output` |
| fold 앙상블 평균 | AI-NT-proBNP (5), von Bachmann (5) | 카드 `ensemble` (keras 쪽만 검증됨) |
| 8-lead 입력 | von Bachmann | `input_contract` (PROPHECG에서 이미 씀, 확인만) |
| waveform 외 공변량 입력 (기존 L2-3) | EchoNext (7개), cavalab MTLR (나이·성별) | `input_contract` |
| 여러 label 출력 중 하나를 고르는 inference_only | EchoNext (12개 중 index 0) | 카드 `output.positive_index` |
| cutoff가 40·50 두 개로 고정, ≤45 표현 불가 | EchoNext, `lvef` | `src/mival/stages/retrieve.py:43` |
| retrieve가 측정값 하나를 cutoff로 자르는 형태만 받음 | `ecg-age`, `potassium`, `ntprobnp`, `mortality`, `delineation` | `src/mival/stages/retrieve.py:150` |
| 분류가 아니면 regression으로 갈라지는 models 분기 | survival, segmentation | `src/mival/stages/models.py:771` |
| study가 모델을 registry 디렉터리 단위로 고름 | 전부 | `src/mival/stages/preprocess.py:250` |
| keras 3 `.keras` 포맷 | ECG2AF, ECG2HF | keras env는 2.7 |
| ONNX·TFLite 형식 | OpenECG | adapter는 torch·keras 둘 |

## 원장 기록

판정 2와 3은 `docs/decisions/adaptations.md`의 L2 목록에 기존 형식(제목, 설명, 발견, 상태 "보고됨.
만들지 않음", 착수 조건)으로 올린다. 원인이 같은 gap은 한 항목으로 묶고 근거 모델을 나열한다.
이미 있는 항목(L2-3 tabular 등)은 새로 만들지 않고 근거 모델만 더한다.

## 하지 않는 것

- 스키마, loader, adapter, stage 코드 변경
- weight 다운로드, 서버 반입, 추론
- 기존 `studies/lvef/study.yaml`, `registry/models/` 수정
- ml4h 모델의 주 endpoint(AF, HF, 뇌졸중) study
- LVEF L1 수정과 전체 실행 (이 작업 다음)
