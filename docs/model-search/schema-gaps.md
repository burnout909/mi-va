# 스키마 gap 판정 (2026-10-06)

[스펙](../superpowers/specs/2026-10-06-task-draft-cards-design.md)대로 카드 17개와 study 6개를
지금 코드에 넣어 본 결과다. 카드는 `registry/drafts/`, study는 `studies/drafts/`에 있다. 원장 번호는
[`adaptations.md`](../decisions/adaptations.md)의 L2 목록이다.

판정: **1** 통과 (뜻대로 실행) · **2** 로드되지만 뜻이 틀림 (대체값으로 실행은 됨) · **3** 표현 불가
(대체값으로도 실행 경로가 성립하지 않거나, 제안 필드 없이는 결과에 의미가 없음).

카드 17개 모두 `load_card`를 통과했다. 판정 2·3은 로드 결과가 아니라 아래 코드 근거로 내렸다.

## 카드

| model_id | task | 판정 | 근거 (코드 위치) | 원장 |
|---|---|---|---|---|
| heartwise-lvef-under50 | LVEF 분류 | 1 | 기존 heartwise-lvef-binary와 같은 계약. **LVEF 실행에 바로 넣을 수 있음** (sha256을 서버 manifest에서 채운 뒤) | — |
| echonext-mini | LVEF 분류 | 3 | 입력이 (B,1,2500,12) 4-D와 tabular 7개. layout은 둘뿐 (`adapters/base.py:35`), tabular 축 없음 | L2-3, L2-9 |
| lima-ecg-age | 회귀 | 2 | 10 s를 4096 샘플로 가운데 패딩. `pad_policy: reject`면 `duration_short`, `zero`면 끝에만 패딩 (`compiler.py:111-117`, Pad anchor 미전달) | L2-9 |
| singstad-ecg-age | 회귀 | 2 | `dtype: int32`가 기록만 되고 적용되지 않음 (`stages/preprocess.py:217`, `signal.py:40`은 float32 강제). 영향 작음 | L2-9 |
| kardionet-k-12lead | 회귀 | 2 | 대상 코호트 통계로 하는 lead별 z-score를 레코드 단위로 근사. wavelet denoising 없음 (`ops.py:257`) | L2-2, L2-9 |
| vonbachmann-k | 회귀 | 3 | 출력이 z-score라 역변환 없이는 mmol/L 라벨과 비교 불가. torch adapter는 `weights[0]`만 읽어 5개 앙상블이 첫 member가 됨 (`torch_adapter.py:122`). 4096 샘플 패딩은 lima와 같음 | L2-8, L2-14, L2-9 |
| ai-ntprobnp | 회귀 | 3 | 출력이 log 값. 역변환·라벨 log 변환 둘 다 없음. 5개 앙상블이 첫 member가 됨 (`torch_adapter.py:122`). z-score 뒤 median 빼기 없음. 앞 2048 샘플 crop은 Crop 기본 anchor `start`와 맞음 | L2-8, L2-14, L2-9 |
| ml4h-ecg2af | 생존 | 3 | `.keras` keras 3 형식, keras env는 2.7 (`keras_adapter.py:75`). head 5개 중 사망 생존곡선 선택 불가. 곡선을 확률로 읽음 (`torch_adapter.py:325-334`와 같은 4종) | L2-12, L2-7, L2-8 |
| ml4h-ecg2hf | 생존 | 3 | ECG2AF와 같음. 라이선스가 출력에도 적용됨 (비상업) | L2-12, L2-7, L2-8 |
| ml4h-ecg2stroke | 생존 | 3 | `.h5`라 로드는 될 수 있으나 head 선택·곡선 해석 불가. 정규화와 bin 폭이 README와 코드에서 다름 | L2-7, L2-8 |
| cavalab-deepsurv-code15 | 생존 | 2 | `regression`으로 적으면 log-risk를 연속 라벨과 비교. baseline hazard가 없어 절대 위험을 못 냄. 가운데 7 s crop 불가 (`compiler.py:109`가 anchor 없이 Crop) | L2-7, L2-8, L2-9 |
| cavalab-mtlr-code15 | 생존 | 3 | MTLR logit 100개를 softmax로 읽음. `positive_index: null`이라 추론에서 `int(None)` (`torch_adapter.py:328`). 나이·성별 입력 없음 | L2-7, L2-8, L2-3, L2-9 |
| openecg-codec-v6 | 분할 | 3 | adapter `onnx` 없음 (`adapters/__init__.py:9`). 샘플별 마스크, single lead, rank 정규화 | L2-12, L2-7, L2-9 |
| hrnetv2-delineation | 분할 | 3 | 샘플별 multi-label 마스크. z-score 뒤 ×0.1인데 gain은 정규화 **전**에 적용되어 사라짐 (`compiler.py:121-125`). `positive_index: null` | L2-7, L2-9 |
| semisegecg-resnet18 | 분할 | 3 | (B,4,2500) 샘플별 softmax. single lead. `positive_index: null` | L2-7, L2-9 |
| semisegecg-vit-tiny | 분할 | 3 | resnet18과 같음 | L2-7, L2-9 |
| heartkit-seg-tcn | 분할 | 3 | 2.56 s 창 하나로 crop됨 (10 s를 창으로 나눠 이어 붙이는 단계 없음). `.keras` 형식. 샘플별 마스크 | L2-9, L2-12, L2-7 |

카드 작성 누락 (gap 아님): 회귀 카드 6개(lima, singstad, kardionet, vonbachmann, ai-ntprobnp,
cavalab-deepsurv)에 `output.value_index`가 없다. 기존 adapter는 이 키를 요구한다
(`torch_adapter.py:327`, `keras_adapter.py:99`). `registry/models/`로 옮길 때 `"value_index": 0`을 넣는다.

## study

| study | 판정 | 막히는 stage | 근거 | 원장 |
|---|---|---|---|---|
| lvef | 3 | models | 기존 arm 4개가 `registry/drafts`에 없음. study는 registry 하나만 받음 (`stages/preprocess.py:250`). EchoNext의 ≤45 cutoff는 열이 없어 `primary`(≤40)로 대체 (`retrieve.py:43`) | L2-13, L2-10 |
| ecg-age | 3 | retrieve (뜻), evaluate | retrieve는 **통과**하지만 `label_concept_id` 기본값 3027172(LVEF)를 조용히 씀 (`retrieve.py:59`). evaluate는 `regression_cuts: []`를 거부 (`evaluate.py:391`). 나이에는 자연스러운 cut이 없음 | L2-10, L2-11 |
| potassium | 3 | retrieve | `label_concept_id: null` → `int(None)` 실패. 하루 미만 창, 높은 값이 양성인 cutoff 방향 표현 불가 (`retrieve.py:150`) | L2-10 |
| ntprobnp | 3 | retrieve | potassium과 같음. 라벨 log 변환 없음 | L2-10, L2-8 |
| mortality | 3 | retrieve, models, evaluate | 사건 여부·추적 기간·censoring 열 없음. arm이 `primary`/`value`를 빌려 씀 (`models.py:216`). inference_only는 회귀/분류로만 갈림 (`models.py:771`). evaluate는 `regression_cuts`를 생략하면 기본값 40을 조용히 씀 (`evaluate.py:388`) | L2-10, L2-7, L2-11 |
| delineation | 3 | retrieve, preprocess, models, evaluate | 기준값이 같은 ECG의 기계 측정 여러 개. 12-lead → single lead 12개 fan-out 없음. 예측이 레코드당 값 하나라는 전제 | L2-10, L2-9, L2-7, L2-11 |

## gap별 묶음

| L2 | gap | 근거 모델·study |
|---|---|---|
| L2-2 | 코호트 단위 정규화 (기존) | kardionet-k-12lead 추가 |
| L2-3 | waveform 외 tabular 입력 (기존) | echonext-mini, cavalab-mtlr-code15 추가 |
| L2-7 | 출력이 레코드당 값 하나뿐 | 생존 5개, 분할 5개, `mortality`, `delineation` |
| L2-8 | 출력을 라벨 척도로 바꾸는 후처리 | vonbachmann-k, ai-ntprobnp, ml4h 3개, cavalab 2개, `ntprobnp` |
| L2-9 | 입력 준비 단계의 표현 범위 | echonext-mini, lima-ecg-age, singstad-ecg-age, kardionet-k-12lead, vonbachmann-k, ai-ntprobnp, cavalab 2개, 분할 5개, `delineation` |
| L2-10 | 라벨이 측정값 하나를 cutoff 둘로 자른 것뿐 | `lvef`, `ecg-age`, `potassium`, `ntprobnp`, `mortality`, `delineation` |
| L2-11 | 평가가 이진·회귀 둘뿐 | `ecg-age`, `mortality`, `delineation` |
| L2-12 | keras 3·ONNX 형식 | ml4h-ecg2af, ml4h-ecg2hf, heartkit-seg-tcn, openecg-codec-v6 |
| L2-13 | study가 registry 디렉터리 하나를 통째로 씀 | `lvef` (모든 초안 study) |
| L2-14 | torch 앙상블이 첫 member만 씀 | vonbachmann-k, ai-ntprobnp |

### 스펙의 "이미 보이는 gap" 18개가 간 곳

| 스펙의 gap | 결과 |
|---|---|
| `output.type`이 넷뿐 | L2-7 |
| 출력 head가 여러 개 | L2-7 |
| survival 출력 인코딩 셋 | L2-7 |
| 생존곡선을 기준 시점 위험으로 줄이는 규칙 | L2-8 |
| baseline hazard 없음 | L2-8 |
| 샘플별 출력 | L2-7 |
| lead를 하나씩 넣는 모델 | L2-9 |
| 출력을 원래 단위로 되돌리는 변환 | L2-8 |
| fold 앙상블 평균 | L2-14. 예상보다 나쁨: torch adapter는 member 하나만 읽는다 |
| 8-lead 입력 | gap 아님. lead 선택으로 처리되고 vonbachmann-k가 로드됨 |
| waveform 외 공변량 | L2-3 (근거 추가) |
| 여러 label 중 하나를 고르는 inference_only | gap 아님. `positive_index: 0`으로 표현됨 (echonext-mini) |
| cutoff가 둘로 고정, ≤45 불가 | L2-10 |
| retrieve가 측정값 하나만 받음 | L2-10 |
| models의 회귀/분류 분기 | L2-7 |
| study가 모델을 registry 디렉터리 단위로 고름 | L2-13 |
| keras 3 `.keras` | L2-12 |
| ONNX·TFLite | L2-12 |

새로 드러난 것 (스펙 표에 없던 것): Crop·Pad anchor가 compiler에서 항상 `start` (L2-9), gain이 정규화
전에 적용되는 순서 (L2-9), `dtype` 미적용 (L2-9), `label_concept_id`와 `regression_cuts`의 LVEF 기본값이
다른 study에서 조용히 쓰임 (L2-10, L2-11).
