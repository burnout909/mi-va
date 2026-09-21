# Adaptation Ledger

실제 데이터를 통과시키면서 프레임워크에 가한 수정의 기록이다.

목적은 기록 자체가 아니라 **한 건에만 맞는 코드가 쌓이는 것을 막는 것**이다.
수정 요구를 만나면 아래 사다리에서 위부터 시도하고, 위에서 해결되면 아래로
내려가지 않는다.

| 레벨 | 정체 | 조치 | 코드 |
|---|---|---|---|
| L0 | 모델·기관마다 다른 게 당연한 값 | ModelCard, study.yaml, registry, configs에 값 추가 | 0줄 |
| L1 | 프레임워크가 틀렸거나 통과시키지 말았어야 할 입력을 통과시킴 | 고치고 회귀 테스트를 반드시 붙인다 | 해당 모듈 |
| L2 | 새 축, 새 개념 | **즉시 하지 않고 보고한다.** | 구조 변경 |

세 단계 모두 human-in-the-loop으로 코드를 수정한다. 분류와 반영은 사람이
판단하고, 판단의 근거를 여기 남긴다.

## 이 문서에 들어가지 않는 것

**환경 구축은 여기 적지 않는다.** 데이터베이스를 세우고 DDL을 맞추고 스키마를
넓힌 일은 프레임워크가 부족했다는 증거가 아니다. 섞으면 "3회 누적 후 추상화"
카운트가 오염된다. 그 기록은
[`../operations/micdm-database.md`](../operations/micdm-database.md)에 있다.

## 이 규율을 지키는 방법

기본은 **코드 리뷰**다. 리뷰어가 볼 것은 하나다. 이 수정이 L0에서 끝날 수
있었는가.

리뷰를 돕는 두 가지가 있다.

- **N=2 불변식** — 모든 수정은 ECGFounder(torch · 12유도 · feature 추출)와
  PROPHECG(keras · 8유도 · 5멤버 앙상블) 양쪽에서 통과해야 한다. 한쪽만
  통과하면 그것은 정의상 quirk이므로 L0으로 내린다. 두 모델이 이미 충분히
  비대칭이라 별도 장치가 필요 없다.
- `tests/golden` — 스냅샷 회귀. 숫자가 바뀌면 커밋 메시지가 이유를 적는다.

`tests/test_no_proper_nouns.py`는 `src/mival/`의 식별자와 문자열 값에서
모델명·기관명을 찾는 기계적 검사다. 주석과 독스트링은 허용한다. 도입 여부는
아직 결정하지 않았다. 붙였을 때 `evaluate.py`의 `OUTCOME_DIAGNOSTIC`(A-2)과
`misclassify.py`의 `stemi_mimic`(A-3) 두 건을 잡았고, 후자는 절반이 오탐이었다.

---

## 기록

### 2026-08-19

#### A-1 · L0 · `local_path`가 다른 기관의 파일시스템 경로

MI-CDM `image_occurrence.local_path`의 값이
`/home/ubuntu/dryou_mount/Datasets/ECG/open_datasets/MIMIC/mimic-iv-ecg-dcm/files/p1000/...`
로 시작한다. ETL을 수행한 장비의 경로이고 DICOM-MIVA에는 없다.

기관마다 다른 것이 당연한 값이므로 L0이다. `configs/aws/*.yaml`에 이미
`retrieval.local_path_root: /scratch/mi-val/dicom`가 선언돼 있으므로, retrieve가
CDM이 준 경로에서 파일 부분을 취해 이 root 아래로 해석한다.

코드에 `dryou_mount`를 적으면 그 순간 site 축이 오염된다. 다른 기관의 MI-CDM은
또 다른 접두사를 가질 것이고, 그때 필요한 것은 코드 수정이 아니라 설정 한 줄이다.

- 조치: retrieve 구현 시 prefix 재해석. 코드 0줄 추가 없음
- 상태: retrieve 구현으로 종료
- 검증: 서로 다른 `local_path_root` 두 개로 같은 cohort가 해석되는 테스트

#### A-2 · 관찰 · `OUTCOME_DIAGNOSTIC = "stemi"`

`src/mival/stages/evaluate.py`가 `outcome` report 축의 기본값으로 이 연구의
질환명을 갖고 있다. 다른 결과를 검증하는 study도 결과 표에 `stemi`를 받는다.

중립 sentinel
(`primary`)로 바꾸거나 evaluation spec에서 읽어야 한다. 값을 바꾸면 기존 산출물의
축 값이 달라지므로 **연구 결정이 필요하다.**

- 상태: **결정 대기**

#### A-3 · 관찰 · `DEFAULT_VERDICTS`의 질환 특정 어휘

`src/mival/stages/misclassify.py`의 기본 verdict 어휘에 `stemi_mimic`이 있다.
여기서 `mimic`은 데이터셋이 아니라 "STEMI 유사 소견"이라는 임상 용어이므로
게이트 관점에서는 절반이 오탐이다.

다만 질환 특정 기본값이 core에 있는 것은 사실이다. 독스트링이 이미 study가
`misclassify.verdicts`로 자기 어휘를 선언한다고 적고 있어, 숨은 선택이 아니라
문서화된 선택이다. 지금은 그대로 둔다.

- 상태: **의도된 기본값으로 유지**

### 2026-09-21

LVEF 과제용 모델 3종(xECG, HeartWise DeepECG-SL LVEF, EchoNext-Mini)을 DICOM-MIVA에
반입하면서 발견한 항목이다. 모델 문서는 `docs/models/`에 있다. 코드는 아직 고치지
않았고 분류만 했다.

#### A-4 · L1 · torch adapter가 ECGFounder 전용 builder를 하드코딩한다

`src/mival/adapters/torch_adapter.py::load`가 `from net1d import Net1D`와 ECGFounder의
생성자 인자를 코드에 직접 갖고 있다. 두 번째 torch 모델을 넣는 순간 드러났다.
새 모델 3종의 weights 형식이 서로 다르다.

| 모델 | 형식 | 만드는 법 |
|---|---|---|
| HeartWise binary | `torch.jit` | `torch.jit.load` (아키텍처 코드 불필요) |
| HeartWise 회귀 | state_dict | `EfficientNet1DV2(num_classes=1, expansion_factors=[1,2,2,2,2,2,2])` + 키 이름 변경 |
| EchoNext-Mini | `{"model": state_dict}` | `ResNet1dWithTabular(7, filter_size=16, num_classes=12)` |
| xECG | safetensors | `xECG(cls_type, config)` |

모델명이 core에 있는 것은 `tests/test_no_proper_nouns.py`가 막으려는 바로 그것이다.
카드의 `x-mival`에 `weights_format`(`jit` / `state_dict` / `safetensors`)과
`builder`(모듈 경로와 인자)를 선언하고 adapter가 그것을 읽게 한다. 회귀 테스트는
ECGFounder 카드가 기존과 같은 module을 만드는지 확인하는 것으로 충분하다.

- 상태: 종료. 카드 `weights_format`/`builder`/`runtime.returns`

#### A-5 · L1 · 출력 유형에 회귀가 없다

`output.type`이 `logits` / `softmax`뿐이다. HeartWise `LVEF_MSE_SL.pt`는 LVEF %를
직접 낸다. `positive_index`와 threshold 정책이 의미를 잃고, evaluation 지표도
AUROC 계열이 아니라 MAE·R²가 된다. 회귀냐 분류냐는 아직 연구 결정 전이므로
(decisions 참조) 결정 전까지 회귀 체크포인트는 카드에 넣지 않는다.

- 상태: 종료. `output.type: regression`, 연구 결정은 회귀·분류 병행

#### A-6 · 관찰 · sigmoid를 어디서 취하는가

HeartWise binary 모델은 logit을 내고 저자 wrapper가 sigmoid를 취한다. 현재 카드
`output.type: logits`가 이 경우를 이미 뜻하므로 L0이다. 다만 EchoNext는 12개 logit
중 index 0만 LVEF라 `positive_index`가 "여러 출력 중 하나"를 가리키게 된다.
ECGFounder의 150 logits와 같은 상황이므로 기존 의미와 충돌하지 않는다.

- 상태: **L0, 카드에서 선언**

#### A-7 · L1 · 라벨이 이진이라는 가정

회귀 arm이 나오면서 `label_value`, `pred_value` 컬럼을 추가했다. 회귀 테스트를
붙였고, 분류 golden은 변경 없이 그대로 통과한다.

- 상태: 종료

#### A-8 · L0 · `input_contract.gain`

HeartWise의 `1/0.0048` 상수(저자 `efficientnet_wrapper.mhi_factor`)를 카드
`input_contract.gain`으로 선언했다. L2-2(코호트 단위 스펙트럼 스케일링)의
임시 조치이고, 코호트 단위 정규화 자체는 여전히 만들지 않는다.

- 상태: 종료

---

## L2 보고 목록

즉시 만들지 않고 여기 올린다. 한 건을 보고 지은 추상은 그 한 건에만 맞으므로,
여러 모델과 여러 기관에서 반복되는 것을 본 뒤에 만든다.

### L2-1 · 반출 경계가 1급 개념이 아니다

신촌 데이터셋은 병원 밖으로 나갈 수 없다. 데이터가 오는 것이 아니라 코드가
가고, 나오는 것은 집계물뿐이다.

`manifest.json`, exclusion ledger, 지표 표는 반출 가능한 형태다. 그런데 6단계
misclassification은 **산출물 자체가 환자 단위**다. `top_errors`, `boundary`,
review CSV는 개별 케이스이고 임상의가 보라고 만든 것이다. 병원 안에서만 볼 수
있다.

지금은 stage 산출물에 "이것이 site 밖으로 나갈 수 있는가"를 표시하는 개념이
없다. 반출 묶음을 만들 때 환자 단위 산출물이 섞이지 않도록 강제할 자리가 필요하다.

- 발견: 2026-08-19, 신촌 접근 조건 확인 중
- 상태: **보고됨. 만들지 않음**
- 착수 조건: 신촌 실행이 실제로 준비될 때. MIMIC 단독 실행에는 필요 없다


### L2-2 · 코호트 단위 정규화

HeartWise 파이프라인의 첫 전처리는 **배치 전체**의 평균 스펙트럼 파워를
`PTBXL_POWER_RATIO`에 맞추는 것이다. 현재 `input_contract.scaling`은 레코드 단위
(`global_zscore`, `per_lead_zscore`)만 있다. 코호트 단위 값은 preprocess 단계의
산출물이 어느 코호트로 계산됐는지에 따라 달라지므로 `config_hash`에 코호트가
들어가야 한다. 한 모델에서만 본 것이라 만들지 않는다.

- 임시 조치 후보: 저자 파이프라인으로 전체 코호트를 한 번 스케일링해 그 factor를
  카드에 상수로 적는다. 그러면 레코드 단위 `scale` op로 환원된다. 이 factor가
  MIMIC 코호트에 의존한다는 것을 카드에 적어야 한다.
- 발견: 2026-09-21
- 상태: **보고됨. 만들지 않음**

### L2-3 · waveform 외 tabular 입력

EchoNext-Mini는 waveform과 함께 연령·성별·기계 계측 5종을 받는다. 현재
`input_contract`는 waveform 하나만 기술한다. tabular 축은 retrieve(어느 테이블에서
가져오는가), preprocess(결측 처리·표준화), models(adapter의 forward 서명) 세 단계를
모두 건드린다.

- 발견: 2026-09-21
- 상태: **보고됨. 만들지 않음**. EchoNext-Mini를 4개 모델에 포함할지가 이 항목의
  착수 조건이다.

### L2-4 · 백엔드가 수치를 바꾸는 모델

xECG의 sLSTM은 `cuda` backend(bf16 커스텀 커널)와 `vanilla` backend(float32 torch)
가 있고, 체크포인트의 파라미터 레이아웃이 backend마다 다르다. 저자 코드는
vanilla용 변환(`_recurrent_kernel_.permute(0, 2, 1)`)을 갖고 있다. 어느 backend로
평가했는지가 재현성 축이 되므로 카드 `runtime`에 backend를 적고 manifest에도
남겨야 한다.

- 발견: 2026-09-21
- 확인 결과 (2026-09-21): cuda backend 컴파일 성공. vanilla backend는 저자 변환만으로는
  출력이 다르고(코사인 0.32), bias까지 재배열하면 일치한다(코사인 1.00000, 상대 오차 0.25%).
  **cuda backend를 정본으로 쓴다.** 카드 `runtime`에 `slstm_backend: cuda`를 적는다.
- 상태: **보고됨. backend는 카드 값(L0)으로 처리. 새 축은 만들지 않음**
