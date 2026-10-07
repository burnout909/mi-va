# Decisions and Open Questions

회의에서 합의된 내용과 추가 논의가 필요한 내용을 구분해 관리한다.

## 합의된 내용

### 연구 전제

- MI-CDM ETL 완료 상태를 전제로 한다.
- cohort는 ATLAS Data4Life에서 정의한다.
- index date와 observation window도 cohort 정의 과정에서 설정한다.

### 파이프라인

- 회의록 기준 6단계로 관리한다.
- 순서는 Retrieve, Profile, Preprocess, Models, Evaluation, Misclassification이다.
- Retrieve의 시작점은 `person_id`, `image_occurrence_id`, `local_path`다.

### 담당 범위

- 규리T: Retrieve, Profile, Preprocess.
- 민성: Models, Evaluation, DICOM-MIVA 구축.
- Misclassification 담당은 아직 정하지 않았다.

### 인프라

- ATLAS version은 Data4Life를 사용한다.
- AWS instance 접속은 RealVNC Viewer를 활용한다.
- DICOM-MIVA는 `dicom-etl`보다 높은 사양으로 구성한다.
- 2026-08-10 사용자 최종 지시에 따라 DICOM-MIVA root는 100 GB로 변경했다.
- persistent EBS는 400 GB이며, 별도 559 GB instance store를 임시 cache로 사용한다.

### ECG channel metadata

- DICOM마다 channel 순서가 다를 수 있다.
- DICOM 내부 channel 순서를 `image_feature_value_order`에 보존한다.
- order가 없는 값은 전체 ECG 수준 값으로 취급한다.

## 방향은 잡혔지만 확정이 필요한 내용

- Profile을 Fingerprint 버전에 기반해 설계한다.
- Models를 TaskSpec 버전에 기반해 설계한다.
- Evaluation을 Discrimination, Calibration, Interpretation으로 구분한다.
- 전체 과정과 결과를 하나의 Technical Report로 제공한다.

## Open Questions

### Preprocess

1. MI-CDM ETL validation 방식과 pydicom 방식 중 무엇을 기본으로 사용할 것인가?
2. 둘 다 지원한다면 동일 결과를 보장하는 공통 기준은 무엇인가?
3. 필수 metadata가 없을 때 default를 사용할 것인가, record를 제외할 것인가?

### Models

1. 사용자가 model architecture와 hyperparameter를 어디까지 바꿀 수 있는가?
2. PROPHECG-STEMI의 amplitude scaling과 5120→5000 변경 이력은 무엇인가?
   현재 `registry/models/prophecg-stemi.json`은 `input_contract.scaling`을
   `none`으로 선언해 두었다. 근거 문서가 없어 잠정 선언한 값이므로,
   확인되면 카드를 고치고 golden 값을 다시 생성해야 한다.
3. archived PTB threshold `0.0768`을 새 cohort에서 다시 추정할 것인가?
4. ECGFounder의 downstream STEMI head와 split protocol은 무엇인가?
5. `registry/models/prophecg-stemi.json`과 `registry/models/ecgfounder.json`의
   `x-mival.pretraining_corpora` 값(`institutional-ed-ecg-2022`,
   `harvard-emory-ecg`)은 잠정값이다. contamination gate(Plan 4)를 켜기 전에
   원 논문·모델 카드와 대조해 확정해야 한다.

2026-08-10 확인: PROPHECG는 Keras 2.7, 8-lead `(5000, 8)`, 2-class
softmax의 5-member mean ensemble이다. ECGFounder artifact와 code revision도
고정했으며 12-lead CUDA forward를 확인했다. 상세 값은 `docs/models/`에 있다.

### Preprocess (구현에서 제기됨)

4. perturbation 축 `powerline_50hz`는 MIMIC에 대해 틀린 값인가?
   spec §4.3은 50 Hz로 적고 있으나 MIMIC-IV의 출처 기관(Beth Israel
   Deaconess, 보스턴)의 상용 전원은 60 Hz다. 50 Hz 간섭은 이 기관 장비가
   만들어낼 수 없는 아티팩트이므로, 이 수준은 robustness를 측정하지 못하고
   해당 축의 한 칸을 낭비한다. `mival.perturbation`은 주파수를 수준 이름에서
   파싱하므로 `powerline_60hz`로 바꾸는 데 코드 변경은 필요 없다.
   확정되면 spec §4.3 표와 study.yaml을 함께 고친다.

5. perturbation의 `amplitude_scale` 축은 정규화 모델에 대해 무의미한가?
   perturbation은 저장된 tensor에 적용된 뒤 입력계약의 `scaling`이 다시
   적용된다. 따라서 `scaling: none`인 PROPHECG에는 실제 gain 오류이지만,
   `global_zscore`인 ECGFounder에는 정확히 아무 효과가 없다. 이것이 옳은
   답이라는 것이 구현 시점의 판단이다 — 실제로 잘못 보정된 ECG도 모델
   자신의 정규화를 통과하므로, 재정규화하지 않으면 배포 환경에서 일어날 수
   없는 robustness 실패를 보고하게 된다. 다만 결과표에서 이 축이 모델별로
   다른 의미를 갖는다는 점을 명시해야 한다.

6. 여러 모델이 한 record에 대해 서로 다르게 판정할 때 STARD flow의
   participant 수를 어떻게 센다고 선언할 것인가?
   exclusion ledger는 record당 한 행을 기록하므로(모델당이 아니라), 모델이
   엇갈리면 `in != out + excluded`가 된다. Preprocess stage가 경고를 내지만
   Figure 1의 계수 규약은 사람이 정해야 한다.

7. powerline 수준에는 이제 기본값이 없다. 무엇으로 선언할 것인가?
   spec §4.3은 `powerline_50hz`로 적지만 전원 주파수는 **건물의 사실**이다.
   MIMIC-IV는 보스턴, 한국 병원도 60 Hz이므로 50 Hz 간섭은 두 사이트 어느 장비도
   만들 수 없는 아티팩트다. 틀린 기본값은 없는 수준보다 나쁘다 — 셀을 채우고
   보고까지 되기 때문이다. 2026-08-14에 기본 grid에서 뺐으므로, study.yaml이
   `noise: [none, baseline_wander, powerline_60hz, emg]`로 선언해야 한다.
   선언하지 않으면 기본 OFAT 조건 수는 14가 아니라 13이다.

### Models (adapter `fit()` 구현에서 제기됨)

1. PROPHECG의 `feature_layer`는 무엇인가?
   카드에 `null`이라 `features()`가 동작하지 않고, 따라서 spec §2.5의
   "PROPHECG head-retrained (`linear_probe`)" arm이 실행되지 않는다. 이 arm은
   ECGFounder linear-probe와의 **대칭 비교**를 위해 존재하므로, 빠지면
   confirmatory 비교 2쌍 중 하나가 사라진다. H5의 `model.summary()`를 확인해
   penultimate layer 이름을 카드에 적으면 코드 변경 없이 해결된다.

2. 5-member ensemble을 어떻게 linear-probe할 것인가?
   `inference_only` arm은 5개 멤버의 확률을 평균한다. 독립 학습된 멤버들의
   표현은 좌표계를 공유하지 않으므로 평균이 무의미하고, 멤버 0만 probe하면
   같은 모델의 두 arm이 서로 다른 용량을 갖게 되어 "대칭" 비교가 성립하지
   않는다. keras adapter는 현재 이 경우를 **거부한다**(추측하지 않는다).
   선택지는 (a) 멤버마다 head를 학습하고 카드가 선언한 방식으로 확률을
   pooling, (b) 멤버 표현을 concat해 head 하나, (c) 단일 멤버로 arm을
   재정의하고 그 사실을 결과에 명시. 판단이 필요하다.

3. PROPHECG 카드의 `training_modes_supported`에서 `full_finetune`을 뺄 것인가?
   keras adapter는 fine-tuning을 구현하지 않았다. spec §2.5의 어떤 arm도
   요구하지 않고, archived Keras 2.7 런타임은 카드가 선언한 대로 CPU 전용이라
   5-member ensemble을 hospital-scale dev split으로 역전파하는 것은 가용
   하드웨어에서 실행 가능하지 않다. 카드가 선언만 하고 실행할 수 없는 상태를
   남길지, 선언을 현실에 맞출지 정해야 한다.

4. fitted head를 디스크에 남길 것인가?

   2026-08-14 결정: **남긴다.** models stage가 `heads/<arm>.npz`에 쓰고
   `train_log.jsonl`의 `fitted_head`가 경로를 기록한다. 이유는 head가 메모리에만
   있으면 학습된 arm에 대해 예측 재현도, attribution overlay도 불가능한데, head는
   작은 배열 4개라 저장 비용이 사실상 0이기 때문이다. 4단계 산출물이 하나 늘었다.
   관련 코드: `src/mival/stages/models.py` (`HEADS_DIR`, `_persist_heads`).

5. 5-member ensemble을 어떻게 linear-probe할 것인가? (위 2번의 후속)

   2026-08-14 결정: **멤버마다 head를 학습하고 카드가 선언한 `ensemble.method`로
   확률 수준에서 pooling한다.** 이유는 이것이 코드의 선택이 아니라 **카드가 이미
   답을 갖고 있던 것**이기 때문이다 — `inference_only`는 확률 수준에서 평균하므로,
   같은 위치에서 pooling하는 probe만이 두 arm을 같은 크기·같은 구조로 만든다.
   `Adapter.feature_set_names()`가 표현 목록을 내고 `_fit_linear_probe`가 그 수만큼
   head를 학습한다. 카드가 `mean_probability`가 아닌 pooling을 선언하면 거부한다.
   관련 코드: `src/mival/adapters/base.py`, `src/mival/adapters/keras_adapter.py`.

6. perturbation 그리드는 카드마다 다를 수 없다.
   `perturbation.axes`는 study 전체에 하나뿐인데, `resample`·`duration` 수준은
   **계약에 상대적**으로만 의미가 있다. 그래서 sampling rate나 duration이 서로 다른
   카드 둘은 하나의 그리드로 함께 검증할 수 없다(`check_grid_against_contract`가
   거부한다). 현재 두 카드 모두 500 Hz / 10 s라 영향이 없지만, 다른 계약의 모델을
   추가하면 C1("카드 하나만 추가")이 이 지점에서 깨진다. 축을 계약 상대값(예:
   `resample: [1.0, 0.5, 0.25]`)으로 재정의할지 판단이 필요하다.

7. `lead_dropout` 수준은 모델마다 다른 손상을 뜻한다.
   `LEAD_DROPOUT_SETS`는 12유도 표준으로 정의되고 perturbation은 **계약 적용 후**의
   tensor에 적용된다. PROPHECG는 8유도(I, II, V1–V6)이므로 `limb_only`는 2유도만
   남기고, ECGFounder(12유도)는 6유도를 남긴다. 같은 셀에서 두 모델을 비교하면 서로
   다른 정도의 손상을 비교하는 것이다. `amplitude_scale`(Preprocess 5번)과 같은
   부류이며, 결과표에 모델별 잔존 유도 수를 함께 적어야 한다.

### Misclassification (Plan 7 구현에서 제기됨)

1. attribution baseline을 무엇으로 선언할 것인가?
   Integrated Gradients(Sundararajan et al. 2017)는 baseline에 상대적인 귀속을
   내므로 baseline이 곧 "무엇과 비교했는가"다. 영상의 관례인 0 baseline은 ECG에서
   **무신호가 아니라 asystole**이며, STEMI 모델이 강한 의견을 갖는 극단적 이상
   파형이다. 그래서 코드에 기본값을 두지 않고 선언을 강제했다. 실무적 선택지는
   (a) dev split의 lead별 평균 — "이 코호트의 전형적 기록 대비"라는 해석,
   (b) lead별 중앙값, (c) `zeros`를 caveat과 함께. 판단이 필요하다.

2. attribution overlay를 논문·보고서에 실을 것인가?
   Adebayo et al.(NeurIPS 2018)은 널리 쓰이는 saliency 방법 여러 개가 **모델
   가중치를 무작위화해도 그림이 거의 변하지 않음**을 보였다. 해부학적으로 그럴듯한
   지도는 그 자체로 근거가 아니다. 현재 구현은 이를 §4.6의 exploratory 검토
   보조 자료로만 쓰고 지표로 보고하지 않는다. Evaluation 범주 4(interpretation)를
   정량화할 것인지와 함께 결정해야 한다.

3. 개별 waveform 그림의 게재가 MIMIC DUA상 허용되는가? (spec §4.6 확인 항목)
   ECG/PPG가 생체 식별자로 인식되며 통상적 익명화 후에도 재식별 정확도가 85%를
   넘는다는 연구가 있다. `cases.parquet`과 `figures/cases/`는 내부 검토용으로
   생성되므로, 외부 공개 시 어디까지 실을 수 있는지 확인이 필요하다.

4. STARD 보고에서 6단계 사례를 어떻게 위치시킬 것인가?
   Misclassification은 exploratory이므로 confirmatory 결과와 섞이면 안 된다.
   Technical Report에서 별도 절로 두고 "사후 선택된 사례"임을 명시하는 방식을
   전제하고 구현했다. 문서 구조 확정 시 함께 결정한다.

### Evaluation

1. primary metric과 secondary metric은 무엇인가?
2. operating threshold를 어떻게 선정할 것인가?
3. confidence interval과 모델 간 통계 비교 방법은 무엇인가?
4. Interpretation을 정량 평가할 것인가, 별도 결과로 보고할 것인가?

### Pipeline과 Report

1. Misclassification을 독립 6단계로 구현할 것인가, Evaluation의 하위 기능으로 둘 것인가?

   2026-08-14 결정: **독립 6단계**. 이유는 두 단계의 인식론적 지위가 반대이기
   때문이다 — Evaluation은 사전 지정된 질문에 답하는 confirmatory 추정이고,
   Misclassification은 사전 지정되지 않은 실패 양식의 exploratory 발견이다. 한
   단계에 섞으면 성능표를 본 뒤 눈에 띄는 subgroup을 사후 선택해 보고하는 오염이
   생긴다. 단계와 파일로 분리하면 구조적으로 차단된다. 근거: spec §4.6,
   medical algorithmic audit (Liu et al., Lancet Digital Health 2022).
   관련 코드: `src/mival/stages/misclassify.py`, `docs/plans/2026-08-14-plan7-misclassification.md`.

2. Misclassification 담당자는 누구인가?
3. Technical Report의 형식은 HTML, Word, PDF 중 무엇인가?
4. 현재 `miva` 5단계 구현을 언제 6단계 문서와 맞출 것인가?

### Data와 재현성

1. “최신 MI-CDM” snapshot을 어떻게 고정할 것인가?
2. S3 dataset manifest와 checksum을 만들 것인가?
3. `mimic-ecg-metadata.csv`의 canonical 위치는 어디인가?
4. ATLAS cohort definition JSON을 어디에 보관할 것인가?

## 결정 기록 방법

Open Question이 해결되면 해당 질문을 지우지 않고 다음 정보를 바로 아래에 추가한다.

- 결정일
- 결정 내용
- 결정 이유
- 참석자
- 관련 코드, 설정 또는 문서
