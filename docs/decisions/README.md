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
3. archived PTB threshold `0.0768`을 새 cohort에서 다시 추정할 것인가?
4. ECGFounder의 downstream STEMI head와 split protocol은 무엇인가?

2026-08-10 확인: PROPHECG는 Keras 2.7, 8-lead `(5000, 8)`, 2-class
softmax의 5-member mean ensemble이다. ECGFounder artifact와 code revision도
고정했으며 12-lead CUDA forward를 확인했다. 상세 값은 `docs/models/`에 있다.

### Evaluation

1. primary metric과 secondary metric은 무엇인가?
2. operating threshold를 어떻게 선정할 것인가?
3. confidence interval과 모델 간 통계 비교 방법은 무엇인가?
4. Interpretation을 정량 평가할 것인가, 별도 결과로 보고할 것인가?

### Pipeline과 Report

1. Misclassification을 독립 6단계로 구현할 것인가, Evaluation의 하위 기능으로 둘 것인가?
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
