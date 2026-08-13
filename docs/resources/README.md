# MI-VAL Resources

규리T가 공유한 자원과 2026-08-10 기준 확인 결과를 정리한다. S3 경로는 데이터 위치이며 이 프로젝트 폴더로 복사하지 않는다.

## 모델

### PROPHECG-STEMI

- 역할: single-task STEMI classification model
- weights: `s3://<bucket>/Users/<user>/STEMI_JKL/BestModelSaved/`
- 관련 논문: <https://www.sciencedirect.com/science/article/pii/S0196064424003275>
- DOI: <https://doi.org/10.1016/j.annemergmed.2024.06.004>
- PMID: `39066765`
- 논문명: Development of Clinically Validated Artificial Intelligence Model for Detecting ST-segment Elevation Myocardial Infarction

확인된 weights는 `.pth`가 아니라 다음 5개의 `.h5` 파일이다.

| 파일 | 크기 | SHA-256 |
|---|---:|---|
| `221204_16431_bestmodel.h5` | 2,407,504 bytes | `316e4fac1ccf994625dfc918f5bb5c6cecc3ff689dcf38b698b4322427754894` |
| `221204_16432_bestmodel.h5` | 2,407,504 bytes | `5754744b3ef464266d7891ab62a403c82ce6314f17578b400157c2e2684d30f3` |
| `221204_16433_bestmodel.h5` | 2,407,504 bytes | `8691d778cb439189a5d19173daed58052d2570b9fe4b3f4ccae10894349f3f39` |
| `221204_16434_bestmodel.h5` | 2,407,504 bytes | `6cb467412d62d4ac542695fdb674f6e760ab0dbd02c4a9f9ad8a665f2fa4690b` |
| `221204_16435_bestmodel.h5` | 2,407,504 bytes | `1fea2e24a4a0acc68141ab4a4cc5b699eab58b0b485e777a226e2e6fd187939d` |

확인 결과:

- embedded runtime metadata: Keras 2.7.0 / TensorFlow backend
- input: `(5000, 8)`, lead order `I, II, V1, V2, V3, V4, V5, V6`
- output: 2-class softmax, positive index 1
- ensemble: 5개 member probability의 산술 평균
- archived PTB-XL revision threshold: `0.0768`
- 2026-08-10 server smoke test: 다섯 모델 load 및 mean inference 성공

남은 확인 사항은 amplitude unit/scaling과 5120-sample training notebook에서
현재 5000-sample weights로 변경된 전처리 이력이다. 상세 내용은
[PROPHECG-STEMI 실행 문서](../models/prophecg-stemi.md)에 기록했다.

### ECGFounder

- Hugging Face revision: `d9b1793951b2342f5f7e84f1ac03cd37f8a08724`
- `12_lead_ECGFounder.pth` SHA-256: `ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997`
- official code commit: `68d25f25e323a4a423b9d9e8ea2e0af3f234bf22`
- 2026-08-10 NVIDIA L4 forward test 성공
- linear probe head의 학습 split, freeze 범위와 STEMI label protocol은 아직 결정해야 한다.

## 데이터

### MIMIC-IV OMOP CDM

- S3: `s3://<bucket>/Datasets/MIMIC-IV_CDM/`
- DDL: <https://github.com/OHDSI/MIMIC/tree/main/etl/ddl>
- 확인된 공개 DDL: `ddl_cdm_5_3_1.sql`, `ddl_voc_5_3_1.sql`
- server checkout: `/opt/ohdsi-mimic`, commit `d209ed37bcc533a69923dfd413fb15d53a6dad58`

### MIMIC-IV MI-CDM Extension

- S3: `s3://<bucket>/Datasets/MIMIC-IV_CDM/Extension/`
- 주요 파일:
  - `CREATE TABLE image_occurrence.txt`
  - `CREATE TABLE image_feature.txt`
  - `image_occurrence.csv`
  - `image_feature.csv`
  - `measurement_ADD.csv`
  - `concept_ADD.csv`

`image_occurrence`의 주요 필드:

- `image_occurrence_id`
- `person_id`
- `procedure_occurrence_id`
- `visit_occurrence_id`
- `wadors_uri`
- `local_path`
- `image_occurrence_date`
- `image_study_uid`
- `image_series_uid`
- `modality_concept_id`

`image_feature`의 주요 추적 필드:

- `image_feature_id`
- `person_id`
- `image_occurrence_id`
- `image_feature_concept_id`
- `image_feature_value_order`
- `image_feature_type_concept_id`
- `image_instance_uid`
- `alg_system`
- `alg_datetime`

Extension DDL 두 파일은 서버의 `/data/mi-val/schemas/mi-cdm-extension/`에
복사했다. 대용량 CSV는 instance profile 연결 전에는 서버로 동기화하지 않는다.

### MIMIC-IV ECG DICOM

- S3: `s3://<bucket>/Datasets/ECG/open_datasets/MIMIC/mimic-iv-ecg-dcm/`
- 확인된 하위 prefix: `files/`
- 2026-08-10 집계: 803,635 objects, 98,453,739,604 bytes (약 91.7 GiB)
- `/scratch`의 현재 usable 521 GiB에 전체 DICOM cache를 수용할 수 있다.

## Repository와 문서

### MI-VAL 기존 구현

- private repository: <https://github.com/kyulee-jeon/miva>
- 현재 Overview: <https://github.com/kyulee-jeon/miva/blob/main/docs/MI-VAL_overview.docx>
- 로컬 복사본: [`../overview/MI-VAL_overview.docx`](../overview/MI-VAL_overview.docx)
- 현재 main에는 Retrieve, Profile, Preprocess, Models, Evaluate의 5개 module이 있다.
- 회의록의 6단계 기준에 맞추려면 Misclassification 문서와 구현을 추가해야 한다.

### MIMIC ECG to DICOM

- repository: <https://github.com/dr-you-group/mimic-iv-ecg-to-dcm>
- MIMIC-IV-ECG WFDB를 DICOM 12-lead ECG Waveform Storage로 변환한다.
- `mimic-ecg-metadata.csv`는 현재 repository main tree에서 발견하지 못했다. 정확한 위치를 추가 확인해야 한다.

## 인프라

### dicom-etl

- Data4Life 설치 대상이다.
- 규리T의 Retrieve, Profile, Preprocess 검증 환경으로 사용한다.
- 접속은 RealVNC Viewer를 사용한다.

### DICOM-MIVA

- 민성 담당 구축 환경이다.
- `dicom-etl`보다 높은 사양으로 구성한다.
- 사용자 최종 지시에 따라 root volume 100 GB.
- persistent EBS 400 GB를 `/data`로 구성했다.
- instance store 559 GB를 `/scratch`로 추가 구성했다.
