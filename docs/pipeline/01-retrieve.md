# 1. Retrieve

## 목적

ATLAS cohort와 MI-CDM을 연결해 검증에 사용할 실제 DICOM 파일을 선택한다.

## 시작 정보

- `person_id`
- `image_occurrence_id`
- `local_path`

## 입력

- ATLAS cohort definition
- index date
- observation/image selection window
- 영상 선택 규칙
- MI-CDM `image_occurrence`
- MI-CDM `image_feature`
- site-local DICOM root

## 처리 내용

1. ATLAS 결과의 `person_id`와 index date를 읽는다.
2. 설정한 window 내 `image_occurrence`를 조회한다.
3. 한 사람에게 여러 ECG가 있을 때 선택 규칙을 적용한다.
4. `local_path`를 실제 DICOM 파일 위치로 해석한다.
5. 파일 존재 여부와 metadata 연결 여부를 확인한다.
6. 단계별 대상자/영상 제외 수와 사유를 기록한다.

## 출력으로 남길 것

- 선택된 `person_id`와 `image_occurrence_id` 목록
- DICOM path 또는 file manifest
- image metadata
- cohort 대비 DICOM 보유율
- attrition 표
- 실행에 사용한 SQL과 설정

## 사용 자원

- OMOP CDM: `s3://<bucket>/Datasets/MIMIC-IV_CDM/`
- MI-CDM Extension: `s3://<bucket>/Datasets/MIMIC-IV_CDM/Extension/`
- ECG DICOM: `s3://<bucket>/Datasets/ECG/open_datasets/MIMIC/mimic-iv-ecg-dcm/`

## 확인할 사항

- cohort definition JSON을 어떻게 export·보관할지
- “최신 MI-CDM” snapshot을 어떤 manifest로 고정할지
- DICOM이 여러 개인 경우 `first`, `last`, `nearest`, `all` 중 어떤 규칙을 사용할지
- 누락된 `local_path`와 존재하지 않는 파일을 어떻게 처리할지
