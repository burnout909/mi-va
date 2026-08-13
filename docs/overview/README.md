# MI-VAL Overview

- 로컬 Word 원본: [MI-VAL_overview.docx](MI-VAL_overview.docx)
- 원본 repository: <https://github.com/kyulee-jeon/miva/blob/main/docs/MI-VAL_overview.docx>
- SHA-256: `b0f05bb80d13c5bab78978690732a7228b8e77e7b34cd530b5f110e1d5c87364`
- 문서 기준일: 2026-08-06
- 주의: 기존 Overview는 5단계로 작성되어 있으나, 같은 날 회의에서는 Misclassification을 포함한 6단계로 정리했다. 이 프로젝트 문서에서는 6단계를 현재 기준으로 사용한다.

## 문제 정의

유망한 의료 AI 모델의 상당수는 논문이나 리더보드에는 존재하지만 실제 임상 데이터에서는 재현되지 않는다. 주요 원인은 알고리즘 자체뿐 아니라 다음 과정이 표준화·기록되지 않는 데 있다.

- 어떤 환자의 어떤 영상을 사용했는가
- 영상을 어떻게 전처리했는가
- 어떤 지표와 하위집단에서 성능을 측정했는가

ECG의 sampling frequency나 lead 구성과 같은 획득 parameter는 모델 성능에 큰 영향을 주지만, 많은 검증 연구에서 충분히 확인되거나 보고되지 않는다.

## 목표

MIMIC 기반 MI-CDM, 즉 OMOP CDM과 ECG-DICOM ETL 결과 위에서 서로 다른 모델을 같은 조건으로 검증할 수 있는 framework를 만든다.

1차 목표는 다음 두 모델을 같은 cohort, 전처리, 평가 조건에서 비교하는 것이다.

- ECGFounder
- PROPHECG-STEMI single-task model

## 설계 방향

- cohort는 기존 ATLAS Data4Life에서 정의한다.
- cohort definition, index date, observation window와 영상 선택 규칙을 기록한다.
- 전처리 방법은 설정 파일과 실행 기록으로 남긴다.
- 모델 입력 조건을 실제 MI-CDM/DICOM metadata와 추론 전에 비교한다.
- 결과는 전체 성능뿐 아니라 acquisition metadata와 임상 subgroup으로 나눠 확인한다.
- 실행에 사용한 cohort, 전처리, 모델, 지표, 환경을 Technical Report에서 함께 확인할 수 있게 한다.

## 파이프라인

1. Retrieve
2. Profile
3. Preprocess
4. Models
5. Evaluation
6. Misclassification

자세한 내용은 [pipeline 문서](../pipeline/README.md)를 참고한다.

## 기대 산출물

- MI-CDM에서 DICOM을 조회하고 가져오는 표준화된 과정
- 전처리 전 데이터 특성을 확인하는 Profile report
- 재현 가능한 ECG preprocess 설정
- ECGFounder와 PROPHECG-STEMI 실행 설정
- discrimination, calibration, interpretation 평가 결과
- 자동화된 misclassification review
- 전체 실행 과정과 결과를 묶은 Technical Report

## 연구 비전

### 이번 연구

병원 내에서 AI 모델 평가를 더 쉽게 도입할 수 있는 validation framework를 개발하고, 2026년 8–10월 내 논문 작업을 마무리하는 것을 목표로 한다.

### 후속 연구

- CDM을 환자별 episode가 구현된 가상 세계로 보고 후향적 연구와 reinforcement learning으로 확장한다.
- 임상의가 직접 모델을 쉽게 학습하고 검증할 수 있는 framework로 확장한다.
