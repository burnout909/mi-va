# 5. Evaluation

## 목적

모델 성능을 일관된 기준으로 평가하고 전체 결과, subgroup 결과와 불확실성을 함께 정리한다.

## 담당

- 김민성

## 평가 범주

### Discrimination

후보 지표:

- AUROC
- AUPRC
- sensitivity
- specificity
- PPV, NPV
- F1
- balanced accuracy
- MCC

### Calibration

후보 지표와 결과:

- calibration plot
- calibration intercept/slope
- Brier score
- 필요 시 기관별 recalibration 결과

### Interpretation

후보 결과:

- 모델이 제공하는 attribution/interpretation 결과
- ECG lead 또는 time region별 설명
- 임상적으로 타당한 영역을 보는지에 대한 검토 기록

## Subgroup

임상 변수뿐 아니라 acquisition metadata별 성능을 확인한다.

- sampling frequency
- channel/lead 구성
- manufacturer 또는 station
- sex
- age group
- 기타 cohort에서 정의한 임상 subgroup

## 비교 원칙

- ECGFounder와 PROPHECG-STEMI에 같은 cohort와 label을 사용한다.
- 같은 source DICOM과 cohort를 사용하되, 모델별 입력 계약에 맞는 서로 다른
  전처리 recipe를 고정한다. ECGFounder 12-lead와 PROPHECG 8-lead tensor를
  동일 tensor로 강제하지 않는다.
- operating threshold 선택 규칙을 미리 기록한다.
- confidence interval과 random seed를 기록한다.
- 전체 성능만으로 결론을 내리지 않고 subgroup 결과를 같이 본다.

## 출력으로 남길 것

- 모델별 전체 성능표
- calibration 결과
- subgroup 성능표
- 모델 간 비교표
- metric 계산 설정과 코드 revision
- Technical Report에 들어갈 표와 그림

## 확인할 사항

- primary/secondary metric
- threshold 선정 방식
- bootstrap 횟수와 confidence interval
- Interpretation을 정량 평가에 포함할지 별도 보고할지
- 모델 간 통계적 비교 방법
