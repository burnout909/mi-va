# 2. Profile

## 목적

전처리 전에 연구자가 cohort와 ECG/DICOM 데이터의 특성, 누락, 이질성을 빠르게 확인할 수 있게 한다.

## 현재 방향

- 민성은 Fingerprint 버전을 기반으로 검토한다.
- 표 형태뿐 아니라 한눈에 skim할 수 있는 report를 지향한다.
- 이상치나 호환성 문제는 전처리나 모델 실행 전에 드러나야 한다.

## 확인할 내용

### Cohort

- 대상자 수와 ECG 보유 대상자 수
- age, sex 등 기본 특성
- index date와 ECG 촬영일의 간격
- label 및 class prevalence

### ECG/DICOM

- sampling frequency
- channel 수와 channel order
- lead 누락
- duration과 sample 수
- sensitivity, unit, baseline 누락
- waveform originality
- manufacturer/station 정보
- 파일 크기와 읽기 실패

### MI-CDM 연결

- `image_occurrence`와 `image_feature` 연결률
- DICOM metadata와 MI-CDM metadata 일치 여부
- `image_feature_value_order`의 누락 또는 중복

## 출력으로 남길 것

- Profile summary
- 분포표와 그림
- cohort Table 1 초안
- data quality flag
- preprocess 전에 제외하거나 검토할 record 목록

## 확인할 사항

- Fingerprint의 어떤 기능을 유지할지
- HTML, Word, Markdown 중 기본 report 형식
- PHI를 포함하지 않는 report export 기준
- quality flag의 warning/error 구분
