# 3. Preprocess

## 목적

DICOM ECG를 모델 입력으로 변환하면서 lead, sampling frequency, sensitivity, unit, baseline 정보를 정확히 반영한다.

## 비교할 방식

### A. MI-CDM ETL validation 방식

- 기존 ETL validation에서 사용한 전처리 흐름을 재사용한다.
- 현재 구현과 일관성을 확인하기 쉽다.

### B. pydicom 기반 방식

- DICOM waveform과 metadata를 직접 읽는다.
- 다른 기관의 DICOM에 적용하기 쉬운지 확인해야 한다.

두 방식에 같은 DICOM을 넣어 waveform 값, lead order, scaling, 실패 record와 처리 속도를 비교한다.

## Channel Definition Sequence 규칙

- 일반적인 12-lead ECG에는 12개의 channel item이 있다.
- DICOM마다 channel 순서가 다를 수 있다.
- DICOM에 기록된 순서대로 `image_feature_value_order`를 저장하고 사용한다.
- order가 없는 값은 전체 ECG 수준의 값으로 취급한다.

Channel별로 함께 다뤄야 하는 값:

- `ChannelLabel`
- `ChannelSourceSequence`
- `ChannelSensitivity`
- `ChannelSensitivityUnitsSequence`
- `ChannelBaseline`

`SamplingFrequency`는 waveform-level 값으로 취급한다.

## 후보 전처리 항목

- lead 확인 및 표준 순서 정렬
- physical unit 변환
- resampling
- filtering
- crop/padding
- normalization
- flat signal 및 품질 불량 record 제외

## 출력으로 남길 것

- 전처리 설정과 적용 순서
- 입력/출력 shape와 dtype
- 적용된 lead order와 sampling frequency
- 제외된 record와 이유
- 전처리된 sample manifest
- 전처리 코드 revision

## 확인할 사항

- 두 방식 중 하나를 고정할지, 공통 규격 아래 두 loader를 허용할지
- PROPHECG-STEMI 원 논문의 정확한 전처리
- ECGFounder가 요구하는 입력 조건
- metadata가 없을 때 default를 허용할지 record를 제외할지
