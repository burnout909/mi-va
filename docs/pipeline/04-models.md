# 4. Models

## 목적

모델의 task, 입력 조건, weights, 실행 환경과 hyperparameter를 기록하고 동일한 전처리 결과에 모델을 실행한다.

## 담당

- 김민성
- TaskSpec 버전을 기반으로 검토한다.

## 검증 모델

### ECGFounder

- foundation model과 downstream head를 구분해 기록한다.
- model/Hugging Face revision, code commit, 1024 feature layer를 고정했다.
- freeze 범위와 head 학습 protocol은 아직 결정해야 한다.

### PROPHECG-STEMI

- single-task STEMI classification model이다.
- 실제 weights는 5개의 `.h5` 파일이다.
- 위치와 checksum은 [resources 문서](../resources/README.md)에 기록했다.
- Keras 2.7, 8-lead, 5-member mean ensemble을 실제 load/forward로 확인했다.

모델별 고정값과 검증 결과는 [models 문서](../models/README.md)에 있다.

## 모델별로 기록할 내용

- model name과 version/revision
- task와 target label
- weights 위치와 checksum
- framework 및 library version
- architecture를 생성하는 코드
- input shape, lead order, sampling frequency, duration, unit
- normalization과 scaling
- output activation과 positive class
- ensemble 방법
- threshold
- device와 batch size
- hyperparameter
- dependency와 실행 명령

## 모델 소스 후보

- local/S3 weights
- GitHub
- Hugging Face
- RSNA/ATLAS `model.json`

## 출력으로 남길 것

- model 설정 문서 또는 model card
- inference environment
- sample-level prediction
- 실행 warning/error
- 처리 시간과 resource 사용량

## 확인할 사항

- 사용자가 architecture와 hyperparameter를 어디까지 변경할 수 있는지
- PROPHECG-STEMI amplitude scaling과 padding 변경 이력
- ECGFounder linear probe를 어디서, 어떤 split으로 학습할지
- TaskSpec과 model 설정을 하나로 둘지 분리할지
