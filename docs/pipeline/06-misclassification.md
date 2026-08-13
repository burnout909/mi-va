# 6. Misclassification

## 목적

모델의 false positive와 false negative 사례를 자동으로 모으고, 오류가 반복되는 조건을 확인한다.

## 검토 대상

- false positive
- false negative
- threshold 주변의 불확실한 사례
- 두 모델의 prediction이 다른 사례
- acquisition metadata 또는 임상 subgroup에서 반복되는 오류

## 사례별로 함께 볼 정보

- `person_id`의 비식별 연구용 참조
- `image_occurrence_id`
- true label과 prediction score
- 사용한 model과 threshold
- ECG waveform 또는 안전하게 생성한 visualization
- sampling frequency와 lead order
- preprocess warning
- cohort/index date context
- 가능한 경우 interpretation 결과

## 출력으로 남길 것

- false positive/negative 목록
- 대표 사례 visualization
- 오류 유형별 집계
- model 간 disagreement 목록
- acquisition/clinical subgroup별 오류 pattern
- 연구자 review 결과와 note

## 확인할 사항

- 독립 module로 구현할지 Evaluation report의 하위 기능으로 둘지
- 담당자
- 임상 review에 노출할 정보와 PHI 보호 기준
- 자동 생성 결과에 수기 review를 어떻게 연결할지
- Technical Report에 개별 사례를 어느 수준까지 포함할지
