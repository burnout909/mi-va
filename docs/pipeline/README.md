# MI-VAL Pipeline

회의록을 기준으로 연구 파이프라인을 6단계로 관리한다.

| 순서 | 단계 | 담당 | 문서 |
|---:|---|---|---|
| 1 | Retrieve | 규리T | [01-retrieve.md](01-retrieve.md) |
| 2 | Profile | 규리T | [02-profile.md](02-profile.md) |
| 3 | Preprocess | 규리T | [03-preprocess.md](03-preprocess.md) |
| 4 | Models | 민성 | [04-models.md](04-models.md) |
| 5 | Evaluation | 민성 | [05-evaluation.md](05-evaluation.md) |
| 6 | Misclassification | 담당 확인 필요 | [06-misclassification.md](06-misclassification.md) |

## 전체 흐름

```text
ATLAS cohort
    ↓
Retrieve → Profile → Preprocess → Models → Evaluation → Misclassification
    └──────────────── Technical Report ────────────────────────────────┘
```

## 공통 기록 원칙

각 단계 문서와 실행 결과에는 최소한 다음 내용을 남긴다.

- 사용한 입력 데이터와 버전
- cohort 및 index window
- 실행 설정과 코드 revision
- 입력·출력 파일 위치
- 제외된 record와 이유
- warning 및 error
- 다음 단계가 확인해야 할 사항

## 현재 구조 차이

- 회의록: Misclassification을 포함한 6단계.
- 기존 Overview와 `kyulee-jeon/miva` main: Evaluate까지 5단계.
- 이 폴더에서는 6단계를 기준으로 문서화하고, 실제 구현 정렬 여부는 별도 결정한다.
