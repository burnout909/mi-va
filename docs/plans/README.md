# MI-VAL 구현 계획

[연구 설계 spec](../spec/2026-08-14-research-design.md)을 6개 plan으로 나눈다. 각 plan은 단독으로 동작·검증 가능한 소프트웨어를 산출한다.

| # | Plan | 산출 | 선행 조건 | 상태 |
|---|---|---|---|---|
| 1 | [Framework core](2026-08-14-framework-core.md) | ModelCard registry, adapter, op 라이브러리, recipe 컴파일러, input contract gate, golden test | 없음 (합성 신호로 검증) | 작성 완료 |
| 2 | Pipeline runner | 단계 CLI 골격, `config_hash`, run manifest, exclusion ledger, artifact 주소 규칙 | Plan 1 | 미작성 |
| 3 | Preprocess stage | perturbation grid, on-the-fly 적용, `preprocess_index`, tensor 저장 | Plan 1, 2 | 미작성 |
| 4 | Models stage | training mode 4종, `fit`, threshold 정책, leakage/contamination gate, prediction 출력 | Plan 1, 2 | 미작성 |
| 5 | Evaluation stage | 범주 4개, bootstrap, paired bootstrap 비교, `metrics_long`, figure | Plan 2, 4 | 미작성 |
| 6 | Retrieve / Profile | cohort SQL, DICOM 경로 해석, acquisition metadata, event-count gate, split 동결 | **Data4Life 설치 + MI-CDM 접근** | 미작성 (차단됨) |
| 7 | Misclassification | selector 4종, 사례 시각화, review 병합 | Plan 4, 5 | 미작성 |

## 순서 근거

- Plan 1은 데이터 의존이 없다. DICOM-MIVA에 이미 두 모델 weights가 검증된 상태로 올라가 있어 합성 신호만으로 전 범위를 검증할 수 있다.
- Plan 6(Retrieve/Profile)이 표에서 뒤에 있는 것은 우선순위가 낮아서가 아니라 **외부 선행 조건에 차단**되어 있기 때문이다. Data4Life가 설치되면 즉시 착수한다.
- Plan 3–5는 실데이터 없이도 합성 cohort fixture로 개발·검증할 수 있다. Plan 6이 늦어져도 병렬 진행이 가능하도록 fixture 기반으로 설계한다.
