# MI-VAL 구현 계획

[연구 설계 spec](../spec/2026-08-14-research-design.md)을 6개 plan으로 나눈다. 각 plan은 단독으로 동작·검증 가능한 소프트웨어를 산출한다.

| # | Plan | 산출 | 선행 조건 | 상태 |
|---|---|---|---|---|
| 1 | [Framework core](2026-08-14-framework-core.md) | ModelCard registry, adapter, op 라이브러리, recipe 컴파일러, input contract gate, golden test | 없음 (합성 신호로 검증) | **구현 완료** (2026-09-14 DICOM-MIVA에서 weights 포함 전 스위트 통과) |
| 2 | Pipeline runner | 단계 CLI 골격, `config_hash`, run manifest, exclusion ledger, artifact 주소 규칙 | Plan 1 | **구현 완료** |
| 3 | [Preprocess stage](2026-08-14-plan3-preprocess.md) | perturbation grid, on-the-fly 적용, `preprocess_index`, tensor 저장 | Plan 1, 2 | **구현 완료** |
| 4 | [Models stage](2026-08-14-plan4-models.md) | training mode 4종, `fit`, threshold 정책, leakage/contamination gate, prediction 출력 | Plan 1, 2 | **구현 완료** (adapter `fit` 포함) |
| 5 | [Evaluation stage](2026-08-14-plan5-evaluation.md) | 범주 4개, bootstrap, paired bootstrap 비교, `metrics_long`, figure | Plan 2, 4 | **구현 완료** (범주 4 interpretation은 스키마만) |
| 6 | [Retrieve / Profile](../superpowers/plans/2026-09-21-lvef-retrieve-profile-regression.md) | cohort SQL, DICOM 경로 해석, acquisition metadata, event-count gate, split 동결 | **Data4Life 설치 + MI-CDM 접근** | **구현 완료 (2026-09-21, LVEF 과제)** |
| 7 | [Misclassification](2026-08-14-plan7-misclassification.md) | selector 4종, 사례 시각화, review 병합, attribution | Plan 4, 5 | **구현 완료** |

## 순서 근거

- Plan 1은 데이터 의존이 없다. DICOM-MIVA에 이미 두 모델 weights가 검증된 상태로 올라가 있어 합성 신호만으로 전 범위를 검증할 수 있다.
- Plan 6(Retrieve/Profile)이 표에서 뒤에 있는 것은 우선순위가 낮아서가 아니라 **외부 선행 조건에 차단**되어 있었기 때문이다. 선행 조건이 풀린 뒤 2026-09-21 LVEF 과제로 구현했다.
- Plan 3–5는 실데이터 없이도 합성 cohort fixture로 개발·검증할 수 있다. Plan 6이 늦어져도 병렬 진행이 가능하도록 fixture 기반으로 설계한다.

## 실행 방식

Plan 1은 Superpowers subagent-driven-development로 실행했다(task 9개 순차, task마다 리뷰어 + fix loop, 17커밋).

Plan 2 이후는 방식을 바꿨다. 계약 계층(Plan 2)은 컨트롤러가 직접 구현하고, 그 위의 stage들은 1페이지 brief만 주고 **병렬 subagent**로 실행한 뒤 컨트롤러가 diff를 직접 검토한다. Plan 1의 비용 대부분이 task마다 리뷰어를 띄우고 findings를 판정한 라운드에서 나왔기 때문이다.

계약 계층을 위임하지 않는 것이 이 방식의 전제다. 병렬 stage들이 전부 같은 계약을 물고 들어가므로, 계약이 틀리면 병렬로 틀린 코드가 나온다. stage마다 파일 소유를 배타적으로 고정해 worktree 없이 충돌을 없앴다.

## 남은 작업

- **study.yaml 선언 필요** — 전원 주파수(`powerline_60hz`)와 attribution baseline은 코드가 알 수 없는 사실이라 기본값이 없다 (decisions Preprocess 7, Misclassification 1)
- ~~DICOM-MIVA 검증~~ — 2026-09-14 완료. 서버 `/data/mi-val/mi-va`에 브랜치를 올려 env 3개(mival 3.12 / ecgfounder 3.10 / prophecg 3.9)에서 전 스위트 통과. `tests/golden/expected.json`은 prophecg env에서 생성했다. PROPHECG `feature_layer`는 `dense_1`(64)로 채웠다
