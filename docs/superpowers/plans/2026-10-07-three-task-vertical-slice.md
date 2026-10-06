# 대표 모델 3개 끝까지 돌리기 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lima 나이, cavalab DeepSurv 1년 사망, SemiSegECG ResNet-18 간격을 retrieve → evaluate까지 서버에서 끝낸다.

**Architecture:** 기존 stage를 그 자리에서 넓힌다. 새 동작은 모두 새 spec 키(`label_source.kind`, 새 `label_def`, 새 `output.type`, `crop_anchor`/`pad_anchor`)가 있을 때만 켜지고, 없으면 지금 동작 그대로다.

**Tech Stack:** Python 3.9 로컬 / 3.10 서버, numpy, pandas, torch, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-three-task-vertical-slice-design.md`

**Execution note:** 사용자가 계획 승인을 위임했고 실행자는 작성자 자신이다 (Native). 그래서 이 계획은 코드 전문 대신 각 task의 동작, 파일, 테스트 이름을 고정한다. 구현은 TDD로 한다.

## Global Constraints

- `registry/models/`, `studies/lvef/`, `registry/drafts/`, `studies/drafts/` 수정 금지.
- 새 키가 없을 때의 동작은 바이트 단위로 지금과 같다. `tests/golden/lvef_recipes.json`(변경 전 캡처)과 일치해야 한다.
- 로컬 전체 테스트 기준선: 651 passed, 13 skipped (2026-10-07). 각 task 끝에 전체 통과.
- 커밋 메시지 끝 attribution 두 줄. GitHub push 금지, 서버 저장소 push만.

## Review Focus

1. LVEF 레시피·라벨·predictions가 새 기본값 때문에 조용히 바뀌는 것 → `test_lvef_recipes_unchanged`, 기존 stage 테스트.
2. 새 label kind에서 LVEF cutoff 열이 0/1로 채워지는 것 → retrieve 테스트가 NaN을 확인.
3. 1년 사망 censoring 경계 (사망일 = 종료일, 마지막 방문 + 365 < ECG + 365) → retrieve 테스트.
4. 마스크 → 간격에서 박동이 하나도 없거나 P파가 없는 ECG → decoder가 NaN을 내고 evaluate가 그 행을 뺀다.
5. survival arm에 threshold·prob 경로가 섞이는 것 → models/evaluate 테스트가 prob=None, threshold 없음 확인.

## File Structure

```
src/mival/contract.py, compiler.py, ops.py        crop/pad anchor (Task 1)
src/mival/stages/retrieve.py                      label_source kinds, one_per_person (Task 2)
src/mival/stages/sql/retrieve_ecg_person.sql      새 kind용 ECG+person+death+last visit 쿼리 (Task 2)
src/mival/stages/profile.py                       stratify_on: none (Task 2)
src/mival/decode/__init__.py, decode/intervals.py 마스크 → PR/QRS/QT ms (Task 3)
src/mival/adapters/torch_adapter.py               output.type risk_score, segmentation_mask (Task 3)
src/mival/stages/models.py, _predictions.py       label_def pr/qrs/qt/survival, 새 label 열 (Task 4)
src/mival/stages/evaluate.py, src/mival/metrics/  survival 범주, 라벨 열 일반화, cuts 매핑 (Task 5)
registry/{regression,survival,segmentation}/      카드 3개 (Task 6)
studies/{ecg-age,mortality,delineation}/study.yaml (Task 6)
tests/test_vertical_slice_*.py                    새 테스트
```

### Task 1: crop/pad anchor

- `InputContract`에 `crop_anchor`, `pad_anchor` (기본 `"start"`, 허용 `start|center`). 잘못된 값은 ValueError.
- `Pad.apply`: `center`면 앞 `deficit // 2`, 뒤 나머지. 기존 `start`/그 외 동작은 그대로.
- `compile_recipe`: `Crop(n, anchor=contract.crop_anchor)`, `Pad(n, mode, anchor=contract.pad_anchor)`.
- Tests: `test_pad_center_splits_deficit`, `test_contract_rejects_unknown_anchor`, `test_compiler_passes_anchors`, `test_lvef_recipes_unchanged` (golden 비교).
- Commit: `feat: input contract declares crop and pad anchors`

### Task 2: retrieve 라벨 종류와 사람당 1건

- `RetrieveSpec`에 `label_kind` (`label_source.kind`, 기본 `measurement`), `horizon_days` (기본 365), `machine_measurements` (경로), `one_per_person` (기본 False).
- `kind == measurement`이면 지금 `run` 경로 그대로. 아니면 `_run_derived`:
  - SQL `retrieve_ecg_person.sql`: ECG(modality 필터) + `person.year_of_birth` + `death.death_date` + 사람별 `max(visit_end_date)`.
  - `age_at_ecg`: `label_value = year(index) - year_of_birth`.
  - `death_within`: `end = min(index + horizon, last_visit_end + 365)`, 사망일 ≤ end면 event=1, time=사망일 − index, 아니면 event=0, time=end − index. time < 0 또는 last visit 없음은 `label_implausible`.
  - `machine_measurement`: csv(`study_id, pr_onset, p_onset, qrs_onset, qrs_end, t_end` 또는 `pr, qrs, qt` 열 — 서버 파일을 보고 맞춤)에서 `local_path`의 `s<study_id>` 세그먼트로 join. 결측·≤0·장비 결측 코드(29999 등)는 `label_implausible`.
  - LVEF cutoff 열은 NaN. `label_datetime`, `label_delta_days`는 NaN.
  - `one_per_person`: `np.random.default_rng(seed)`로 사람당 1행, 나머지 `not_selected`.
- `COHORT_INDEX_COLUMNS` + kind별 추가 열(`label_event`, `label_time_days`, `label_pr`, `label_qrs`, `label_qt`)로 기록.
- `REASON_CODES`에 `not_selected`.
- profile: `stratify_on: none`이면 상수 층 하나, gate는 `min_test_positives`가 0일 때 통과.
- Tests: `test_age_label`, `test_death_within_event_and_censoring` (경계 3개), `test_machine_measurement_join_and_exclusion`, `test_one_per_person_is_seeded`, `test_derived_kind_leaves_lvef_cutoffs_nan`, `test_profile_stratify_none`.
- Commit: `feat: retrieve derives age, one-year death and machine-measured intervals`

### Task 3: 출력 종류와 간격 decoder

- `mival/decode/intervals.py`: `intervals_from_mask(mask: (N,T) int, fs) -> (N,3) float [pr, qrs, qt] ms`. 클래스 0 none, 1 P, 2 QRS, 3 T. 박동 = QRS run. 각 QRS에 대해 직전 P 시작(이전 QRS 이후), 직후 T 끝(다음 QRS 이전). PR = QRS onset − P onset, QRS = QRS end − onset, QT = T end − QRS onset. 박동별 중앙값. 해당 박동이 없으면 NaN. 짧은 run(< `min_run_ms`, 기본 10 ms)은 무시.
- torch adapter `_score_from_outputs`: `risk_score` → `outputs[:, value_index or 0]`; `segmentation_mask` → forward가 argmax 후 `intervals_from_mask`로 (N,3) 반환 (`attribute` 미지원 오류).
- forward 앞단의 positive_index 검사를 새 두 type에서는 건너뛴다.
- Tests: `test_intervals_from_synthetic_mask`, `test_intervals_nan_without_p`, `test_torch_forward_risk_score`, `test_torch_forward_segmentation_mask` (작은 nn.Module).
- Commit: `feat: torch adapter reads risk scores and segmentation masks`

### Task 4: models label_def

- `PREDICTION_COLUMNS`에 `label_event`, `label_time_days`, `label_pr`, `label_qrs`, `label_qt`를 `label_value` 뒤에 추가.
- `REGRESSION_LABEL_DEFS = ("value", "pr", "qrs", "qt")`, `SURVIVAL_LABEL_DEF = "survival"`. `Arm.is_regression`, `Arm.is_survival`, `label_column`(survival → `label_event`), `required_label_columns`(survival → event, time_days).
- survival arm은 `inference_only`만 허용.
- `_labels_from`: 연속 열(`label_value`, `label_time_days`, `label_pr/qrs/qt`)은 float, 나머지는 int.
- cohort_index에 새 열이 없으면 NaN으로 읽는다 (LVEF cohort 호환).
- inference_only 짝 검사 표 (spec). `_predict`에 `column` 인자: 2-D 출력이면 `INTERVAL_ORDER.index(label_def)` 열.
- threshold: binary만. `_rows`: regression·survival은 `pred_value`, prob/logit None.
- `_predictions.is_regression` 일반화, `is_survival` 추가, `fit_operating_thresholds`는 survival도 건너뜀.
- Tests: `test_arm_survival_requires_inference_only`, `test_regression_arm_on_interval_label`, `test_survival_rows_have_no_prob`, `test_inference_only_pairing_table`, `test_lvef_cohort_without_new_columns_still_loads`.
- Commit: `feat: models runs interval-regression and survival arms`

### Task 5: evaluate

- 회귀: `_regression_rows`는 이미 `label_column`을 받는다. `regression_cuts`를 리스트 또는 `label_def`별 매핑으로 받고 `cuts_for(label_def)`로 고른다.
- 생존: `mival/metrics/survival.py` `harrell_c(time, event, risk)`, `auroc_at_horizon(time, event, risk, horizon)` (horizon 전 censored 제외). `_survival_rows`: category `survival`, 사람 단위 bootstrap, n_events = event 합. `CATEGORIES`에 `survival` 추가, metrics registry에 등록.
- survival arm은 figures·comparisons에서 건너뛴다.
- Tests: `test_harrell_c_known_example`, `test_auroc_at_horizon_excludes_early_censored`, `test_regression_cuts_mapping`, `test_evaluate_survival_rows`.
- Commit: `feat: evaluate scores survival arms and per-label regression cuts`

### Task 6: 카드, study, 서버 실행

- 서버: `machine_measurements.csv` 반입(`/data/mi-val/datasets/mimic-iv-ecg/`), sha256 기록. weight 3개 확인(agent 반입 결과). wrapper 모듈(`/data/mi-val/models/<id>/wrapper.py`)로 checkpoint 정규화·필터 처리, forward 확인.
- 카드 `registry/{regression,survival,segmentation}/<id>.json`, study 3개.
- 순서: 서버 push → 샘플 1,000건(retrieve 전체, profile, preprocess/models/evaluate는 샘플 cohort_index) → 디버깅 → LVEF 체인 종료 확인 → 전체 실행.
- 결과 기록: `docs/operations/vertical-slice-run.md`, 원장 L0/L1.
- Commit: `feat: cards and studies for age, mortality and delineation` + 운영 기록 커밋.
