# LVEF 결과 덱 계획

기존 `docs/mival-options-slides.html`(2026-08-14, STEMI 옵션 덱 14장)과 같은 형식으로
만든다. 같은 CSS, 같은 조작(←/→, 클릭, THEME), 같은 부품(svg 흐름도, 표, `.code`
블록, 눌러서 바뀌는 `.knob`/`.cell`+`.box`)만 쓴다. 새 파일은
`docs/mival-lvef-results-slides.html`.

덱이 말하려는 한 문장:
**"MI-VAL은 모델 카드와 study 파일만으로 임상 ECG 모델을 검증하는 모듈이고,
LVEF 과제에서 4개 모델을 6단계 끝까지 돌려 각 모델의 성능과 오분류를 얻었다."**

## 흐름 (사용자가 정한 순서)

1. 목적 — 내가 기획한 모듈이 무엇을 위한 것인가
2. 구성 — 단계와 산출물의 직관적 흐름 (시각)
3. 코드 — 어떻게 선언하는가 (눌러서 코드 블록)
4. 실제 실행 — 데이터 소스, retrieve 결과, 그중 1,000건 샘플을 돌린 것
5. 모델별 결과
6. misclassification

## 지금 있는 재료와 빠진 재료

| 재료 | 상태 | 출처 |
|---|---|---|
| retrieve·profile 전체 코호트 수치 | 있음 | `docs/operations/lvef-first-run.md` |
| preprocess 샘플 1,000건 recipe 4종 | 있음 | 같은 문서 + `recipes.json` |
| models 샘플, ecgfounder env 4 arm (HeartWise 2 + ECGFounder 2) | 있음 | 서버 `runs/lvef/models/ad943c170e91e09e` |
| models 샘플, xecg env 2 arm | 있음(예측 파일) | 서버 `runs/lvef/models/` xecg 실행 |
| evaluate 샘플, 6 arm 전부 | 있음 | `runs/lvef/evaluate/d99958c3e4a2c43b` (두 env 예측 합친 `_sample/predictions_sample_all.parquet`, 3분 18초) |
| misclassify 결과, selector 6개 | 있음 | `runs/lvef/misclassify/97cee48f8d4539bc` (study 커밋 `53f71f8`, 123 cases) |
| 그림·집계 사본 | 있음 | `docs/figures/lvef-sample/` (png 6, `metrics_long_all.csv`, `misclassify_summary.csv`, `selectors.json`) |
| perturbation 열화 셀 | **없음, 구조적** | 어느 stage도 grid를 실행하지 않음. 덱에서는 "선언은 되나 실행은 안 됨"으로 정직하게 표시 |
| 전체 코호트 models/evaluate | 하지 않음(사용자 결정) | 실행 시간 추정만 보여줌 |

2026-09-22 사용자 승인으로 xECG evaluate와 misclassify를 샘플 예측으로 채웠다(새 추론 없음).
덱에 빈 칸은 perturbation 셀(구조적)뿐이다.

## 슬라이드별 계획

### 1. 표지
- `h1` "MI-VAL · LVEF 과제 실행", `.meta` "HeartWise LVEF ×2 · ECGFounder · xECG · 2026-09-22"

### 2. 목적 — 무엇을 위한 모듈인가
- `.lede` 한 단락 + `ul` 3개.
  - 병원 ECG 모델은 학습 기관 밖에서 얼마나 맞는지 알 수 없다. 모델마다 입력 형식·런타임·출력이 달라 비교 자체가 어렵다.
  - MI-VAL은 **모델을 코드에 쓰지 않는다.** 모델은 카드(JSON) 하나, 연구 설계는 study(YAML) 하나로 선언하고 6단계가 그것만 읽는다.
  - 목표는 모델 12개 × 기관 2개(MIMIC → 신촌). 지금은 LVEF 과제로 4개.
- 출처: `docs/mival-options-slides.html` 2·14장, 메모리 `mival-validation-scope`.

### 3. 구성 — 6단계 흐름 (시각)
- 기존 2장의 svg를 그대로 가져오되 산출물 이름을 LVEF 실행에서 실제로 나온 파일명으로 바꾼다:
  `cohort_index` → `cohort_split`·`profile_summary` → `tensors/`·`preprocess_index`·`recipes.json` → `predictions/`·`train_log.jsonl`·`heads/` → `metrics_long`·`figures/` → `cases`·`review_template`.
- 아래 `ul` 2개: 카드는 "모델이 무엇인가", study는 "무엇을 물을 것인가". 단계는 파일로만 이어진다(한 단계의 출력 경로가 다음 단계의 `--input`).

### 4. 이번 과제의 4개 모델 (카드 표)
- 기존 3장 표 형식. 열: heartwise-lvef-binary · heartwise-lvef-regression · ecgfounder · xecg.
- 행: adapter(torch 4개 모두), runtime env(ecgfounder env py3.10/torch 2.13 ×3, xecg env py3.12/torch 2.8), leads(12), sampling_rate(250·250·500·100 Hz), gain(HeartWise ×208.33, 나머지 없음), scaling(none·none·global_zscore·none), output(binary·regression·150 logits·1024-dim feature), 쓸 수 있는 training_mode(inference_only ×2, linear_probe ×2), weights 형식(torch.jit ×2, state_dict, HF safetensors).
- 정확한 값은 `registry/models/*.json`에서 옮긴다. `.warn`으로 표시할 것: xECG는 sLSTM CUDA 확장 빌드가 필요(env 변수 4개), HeartWise는 데이터셋 단위 스펙트럼 정규화 주의.

### 5. 코드 — study.yaml (인터랙티브)
- `.knobs`로 `retrieve · profile · preprocess · models · evaluate · misclassify` 6개 버튼, 누르면 `.code`에 `studies/lvef/study.yaml`의 해당 블록이 주석과 함께 나온다(기존 7·10장 스타일, `.k`/`.c`/`.h` 강조).
- 강조할 줄: `modality_concept_id: 4145308`(CDM에 CXR 섞임), `label_def: value`(회귀는 새 축이 아니라 label_def 값), `perturbation.axes` baseline-only 주석, `regression_cuts: [40]`.
- 출처: `studies/lvef/study.yaml` 원문.

### 6. 코드 — 모델 카드 하나 (인터랙티브)
- `.knobs` 4개(모델별), 누르면 그 카드 JSON의 핵심 필드만 발췌해 `.code`로. 아래 `.box`에 "이 카드 때문에 코드가 바뀐 것 0줄 / L1 n건" 한 줄.
- 출처: `registry/models/*.json`, `docs/decisions/adaptations.md` A-4~A-8.

### 7. 실제 실행 — 데이터 소스와 retrieve
- 표: MI-CDM `cdm.image_occurrence` 1,011,623행 → ECG(modality 4145308) 796,617 → ±7일 LVEF 있음 182,960 → 5% 미만 제외 33 → **182,927 ECG / 44,179명**, LVEF≤40 양성 44,056(24.1%).
- LVEF 라벨 출처: 협업자가 OMOP measurement로 적재한 `measurement_lvef_ADD`(147,431행, concept 3027172). DICOM은 S3 → `/scratch` 캐시 800,035파일 99 GB.
- 작은 svg 또는 표로 STARD식 흐름. `label_delta_days`가 0·1일에 몰림.
- 출처: `docs/operations/lvef-first-run.md`, `micdm-database.md`.

### 8. 실제 실행 — profile과 1,000건 샘플
- profile: 사람 단위 split, dev 35,343명/146,757 ECG, test 8,836명/36,170 ECG, gate 통과(test 양성 8,673 ≥ 100).
- **샘플 기준을 정직하게**: `cohort_index` 앞 1,000행(`image_occurrence_id` 최소 1,000건, 무작위·층화 아님). 272명, 양성률 19.6%(전체 24.1%). split은 전체 `cohort_split`을 조인해 794/206.
- `.key` 항목: 이 샘플은 "끝까지 도는가"를 보는 용도이며 성능 추정치가 아니다. 전체 실행 추정: preprocess 26분(130 GB), models 3~4 h(xECG가 병목), evaluate 0.5~2 h.

### 9. preprocess — 4개 recipe
- 표: recipe 4종(HeartWise 2개가 공유). resample 500→250/100, gain, normalize, lead 선택. `excluded 0`.
- 그리드 경고 3건을 `.warn`으로 보여주고 "perturbation 셀은 grid만 기록되고 실행되지 않음"을 명시(L2 후보).

### 10. models — 6 arm, env 2개
- `.cells` 행렬: 행 4모델 × 열 `inference_only · linear_probe`. ✓ 6칸, 누르면 `.box`에 그 arm이 무엇을 했는지(HeartWise: 카드 head 그대로 + dev에서 임계값 refit; probe: backbone 고정, 5-fold 내부 CV로 lr {0.01, 0.001} 선택, NumPy 로지스틱/선형 head).
- 한 줄: study는 하나지만 env가 둘이라 models는 env별로 두 번 돌리고 manifest를 따로 남긴다(L2-5). 첫 실행에서 A-10(상대 `tensor_path`) 발견·고침.

### 11. 결과 — 분류 (label_def primary, test 206건)
- 표: heartwise-lvef-binary · ecgfounder probe · xecg probe. 행: AUROC(95% CI) · AUPRC · sens@refit_sens95의 spec · brier · net benefit@0.1.
- 값(test 206, refit_sens95 임계값): HeartWise AUROC 0.919(0.844~0.972) AUPRC 0.734 spec 0.399 · ECGFounder 0.898(0.802~0.961) 0.720 spec 0.259 · xECG 0.903(0.803~0.974) 0.736 spec 0.405. brier 0.120 / 0.110 / 0.101.
- `docs/figures/lvef-sample/roc_pr.png`, `calibration.png`, `decision_curve.png`를 `<img>`로.

### 12. 결과 — 회귀 (label_def value, test 206건)
- 표: heartwise-lvef-regression · ecgfounder probe · xecg probe. 행: MAE · RMSE · R² · AUROC@cut40(회귀 예측을 40으로 자른 판별력).
- 값: HeartWise MAE 7.77(5.83~10.22) RMSE 10.94 R² 0.470 auroc_below@40 0.912 · xECG MAE 8.92 RMSE 11.62 R² 0.402 auroc@40 0.885 · ECGFounder MAE 10.97 RMSE 14.42 R² 0.080 auroc@40 0.894.
- 읽는 법: ECGFounder probe는 순위(auroc@40 0.89)는 좋은데 값(R² 0.08)은 못 맞춘다. 선형 head + 표준화 feature의 한계인지 lr grid 문제인지는 전체 실행에서 본다.
- `docs/figures/lvef-sample/regression_scatter.png`.

### 13. misclassification — 구성 원칙과 결과
- 원칙(사용자 결정 2026-09-22): **모델별로 독립**. selector는 한 모델의 예측 안에서만 정의하고, 두 모델을 교차하는 contrast는 두지 않는다.
- `.knobs`: 모델 4개. 누르면 `.code`에 그 모델의 selector 선언, `.box`에 결과 요약(FP/FN 상위 k의 LVEF 분포, 임계값 근처 건수).
- 선언 초안(모델당 동일 형태):
  ```yaml
  misclassify:
    selectors:
      - {name: hw_binary_errors, type: error, k: 20,
         pattern: {model_id: heartwise-lvef-binary, training_mode: inference_only, label_def: primary, perturbation_id: baseline}}
      - {name: hw_binary_boundary, type: boundary, epsilon: 0.05, k: 20,
         pattern: {model_id: heartwise-lvef-binary, training_mode: inference_only, label_def: primary, perturbation_id: baseline}}
      # ecgfounder / xecg: model_id, training_mode: linear_probe 로 같은 두 개
  ```
  회귀 arm(label_def value)은 확률이 없어 error/boundary 대상이 아니다. 회귀 오차 상위는 지금 selector 종류에 없다(L1 후보: `type: residual`).
- 결과(`misclassify_summary.csv`, test 206건, 임계값 refit_sens95 = HeartWise 0.065 · ECGFounder 0.069 · xECG 0.074):
  - error 후보: HeartWise FP 95 / FN 0, ECGFounder FP 117 / FN 1, xECG FP 94 / FN 2. sens95로 임계값을 잡았으니 **오분류는 거의 전부 FP**이고, FN은 셋 다 LVEF 21인 같은 환자 계열(ECGFounder 1건, xECG 2건).
  - FP 상위 20의 LVEF 중앙값 50~55(45~75). 즉 "경계 근처(40~50)를 양성으로 올린" 것이 대부분이고, LVEF 60 이상을 강하게 양성이라 한 것은 xECG(prob 최대 0.955)에서 두드러진다.
  - boundary(±0.05): HeartWise 20건 중 FP 7 / TN 13, ECGFounder FP 11 / TN 9, xECG FP 7 / TN 12 / FN 1.
  - 덱 13장 `.box`에는 selector별 (FP, FN, 후보 수, LVEF 범위)를 표로, 아래에 "FN 3건은 전부 LVEF 21"을 `.key`로.
- 덱에서 밝힐 것: `selectors.json`의 verdict 어휘가 아직 STEMI용(`stemi_mimic`, `reperfused`)이다. study별 verdict 목록은 L0로 열어야 한다. `attribution_status`는 비어 있다(preprocess_index를 넘기지 않아 파형 그림 없음).

### 14. 이번 실행에서 프레임워크가 배운 것
- 기존 13장(L0/L1/L2 knob) 형식. L0: 카드 4장, study 1개. L1: A-7(Decimal), A-8(modality 필터), A-10(상대 tensor_path), safe_globals. L2: L2-5 env 강제 없음, L2-6 code_path 충돌, perturbation 미실행, CLI `--input` 파일만.
- 연구 결정 대기: A-9 `label_sens2` 구조적 동일.

### 15. 다음
- 전체 코호트 실행(추정 5~7 h). 그 전에 linear probe feature 캐시(L1)로 xECG 3~4 h → 15분.
- 신촌 전이.

## 결정된 것 (2026-09-22)

1. xECG 포함 evaluate: 함. `d99958c3e4a2c43b`.
2. selector 6개(모델별 error + boundary) study에 넣고 misclassify: 함. `97cee48f8d4539bc`.
3. 그림·집계를 `docs/figures/lvef-sample/`에 복사: 함. record 단위 `cases.parquet`는 repo에 넣지 않았다.

다음 단계: 이 계획대로 `docs/mival-lvef-results-slides.html` 작성.
