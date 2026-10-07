# 논문 방식 그대로 재현하고, 그 추출을 자동화하기

2026-10-07. 지금까지 돌린 모델은 task마다 우리가 정한 공통 기준(LVEF ≤ 40, 1년 사망, 기계 측정
간격 등)으로 평가했다. 논문은 각자 다른 라벨 정의, cutoff, 지표로 보고했기 때문에
[모델 결과 표](../../mival-model-results.xlsx)의 "논문 vs MIMIC"은 같은 기준끼리의 비교가 아니다.

이 작업은 두 부분이다 (사용자 결정 2026-10-07: "C").

- **A. 정답 세트:** 논문과 코드를 읽고 모델마다 논문 방식의 study를 손으로 만든다. 프레임워크로
  돌리고 논문 보고값과 같은 조건에서 비교한다.
- **B. 자동 추출:** 논문 PDF와 repo를 넣으면 같은 형식의 study 초안이 나오는 절차를 만든다.
  A의 정답과 필드 단위로 채점한다.

A 없이 B를 하면 자동 추출이 맞았는지 판단할 기준이 없다. 그래서 A를 먼저 한다.

## 성공 기준

- 정답 16건이 서버에서 전체 코호트로 돌고, 각 논문 claim마다 판정(재현됨 / 차이 있음 / 비교 불가)과
  조건 차이가 붙은 비교표가 나온다.
- 기존 공통 study 6개(`lvef`, `ecg-age`, `mortality`, `delineation`, `ntprobnp`, `potassium`)의 stage별
  `config_hash`와 기존 테스트 결과가 바뀌지 않는다. 새 필드는 모두 기본값이 지금 동작과 같다.
- B의 자동 추출이 평가용 8건에서 필드별 일치율, 지어낸 정보 수, 근거 유효성으로 채점된다.

## 정답 세트 (16건)

결정은 모두 2026-10-07 대화에서 사용자가 했다.

| 유형 | paper_id (study) | arm | 논문 방식 | 옮길 때 다른 것 |
|---|---|---|---|---|
| LVEF | `heartwise` | DeepECG-SL ≤40 | LVEF ≤ 40 AUROC 0.900 (MHI 내부) | ECG–심초음파 간격은 논문 값으로 |
| LVEF | `ecgfounder` | linear probe | LVEF < 50 AUROC 0.8674, MAE 7.0117 (MIMIC-IV-ECG test) | 저자는 full fine-tune, 우리는 기존 linear probe (결정 E1). 라벨과 test는 저자 것 |
| 회귀 | `lima-age` | lima-ecg-age | MAE 8.38, R² 0.71 (CODE-15%) | 인구집단 |
| 회귀 | `singstad-age` | singstad-ecg-age | MAE 8.3 (PTB-XL) | 인구집단 |
| 회귀 | `ai-ntprobnp` | ai-ntprobnp | Pearson R 0.566 (HCHS) | 인구집단. log 척도 여부 확인 필요 |
| 회귀 | `vonbachmann-k` | vonbachmann-k | MAE 0.285, R 0.582, AUROC <3.5 0.809, >5.5 0.892 (5 seed 평균) | 우리는 공개 weight 하나 |
| 회귀 | `kardionet-k` | kardionet-k-12lead | ESRD MAE 0.527, AUROC >6.5 0.852 | ESRD 정의는 우리가 정함 |
| 생존 | `cavalab-code15` | DeepSurv, MTLR | Antolini C 0.80 / 0.83 (CODE-15, 5 seed 중앙값으로 추정) | 1년 추적 (사용자 결정) |
| 생존 | `cavalab-mimic` | DeepHit ResNet, DeepHit InceptionTime | Antolini C 0.77 / 0.78 (MIMIC-IV test) | 없음. 같은 데이터, 같은 split |
| 생존 | `lima-survival` | lima-ecg-age 파생 | 예측 나이 − 실제 나이 > 8년 HR 1.79, Cox(나이+성별+ECG-age) 1년 AUC 0.80 | 인구집단 |
| 분할 | `semiseg` | ResNet-18, ViT-Tiny | 간격 MAE PR/QRS/QT | 정답이 사람 주석 → 기계판독 |
| 분할 | `openecg` | codec v6 | 경계 F1 0.855, 시점 오차 11.1 ms (LUDB) | 정답이 기계판독 시점 (12-lead 전체 기준) |

정답 study는 12개, arm으로 세면 16건이다 (cavalab 두 study가 각 2 arm, SemiSeg 2 arm. Lima 논문 하나가
회귀와 생존 study 2개로 나뉜다).

뺀 것과 이유:

- ml4h ECG2AF/HF/Stroke: 사망 head의 논문 수치가 없다.
- HRNetV2, HeartKit: 논문이 없다.
- 다른 생존 모델: 공개 weight, 논문 사망 수치, MIMIC 1년 재현 가능의 세 조건을 모두 만족하는 것이
  없었다 (AIRE, Raghunath, Alberta, CGMH, MDS-ED는 weight 비공개. SEER는 5년 심혈관 사망).

### 조사로 확인한 논문 사실

정답 study를 쓸 때 근거가 되는 사실이다. 정답의 `evidence.yaml`에 원문 위치와 함께 다시 적는다.

- **ECGFounder** (NEJM AI 2025, arXiv 2410.04133 v4, GitHub `PKUDigitalHealth/ECGFounder`)
  - 미세조정 LVEF weight는 공개되지 않았다. HF에는 사전학습 checkpoint만 있다.
  - 라벨은 `csv/LVEF.csv` (75,412행, 19,486명, `subject_id, waveform_path, LVEF, class`)다.
    MIMIC-IV-Notes 퇴원기록에서 뽑은 LVEF이고, ECG–심초음파 간격은 같은 입원이라는 것 외에 없다.
  - `class` 1이 LVEF ≥ 50(정상)이다. 양성 클래스가 뒤집혀 있으니 우리는 LVEF < 50을 양성으로 둔다.
  - split: subject_id 순 정렬 후 `train_test_split(test_size=0.2, shuffle=False)`, 남은 20%를 다시
    50/50. 재구성 결과 60,329 / 7,541 / 7,542. 논문 Table S5는 60,333 / 7,539 / 7,541이다.
  - 저자 코드는 test 세트로 checkpoint를 고른다.
  - 필터가 논문(1–30 Hz)과 코드(0.67–40 Hz, 50 Hz notch, median baseline)에서 다르다.
  - 라벨에 100 초과(최대 7,065)와 0–5 값이 섞여 있다.
- **cavalab** (BioData Mining 19:6 2026, arXiv 2406.17002 v4, GitHub `cavalab/ecg-survival-benchmark`)
  - 지표는 PyCox Antolini 시간 의존 concordance (KM 가중 censoring)다. Harrell C가 아니다.
  - MIMIC 코호트 (`MIMIC_IV_PreProcess_Jan2025.py`):
    - ECG는 전부 사용한다 (785,035 / 795,546).
    - 사망 출처는 hosp v2.2 `patients.dod`이고, 시간 제한이 없다.
    - censoring 시점은 다음 규칙이다: 입원이 없으면 마지막 ECG, 있으면 max(마지막 퇴원 + 365일, 마지막 ECG).
    - 사건까지의 시간은 최소 1일로 둔다.
    - split은 사람 단위 64/16/20이다. test는 seed 12345로 고정이고, 학습 seed는 10–14다.
  - 전처리 (코드 기준):
    - 신호는 mV 단위다.
    - 10초 5,000 샘플을 `scipy.signal.resample`로 4,000 샘플로 줄이고, 앞뒤에 0을 48개씩 붙인다.
    - 필터는 쓰지 않는다.
    - z-score는 학습셋 통계로 한다.
    - 논문 v4는 "7초"라고 적었지만, 코드와 v1/v3를 따른다.
  - Zenodo 16877773의 MIMIC weight 2개는 DeepHit이다. 파일명은 InceptionTime "no Dem"인데, 논문
    Table 3은 같은 모델을 ECG + 나이 + 성별로 적었다. 어느 seed인지는 미기재다.
  - CODE-15 Table 3 구간은 Table S1의 IQR과 같아서, 5 seed 중앙값(IQR)으로 본다.
    [모델 결과 표](../../mival-model-results.xlsx)의 "5 seed 중 최고"는 고친다.
- **Kardio-Net** (JACC EP 2024, medRxiv 2024.05.08.24307064 v1)
  - 짝짓기: ECG와 칼륨이 1시간 안이고, ECG마다 가장 가까운 검사를 짝짓는다. ECG는 전부 쓴다.
  - 고칼륨 기준은 K > 6.5 mEq/L 하나다.
  - split: 사람 단위 80/10/10이고, ESRD test는 147명이다.
  - ESRD 정의는 논문 본문에 없다.
- **von Bachmann** (Sci Rep 2024, PMC11222546)
  - 짝짓기: ECG와 채혈이 ±60분 안이다. 검증과 test는 사람당 첫 ECG만 쓴다.
  - 보고값은 5 random seed 평균(SD)이다.
  - 같은 표에서 MSE와 MAE가 같은 숫자로 읽힌 곳(temporal test)은 원문으로 다시 확인한다.

## ① 정답 study 형식

```
studies/repro/<paper_id>/study.yaml     프레임워크가 실행하는 파일
studies/repro/<paper_id>/claims.yaml    논문 보고값, 조건 차이. 비교 단계만 읽음
studies/repro/<paper_id>/evidence.yaml  필드마다 근거 (원문 인용, 위치)
studies/repro/<paper_id>/notes.md       옮길 수 없었던 것, 우리가 정한 것
```

`claims.yaml`:

```yaml
paper: {id: chiu2024-kardionet, doi: 10.1016/j.jacep.2024.07.023, version_read: "medRxiv v1"}
claims:
  - id: c1
    arm: kardionet-k-12lead
    metric: auroc
    cut: ">6.5"
    subgroup: esrd
    value: 0.852
    ci: [0.745, 0.956]
    aggregate: single          # single | mean_over_seeds | median_over_seeds | best_of_seeds
    cohort: "Cedars-Sinai ESRD test"
    source: "medRxiv v1 Table 2"
deviations:
  - {field: retrieve.subgroups_def.esrd, kind: our_choice, why: "논문에 ESRD 정의가 없음", claims: [c1]}
```

`deviations.kind`:

| kind | 뜻 |
|---|---|
| `our_choice` | 논문에 정보가 없어 우리가 정함 |
| `data_limit` | MIMIC 데이터 한계로 다르게 함 |
| `different_reference` | 정답 출처가 다름 |
| `different_model` | 같은 weight가 아님 (ECGFounder probe, von Bachmann seed 하나) |
| `paper_vs_code` | 논문과 코드가 다르고, 둘 중 하나를 따름 |

`evidence.yaml`은 study.yaml과 claims.yaml의 필드 경로를 key로 쓴다:

```yaml
retrieve.window_minutes:
  value: 60
  quote: "ECGs ... paired with a potassium value obtained within 1 hour"
  where: "medRxiv v1, Methods, Study population"
  status: stated            # stated | inferred | not_stated | our_choice
  reviewed_by: null         # 사람이 원문과 대조하면 이름과 날짜
```

### 정답 검토는 나중에 (사용자 결정 2026-10-07)

정답은 Claude가 만들고, 지금은 사람이 확인하지 않는다. 대신 나중에 제대로 확인할 수 있게 남긴다.

- 모든 정답 필드에 `evidence.yaml` 항목을 쓴다. `inferred`와 `our_choice`는 이유를 적는다.
- `docs/repro/gold-review.md`에 검토표를 만든다. 열은 paper_id, 필드, 값, 원문 인용, 위치, status,
  확인(빈칸)이다.
  - 라벨 정의, cutoff, 간격, split, 보고값처럼 결과를 바꾸는 필드를 위에 둔다.
- 원문 사본(PDF와 추출한 텍스트)은 `docs/repro/sources/<paper_id>/`에 둔다. 저작권 때문에 git에
  넣지 않는다. `.gitignore`에 추가하고, 받은 위치를 `sources.md`에 적는다.
- 비교 리포트와 B의 채점 리포트 머리에 "정답은 사람이 검토하지 않음 (검토 N / 전체 M 필드)"를
  표시한다. 이 수는 `reviewed_by`에서 센다.
- 같은 모델(Claude)이 정답과 자동 추출을 둘 다 하면 같은 실수가 "일치"로 채점될 수 있다. 이 위험은
  검토 전까지 남는다. 채점 리포트에 이 사실을 적는다.

## ② 추가할 기능

모든 새 필드는 기본값이 지금 동작과 같다. 수정 분류로 L2이고, 이 스펙 승인이 L2 승인이다.

### retrieve

| # | 필드 | 동작 | 필요한 곳 |
|---|---|---|---|
| R1 | `window_minutes`, `window_direction` (`both`/`before`/`after`), `pairing: nearest` | 분 단위 간격. ECG마다 시간상 가장 가까운 값을 짝짓는다. measurement의 `measurement_datetime`을 쓴다. 시각이 없는 행은 제외하고 그 수를 manifest에 남긴다 | 칼륨 2개, HeartWise, AI-NT-proBNP |
| R1 | `one_per_person: false` + `test_first_per_person` | ECG 전부 사용. 평가 세트만 사람당 첫 ECG로 줄이는 옵션 (von Bachmann) | 칼륨 2개, cavalab-mimic, ECGFounder |
| R2 | `label_source: {kind: file, path, key: waveform_path, split_column}` | 외부 라벨 파일. split 열이 있으면 profile이 그 split을 그대로 쓴다 | ECGFounder, cavalab-mimic |
| R3 | `subgroups_def: {<name>: {conditions: [...], procedures: [...], window: ever_before}}` | OMOP condition/procedure로 person 단위 부분집단 플래그를 만든다 | Kardio-Net ESRD |
| R4 | `label_source: {kind: death_followup, censor_rule: cavalab}` | 추적 기간 제한 없음, cavalab censoring 규칙. 기존 `death_within`은 그대로 | cavalab-mimic |
| R5 | 분할 라벨에 기계판독 시점 (`p_onset, p_end, qrs_onset, qrs_end, t_end`) | `machine_measurements.csv`에서 같이 읽는다 | OpenECG |

split 파일 만들기는 study 밖의 스크립트로 한다 (`scripts/repro/make_split_<paper_id>.py`). 저자 코드의
split 로직을 그대로 옮기고, 결과 건수를 논문과 대조해 notes.md에 적는다.

### evaluate

| # | 필드 | 동작 | 필요한 곳 |
|---|---|---|---|
| M1 | 회귀 지표에 `pearson`, `spearman` 추가. `label_transform: log10` (예측과 라벨 둘 다) | | AI-NT-proBNP, von Bachmann, Lima |
| M2 | `regression_cuts`와 분류 cutoff에 문자열 방향 허용 (`">6.5"`, `"<3.5"`, `"<=40"`) | 숫자만 주면 지금 동작 | 칼륨 2개, LVEF 2개 |
| M3 | 생존 지표에 `antolini_c` | PyCox와 같은 정의. 생존곡선 출력이 필요하다. 위험 점수만 내는 모델은 계산하지 않고 이유를 남긴다 | cavalab 4개 |
| M4 | 분할 지표에 `fiducial_error` (시점별 평균 오차와 SD), `boundary_f1` (허용 오차 `tolerance_ms`, 기본 150) | 모델 마스크에서 시점을 뽑는다 (`decode/intervals.py`) | OpenECG |
| M5 | `derived_arms`: 다른 arm 예측값으로 파생 변수를 만든다 (예: `pred_age - true_age > 8`). `cox: {covariates: [...], report: [hr, auc_at]}` | | Lima 생존 |

### models

| # | 무엇 | 필요한 곳 |
|---|---|---|
| D1 | cavalab MIMIC DeepHit 카드 2개 (`registry/repro/`). 기존 `mival_wrap_cavalab.py`를 넓혀 DeepHit 출력(이산 시간 PMF → 생존곡선)을 처리한다 | cavalab-mimic |
| D2 | 기존 카드는 다시 쓴다. 논문 방식 study의 `models.registry`는 기존 registry 디렉터리를 가리킨다 | 나머지 |

## ③ 비교 리포트

evaluate 마지막에 `claims.yaml`이 있으면 비교를 만든다. 출력은 `comparison.csv`, `comparison.md`이고,
전체를 모은 `docs/repro/summary.xlsx`는 스크립트로 만든다.

| paper | arm | metric | cut | subgroup | 논문 | 재현 (95% CI) | 차이 | 판정 | 조건 차이 |
|---|---|---|---|---|---|---|---|---|---|

- 짝짓기는 (arm, metric, cut, subgroup)으로 한다. 짝이 없는 claim도 행으로 남기고 "계산 못 함: 이유"를 적는다.
- 판정은 셋이다.
  - **재현됨:** 논문 값이 재현 95% CI 안이거나, 차이가 허용 범위 안이다.
  - **차이 있음:** 위가 아니다.
  - **비교 불가:** deviation에 `different_reference`가 있고 지표 정의가 달라 숫자를 놓을 수 없다.
- 허용 범위 기본값은 AUROC/C-index ±0.02, MAE·시점 오차 ±10%, 상관계수 ±0.05다. study에서
  `compare.tolerance`로 바꿀 수 있다.
- deviation이 있어도 판정은 낸다. 표에서 조건 차이 열을 항상 같이 보여준다.

## 실행 방식

차수별로 구현하고, 서버 실행은 queue로 한다 (사용자 결정 2026-10-07).

| 차수 | 기능 | 정답 study |
|---|---|---|
| 0 | 정답 study, claims, evidence 작성. 원문 수집. `gold-review.md` | 12개 전부 (실행 전 형식만) |
| 1 | M1, M2, ③ 비교 | heartwise, lima-age, singstad-age, ai-ntprobnp |
| 2 | R1, R2, R3 | vonbachmann-k, kardionet-k, ecgfounder |
| 3 | R4, R5, M3, M4, D1 | cavalab-code15, cavalab-mimic, semiseg, openecg |
| 4 | M5 | lima-survival |

- 각 차수는 구현 → 테스트 → 샘플 1,000건 end-to-end → 서버 queue에 전체 실행 등록 → 다음 차수
  구현으로 넘어간다.
- 구현은 차수 순서대로 한다. 1~3차가 같은 파일(`retrieve.py`, `evaluate.py`, metrics)을 고친다.
- 서버 queue: `/data/mi-val/queue/`에 `<순번>-<study>.sh`를 넣으면 tmux `rq` worker 하나가 순서대로
  실행한다. 로그는 `/data/mi-val/queue/log/`에 남는다. 실패하면 `.failed`를 붙이고 다음으로 넘어간다.
- 차수마다 기존 공통 study의 `config_hash`가 그대로인지 테스트로 확인한다.
- 실행 규칙은 [회귀·생존·분할 스펙](2026-10-07-three-task-vertical-slice-design.md)과 같다:
  - push는 서버 repo에만 하고 GitHub에는 하지 않는다.
  - 데이터 삭제, 새 인증, 이 스펙 밖의 L2, 같은 오류 3회면 멈춘다.
  - 인스턴스는 끄지 않는다.

## ④ B단계: 자동 추출과 채점

**입력:** 논문 PDF(보충자료 포함)와 repo 주소. weight 위치가 있으면 그것도 준다.

**출력:** 정답과 같은 형식의 `study.yaml`, `claims.yaml`, `evidence.yaml`과 카드 초안 `model.json`.
`notes.md`는 출력하지 않는다.

**추출기:**

- 지침서 `docs/repro/extraction-guide.md`와 출력 스키마를 받은 Claude 서브에이전트다.
- 정답(`studies/repro/`)이 없는 별도 작업 공간에서 돌린다. 지침서는 정답 파일을 인용하지 않는다.
- 논문에 없는 정보는 `not_stated`로 둔다. 추측으로 채우면 감점이다.
- 논문마다 3회 돌려 일관성을 본다.

**채점** (`scripts/repro/score_extraction.py <자동> <정답>`):

| 항목 | 방식 |
|---|---|
| 실행 조건 필드 | 일치 / 틀림 / 누락 |
| claims 값 | 숫자 정확히 일치. 출처 표도 일치하는지 |
| 지어낸 정보 | 정답 `not_stated`인데 값을 채운 수 |
| 근거 유효성 | 인용문이 원문 텍스트에 있는지 (공백 정규화 후 부분 문자열) |
| deviation 탐지 | 정답 deviation 중 찾아낸 비율 |

**개발용과 평가용:**

| 세트 | paper_id |
|---|---|
| 개발용 7 | heartwise, lima-age, ai-ntprobnp, vonbachmann-k, cavalab-code15, semiseg, lima-survival |
| 평가용 5 (+2) | ecgfounder, singstad-age, kardionet-k, cavalab-mimic, openecg (+ 정답 없는 둘, 아래) |

- 지침서는 개발용 결과로만 고친다. 평가용 점수가 최종 성능이다.
- 평가용에는 정답에 없는 HeartWise <50 head와 xECG를 넣는다. 자동 추출이 "재현 불가"와
  `not_stated`를 맞게 표시하는지 본다. 이 둘은 정답 study가 없으므로 사람이 판정할 항목으로
  `gold-review.md`에 같이 둔다.
- B 순서는 다음과 같다: B1 지침서 + 스키마 + 개발용 튜닝 → B2 평가용 추출과 채점 →
  (선택) 평가용 2~3건은 자동 study를 실제로 돌려 판정이 정답과 같은지 본다.

## 하지 않는 것

- 미세조정과 학습. ECGFounder는 기존 linear probe를 쓴다 (결정 E1).
- 기존 공통 study의 결과와 설정 변경.
- ml4h, HRNetV2, HeartKit의 논문 방식 study.
- 원문 PDF를 git에 넣는 것.
