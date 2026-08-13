# MI-VAL 연구 설계 Spec

- 작성일: 2026-08-14
- 상태: 초안 (팀 검토 대기)
- 범위: MIMIC-IV MI-CDM 전체 실행 + 다기관 이식은 설계까지
- 관련 문서: [pipeline](../pipeline/README.md) · [decisions](../decisions/README.md) · [models](../models/README.md) · [resources](../resources/README.md)

---

## 0. 요약

MI-VAL은 OMOP MI-CDM 위에서 공개 ECG 모델을 **독립적으로, 사전학습 오염을 자동 탐지하며, 기관을 옮겨 재현 가능하게** 평가하는 실행형 framework다.

첫 적용 사례는 ECGFounder와 PROPHECG-STEMI의 STEMI 탐지 성능 비교이며, 파이프라인은 회의록 기준 6단계(Retrieve, Profile, Preprocess, Models, Evaluation, Misclassification)를 유지한다.

이 문서는 앞부분이 연구 프로토콜(무엇을, 왜), 뒷부분이 단계별 실행 계약(어떻게)이다.

---

## 1. 연구 목적과 주장

### 1.1 배경

Han 등의 npj Digital Medicine 리뷰(2026)는 공개 ECG/PPG foundation model 분야의 상태를 다음과 같이 정리했다. 이 문장들이 MI-VAL의 존재 이유다.

| 리뷰가 보고한 문제 | 원문 취지 |
|---|---|
| 표준 평가 framework 부재 | "Current FMs lack a standardized evaluation framework." 이질적 데이터셋·지표로 평가돼 모델 간 비교가 불가능 |
| 자체 평가 편향 | 임상 평가가 대부분 모델 개발자 본인에 의해 수행됨. 최근 종합 벤치마크조차 최고 성능 모델을 소개한 연구가 직접 수행 |
| 평가 오염 | 벤치마크된 8개 모델 중 6개가 사전학습에 포함된 데이터셋으로 평가됨 |
| 데이터 편중 | MIMIC-IV-ECG·CSN·CPSC·PTB/PTB-XL 네 코퍼스가 전체 공개데이터 사용의 약 70%. MIMIC-IV-ECG는 검토 대상 12개 FM 중 9개가 사용 |
| fairness 미측정 | 16개 FM 중 subgroup 성능 격차를 본 모델은 2개뿐, 축도 age/sex/skin tone에 한정 |
| explainability 부실 | representation 수준 9개, prediction 수준 5개, 둘 다 3개. 정량 localization 지표나 무작위 사례 체계 평가를 쓴 모델은 없음 |
| shortcut 미검증 | 벤치마크된 8개 모델 전부가 생물학적 성별을 AUROC 0.8 이상으로 예측하지만, 실제 예측 시 그 정보에 의존하는지는 검증된 적 없음 |
| 생태계 부재 | 표준화된 telemetry, model registry, 조율된 보고 체계가 의료 FM에는 아직 존재하지 않음 |

리뷰는 해결 방향도 명시했다. **held-out 데이터에서의 독립 벤치마크**, **모델별 입력 형식과 전처리를 수용하는(균일 파이프라인을 강요하지 않는) 벤치마크**, 그리고 **데이터를 비공개로 두고 안전한 환경에서 모델을 평가하는 controlled-access 평가 플랫폼**이다.

MI-VAL의 설계는 이 세 요구를 각각 구현한 것이다. 즉 이 연구의 주장은 "우리 framework가 좋다"가 아니라 **"리뷰가 요구한 것을 실행 가능한 소프트웨어로 만들었다"**이다.

### 1.2 주장

| ID | 주장 | 근거가 되는 산출물 |
|---|---|---|
| **C1** | 모델은 ModelCard 하나로 등록되고, 전처리는 모델의 입력 계약에서 컴파일된다. 균일 파이프라인을 강요하지 않는다 | registry, `compile(input_contract)`, input contract gate |
| **C2** | 사전학습 코퍼스와 평가 cohort의 교집합을 자동 판정한다 | ModelCard `pretraining_corpora`, contamination gate, manifest flag |
| **C3** | 커맨드 한 줄로 다른 기관 MI-CDM에서 전체 파이프라인이 재실행된다 | site 축, 이식 절차(§5), 컨테이너 + lock |
| **F1** | 기관·adaptation·acquisition 축에서 모델 순위가 뒤집힌다 → 단일 벤치마크 순위는 신뢰할 수 없다 | run_key 축별 성능표, 순위 역전 분석 |

C1–C3는 구현으로 통제 가능하다. F1은 데이터가 답하며, 순위가 뒤집히지 않아도 그것 자체가 결과다.

리뷰는 이미 adaptation regime에 따른 순위 역전을 보고했다(ECG-CPC는 fine-tuning·frozen에서 우위, linear probing에서 열위. ST-MEM은 linear probing에서 우위. ECG-JEPA만 세 조건에서 안정). F1은 이 현상을 **site 축과 acquisition 축으로 확장**하는 것이다.

### 1.3 이번 spec의 범위

**포함**
- MIMIC-IV MI-CDM에서의 전체 6단계 실행
- ECGFounder, PROPHECG-STEMI 완전 온보딩
- 추가 공개 FM을 등록만으로 붙일 수 있는 registry 구조
- site 축을 포함한 run_key 설계와 다기관 이식 절차 명세

**제외 (설계만 하고 실행하지 않음)**
- 세브란스 데이터 실행 — IRB/DUA/MI-CDM ETL 선행 필요. §5에 이식 절차만 명세하고, 일정이 허락하면 붙인다
- 전향적 임상 평가 — DECIDE-AI 범위. future work로 기술
- privacy risk 평가(membership/attribute inference) — 리뷰가 지적한 항목이나 이번 범위 밖. limitation에 명시

---

## 2. 연구 프로토콜

### 2.1 연구 유형과 준수 표준

**진단정확도 연구(diagnostic accuracy study)**다. ECG로 현재 STEMI 상태를 판정하는 과제이며, 미래 사건을 예측하는 prognostic 과제가 아니다.

따라서 OHDSI PatientLevelPrediction의 time-at-risk 규약(index 이전 데이터만 predictor로 사용, outcome은 index 이후 TAR에서 관찰)을 따르지 않는다. 대신 다음을 준수한다.

| 표준 | 역할 |
|---|---|
| STARD 2015 | 진단정확도 보고. 특히 참가자 flow diagram과 index test–reference standard 간 간격 보고 |
| QUADAS-2 | 비뚤림 평가. flow and timing 영역의 "적절한 간격" 신호질문에 답할 수 있어야 함 |
| TRIPOD+AI | 모델 개발·검증 보고 |
| MI-CLAIM | AI 모델링 최소 보고 |
| FUTURE-AI | 신뢰성 6원칙. §4.5의 축 구조가 이를 구현 |

원고 부록에 STARD·TRIPOD+AI·MI-CLAIM 체크리스트를 첨부한다.

### 2.2 Cohort

**정의 도구**: OHDSI ATLAS (Data4Life 인스턴스). cohort definition JSON을 repo에 동결한다.

ATLAS를 쓰는 이유는 네 가지다.

1. 정의가 코드가 아니라 portable JSON으로 남아, 같은 정의가 같은 환자를 재현한다
2. 다른 OMOP 사이트에 그대로 로드하면 같은 cohort가 재생성된다 — C3(이식)의 전제
3. 임상의가 직접 정의를 편집할 수 있다
4. label 정의 4종을 CohortDiagnostics로 표준 비교할 수 있다 — §2.3의 민감도 분석이 여기 의존한다

MI-CDM은 imaging feature를 clinical domain 테이블(measurement)에 넣고 concept을 추가하는 방식으로 설계되어, 기존 OHDSI 도구가 계속 동작한다. `image_occurrence`는 `procedure_occurrence_id`를 보유하므로 ATLAS는 procedure 도메인으로 target cohort를 표현할 수 있다.

**역할 분담**

| 담당 | 내용 |
|---|---|
| ATLAS (JSON 동결) | target cohort(ECG 시행 성인 방문), outcome phenotype 4종 |
| 우리 코드 (Retrieve 이후) | `procedure_occurrence_id` → `image_occurrence` → `local_path` 해석, DICOM 로딩, split, 전처리, 모델, 평가 |

**검증 필요 전제**: MIMIC OMOP 변환에서 ECG가 `procedure_occurrence`에 신뢰성 있게 적재되어 있는지 실측되지 않았다. 미적재 시 target cohort는 `image_occurrence` 직접 SQL로 내려가고 ATLAS는 outcome 전용이 된다. → Retrieve 단계 확인 항목(§4.1).

**진입 기준**
- MI-CDM `image_occurrence`에 ECG DICOM이 존재하는 성인 방문
- index: ECG acquisition datetime
- ECG 선택: visit당 첫 ECG 1건을 index ECG로 사용. 나머지는 보관하되 primary 분석에서 제외(다중 ECG는 secondary)

### 2.3 Label 정의

**Primary는 visit-level STEMI ICD 정의**다.

시간창을 primary로 두지 않는 이유: MIMIC-IV의 ICD 진단은 입원 단위로 청구된 퇴원 진단이라 OMOP 변환 시 condition 날짜가 입원일로 붙는 경우가 많다. 시간 정밀도가 실재하지 않는데 시간창을 primary로 쓰면 존재하지 않는 정밀도를 가정하게 된다.

| 구분 | 정의 | 지위 |
|---|---|---|
| Primary | 동일 visit 내 STEMI ICD-10 (I21.0–I21.3 계열) | 주 분석 |
| Sensitivity 1 | Primary + index ECG 기준 시간창 제약 | Profile에서 날짜 분포 실측 후 채택 여부 결정 |
| Sensitivity 2 | ICD + troponin 상승 동반 | 민감도 |
| Sensitivity 3 | ICD + PCI/재관류 시술 코드 동반 | 민감도 |
| Negative | STEMI 코드 없음 | — |
| NSTEMI (I21.4) | negative에 포함하되 hard-negative subgroup으로 별도 보고 | 별도 |

label 정의는 run_key의 한 축이므로, 네 정의 전체에 대해 동일한 평가가 자동 반복된다.

**STARD/QUADAS-2 대응**: index test(ECG)와 reference standard(진단) 사이 간격을 Profile에서 실측해 보고한다. 간격이 보고 불가능하면 그 사실 자체를 limitation으로 기술한다.

### 2.4 Split과 event-count gate

**Split 원칙**
- 분할 단위는 **환자(person)**. 분석 단위는 ECG이므로 환자 단위로 나누지 않으면 누출이 발생한다
- outcome-stratified
- Profile 단계 종료 시 `cohort_split.parquet`으로 **동결**하고 이후 모든 단계는 읽기 전용으로 접근한다
- seed를 manifest에 기록한다

**dev / val / test의 실체.** 3분할이되 val은 독립 파일이 아니라 **dev 내부 k-fold의 hold-out fold**로 구현한다.

```text
cohort → dev (k-fold CV: 매 fold의 train / val) + test (동결, 한 번만 접촉)
```

val을 별도 고정 분할로 떼지 않는 이유는 단일 무작위 val이 하이퍼파라미터 선택을 불안정하게 만들고 데이터를 낭비하기 때문이다. `cohort_split.parquet`은 `person_id | split(dev|test) | fold(0..k-1, test는 null)`를 담으며, 이 한 파일이 dev/val/test를 모두 규정한다.

**비율을 미리 고정하지 않는다.** TRIPOD 및 Steyerberg/Collins 계열은 무작위 split-sample을 비효율적·불안정하다고 보고 전체 데이터 학습 + bootstrap optimism 보정을 권한다. 동시에 검증 표본의 event 수 하한은 100–200건으로 제시된다(Vergouwe 2005, Collins 2016).

따라서 **event-count gate**를 둔다.

```text
1. label 정의별 event 수를 센다
2. test set event ≥ 100을 만족하는 최소 test 비율을 역산한다
3. 만족 가능 → held-out test 확정 (기본 목표 20%)
4. 불가능 → held-out 포기, 전량 k-fold CV + bootstrap optimism 보정으로 전환
```

held-out을 완전히 버리지 않는 이유는, PROPHECG as-is는 전량이 external이므로 두 모델을 같은 지면에서 비교하려면 공통 held-out이 하나 필요하기 때문이다. 다만 이는 **비교용**이고, 학습을 수반하는 arm의 자체 성능 추정은 dev 내부 CV + bootstrap으로 이중 보고한다.

**Secondary split**: MIMIC-IV의 `anchor_year_group`을 이용한 era split으로 시간적 heterogeneity를 정량화한다. 비무작위 분할은 단일 성능 추정이 아니라 heterogeneity 정량화 목적으로만 쓴다.

### 2.5 Endpoint와 비교 구조

**Primary endpoint**: held-out test set에서의 AUROC (label primary 정의 기준).

**주요 비교 arm 4개**

| arm | training mode | 성격 |
|---|---|---|
| PROPHECG as-is | inference_only | 순수 external validation |
| PROPHECG head-retrained | linear_probe | 대칭 비교용 |
| ECGFounder linear-probe | linear_probe | in-domain 적응 |
| ECGFounder full-finetune | full_finetune | in-domain 적응 상한 |

**비대칭의 명시적 처리.** PROPHECG as-is는 이 cohort를 본 적이 없고, 학습을 수반하는 arm은 같은 기관 dev를 봤다. 따라서 **"ECGFounder가 더 높으면 그것은 모델 우위가 아니라 domain adaptation 효과일 수 있다"**를 결과 해석 규칙으로 사전 고정한다. PROPHECG head-retrained arm이 이 교란을 상쇄하기 위해 존재한다.

**Confirmatory 비교는 사전 지정한 2쌍**으로 한정하고 나머지는 exploratory로 표기한다.

1. PROPHECG as-is vs ECGFounder linear-probe (실사용 시나리오)
2. PROPHECG head-retrained vs ECGFounder linear-probe (대칭 조건)

---

## 3. 파이프라인 아키텍처

### 3.1 실행 모델

각 단계는 독립 CLI다. 입력은 `(앞 단계 artifact) + (spec 파일)`, 출력은 `(artifact) + (run manifest)`. 단계 간 결합은 파일과 스키마뿐이며 코드 import는 없다.

artifact 배치: `runs/<study_id>/<stage>/<config_hash>/{artifacts,manifest.json,logs}`

`config_hash`는 spec과 입력 artifact checksum의 해시다. 같은 입력·같은 설정이면 재실행이 skip되며 `--force`로 무시한다.

**저장 위치**: 모든 artifact는 `/data`(400 GB EBS)에 둔다. `/scratch`(instance store)는 DICOM 원본 캐시 전용이며 영속성을 가정하지 않는다.

### 3.2 Spec 계층

```text
studies/stemi-mimic-v1/
  study.yaml              # 최상위: 아래 spec 참조 + global seed + site
  cohort/
    target.json           # ATLAS export
    outcome_primary.json
    outcome_sens{1,2,3}.json
  preprocess/
    perturbation.yaml     # sweep grid (모델 무관)
  eval/
    eval.yaml             # 지표, bootstrap, subgroup, case selector
registry/
  models/<model_id>.json  # ROADMAP + x-mival (데이터, 코드 아님)
```

전처리 recipe는 사람이 쓰지 않는다(§4.3). 모델별 YAML이 없으므로 모델 추가 시 신규 파일은 ModelCard 하나뿐이다.

### 3.3 단계 계약

| # | 단계 | 입력 | 핵심 출력 |
|---|---|---|---|
| 1 | Retrieve | cohort JSON, MI-CDM | `cohort_index.parquet` |
| 2 | Profile | cohort_index, DICOM | `acquisition_metadata.parquet`, `profile_report.html`, event-count gate 결과, `cohort_split.parquet` |
| 3 | Preprocess | cohort_index, ModelCard 입력계약, perturbation | `tensors/<recipe_id>/`, `preprocess_index.parquet` |
| 4 | Models | tensors, split, ModelCard | `predictions/<run_key>.parquet`, 학습 weights, train log |
| 5 | Evaluation | predictions, labels, split | `metrics_long.parquet`, figure |
| 6 | Misclassification | predictions, metrics, tensors | `cases.parquet`, 사례 시각화, 집계 |

담당은 회의록 기준(규리T: 1–3, 민성: 4–6)이나, 실무상 전 단계에 함께 관여한다.

### 3.4 run_key — 결과 공간의 좌표

```text
run_key = site × model_id × training_mode × recipe_id
        × perturbation_id × label_def × split × fold
```

5·6단계는 이 축으로 groupby만 한다. 축이 늘어도 평가·감사 코드는 바뀌지 않는다. `site`는 이번 실행에서 `mimic` 단일 값이지만 축으로 존재해야 §5의 이식이 성립한다.

**run_key(실행 좌표)와 보고 축(§4.5)의 관계.** 둘은 같지 않다.

| 종류 | 구성 | 성격 |
|---|---|---|
| **run_key** | site, model_id, training_mode, recipe_id, perturbation_id, label_def, split, fold | **추론을 한 번 더 돌려야** 값이 생기는 축. prediction 파일을 가른다 |
| **보고 전용 축** | subgroup, outcome | 기존 prediction을 **다시 나누거나 다른 label을 붙이면** 되는 축. 추론 재실행 불필요 |

`subgroup`은 예측을 바꾸지 않고 집계만 나눈다. `outcome`(prognostic validity)은 같은 prediction에 다른 label을 조인한다. 따라서 연산 비용은 run_key 축에서만 곱해지고, 보고 축은 거의 공짜다. perturbation을 관찰적 subgroup이 아니라 별도 실행 축으로 둔 대가가 여기서 드러난다.

### 3.5 Gate 4종 (fail-fast)

| gate | 위치 | 조건 | 실패 시 |
|---|---|---|---|
| **Input contract** | 3→4 | ModelCard 입력계약과 실제 tensor metadata 일치 | 해당 record 제외 + ledger 기록 |
| **Event-count** | 2 | test set event ≥ 100 | split 전략을 전량 CV로 자동 분기 |
| **Leakage** | 4 | 학습에 쓰인 person_id가 test split에 부재 | 즉시 중단 |
| **Contamination** | 4 | ModelCard `pretraining_corpora` ∩ 평가 cohort 출처 = ∅ | 중단하지 않고 manifest·결과표에 `contaminated=true` 표시 |

contamination gate는 리뷰가 지적한 "8개 중 6개가 사전학습 데이터로 평가됨" 문제에 대한 직접 대응이다. 차단이 아니라 **표시**하는 이유는, 오염된 조건의 성능도 보고 가치가 있고 오염 여부가 결과 해석의 축이기 때문이다.

MIMIC-IV-ECG는 검토된 12개 FM 중 9개의 사전학습에 포함되었으므로, 이 gate는 MIMIC 실행에서 실제로 켜진다. 이것이 오염되지 않은 site(§5)가 필요한 정량적 근거가 된다.

### 3.6 Exclusion ledger

모든 단계는 record를 버릴 때 `exclusions.parquet`에 append한다.

```text
image_occurrence_id | person_id | stage | reason_code | detail | timestamp
```

단일 테이블이므로 **STARD 참가자 flow diagram이 여기서 자동 생성된다.** 논문 Figure 1이 코드 산출물이 된다.

### 3.7 Run manifest

```text
run_id, stage, study_id, site, config_hash
git_commit + dirty_flag
input_artifacts[]  {path, sha256}
output_artifacts[] {path, sha256}
env {python, 주요 lib version, CUDA/driver, GPU model, container digest}
seed
counts {in, out, excluded}
contamination {flag, overlapping_corpora[]}
compute {wall_time, peak_memory, throughput, gpu_hours}
warnings[], errors[]
started_at, ended_at
```

`compute` 블록은 리뷰가 요구한 표준 보고 항목 중 "연산 효율 지표"에 해당한다.

Technical Report는 별도로 작성하는 문서가 아니라 **manifest + exclusion ledger + 5·6단계 산출물을 렌더링한 결과물**이다. 형식은 HTML을 우선한다. 손으로 쓰는 부분은 해석뿐이다.

---

## 4. 단계별 명세

### 4.1 Retrieve

**출력** `cohort_index.parquet`

```text
person_id | visit_occurrence_id | image_occurrence_id | procedure_occurrence_id
| local_path | index_datetime | label_primary | label_sens1..3 | is_index_ecg
```

**확인 항목**
- MIMIC OMOP 변환에서 ECG가 `procedure_occurrence`에 적재되어 있는가 (§2.2의 전제)
- `image_occurrence.local_path`가 실제 DICOM에 해석되는가
- `mimic-ecg-metadata.csv`의 canonical 위치

### 4.2 Profile

**출력**
- `acquisition_metadata.parquet` — sampling frequency, lead order, duration, unit/sensitivity, manufacturer/station, DICOM 내부 channel 순서(`image_feature_value_order` 보존)
- `profile_report.html`
- event-count gate 결과
- `cohort_split.parquet` (동결)

**추가 산출 (이 연구 특유)**

1. **label 날짜 분포 실측** — condition start가 입원일에 몰려 있는지 확인해 sensitivity 1(시간창) 채택 여부를 결정한다
2. **acquisition 균질성 보고** — MIMIC-IV-ECG는 12-lead/10초/500 Hz로 균질할 것으로 예상된다. 균질하다는 사실 자체를 결과로 기록하며, 이것이 §4.3 perturbation이 관찰적 subgroup을 대체하는 근거가 된다
3. **사전 지정 hidden stratum 유도 가능성** — LBBB/paced, LVH, early repolarization, 심막염, 심방세동, low voltage, lead 오부착을 MIMIC-IV-ECG 기계 판독문에서 유도할 수 있는지 실측한다(§4.6)

### 4.3 Preprocess

**Op 라이브러리** (공용 코드, 모델 무관). 적용 순서를 고정한다.

```text
scale_unit → filter → resample → select/reconstruct_leads → crop/pad → normalize
```

순서가 다르면 같은 파라미터라도 결과가 달라져 chain 간 비교가 불가능하다.

**recipe는 사람이 쓰지 않고 컴파일된다.**

```text
recipe = compile(model.input_contract, source_metadata)
```

예: PROPHECG의 입력계약 `{leads:[I,II,V1..V6], fs:500, duration:10s, unit:mV}`이면
`[scale_unit(mV), resample(500→500), select_leads(8), crop(5000)]`이 생성된다.

새 모델 추가 시 전처리 YAML도 파이프라인 코드도 쓰지 않는다. **이 설계는 리뷰의 요구 — "벤치마크는 모델별 입력 형식과 전처리를 수용해야 하며, 균일 파이프라인을 강요하면 특정 아키텍처가 불리해진다" — 를 직접 구현한 것이다.**

**컴파일 실패 = input contract gate**

| 실패 유형 | 정책 | reason_code |
|---|---|---|
| 요구 lead 부재, 재구성 불가 | 제외 | `lead_unavailable` |
| source fs < target fs | 기본 거부(`allow_upsample: false`), 명시 허용 시만 통과 | `upsample_required` |
| duration 부족 | `pad_policy` 따름, 기본 제외 | `duration_short` |
| unit/sensitivity 미기재 | 제외 (default 추정 금지) | `unit_missing` |

마지막 항목은 decisions의 Preprocess Open Question 3("필수 metadata가 없을 때 default를 쓸 것인가 record를 제외할 것인가")에 대한 답이다. 추정한 unit은 조용히 틀리며, 틀린 것을 검증할 방법이 없다.

**Perturbation**

핵심 규칙: **변형 후 다시 입력계약 형태로 되돌린다.** 500 Hz → 125 Hz로 낮춘 뒤 다시 500 Hz로 복원해 모델에 넣는다. 그래야 측정 대상이 정보 손실뿐이고 shape 불일치가 섞이지 않는다.

```text
axes:
  resample:        [500, 250, 125, 100]
  lead_dropout:    [none, drop_V3V4, precordial_only, limb_only]
  duration:        [10, 5, 2.5]
  amplitude_scale: [1.0, 0.5, 2.0]
  noise:           [none, baseline_wander, powerline_50hz, emg]
```

- 기본 정책은 **OFAT(한 번에 한 축)**. 4~5축 × 3~4수준 = baseline 포함 약 13~16조건. 4 arm × 13 = 52 run으로 감당 가능하다. full cartesian은 opt-in
- **perturbation tensor는 디스크에 저장하지 않고 dataloader에서 on-the-fly 적용**한다. 재현성은 `seed + perturbation_id`로 보장한다
- lead_dropout 축은 리뷰가 보고한 reduced-lead 성능(ECGFounder single-lead variant, HeartLang, ECG-JEPA, ST-MEM이 단일 lead에서 multilabel 진단 AUROC 0.80 이상)과 직접 비교 가능하다

perturbation은 medical algorithmic audit의 **adversarial testing**에 해당하며, FUTURE-AI의 **Robustness** 원칙을 구현한다.

### 4.4 Models

**ModelCard = ROADMAP 준수 + `x-mival` 실행 확장**

RSNA ATLAS의 `model.json`(ROADMAP 온톨로지 기반)은 발견·공유용 카드이며 실행 계약이 아니다. `Input`은 자유서술이고, 학습 데이터 전용 섹션이 없으며, `Model performance`는 지표 이름 enum과 자유서술 comment로만 구성된다. 따라서 서술 레이어는 ROADMAP을 그대로 따르고 실행에 필요한 것은 확장 네임스페이스에 둔다.

```text
x-mival:
  adapter: keras | torch
  weights: [{uri, sha256, role}]
  ensemble: {method: mean_probability, members: 5}
  input_contract: {leads, sampling_rate_hz, duration_s, unit, scaling, layout, dtype}
  output: {type: softmax, positive_index: 1}
  feature_layer: <name>
  threshold: {value, provenance, status}
  runtime: {framework, python, device}
  training_modes_supported: [...]
  pretraining_corpora: [...]          # contamination gate의 입력
```

`pretraining_corpora`를 **필수 필드**로 둔다. 리뷰가 지적한 평가 오염 문제는 이 정보가 카드에 없어서 자동 판정이 불가능했기 때문에 발생한다.

발행 시에는 ROADMAP 호환 model.json으로 export한다(RSNA ATLAS에 게재하면 DOI 부여 가능).

**Adapter interface** — backend는 이 뒤로 숨는다.

```text
load(weights_ref) → handle
forward(handle, batch) → prob
features(handle, batch) → embedding
trainable_groups(handle) → named groups
fit(handle, data, mode, hparams) → handle
attribute(handle, batch) → saliency        # optional
```

**Training mode**

| mode | 학습 대상 | dev 사용 | 필수 기록 |
|---|---|---|---|
| `inference_only` | 없음 | 미사용 | weights checksum |
| `linear_probe` | 최종 head | dev + 내부 CV | feature_layer, head hparams |
| `partial_unfreeze` | 지정 group | dev + 내부 CV | unfreeze group 목록 |
| `full_finetune` | 전체 | dev + 내부 CV | 전체 hparams |

하이퍼파라미터와 early stopping은 dev 내부 CV로만 결정한다. test는 마지막 한 번만 접촉하며 접촉 사실을 manifest에 기록한다.

**Threshold 정책** (decisions의 Evaluation Open Question 2에 대한 답)

| 정책 | 내용 |
|---|---|
| `legacy` | 원 논문 값 그대로 (PROPHECG archived PTB revision `0.0768`) |
| `refit_youden` | dev에서 Youden J 재추정 |
| `refit_sens95` | dev에서 sensitivity 95% 지점 |

셋 다 보고하되 **primary는 `refit_sens95`**로 한다. STEMI는 miss 비용이 압도적이라 민감도·특이도를 대칭 취급하는 Youden은 임상적으로 부적절하다. **test set에서는 threshold를 선택하지 않는다.**

**출력** `predictions/<run_key>.parquet`

```text
image_occurrence_id | person_id | split | fold | label_primary | label_sens1..3
| prob | logit | model_id | training_mode | recipe_id | perturbation_id | seed | site
```

### 4.5 Evaluation

**구조: 범주 4개 × 축 6개.** subgroup·perturbation·fairness는 범주가 아니라 축이며, 4개 범주 전부를 그 축으로 자른다.

**범주 (무엇을 재는가)**

| # | 범주 | 지표 | 근거 |
|---|---|---|---|
| 1 | Discrimination | AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, MCC, balanced accuracy, discrimination slope | Steyerberg 2010 |
| 2 | Calibration | CITL(mean) → intercept/slope(weak) → flexible curve(moderate), Brier + scaled Brier + Murphy 분해 | Steyerberg 2010, Van Calster 2019 |
| 3 | Clinical utility | decision curve, net benefit vs treat-all/treat-none | Vickers, Steyerberg 2010 |
| 4 | Interpretation | attribution, lead/time region, 임상 타당성 (정성) | FUTURE-AI Explainability |

**"Overall performance"를 별도 층으로 두지 않는 이유**: Brier score는 독립 성질이 아니라 합성량이다. Murphy 분해에 따라 `Brier = reliability − resolution + uncertainty`이며, reliability는 calibration, resolution은 discrimination에 대응한다. 따라서 Brier를 Calibration 층에 두고 분해값을 함께 보고한다. STEMI처럼 유병률이 낮으면 Brier가 uncertainty 항에 지배되므로 **scaled Brier(IPA)를 반드시 병기**한다.

**Clinical utility를 별도 범주로 두는 이유**: net benefit은 순위 기반이 아니어서 discrimination이 아니고, 예측–관측 일치가 아니어서 calibration이 아니다. "이 threshold에서 이 모델을 쓰는 것이 안 쓰는 것보다 나은가"라는 결정이론적 질문이다. STEMI는 miss 비용이 커서 임상적으로 타당한 threshold 대역이 낮은 쪽(약 1–10%)이며, 그 대역에서 treat-all 대비 net benefit이 낮으면 AUROC가 높아도 임상적으로 무가치하다.

**축 (어디서 재는가)**

```text
× site           (mimic / 향후 추가 기관)
× subgroup       (sex, age, 임상 / sampling freq, lead, manufacturer / 사전 지정 hidden stratum)
× perturbation   (§4.3)
× label 정의     (primary + sensitivity 3)
× outcome        (STEMI 진단 / prognostic validity)
× split          (dev CV / held-out test / era)
```

이는 §3.4의 run_key와 동일한 축이다. 평가 코드는 "범주 4개를 계산하는 함수" 하나이며 나머지는 groupby다.

**FUTURE-AI 매핑**

| 원칙 | MI-VAL 위치 |
|---|---|
| Fairness | 축 (임상 subgroup) |
| Robustness | 축 (perturbation) |
| Universality | 축 (site / era) |
| Usability | 범주 3 |
| Explainability | 범주 4 |
| Traceability | 범주 아님 — manifest·ledger (§3.6–3.7) |

**Prognostic validity (outcome 축의 값)**

모델 score가 downstream outcome(30일 사망, 24시간 내 PCI, 재관류까지의 시간)을 예측하는지 평가한다. 별도 범주가 아니라 outcome 축의 값이므로 추가 구현 부담은 outcome cohort 정의 2–3개뿐이다.

목적은 label noise 방어다. ICD가 음성인데 모델이 양성으로 예측한 환자의 실제 예후가 나빴다면, 그것은 모델 오류가 아니라 label 오류라는 증거가 된다.

**Shortcut 진단 (신규)**

리뷰는 벤치마크된 8개 모델 전부가 embedding에서 생물학적 성별을 AUROC 0.8 이상으로 예측하지만 실제 예측 시 그 정보에 의존하는지는 검증된 적 없다고 지적했다. 다음을 추가한다.

1. **Representation probe** — 각 모델의 embedding에서 sex, age band, manufacturer/station을 linear probe로 예측. 인코딩 여부 측정
2. **의존 여부 검정** — 위에서 인코딩이 확인된 속성에 대해, 해당 속성으로 층화한 성능 격차(Fairness 축)와 attribution 분포를 대조

인코딩과 의존은 다른 문제이므로 둘을 분리해 보고한다.

**불확실성과 비교**

- **Bootstrap**: person-level stratified bootstrap 2,000회, percentile CI. 환자당 ECG가 복수일 수 있으므로 재표집 단위는 반드시 person
- **모델 간 비교**: DeLong은 clustered data에 부적절하므로 **paired bootstrap of difference**(person 단위 재표집)를 primary로 한다. net benefit 차이도 동일
- **다중비교**: §2.5의 사전 지정 2쌍만 confirmatory. 나머지는 exploratory로 명시. Bonferroni 남발 대신 사전 지정 + CI 보고
- **Subgroup 억제**: `n_events < 10`이면 계산하되 `suppressed=true`로 표시하고 결론에 사용하지 않는다

**출력** `metrics_long.parquet` (tidy long format)

```text
run_key 컬럼들 | category | metric | value | ci_lo | ci_hi | n | n_events | suppressed | contaminated
```

long format이므로 축이 늘어도 스키마가 불변이다.

**Figure (전부 코드 산출물)**: STARD flow(ledger 자동 생성) · ROC/PR · flexible calibration curve · decision curve · perturbation 열화 곡선 · subgroup forest plot · 순위 역전 도표(F1)

### 4.6 Misclassification

**근거**: 이 단계는 **medical algorithmic audit**(Liu, Glocker, McCradden, Ghassemi, Denniston, Oakden-Rayner, Lancet Digital Health 2022)의 구현이다. audit이 제시한 세 접근이 MI-VAL 설계와 1:1로 대응한다.

| audit 접근 | MI-VAL 구현 |
|---|---|
| exploratory error analysis | `error`, `boundary` selector |
| subgroup testing | subgroup 축 |
| adversarial testing | perturbation 축 |

**왜 필요한가**: Oakden-Rayner 등(ACM CHIL 2020)은 집계 지표가 임상적으로 중요한 소집단의 실패를 은폐함을 실증했다. 기흉 검출에서 전체 AUROC 0.87인 모델이 흉관 있는 사례 0.94, 없는 사례 0.77이었고 test set 양성의 80%가 흉관을 동반했다. 모델이 학습한 것은 질환이 아니라 치료 흔적이었으며 집계 지표로는 검출 불가능하다.

**왜 5단계와 분리하는가**: 목적이 다르다. Evaluation은 사전 지정된 질문에 답하는 성능 *추정*(confirmatory)이고, Misclassification은 예상하지 못한 실패 양식의 *발견*(exploratory)이다. 한 단계에 섞으면 성능표를 본 뒤 눈에 띄는 subgroup을 사후 선택해 보고하는 오염이 발생한다. 단계와 파일로 분리하면 구조적으로 차단된다. 이는 decisions의 "Misclassification을 독립 단계로 둘 것인가"에 대한 답이다.

**사전 지정 failure mode (FMEA)** — 데이터를 보기 전에 고정한다.

| stratum | 우려 |
|---|---|
| LBBB / paced rhythm | 고전적 STEMI mimic. Sgarbossa 기준 영역 |
| LVH, early repolarization, 심막염 | ST 상승 mimic |
| post-PCI / 재관류 후 ECG | label은 STEMI이나 파형은 정상화. 흉관 사례와 동형의 함정 |
| 심방세동, low voltage, lead 오부착 | 신호 품질 교란 |

유도 가능성은 Profile에서 실측한다(§4.2).

**케이스 선정 selector 4종** — `eval.yaml`에 선언하며 코드에 모델을 하드코딩하지 않는다.

| selector | 정의 |
|---|---|
| `error` | reference run_key의 FP 상위 k / FN 하위 k |
| `boundary` | `abs(prob − threshold) < ε` |
| `contrast` | run_key 패턴 a·b 간 `label_flip` 또는 `abs(Δprob) > τ` |
| `instability` | 같은 패턴 내 fold/seed 간 `prob_std > τ` |

`contrast`가 축을 특정하지 않으므로 다음이 모두 같은 코드로 산출된다.

- `model_id` 축 → 모델 간 불일치
- `training_mode` 축 → fine-tuning이 실제로 무엇을 바꿨는가
- `perturbation_id` 축 → 500 Hz에서 맞고 125 Hz에서 뒤집힌 사례. perturbation 열화 곡선의 정성적 증거
- `label_def` 축 → 정의에 따라 정답이 바뀌는 사례. label noise의 실물
- `site` 축 → 기관 간 불일치 (§5 이후)

즉 6단계는 특정 비교를 수행하는 모듈이 아니라 **run_key 공간에서 흥미로운 지점을 뽑는 질의 엔진**이다.

**케이스별 동반 정보**: waveform plot, acquisition metadata, preprocess warning, attribution overlay, **label 정의 4종의 값**, cohort/index context.

**수기 review 연결**: 자동 생성 `cases.parquet` + 사람이 채우는 `review.csv`(verdict, note)를 `case_id`로 병합한다. 자동/수기 경계가 파일로 분리되어 재실행 시 review가 유실되지 않는다.

**확인 항목**: MIMIC은 공개 데이터이나 개별 waveform 그림의 논문 게재가 DUA상 허용되는지 확인이 필요하다. 리뷰는 ECG/PPG가 생체 식별자로 인식되며 통상적 익명화 후에도 재식별 정확도가 85%를 넘는 연구가 있다고 지적했다.

---

## 5. 다기관 이식 (설계만)

C3를 성립시키는 요소는 다음과 같다. 이번 범위에서는 구조만 확보하고 실행하지 않는다.

**site 축**: `study.yaml`의 `site` 필드가 run_key 전 구간을 통과한다. 성능표는 site별로 자동 분리된다.

**이식 시 교체되는 것과 유지되는 것**

| 교체 | 유지 |
|---|---|
| CDM 접속 정보 | ATLAS cohort JSON (동일 정의 재생성) |
| DICOM 경로 해석 규칙 | ModelCard, 입력계약, op 라이브러리 |
| site 식별자 | perturbation grid, eval.yaml, 지표 정의 |
| — | split 규칙, gate 임계값 |

**이식 절차**

```text
1. 대상 기관에 MI-CDM ETL 완료 (선행 조건, 이번 범위 밖)
2. ATLAS에 cohort JSON 로드 → target/outcome cohort 재생성
3. study.yaml의 site와 CDM 접속만 교체
4. 전 단계 실행 → site 축이 추가된 metrics_long 산출
5. 기존 site 결과와 병합해 순위 역전 분석 (F1)
```

**이 설계의 근거**: 리뷰는 오염되지 않은 held-out 평가셋 확보를 위해 "데이터는 비공개로 두고 모델을 안전한 환경에서 평가하는 controlled-access 평가 플랫폼"이 필요할 수 있다고 했다. MI-VAL을 기관 내부에서 실행하는 것이 그 구현이다.

**정량적 근거**: MIMIC-IV-ECG는 검토된 12개 FM 중 9개의 사전학습에 포함되었다. contamination gate가 MIMIC에서 몇 개 모델에 켜지는지를 보고하면, 오염되지 않은 site가 왜 필요한지가 수치로 제시된다.

**선행 작업 (일정 확인 필요)**: IRB, DUA, 대상 기관 MI-CDM ETL, ECG DICOM 변환.

---

## 6. 재현성과 테스트

**결정론**: 전역 seed → 단계별 파생 seed, cudnn deterministic, 데이터 순서 고정.

**환경**: backend별 lock 파일 2종(`keras27`, `torch`) + 컨테이너 이미지 digest를 manifest에 기록.

**테스트 4종**

| 종류 | 내용 | 실행 |
|---|---|---|
| unit | op별 수치 검증 (resample 후 길이·주파수, lead 순서, crop anchor) | 매 커밋 |
| contract | 모든 ModelCard에 대해 합성 신호로 compile + forward 성공 | 매 커밋 |
| **golden** | 고정 합성 ECG 10건에 대한 모델 출력 해시 대조 | 매 커밋 |
| e2e | fixture cohort 50건으로 6단계 전체 통과 | nightly |

golden test가 가장 중요하다. weights 교체나 라이브러리 드리프트로 예측이 조용히 바뀌는 것을 잡는 유일한 장치다.

---

## 7. 문헌 근거 매핑

리뷰가 제시한 표준 FM 보고 8항목에 대한 MI-VAL의 대응이다.

| 리뷰의 요구 항목 | MI-VAL 대응 | 상태 |
|---|---|---|
| (i) 사전학습·평가 데이터셋과 아키텍처 투명 기술 | ModelCard + `pretraining_corpora` + Profile report | 충족 |
| (ii) 모델 artifact와 추론 코드·adaptation 스크립트 제공 | registry + adapter + training mode | 충족 |
| (iii) 표준 성능 지표 | §4.5 범주 4개 | 충족 |
| (iv) 명시적 shortcut learning 평가 | §4.5 shortcut 진단 (representation probe + 의존 검정) | 부분 충족 |
| (v) 인구학적 subgroup fairness 분석 | subgroup 축 + 억제 규칙 | 충족 |
| (vi) privacy risk 평가 | 이번 범위 밖 | 미충족 (limitation 기술) |
| (vii) 연산 효율 지표 | manifest `compute` 블록 | 충족 |
| (viii) 설계 선택을 분리하는 ablation | training mode × perturbation 축이 부분적으로 수행 | 부분 충족 |

**설계 요소별 근거**

| 설계 요소 | 근거 |
|---|---|
| 진단정확도 프레이밍, flow diagram, 간격 보고 | STARD 2015, QUADAS-2 |
| cohort를 ATLAS JSON으로 동결 | The Book of OHDSI Ch.10 |
| PLP time-at-risk를 따르지 않는 이유 | The Book of OHDSI Ch.13 |
| split 대신 CV+bootstrap 병행, event ≥100 | TRIPOD, Steyerberg/Collins 내부검증 문헌, Vergouwe 2005, Collins 2016 |
| 지표 4범주 | Steyerberg 2010 |
| calibration 위계와 moderate 목표 | Van Calster 2019 |
| Brier를 Calibration에 배치 | Murphy 분해 |
| decision curve / net benefit | Vickers, Steyerberg 2010 |
| 신뢰성 축(F/U/R/E) | FUTURE-AI |
| 최소 보고 | MI-CLAIM, TRIPOD+AI |
| ModelCard 스키마 | ROADMAP / RSNA ATLAS `model.json` |
| MI-CDM 위에서 OHDSI 도구 재사용 | MI-CDM extension 문헌, JAMIA 2025 |
| 6단계 audit | Liu et al. Lancet Digit Health 2022 |
| 집계 지표의 은폐 | Oakden-Rayner et al. ACM CHIL 2020 |
| 모델별 전처리 수용, 오염 판정, controlled-access | Han et al. npj Digit Med 2026 |

---

## 8. Open Questions와 이 spec의 결정

decisions/README.md의 미해결 질문 중 이 spec이 답한 것.

| 질문 | 이 spec의 결정 | 위치 |
|---|---|---|
| Preprocess: 필수 metadata 부재 시 default vs 제외 | 제외. default 추정 금지 | §4.3 |
| Models: 사용자가 architecture/hyperparameter를 어디까지 바꾸는가 | training mode 4종으로 고정. dev 내부 CV로만 선택 | §4.4 |
| Models: archived PTB threshold 0.0768 재추정 여부 | 세 정책 모두 보고, primary는 `refit_sens95` | §4.4 |
| Models: ECGFounder downstream head와 split protocol | linear_probe/full_finetune, dev/val/test 3분할, split은 Profile에서 동결 | §2.4, §4.4 |
| Evaluation: primary/secondary metric | primary = held-out test AUROC. 범주 4개 전체 보고 | §2.5, §4.5 |
| Evaluation: threshold 선정 | 위와 동일 | §4.4 |
| Evaluation: CI와 통계 비교 | person-level stratified bootstrap 2,000회, paired bootstrap of difference | §4.5 |
| Evaluation: Interpretation을 정량 평가할 것인가 | 범주 4로 두되 정성 보고. shortcut 진단만 정량 | §4.5 |
| Pipeline: Misclassification을 독립 단계로 둘 것인가 | 독립 단계. confirmatory/exploratory 분리가 이유 | §4.6 |
| Pipeline: Technical Report 형식 | HTML 우선, 렌더링 산출물로 고정 | §3.7 |

**여전히 미해결**

- MIMIC OMOP에서 ECG가 `procedure_occurrence`에 적재되어 있는가 (§2.2)
- condition 날짜 분포 — sensitivity 1(시간창) 채택 여부 (§2.3)
- 사전 지정 hidden stratum을 기계 판독문에서 유도 가능한가 (§4.2)
- `mimic-ecg-metadata.csv` canonical 위치
- "최신 MI-CDM" snapshot 고정 방법
- 개별 waveform 게재의 DUA 허용 여부 (§4.6)
- Misclassification 담당자
- 세브란스 이식의 IRB/DUA 일정 (§5)

---

## 9. 참고문헌

1. Han A, Tohyama T, Yoon D, Paik K, Gow B, Celi LA, Lee H, Lee HC. Review of open foundation models and datasets for ECG and PPG waveforms. *npj Digital Medicine*. 2026. doi:10.1038/s41746-026-03101-7
2. Steyerberg EW, Vickers AJ, Cook NR, et al. Assessing the performance of prediction models: a framework for traditional and novel measures. *Epidemiology*. 2010;21(1):128–138.
3. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Medicine*. 2019;17:230.
4. Vickers AJ, van Calster B, Steyerberg EW. A simple, step-by-step guide to interpreting decision curve analysis. *Diagnostic and Prognostic Research*. 2019;3:18.
5. Liu X, Glocker B, McCradden MM, Ghassemi M, Denniston AK, Oakden-Rayner L. The medical algorithmic audit. *The Lancet Digital Health*. 2022;4(5):e384–e397.
6. Oakden-Rayner L, Dunnmon J, Carneiro G, Ré C. Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. *ACM CHIL*. 2020.
7. Norgeot B, Quer G, Beaulieu-Jones BK, et al. Minimum information about clinical artificial intelligence modeling: the MI-CLAIM checklist. *Nature Medicine*. 2020;26:1320–1324.
8. Lekadir K, et al. FUTURE-AI: international consensus guideline for trustworthy and deployable artificial intelligence in healthcare. *BMJ*. 2025.
9. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024.
10. Bossuyt PM, Reitsma JB, Bruns DE, et al. STARD 2015: an updated list of essential items for reporting diagnostic accuracy studies. *BMJ*. 2015;351:h5527.
11. Whiting PF, Rutjes AWS, Westwood ME, et al. QUADAS-2: a revised tool for the quality assessment of diagnostic accuracy studies. *Annals of Internal Medicine*. 2011;155(8):529–536.
12. The Book of OHDSI. Chapter 10 (Defining Cohorts), Chapter 13 (Patient-Level Prediction).
13. ROADMAP: An Ontology of Medical AI Models and Datasets. *Radiology: Artificial Intelligence*. 2026. / RSNA ATLAS `model.json` schema.
14. Breaking data silos: incorporating the DICOM imaging standard into the OMOP CDM to enable multimodal research. *JAMIA*. 2025;32(10):1533.
15. Development of Medical Imaging Data Standardization for Imaging-Based Observational Research: OMOP Common Data Model Extension. 2024.
16. Murphy AH. A new vector partition of the probability score. *Journal of Applied Meteorology*. 1973;12:595–600.
17. Sgarbossa EB, Pinski SL, Barbagelata A, et al. Electrocardiographic diagnosis of evolving acute myocardial infarction in the presence of left bundle-branch block. *NEJM*. 1996;334:481–487.
18. PROPHECG-STEMI 원 논문: Development of Clinically Validated Artificial Intelligence Model for Detecting ST-segment Elevation Myocardial Infarction. *Annals of Emergency Medicine*. 2024. PMID 39066765.

> 참고문헌 5, 6, 8, 9, 12–15의 권·호·면수는 원문 대조 후 최종 확정한다.

---

## 10. 다음 단계

1. 이 spec을 팀이 검토하고 §8의 미해결 항목에 담당·기한을 붙인다
2. `docs/decisions/README.md`에 이 spec이 내린 결정을 기록 규칙(결정일·내용·이유·참석자·관련 문서)대로 반영한다
3. `docs/pipeline/*.md`를 이 spec과 정렬한다
4. 구현 계획을 별도 문서로 작성한다
