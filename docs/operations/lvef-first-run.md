# LVEF 첫 서버 실행 (retrieve, profile, preprocess 샘플)

2026-09-21, DICOM-MIVA 서버(`/data/mi-val/mi-va`, branch `feat/framework-core`)에서
`studies/lvef`를 실제 MI-CDM 데이터베이스와 DICOM 캐시로 처음 끝까지 돌린 기록이다.

## 준비: 코드와 인덱스

retrieve에 두 가지 변경이 먼저 들어갔다.

1. **modality 필터** (commit `bda29d9`, "feat: retrieve filters image_occurrence
   by modality"). `cdm.image_occurrence`가 ECG와 MIMIC-CXR 흉부 X선을 같은
   테이블에 섞어 담고 있어(`docs/operations/dicom-cache.md` 참고),
   `RetrieveSpec`에 필수 `modality_concept_id`를 추가하고 SQL 바깥 쿼리에
   `where i.modality_concept_id = %(modality_concept_id)s`를 달았다.
   `studies/lvef/study.yaml`에 `modality_concept_id: 4145308`을 넣었다.
2. **Postgres numeric 컬럼 변환** (commit `18c7089`, "fix: retrieve converts
   Postgres numeric columns to float"). 첫 실제 DB 실행에서
   `write_summary`의 `quantile()` 호출이
   `TypeError: unsupported operand type(s) for *: 'decimal.Decimal' and 'float'`로
   죽었다. psycopg가 Postgres `numeric`(측정치 `value_as_number`)을
   `decimal.Decimal`로 돌려주는데, numpy가 `Decimal`과 `float`을 함께 연산하지
   못해서다. `query_postgres`가 DataFrame을 만든 직후 `_decimals_to_float`로
   Decimal 값을 가진 모든 컬럼을 float64로 정규화하도록 고쳤다. 컬럼이나
   라벨 이름을 하나도 알지 않는 일반적인 고정이다.

두 커밋 모두 `python3 -m pytest tests -q`가 684 passed, 12 skipped로 끝난 뒤
서버로 push했다(전자는 683 passed였고 후자가 새 테스트 1개를 더했다).

인덱스는 retrieve가 `cdm.measurement`를 person_id로 훑는 lateral join을 감당할
수 있도록 미리 만들었다. 이름, 정의, 빌드 시간, `explain` 결과는
[`micdm-database.md`](micdm-database.md)의 "인덱스" 절에 있다. 요약하면
`measurement_lvef_person_date_idx`, 2026-09-21, 6분 45초, retrieve SQL의
lateral 서브쿼리가 실제로 이 인덱스를 쓴다.

## 실행

```bash
cd /data/mi-val/mi-va

/data/mi-val/envs/mival/bin/mival run retrieve \
  --study studies/lvef --runs-root /data/mi-val/runs

/data/mi-val/envs/mival/bin/mival run profile \
  --study studies/lvef --runs-root /data/mi-val/runs \
  --input cohort_index=/data/mi-val/runs/lvef/retrieve/72370acd5dc64a06/artifacts/cohort_index.parquet

# 1,000행 샘플 (pandas로 cohort_index.parquet 앞 1,000행을 같은 컬럼으로 저장)
/data/mi-val/envs/mival/bin/mival run preprocess \
  --study studies/lvef --runs-root /data/mi-val/runs \
  --input cohort_index=/data/mi-val/runs/lvef/_sample/cohort_index_sample.parquet
```

retrieve는 처음 한 번 Decimal 버그로 실패했다(`config_hash 72370acd5dc64a06`,
`manifest.failed.json`, wall_time 20.9초). 고치고 같은 study.yaml로 다시
돌리자 config_hash는 그대로였고(스펙과 입력 체크섬만 해시에 들어가므로) 이번에는
성공해 같은 디렉터리에 `manifest.json`을 썼다.

## config_hash와 커밋

| stage | config_hash | git commit (manifest 기준) | wall_time_s |
|---|---|---|---|
| retrieve | `72370acd5dc64a06` | `18c7089` | 19.116 |
| profile | `f6e1c93a1893fc8d` | `18c7089` | 0.731 |
| preprocess (1,000건 샘플) | `d1b1addef1d41b13` | `18c7089` | 7.424 |

세 실행 모두 `git.dirty: false`다. retrieve가 두 개의 lateral 쿼리(±7일, ±30일
창)를 796,617행에 대해 돌리는 데 20초가 채 안 걸린 것은
`measurement_lvef_person_date_idx` 덕분이다. 인덱스 없이 돌렸다면 새로 만든
파셜 인덱스가 없는 `cdm.image_occurrence` 쪽 seq scan과는 비교가 안 될 만큼
느렸을 것이다(brief는 40분을 기준으로 잡았지만 실제로는 초 단위였다).

## retrieve 결과

`retrieve_summary.json` (`/data/mi-val/runs/lvef/retrieve/72370acd5dc64a06/artifacts/`):

| 항목 | 값 |
|---|---|
| `n_ecg` (counts.in) | 796,617 |
| `n_labelled` | 182,960 |
| `n_out` (counts.out) | 182,927 |
| `n_persons_out` | 44,179 |

제외 사유:

| reason_code | 건수 |
|---|---|
| `label_missing` | 613,657 |
| `label_implausible` | 33 |
| `path_unresolvable` | 0 |
| `file_missing` | 0 |

dispatch가 준 기대치(counts.in 796,617)와 정확히 일치한다. `file_missing`이
0이라는 것은 DICOM 캐시(800,035개 파일)가 retrieve가 필요로 한 파일을 전부
갖고 있었다는 뜻이다(`require_local_file: true`).

라벨 분포(`label_primary` cutoff 40, `label_sens1` cutoff 50):

| 항목 | 값 |
|---|---|
| `label_primary` 양성 | 44,056 |
| `label_sens1` 양성 | 54,817 |
| `label_sens2` 양성 | 44,056 |

`label_sens2`가 `label_primary`와 완전히 같은 것은 버그가 아니다. ±30일
창에서 "가장 가까운" 값을 고르는 규칙은, 이미 ±7일 안에서 최근접을 찾은
레코드라면 항상 같은 값을 고른다(±7일 안의 최근접이 존재하면 그보다 먼
값이 전체 최근접이 될 수 없다). `label_sens2`는 retrieve가 남긴 레코드가
아니라 애초에 `label_missing`으로 빠진 레코드를 구분하는 축이라, 지금
`cohort_index`에는 차이가 보이지 않는다.

`label_value_quantiles`: p5=20.0, p25=45.0, p50=55.0, p75=60.0, p95=71.0.

`label_delta_days` 분포(일 단위, ECG 대비 LVEF 측정일 차이, 절대값 7 이하):

```
-7: 3,751   -6: 4,119   -5: 4,574   -4: 5,543   -3: 7,013   -2: 9,076   -1: 15,045
 0: 43,672   1: 43,721   2: 18,579   3: 10,494   4: 6,360    5: 4,227    6: 3,486   7: 3,267
```

0일과 1일에 몰려 있다(같은 날 또는 다음 날 측정이 가장 흔하다). 대칭에 가깝지만
음수 쪽(ECG보다 LVEF가 먼저)이 양수 쪽보다 조금 적다.

## profile 결과

`profile_summary.json` (`/data/mi-val/runs/lvef/profile/f6e1c93a1893fc8d/artifacts/`):

| split | n_persons | n_ecgs | n_positive_ecgs |
|---|---|---|---|
| dev | 35,343 | 146,757 | 35,383 |
| test | 8,836 | 36,170 | 8,673 |

gate: `min_test_positives=100`, `test_positives=8673`, `passed: true`.

`label_value_quantiles`, `label_delta_days`는 retrieve와 같다(사람 단위로
나눴을 뿐 값 자체는 그대로다). `ecgs_per_person`은 1건인 사람이 11,830명으로
가장 많고, 최대 89건인 사람이 1명 있다.

## preprocess 1,000건 샘플

`cohort_index.parquet`의 앞 1,000행을 `cohort_index_sample.parquet`로 저장해
돌렸다(`/data/mi-val/runs/lvef/_sample/cohort_index_sample.parquet`).

| 항목 | 값 |
|---|---|
| `in` / `out` | 1,000 / 1,000 |
| `index_rows` | 5,000 (레코드 1,000 × 모델 5) |
| `tensors` | 4,000 |
| `recipes` | 4 |
| `models` | 5 |
| `perturbations` | 8 |
| `excluded` (모든 reason_code) | 0 |

`unit_missing`, `rate_unsupported` 둘 다 0건으로 기대와 같다. `recipes`가
`models`(5)보다 적은 것은 heartwise-lvef-binary와 heartwise-lvef-regression이
둘 다 250 Hz, 12-lead, gain 있는 같은 입력 계약이라 recipe 하나를 공유하기
때문이다(`model_ids: ["heartwise-lvef-binary", "heartwise-lvef-regression"]`).

`recipes.json`에서 확인한 것:

- xECG: `resample -> 100.0 Hz` (500 Hz DICOM에서 100 Hz로), `normalize: none`
- HeartWise(두 모델 공유): `resample -> 250.0 Hz`, `gain factor: 208.3333`,
  `normalize: none`
- ECGFounder: 500 Hz 그대로, `normalize: global_zscore`
- PROPHECG-STEMI: 500 Hz, 8-lead(`I, II, V1..V6`)만 선택, `normalize: none`

grid 경고(perturbation 규칙 3, 실패가 아니라 경고):

```
model 'heartwise-lvef-binary': perturbation axis 'resample' has baseline level 500 above the input contract's sampling rate (250.0). The cell is the identity, which is correct, but its name claims a value this model never receives.
model 'heartwise-lvef-regression': perturbation axis 'resample' has baseline level 500 above the input contract's sampling rate (250.0). The cell is the identity, which is correct, but its name claims a value this model never receives.
model 'xecg': perturbation axis 'resample' has baseline level 500 above the input contract's sampling rate (100.0). The cell is the identity, which is correct, but its name claims a value this model never receives.
```

로컬에서 `mival.perturbation.check_grid_against_contract`를 5개 카드에 대해
직접 불러 먼저 확인한 것과 서버 실행에서 나온 경고가 정확히 같다. 500 Hz
기준(ecgfounder, prophecg-stemi)은 경고가 없다.

## 발견한 문제

1. **Decimal/float TypeError (고침)**. 위 "준비" 절 참고. `query_postgres`가
   psycopg의 raw fetch 결과를 그대로 pandas DataFrame으로 만들면서, Postgres
   `numeric` 컬럼이 `decimal.Decimal` object dtype으로 들어왔다. 비교
   연산(`<=`, `.isna()`)은 Decimal과 float 사이에 통하지만 `numpy.quantile`의
   보간(`Decimal * float`)은 안 된다. `profile.py`의 `write_summary`도 같은
   `.quantile()` 호출을 하므로, retrieve만 고치지 않았다면 profile에서
   똑같이 터졌을 것이다. `_decimals_to_float`가 `query_postgres`의 결과에
   한 번 적용되므로 이후 모든 stage는 `label_value`를 float64로 받는다
   (`cohort_index_sample.parquet`에서 `label_value: float64` 확인함).
2. **grid 경고 3건은 예상된 것이다.** Task 8 ruling에서 이미 baseline-only
   axes를 쓰기로 했고(`studies/lvef/study.yaml`의 주석 참고), 250 Hz와
   100 Hz 카드가 500 Hz baseline보다 낮은 rate라 경고가 난다. 실패가 아니라
   경고이므로 preprocess는 정상 완료됐다.
3. **`label_missing`이 613,657건(796,617의 약 77%)으로 크다.** LVEF가 있는
   환자(147,431행, 74,612명)보다 ECG(796,617행)가 훨씬 많은 코호트라
   자연스러운 결과로 보이지만, `micdm-database.md`의 "±7일 내 LVEF가 있는
   ECG는 241,450건"이라는 이전 메모와 지금의 182,927(kept, label_implausible
   33건 제외 전 182,960)을 정확히 겹쳐 보지는 못했다. 같은 COUNT를 다시
   내보진 않았고, retrieve 자체의 exclude 순서(label_missing이 먼저 걸러지고
   그 다음 label_implausible)가 다르므로 산술은 맞는다고 본다. 이상 신호는
   아니라고 판단했지만 재확인 여지는 남긴다.

## 협업자에게 물어볼 것

새로 발견한 것은 없다. `micdm-database.md`의 "LVEF 적재" 절 "확인이 남은 것"에
있는 두 항목이 여전히 유효하다.

- (person, datetime) 중복 4쌍(값이 55/60으로 다른 1쌍 포함)
- 10% 이하 값 527건(0.0이 18건). 0.0을 실측으로 볼지 label 정의에서 정한다
