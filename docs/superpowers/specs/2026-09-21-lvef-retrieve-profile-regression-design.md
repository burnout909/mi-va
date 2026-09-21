# LVEF 과제: Retrieve, Profile, 회귀 출력 설계

날짜: 2026-09-21
상태: 설계 승인. 구현 계획: `../plans/2026-09-21-lvef-retrieve-profile-regression.md`

## 0. 한 줄 요약

MI-CDM에서 ECG와 LVEF 라벨을 끌어와 `cohort_index.parquet`를 만드는 Retrieve,
split을 동결하는 Profile, DICOM loader, 회귀 출력 유형, 그리고 torch adapter의
builder 선언(A-4)을 추가한다. 4개 모델(ECGFounder, xECG, HeartWise 이진 ≤40,
HeartWise 회귀)로 분류와 회귀를 같은 파이프라인에서 돌린다.

## 1. 목적과 범위

목적은 LVEF 과제를 실제로 돌리면서 프레임워크를 깎는 것이다. 새 개념은
최소로 만들고, 기존 stage 계약(`Stage`, `required_inputs`, `run`, ledger,
manifest)에 맞춘다.

이번 범위:

- Retrieve stage (신규)
- Profile stage (신규)
- DICOM loader 함수 1개
- 회귀 출력 유형: 라벨 컬럼 `label_value`, 예측 컬럼 `pred_value`, 카드
  `output.type: regression`, adapter fit의 MSE 목적함수, evaluate의 회귀 지표
- torch adapter의 builder 선언 (A-4)
- ModelCard 3장(xecg, heartwise-lvef-binary, heartwise-lvef-regression)과
  `studies/lvef/study.yaml`

이번 범위 밖:

- EchoNext-Mini와 tabular 입력 (L2-3, 보고 항목 유지)
- 코호트 단위 정규화 (L2-2). HeartWise의 상수 배율은 카드 `input_contract.gain`으로 환원
- Profile HTML report. 표 하나(`profile_summary.json`)로 대신한다
- 회귀 arm의 Misclassification. 6단계 selector는 확률 위에 정의돼 있어 회귀
  arm은 경고와 함께 건너뛴다. 잔차 기반 selector는 후속 항목
- DICOM 92 GiB 다운로드. 운영 작업이고 `docs/operations/`에 기록한다

## 2. 코드 스타일 원칙

이 spec의 코드는 기존 stage보다 짧고 읽기 쉽게 쓴다.

- 함수 이름이 문서다. docstring은 한 줄. 주석은 "왜"만 적는다
- SQL은 `.sql` 파일로 분리한다. 파이썬에 SQL 문자열을 넣지 않는다
- pandas로 처리한다. dict 순회로 테이블을 다시 만들지 않는다
- 방어 코드는 계약 위반(컬럼 누락, 알 수 없는 값)에만 둔다
- 목표 분량: retrieve 120줄, profile 80줄, DICOM loader 40줄 안팎

## 3. 데이터 계약

### 3.1 `cohort_index.parquet` (Retrieve 출력)

| 컬럼 | 형 | 뜻 |
|---|---|---|
| `image_occurrence_id` | int | ECG 식별자 |
| `person_id` | int | |
| `local_path` | str | `local_path_root` 아래로 재해석된 DICOM 경로 |
| `index_datetime` | datetime | ECG 취득 시각 |
| `label_datetime` | datetime | 선택된 LVEF 측정 시각 |
| `label_delta_days` | int | `label_datetime - index_datetime` (일) |
| `label_value` | float | LVEF % |
| `label_primary` | int | `label_value <= 40` |
| `label_sens1` | int | `label_value < 50` |
| `label_sens2` | float (0/1, NaN) | `label_value <= 40`, 단 창을 ±30일로 넓혀 선택한 LVEF 기준. 30일 안에도 없으면 NaN |
| `label_sens3` | float (NaN) | 비워둔다. 스키마 호환용 |

기존 Preprocess가 요구하는 세 컬럼(`image_occurrence_id`, `person_id`,
`local_path`)과 Models가 요구하는 `label_*` 네 컬럼은 그대로다.
`label_value`만 추가된다.

### 3.2 `cohort_split.parquet` (Profile 출력)

기존 계약 그대로: `person_id`, `split` (`dev` | `test`), `fold` (dev만 0..k-1,
test는 null).

### 3.3 `predictions/<run_key>.parquet` (Models 출력, 변경)

`PREDICTION_COLUMNS`에 `label_value`(float)와 `pred_value`(float)를 추가한다.

- 분류 arm: `prob`, `logit` 채움. `pred_value` null
- 회귀 arm: `pred_value` 채움. `prob`, `logit` null. 확률을 지어내지 않는다

### 3.4 run_key

축을 추가하지 않는다. 회귀 arm은 `label_def=value`로 구분된다.
`Arm.label_column`이 `label_value`를 가리키면 그 arm은 회귀다.

## 4. Retrieve stage

파일: `src/mival/stages/retrieve.py`, `src/mival/stages/sql/retrieve_ecg_lvef.sql`

### 4.1 spec (study.yaml `retrieve:`)

```yaml
retrieve:
  dsn_env: /data/mi-val/secrets/micdm.env   # PG* 변수를 가진 env 파일
  schema: cdm
  label_concept_id: 3027172                 # LOINC 10230-1, LVEF
  window_days: 7                            # label_primary, label_sens1 창
  window_days_sens2: 30                     # label_sens2 창
  implausible_below: 5                      # 이 값 이하 LVEF는 제외
  local_path_root: /scratch/mi-val/dicom
  require_local_file: true
```

`dsn_env`의 내용(비밀번호)은 config_hash에 들어가지 않는다. 경로 문자열만
들어간다. DB snapshot 식별은 산출물의 `retrieve_summary.json`에 적재 행 수와
`max(measurement_id)`를 남기는 것으로 한다.

### 4.2 SQL

한 파일, 파라미터 세 개(`schema`, `label_concept_id`, `window_days`).
`image_occurrence`의 모든 ECG에 대해 lateral join으로 `person_id`가 같고
날짜 차이가 창 안인 LVEF 중 최근접 1건(동률이면 이른 시각)을 붙인다. 창 안에
없는 ECG는 라벨 null로 남긴다. 창 밖 제외는 파이썬에서 ledger에 적기 위해서다.

`window_days_sens2`로 같은 SQL을 한 번 더 실행해 `label_sens2`를 만든다.

### 4.3 처리 순서

1. env 파일을 읽어 psycopg 접속. 접속 정보는 로그와 manifest에 남기지 않는다
2. SQL 2회 실행 (7일, 30일) → DataFrame
3. `local_path` 재해석: CDM 값에서 `files/` 이후 부분을 취해 `local_path_root`
   아래로 (A-1). `files/`가 없으면 `path_unresolvable`
4. 제외 (ledger, 순서 고정)
   - `label_missing`: 7일 창 안에 LVEF 없음
   - `label_implausible`: `label_value <= implausible_below`
   - `path_unresolvable`: 3에서 실패
   - `file_missing`: `require_local_file`이고 파일이 없음
5. 라벨 컬럼 계산, `cohort_index.parquet` 저장
6. 실행한 SQL 두 개를 파라미터가 채워진 형태로 `sql/` 아래에 복사
7. `retrieve_summary.json`: 단계별 남은 수, 라벨 분포(양성 수, LVEF 분위수),
   `label_delta_days` 분포

`reason_codes = {label_missing, label_implausible, path_unresolvable, file_missing}`

### 4.4 의존성

`pyproject.toml`에 optional extra `db = ["psycopg[binary]>=3.1"]`. 모듈 상단에서
import하지 않는다. `run` 안에서 import한다 (기존 pandas 규칙과 같다).

## 5. Profile stage

파일: `src/mival/stages/profile.py`

### 5.1 spec

```yaml
profile:
  test_fraction: 0.2
  n_folds: 5
  stratify_on: label_primary
  min_test_positives: 100      # event-count gate
```

### 5.2 처리

1. `cohort_index` 입력. person 단위로 집계: 사람마다 양성 ECG가 하나라도
   있으면 양성
2. `stratify_on`으로 층화해 `test_fraction`만큼 test, 나머지 dev. seed는 ctx.seed
3. dev를 `n_folds`로 층화 분할, `fold` 부여
4. event-count gate: test의 양성 ECG 수가 `min_test_positives` 미만이면
   `GateError`로 실패. 통과 여부와 수치는 summary에 적는다
5. `cohort_split.parquet` 저장
6. `profile_summary.json`: split별 person 수, ECG 수, 양성 수, `label_value`
   분위수, `label_delta_days` 분포, 사람당 ECG 수 분포

`acquisition_metadata.parquet`는 이번에 만들지 않는다. Preprocess가 loader에서
읽는 `SourceMetadata`(Hz, lead, 길이, unit)를 `preprocess_index`에 이미
기록하므로, 균질성 확인은 그 표로 한다.

`reason_codes`는 비어 있다. Profile은 레코드를 제외하지 않는다.

## 6. DICOM loader

파일: `src/mival/stages/preprocess.py`에 `load_dicom_record` 추가

- pydicom으로 `WaveformSequence[0]`을 읽어 `(n_leads, n_samples)` float32
- lead 이름은 `ChannelSourceSequence[0].CodeMeaning`에서, 표준 12 이름으로
  정규화 (`Lead I` → `I`)
- 단위는 `ChannelSensitivityUnitsSequence[0].CodeValue`(`uV` 또는 `mV`)를
  그대로 `SourceMetadata.unit`에. 변환은 기존 `ScaleUnit` op가 한다
- `sampling_rate_hz = SamplingFrequency`
- 실제 MIMIC DICOM 한 파일로 값을 확인한 뒤 고정한다. 확인 결과는
  `docs/operations/`에 적는다

Preprocess spec에 `loader: dicom | npz` 키를 두고 `PreprocessStage`가 이름으로
고른다. 지금은 생성자 주입만 있어 CLI에서 DICOM을 고를 방법이 없다.

optional extra `dicom = ["pydicom>=3.0"]`. 실측: `waveform_array`가 sensitivity를
적용해 mV를 돌려주고, MIMIC 파일의 채널 순서는 aVF가 aVL보다 앞이며 MDC 코드로
lead 이름을 정한다.

## 7. 회귀 출력 유형

### 7.1 ModelCard

`output.type`에 `regression`을 추가한다. 회귀 카드의 output:

```json
{"type": "regression", "n_outputs": 1, "value_index": 0, "unit": "%"}
```

출력을 그대로 값으로 쓴다. HeartWise 회귀 체크포인트는 실측상 LVEF %를
직접 낸다 (`docs/models/heartwise-lvef.md`). 0..1을 내는 체크포인트가 나타나면
그때 `postprocess` 키를 L0으로 추가한다. 미리 만들지 않는다.

### 7.2 adapter

- `forward`는 분류와 같이 `(batch,)` float를 낸다. 카드가 `regression`이면
  `value_index` 열을 그대로 낸다
- `fit`: `FitData`에 `objective` (`bce` | `mse`)를 추가. Models가 arm의
  label_def로 정한다. linear probe head는 `mse`일 때 선형 회귀 (ridge,
  weight_decay를 λ로). fine-tune은 loss만 바꾼다
- `pos_weight`와 `early_stopping_metric`은 `mse`에서 무시하고 `val_mae`로
  early stopping
- torch와 keras adapter 둘 다 구현한다 (N=2 불변식)

### 7.3 Models stage

- `label_value`를 float로 읽는다. 다른 label 컬럼은 int 그대로
- 회귀 arm은 `fit_thresholds`를 건너뛴다. train_log의 `thresholds`는 빈 dict,
  `primary_threshold_policy`는 null
- `inference_only` 회귀 arm은 카드가 `regression`이어야 한다. 아니면
  `ValueError`
- contamination gate와 leakage gate는 그대로

### 7.4 Evaluate stage

- arm의 label_def가 `value`면 지표 집합을 바꾼다: `mae`, `rmse`, `r2`,
  그리고 `auroc_below@40` (`label_value <= 40` 대 `40 - pred_value`. 컷은
  evaluate spec `regression_cuts`, 기본 `[40]`). bootstrap CI는 기존 기계 그대로
- calibration, utility 범주는 회귀 arm에서 생략한다
- `metrics_long`의 `metric` 값에 위 네 개가 추가된다. `category_of`에
  `regression` 범주를 추가한다
- 그림: 예측 대 실측 산점도 1장. 기존 ROC 그림은 분류 arm만

### 7.5 Misclassify stage

회귀 arm을 만나면 `warnings`에 적고 건너뛴다.

## 8. torch adapter builder 선언 (A-4)

카드 `x-mival`에 다음을 추가한다.

```json
"weights_format": "state_dict",          // state_dict | checkpoint_state_dict | safetensors | jit
"builder": {"module": "net1d:Net1D", "kwargs": {...}},
"state_dict_key": "state_dict",           // checkpoint 안의 키. 없으면 최상위
"rename_keys": {"classifier.fc1.": "classifier.3."}   // 선택
```

`load`는 `code_path`를 sys.path에 넣고 `builder.module`을 import해 `kwargs`로
만든 뒤 형식에 맞게 가중치를 넣는다. `jit`이면 builder 없이
`torch.jit.load`. ECGFounder 카드에 builder를 채우고, 기존 `load`와 같은
module이 나오는지(state_dict 키와 shape 일치) 회귀 테스트로 확인한다.

`n_outputs` 검증은 유지한다. 단 어느 텐서에서 읽을지는 카드
`output.weight_key`(예: `dense.weight`)로 선언한다. 없으면 검증을 건너뛴다.

## 9. ModelCard와 study.yaml

카드 3장 추가: `registry/models/xecg.json`, `heartwise-lvef-binary.json`,
`heartwise-lvef-regression.json`. 입력 계약은 `docs/models/*.md`의 실측값.
HeartWise 두 장은 `input_contract.gain: 208.3333`(저자의 `1/0.0048`)을 적는다.
`gain`은 단위 변환 뒤, 정규화 앞에 곱하는 상수이고 기본값 1.0이다. 저자의 코호트
단위 스펙트럼 스케일링(L2-2)은 적용하지 않고 `notes`에 그 사실을 적는다.

`studies/lvef/study.yaml`:

```yaml
study_id: lvef
site: dicom-miva
seed: 20260921
stages:
  retrieve: {…§4.1}
  profile: {…§5.1}
  preprocess: {loader: dicom, registry: registry/models}
  models:
    arms:
      - {model_id: heartwise-lvef-binary, training_mode: inference_only, label_def: primary}
      - {model_id: heartwise-lvef-regression, training_mode: inference_only, label_def: value}
      - {model_id: ecgfounder, training_mode: linear_probe, label_def: primary}
      - {model_id: ecgfounder, training_mode: linear_probe, label_def: value}
      - {model_id: xecg, training_mode: linear_probe, label_def: primary}
      - {model_id: xecg, training_mode: linear_probe, label_def: value}
  evaluate: {…}
```

fine-tune arm은 GPU 시간을 본 뒤 추가한다. 코드 변경 없이 arm 한 줄이다.

## 10. 오류 처리

- DB 접속 실패, SQL 오류: 그대로 예외. wrapper가 `manifest.failed.json`을 쓴다
- 라벨이 전부 null인 경우(concept_id 오타 등): `retrieve`가 `ValueError`로
  즉시 실패. 조용히 빈 코호트를 내지 않는다
- event-count gate 미달: `GateError`. 수치가 summary에 남는다
- DICOM 읽기 실패: 기존 preprocess 제외 코드로. 새 코드 없음
- 회귀 arm에 `regression`이 아닌 카드: `ValueError`

## 11. 테스트

- `test_stage_retrieve.py`: DB 없이 검증한다. SQL 실행을 함수 하나로 분리해
  DataFrame을 주입. 창 안/밖, 최근접 선택, implausible, 경로 재해석(두 root),
  파일 존재 여부 각각 한 케이스
- `test_stage_profile.py`: person 단위 분리(같은 사람이 dev와 test에 동시에
  없음), 층화 비율, seed 재현성, gate 실패
- `test_dicom_loader.py`: pydicom으로 만든 합성 DICOM 하나. lead 이름 정규화,
  unit 전달
- `test_torch_adapter.py`: builder 선언 로드가 기존 ECGFounder 로드와 같은
  state_dict를 만든다
- `test_adapter_training.py`: `objective: mse` linear probe가 합성 선형 데이터에
  서 계수를 복원한다. torch와 keras 둘 다
- `test_stage_models.py`, `test_stage_evaluate.py`: 회귀 arm 한 개가 끝까지
  흘러 `mae`, `auroc_at_40` 행이 나온다. 분류 arm의 golden은 바뀌지 않는다
- `test_golden.py`: 기존 스냅샷 변경 없음이 통과 조건

## 12. 구현 순서

1. A-4 builder 선언 + ECGFounder 회귀 테스트 (다른 것과 독립)
2. Retrieve + 테스트. DB에서 한 번 실행해 `retrieve_summary.json` 확인
3. Profile + 테스트
4. 회귀 출력 유형: modelcard → adapter fit/forward → models → evaluate →
   misclassify 순
5. DICOM 다운로드(운영) 와 loader. 실제 파일로 unit, lead 이름 확정
6. 카드 3장, study.yaml, 첫 end-to-end 실행

1과 2, 3은 병렬 가능하다. 4는 1 뒤에. 5는 다운로드가 끝나는 대로.

## 13. Adaptation ledger 반영

- A-1: retrieve 구현으로 종료
- A-4: builder 선언으로 종료
- A-5: `regression` 출력 유형으로 종료 (연구 결정: 회귀와 분류 병행)
- L2-2, L2-3: 그대로 보고 항목
- 새 항목: `label_value`와 `pred_value`가 스키마에 들어간 것은 L1 (라벨이
  이진이라는 가정을 프레임워크가 갖고 있었다)
