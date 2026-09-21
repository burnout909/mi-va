# MI-CDM 데이터베이스

DICOM-MIVA에 MIMIC-IV OMOP CDM과 MI-CDM 확장을 세운 기록이다.

**이 문서는 환경 구축 기록이지 프레임워크 수정 기록이 아니다.** 아래 항목들은
[`../decisions/adaptations.md`](../decisions/adaptations.md)에 들어가지 않는다.

## 구성

| 항목 | 값 |
|---|---|
| 위치 | `<instance-id>` (DICOM-MIVA), localhost 전용 |
| 엔진 | PostgreSQL 18.4 (Ubuntu 26.04 패키지) |
| data directory | `/data/pgdata` (400 GiB EBS. stop 후에도 보존) |
| database / schema | `micdm` / `cdm` |
| role | `mival` |
| 자격증명 | `/data/mi-val/secrets/micdm.env` (0600) |
| 튜닝 | `/etc/postgresql/18/main/conf.d/mival.conf` |

`listen_addresses = localhost`다. 외부에서 붙으려면 SSH 터널을 쓴다.

`/data/pgdata`는 `mi-val-storage.service`가 마운트하는 EBS 위에 있다. 기본
unit은 그 mount를 기다리지 않아 부팅마다 "`/data/pgdata` is not accessible"로
실패했다 (2026-09-14 확인). `/etc/systemd/system/postgresql@18-main.service.d/
after-storage.conf`에 `After=` / `Requires=mi-val-storage.service`를 두어
순서를 고정했다.

`synchronous_commit = off`는 **적재 전용 설정**이다. 적재가 끝나면 `on`으로
되돌린다.

```bash
ssh -i <key>.pem -p 2022 ubuntu@<host>
set -a; . /data/mi-val/secrets/micdm.env; set +a
psql
```

## 데이터 출처

| 대상 | S3 |
|---|---|
| OMOP core | `s3://<bucket>/Datasets/MIMIC-IV_CDM/` |
| MI-CDM 확장 | `s3://<bucket>/Datasets/MIMIC-IV_CDM/Extension/` |

전송은 **presigned URL**을 쓴다. 인스턴스에 IAM instance profile이 붙어 있지
않고(`iam:PassRole` 미승인, `infra/aws/IAM_REQUEST.md` 1번), 서버에서
`aws sts get-caller-identity`가 credentials를 찾지 못한다. 로컬에서 서명한
링크를 서버가 `curl`로 받으면 **자격증명이 서버에 남지 않고** S3에서 직접
받으므로 랩탑 대역폭도 거치지 않는다.

## 적재 중 맞춰야 했던 것

MIMIC-IV의 실제 값이 OMOP 규격을 여러 곳에서 벗어난다. 전부 스키마 쪽을
넓혀서 해결했고, 데이터는 변형하지 않았다.

### E-1 · 확장 DDL 두 벌이 서로 다르다

`~/Desktop/youlab/mi-cdm/micdm-etl/ddl/extension/`의 `image_feature`에는
`image_feature_value_order`와 `image_instance_uid`가 없고, UID를
`"image_study_UID"`로 대소문자 혼용 인용한다. S3 Extension이 CSV와 함께
동봉한 `CREATE TABLE image_feature.txt`에는 둘 다 있고 컬럼명이 소문자다.

`image_feature_value_order`는 2026-08-06 회의에서 DICOM 채널 순서를 보존하기로
한 바로 그 필드다. 없으면 lead order를 잃는다.

**S3가 동봉한 DDL을 채택했다.** 데이터와 같은 곳에서 온 스키마가 권위다.

### E-2 · OMOP 5.3이지 5.4가 아니다

`visit_occurrence`에 `admitting_source_concept_id`가 있다. 5.4에서는
`admitted_from_concept_id`로 바뀐 컬럼이다. `measurement`에는 5.4가 없앤
`measurement_time`이 있다.

`micdm-etl`이 들고 있는 것은 5.4 DDL이고, 서버 `/opt/ohdsi-mimic/etl/ddl/`의
5.3 DDL은 BigQuery 방언(`INT64`, `CREATE OR REPLACE TABLE`)이라 쓸 수 없다.
OHDSI CommonDataModel의 PostgreSQL 5.3 DDL을 받아 적용했다.

적재 전 **43개 CSV의 헤더를 전수 대조**했다. 컬럼 이름과 순서 불일치 0건.

### E-3 · id가 int64 해시이고 음수를 포함한다

`person_id = -7819202826128270750`. OMOP DDL은 `person_id integer`다.

`_id`로 끝나고 `_concept_id`가 아닌 컬럼 175개를 `bigint`로 넓혔다. concept id는
진짜 OMOP vocabulary id이므로 `integer`로 두었다.

`fact_relationship.fact_id_1`과 `fact_id_2`는 이 규칙이 놓쳤다. `_1`, `_2`로
끝나기 때문이다. 별도로 넓혔다.

### E-4 · 문자열이 규격 길이를 넘는다

- `vocabulary_id`가 `varchar(20)`인데 `mimiciv_meas_chartevents_value` (30자)
- `observation.value_as_string`이 `varchar(60)`인데 67자 DRG 설명이 들어온다

`cdm`의 254자 이하 문자열 컬럼을 전부 `varchar(2000)`으로 넓혔다. 분석용
사본이므로 길이 제약이 지킬 의미를 갖지 않는다.

### E-5 · `note_nlp.offset`이 예약어

PostgreSQL 예약어라 컬럼 목록에서 인용이 필요하다.

## 재현

```bash
# 로컬에서 매니페스트 생성 (헤더 전수 대조 포함) 후 서버에서 적재
scripts/  # TODO: 이번 실행 스크립트를 저장소로 옮긴다
```

## LVEF 적재 (2026-09-21)

LVEF 과제의 label 원천이다. MIMIC-IV 코어 OMOP에는 LVEF 수치가 없어(measurement·observation
0건) 협업자가 OMOP measurement 형식으로 만들어 S3에 올렸고, 그것을 `cdm.measurement`에
append했다.

| 항목 | 값 |
|---|---|
| 원본 | `s3://<bucket>/Datasets/MIMIC-IV_CDM/Extension/measurement_lvef_ADD.csv` (2026-09-21 16:44 KST) |
| 서버 사본 | `/data/mi-val/datasets/micdm-extension/measurement_lvef_ADD.csv`, SHA-256 `ce2d2edf…3b40a2` |
| manifest | `/data/mi-val/manifests/measurement_lvef_ADD.json` |
| 행 / 환자 | 147,431 / 74,612 |
| concept / 단위 / type | 3027172 (LOINC 10230-1) / 8554 (%) / 32817 (EHR) |
| `measurement_source_value` | `lvef` |
| `visit_occurrence_id` | 전부 NULL (원 데이터에 hadm_id 없음) |
| 값 | 0~90, 중앙값 55, 결측 0 |

절차: staging 테이블(`like cdm.measurement`)에 COPY → 컬럼 집합·순서 대조 → `measurement_id`
충돌 검사(기존 행과 0건) → 단일 트랜잭션 INSERT → staging 삭제 → `ANALYZE cdm.measurement`.
되돌릴 때는 `measurement_concept_id = 3027172 and measurement_source_value = 'lvef'`로 삭제한다.

ECG와의 연결은 FK가 아니라 `person_id` + 날짜 차이다. ±7일 내 LVEF가 있는 ECG는 241,450건,
±30일은 329,557건이다. 시간창과 최근접 선택 규칙은 study.yaml의 label 정의로 둔다.

확인이 남은 것:

- (person, datetime) 중복 4쌍 (값이 55/60으로 다른 1쌍 포함)
- 10% 이하 값 527건 (0.0이 18건). 0.0을 실측으로 볼지 label 정의에서 정한다
- 협업자에게 원 소스(MIMIC-IV-ECHO 구조화 측정 변수명, 구간값 처리)를 확인한다

## 인덱스

retrieve의 lateral join이 `cdm.measurement`를 person_id로 훑는다. 이 테이블에는 전혀
인덱스가 없었고(`cdm.measurement`, `cdm.image_occurrence` 둘 다), LVEF concept만 세는
단순 count도 2분을 넘겼다(전체 스캔).

| 항목 | 값 |
|---|---|
| 이름 | `measurement_lvef_person_date_idx` |
| 정의 | `CREATE INDEX measurement_lvef_person_date_idx ON cdm.measurement USING btree (person_id, measurement_date) WHERE (measurement_concept_id = 3027172)` |
| 날짜 | 2026-09-21 |
| 빌드 시간 | 6분 45초 (15:06:35Z 시작, 15:13:20Z `CREATE INDEX` 완료, `pg_stat_progress_create_index`로 진행률 확인, scanning table 단계가 대부분) |
| 크기 | 4552 kB |
| 대상 테이블 크기 | `cdm.measurement` 54 GB (인덱스 전혀 없음) |

`explain`으로 retrieve SQL(schema=cdm, label_concept_id=3027172, window_days=7,
modality_concept_id=4145308)을 확인했다. lateral 서브쿼리가 이 인덱스를 쓴다:

```
->  Index Scan using measurement_lvef_person_date_idx on measurement m_1
      Index Cond: (person_id = i.person_id)
      Filter: ((value_as_number IS NOT NULL) AND (abs((measurement_date - i.image_occurrence_date)) <= 7))
```

바깥 `image_occurrence`는 `modality_concept_id`로 거르는 조건에 인덱스가 없어 Seq Scan이지만,
한 번만 도는 스캔이라 비용이 lateral 쪽보다 훨씬 작다(cost 53741 대 801570 전체 플랜).
`image_occurrence`에 인덱스가 필요해지면 별도로 판단한다.

## 남은 것

- [ ] 적재 완료 후 `synchronous_commit = on` 복구
- [x] `cdm.measurement`에 `measurement_lvef_person_date_idx` 생성 (2026-09-21, 위 "인덱스" 참고).
      `cdm.image_occurrence`는 아직 인덱스가 없다
- [ ] `ANALYZE`
- [ ] 실패한 5개 테이블 재적재: `note_nlp`, `vocabulary`, `specimen`,
      `fact_relationship`, `observation` (전부 스키마 수정 후 대상)
- [ ] `micdm-etl`의 확장 DDL과 S3판 불일치를 규리T와 확인
- [ ] `modality_concept_id`가 S3판 `4145308`, `micdm-etl` 리포트 `1028987`로
      다른 이유 확인
