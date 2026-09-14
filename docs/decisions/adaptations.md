# Adaptation Ledger

실제 데이터를 통과시키면서 프레임워크에 가한 수정의 기록이다.

목적은 기록 자체가 아니라 **한 건에만 맞는 코드가 쌓이는 것을 막는 것**이다.
수정 요구를 만나면 아래 사다리에서 위부터 시도하고, 위에서 해결되면 아래로
내려가지 않는다.

| 레벨 | 정체 | 조치 | 코드 |
|---|---|---|---|
| L0 | 모델·기관마다 다른 게 당연한 값 | ModelCard, study.yaml, registry, configs에 값 추가 | 0줄 |
| L1 | 프레임워크가 틀렸거나 통과시키지 말았어야 할 입력을 통과시킴 | 고치고 회귀 테스트를 반드시 붙인다 | 해당 모듈 |
| L2 | 새 축, 새 개념 | **즉시 하지 않고 보고한다.** | 구조 변경 |

세 단계 모두 human-in-the-loop으로 코드를 수정한다. 분류와 반영은 사람이
판단하고, 판단의 근거를 여기 남긴다.

## 이 문서에 들어가지 않는 것

**환경 구축은 여기 적지 않는다.** 데이터베이스를 세우고 DDL을 맞추고 스키마를
넓힌 일은 프레임워크가 부족했다는 증거가 아니다. 섞으면 "3회 누적 후 추상화"
카운트가 오염된다. 그 기록은
[`../operations/micdm-database.md`](../operations/micdm-database.md)에 있다.

## 이 규율을 지키는 방법

기본은 **코드 리뷰**다. 리뷰어가 볼 것은 하나다. 이 수정이 L0에서 끝날 수
있었는가.

리뷰를 돕는 두 가지가 있다.

- **N=2 불변식** — 모든 수정은 ECGFounder(torch · 12유도 · feature 추출)와
  PROPHECG(keras · 8유도 · 5멤버 앙상블) 양쪽에서 통과해야 한다. 한쪽만
  통과하면 그것은 정의상 quirk이므로 L0으로 내린다. 두 모델이 이미 충분히
  비대칭이라 별도 장치가 필요 없다.
- `tests/golden` — 스냅샷 회귀. 숫자가 바뀌면 커밋 메시지가 이유를 적는다.

`tests/test_no_proper_nouns.py`는 `src/mival/`의 식별자와 문자열 값에서
모델명·기관명을 찾는 기계적 검사다. 주석과 독스트링은 허용한다. 도입 여부는
아직 결정하지 않았다. 붙였을 때 `evaluate.py`의 `OUTCOME_DIAGNOSTIC`(A-2)과
`misclassify.py`의 `stemi_mimic`(A-3) 두 건을 잡았고, 후자는 절반이 오탐이었다.

---

## 기록

### 2026-08-19

#### A-1 · L0 · `local_path`가 다른 기관의 파일시스템 경로

MI-CDM `image_occurrence.local_path`의 값이
`/home/ubuntu/dryou_mount/Datasets/ECG/open_datasets/MIMIC/mimic-iv-ecg-dcm/files/p1000/...`
로 시작한다. ETL을 수행한 장비의 경로이고 DICOM-MIVA에는 없다.

기관마다 다른 것이 당연한 값이므로 L0이다. `configs/aws/*.yaml`에 이미
`retrieval.local_path_root: /scratch/mi-val/dicom`가 선언돼 있으므로, retrieve가
CDM이 준 경로에서 파일 부분을 취해 이 root 아래로 해석한다.

코드에 `dryou_mount`를 적으면 그 순간 site 축이 오염된다. 다른 기관의 MI-CDM은
또 다른 접두사를 가질 것이고, 그때 필요한 것은 코드 수정이 아니라 설정 한 줄이다.

- 조치: retrieve 구현 시 prefix 재해석. 코드 0줄 추가 없음
- 상태: retrieve 미구현이므로 **대기**
- 검증: 서로 다른 `local_path_root` 두 개로 같은 cohort가 해석되는 테스트

#### A-2 · 관찰 · `OUTCOME_DIAGNOSTIC = "stemi"`

`src/mival/stages/evaluate.py`가 `outcome` report 축의 기본값으로 이 연구의
질환명을 갖고 있다. 다른 결과를 검증하는 study도 결과 표에 `stemi`를 받는다.

중립 sentinel
(`primary`)로 바꾸거나 evaluation spec에서 읽어야 한다. 값을 바꾸면 기존 산출물의
축 값이 달라지므로 **연구 결정이 필요하다.**

- 상태: **결정 대기**

#### A-3 · 관찰 · `DEFAULT_VERDICTS`의 질환 특정 어휘

`src/mival/stages/misclassify.py`의 기본 verdict 어휘에 `stemi_mimic`이 있다.
여기서 `mimic`은 데이터셋이 아니라 "STEMI 유사 소견"이라는 임상 용어이므로
게이트 관점에서는 절반이 오탐이다.

다만 질환 특정 기본값이 core에 있는 것은 사실이다. 독스트링이 이미 study가
`misclassify.verdicts`로 자기 어휘를 선언한다고 적고 있어, 숨은 선택이 아니라
문서화된 선택이다. 지금은 그대로 둔다.

- 상태: **의도된 기본값으로 유지**

---

## L2 보고 목록

즉시 만들지 않고 여기 올린다. 한 건을 보고 지은 추상은 그 한 건에만 맞으므로,
여러 모델과 여러 기관에서 반복되는 것을 본 뒤에 만든다.

### L2-1 · 반출 경계가 1급 개념이 아니다

신촌 데이터셋은 병원 밖으로 나갈 수 없다. 데이터가 오는 것이 아니라 코드가
가고, 나오는 것은 집계물뿐이다.

`manifest.json`, exclusion ledger, 지표 표는 반출 가능한 형태다. 그런데 6단계
misclassification은 **산출물 자체가 환자 단위**다. `top_errors`, `boundary`,
review CSV는 개별 케이스이고 임상의가 보라고 만든 것이다. 병원 안에서만 볼 수
있다.

지금은 stage 산출물에 "이것이 site 밖으로 나갈 수 있는가"를 표시하는 개념이
없다. 반출 묶음을 만들 때 환자 단위 산출물이 섞이지 않도록 강제할 자리가 필요하다.

- 발견: 2026-08-19, 신촌 접근 조건 확인 중
- 상태: **보고됨. 만들지 않음**
- 착수 조건: 신촌 실행이 실제로 준비될 때. MIMIC 단독 실행에는 필요 없다
