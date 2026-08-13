# MI-VAL

MI-CDM 기반 의료 AI validation framework 프로젝트 문서 모음이다.

현재 단계에서는 구현 명세를 엄격하게 고정하기보다, 회의에서 나온 내용과 공유받은 자원을 찾기 쉬운 폴더 구조로 정리한다.

## 문서 구조

```text
docs/
├── overview/       프로젝트 배경과 전체 방향
├── meetings/       날짜별 회의록
├── resources/      S3, repository, 논문, 모델 자원
├── pipeline/       6단계 파이프라인별 작업 문서
├── tasks/          미팅 단위 담당 업무
├── decisions/      합의사항과 추가 논의가 필요한 항목
├── models/         모델별 실행 계약과 검증 결과
└── operations/     AWS instance 구성과 접속 방법
```

## 빠른 링크

- [프로젝트 Overview](docs/overview/README.md)
- [Overview Word 원본](docs/overview/MI-VAL_overview.docx)
- [2026-08-06 회의록](docs/meetings/2026-08-06.md)
- [규리T 공유 자원](docs/resources/README.md)
- [파이프라인 문서](docs/pipeline/README.md)
- [2026-08-14 미팅 전 Task](docs/tasks/2026-08-14.md)
- [결정사항과 Open Questions](docs/decisions/README.md)
- [모델 실행 명세](docs/models/README.md)
- [AWS DICOM-MIVA 운영 상태](docs/operations/aws-instance.md)

## 현재 기준

- MI-CDM ETL이 완료된 환경을 전제로 한다.
- cohort는 ATLAS Data4Life에서 정의한다.
- 연구 파이프라인은 회의록 기준 6단계로 관리한다.
- 검증 대상은 ECGFounder와 PROPHECG-STEMI single-task model이다.
- 실제 임상 데이터나 모델 binary는 이 폴더에 복사하지 않고 위치, revision,
  checksum과 확인 결과만 문서화한다.
