# DICOM 캐시 (DICOM-MIVA 서버)

MI-CDM `image_occurrence.local_path`가 가리키는 파일의 서버 사본이다. retrieve가
`local_path_root` 아래로 경로를 재해석한다 (ledger A-1).

| 항목 | 값 |
|---|---|
| 위치 | `/scratch/mi-val/dicom/files/p1000 .. p1999/<person>/<study>/<id>.dcm` |
| 출처 | `s3://<bucket>/Datasets/ECG/open_datasets/MIMIC/mimic-iv-ecg-dcm/files` |
| 받은 날 | 2026-09-21 12:12 ~ 13:00 UTC |
| 파일 수 | 800,035 `.dcm` (S3 prefix 전체 객체는 803,635) |
| 용량 | 99 GB |
| 방법 | 노트북에서 만든 36시간 STS 임시 토큰을 서버 셸 env에만 넣고 `aws s3 sync --only-show-errors` (`max_concurrent_requests 64`). 오류 0건. 토큰은 파일에 남기지 않았다 |

## 알아둘 것

- `/scratch`는 인스턴스 stop 후 비워진다. 다시 받으려면 같은 방법으로 약 50분이 걸린다.
- CDM `image_occurrence` 1,011,623행 중 ECG(`modality_concept_id = 4145308`)는 796,617행이다.
  나머지 215,006행은 MIMIC-CXR 흉부 X선이고 `local_path`가 `s3://.../MIMIC-CXR/...`로
  시작한다. retrieve는 modality로 걸러야 하며, 걸러지 않으면 이 행들이 `file_missing`으로
  제외 원장에 섞여 들어간다.
- 파일 수 확인: `find /scratch/mi-val/dicom/files -name '*.dcm' | wc -l`
