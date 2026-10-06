# 회귀·생존·분할 15개 모델 실행 기록 (2026-10-07)

[스펙](../superpowers/specs/2026-10-07-three-task-vertical-slice-design.md)과
[계획](../superpowers/plans/2026-10-07-three-task-vertical-slice.md)의 실행 기록이다. 사용자 결정
(2026-10-07)으로 대표 3개를 먼저 끝까지 돌린 뒤 나머지 12개를 붙였고, 전체 코호트 실행은 15개를 모두
통합한 다음 한 번에 시작했다.

## study와 모델

| study | 라벨 | 코호트 (사람당 ECG 1건) | 모델 |
|---|---|---|---|
| `ecg-age` | ECG 연도 − 생년 | 158,566 | lima-ecg-age (torch), singstad-ecg-age (keras 2.7) |
| `potassium` | 같은 날 혈청 칼륨 (concept 3023103) | 124,167 | kardionet-k-12lead, vonbachmann-k (5-fold) |
| `ntprobnp` | ±1일 NT-proBNP, MIMIC labevents 50963 | 20,974 | ai-ntprobnp (5-fold) |
| `mortality` | 1년 사망 (사건, 일수) | 156,484 (사망 5,902) | cavalab DeepSurv, cavalab MTLR (나이·성별), ml4h ECG2AF·ECG2HF·ECG2Stroke 사망 head (keras 3) |
| `delineation` | 기계 측정 PR/QRS/QT, lead II | 152,985 | SemiSegECG ResNet-18·ViT-Tiny, HRNetV2, OpenECG codec v6, HeartKit TCN (keras 3) |

카드는 study별 registry 디렉터리(`registry/age`, `potassium`, `ntprobnp`, `survival`,
`segmentation`)에 있다. 모델별 전처리·출력 처리는 `registry/wrappers/`(서버 사본
`/data/mi-val/models/_wrappers`)에 있다.

## 샘플 결과 (1,000건, test 400건)

샘플은 동작 확인용이다. 사건 수가 적은 생존 지표는 CI가 넓다.

| study | 모델 | 지표 |
|---|---|---|
| ecg-age | Lima | MAE 9.6 y, R² 0.60, bias +3.9 |
| | Singstad | MAE 9.9 y, R² 0.58, bias +1.1 |
| potassium | Kardio-Net | MAE 0.43, R² −0.03, bias +0.18 |
| | von Bachmann | MAE 0.36, R² 0.14, bias −0.18 |
| ntprobnp | AI-NT-proBNP | Spearman 0.56; 예측 중앙값 161 vs 실제 1,312 pg/mL (일반 인구 학습 → 입원 코호트) |
| mortality (사건 13) | DeepSurv / MTLR / ECG2AF / ECG2HF / ECG2Stroke | C-index 0.89 / 0.87 / 0.89 / 0.84 / 0.83 |
| delineation | HRNetV2 | MAE PR 13, QRS 8.6, QT 12.7 ms |
| | OpenECG | QRS bias +10, QT MAE 15 ms |
| | HeartKit | QT MAE 14.6, QRS bias +12 ms |
| | SemiSeg ViT / ResNet | QRS bias +21 / +27 ms, QT bias +18 / +26 ms |

SemiSegECG의 QRS·QT 일정 bias는 학습 라벨(LUDB 등)과 장비 알고리즘의 경계 정의 차이로 본다.
버그로 보지 않았지만 확인하지 않았다.

## 데이터에서 드러난 것

- **NT-proBNP 값이 MI-CDM에 없다.** concept 3029187 행 98,206건 모두 `value_as_number`가 비어 있고
  `value_source_value`가 `___`다. MIMIC labevents의 `value`(비식별화 텍스트)를 옮기고 `valuenum`을
  옮기지 않은 ETL 문제다. 공유 DB는 고치지 않고 원본 `mimic-iv-3.1/hosp/labevents.csv`에서
  itemid 50963을 뽑아 `/data/mi-val/datasets/mimic-iv/ntprobnp_labels.csv`(95,124건, person 매핑
  후)를 만들었다 (`scripts/build_ntprobnp_labels.py`). MI-CDM 관리 쪽에 알릴 일이다.
- **measurement의 source 값은 MIMIC itemid다.** 칼륨은 50971(검사실)과 227442(ICU chartevents)가
  한 concept에 섞여 있다.
- **1년 사망의 추적 종료**는 min(ECG + 365일, 마지막 방문 종료 + 365일)이다. MI-CDM에서 사망일이
  마지막 방문 종료일 뒤인 경우는 없었다(방문 종류 무관).
- **Lima와 cavalab의 입력 단위는 mV다.** CODE README의 1e-4 V와 다르다 (원장 A-12).

## 서버에서 바꾼 것

| 무엇 | 내용 |
|---|---|
| NVIDIA 모듈 | 커널 7.0.0-1012용 `linux-modules-nvidia-595-server-open` 설치 (자동 업데이트로 커널만 올라가 GPU가 안 잡혔음). 드라이버 595.91.07 |
| DB index | `measurement_k_ntprobnp_person_date_idx` on `cdm.measurement(person_id, measurement_date) where concept in (3023103, 3029187)` |
| ecgfounder env | einops 0.8.2, opendsp 0.1.0, PyWavelets 1.8.0, xmltodict 1.0.4 추가 |
| 새 env | `/data/mi-val/envs/keras3`: Python 3.11, TensorFlow 2.19.1 + keras 3.10, GPU |
| 코드 | 작업 트리 `/data/mi-val/mi-va-vs` (branch `feat/vertical-slice`, `vs-incoming`으로 push 후 reset). LVEF 체인이 쓰는 `/data/mi-val/mi-va`는 건드리지 않음 |
| 파생 weight | cavalab 2개, OpenECG, Kardio-Net은 wrapper용 state_dict로 변환(`mival_state_dict.pt`, 카드에 원본 sha256과 함께 기록) |
| 데이터 | `/data/mi-val/datasets/mimic-iv-ecg/machine_measurements.csv`, `/data/mi-val/datasets/mimic-iv/ntprobnp_*` |

## 실행 방법

`scripts/vs_run.sh <study> [N|all]`. 카드의 `runtime.env`별로 models를 나눠 돌리고 예측을 합쳐
evaluate한다. `all`은 runs root를 `/scratch/mi-val/runs`로 쓰고(tensor 약 160 GB, `/data` 여유
부족) 끝나면 models·evaluate를 `/data/mi-val/runs/<study>/`로 복사한다.

전체 실행: 2026-10-06 19:15 UTC 시작, tmux `vsfull`, 순서 ecg-age → mortality → delineation →
potassium → ntprobnp. 결과는 아래에 채운다.

## 전체 실행 결과

test split, subgroup all, 사람 단위 bootstrap 2,000회 95% CI. 시간은 GPU를 LVEF 실행과 나눠 쓴 값이다.

### ecg-age (test 31,713; 39분)

| 모델 | MAE (y) | R² | bias |
|---|---|---|---|
| Lima | 9.90 (9.82–9.98) | 0.579 (0.571–0.587) | +3.39 |
| Singstad | 10.04 (9.86–10.25) | −0.086 (−0.465–0.230) | +1.76 |

Singstad의 R²는 0.3%(90건)의 극단 예측(100~1,144세)이 무너뜨린다. MAE와 중앙값은 정상이다.
잡음 많은 입력에 대한 출력 범위 제한 없는 모델의 반응으로 보지만, 저자 FFT 리샘플링과 이 파이프라인의
polyphase 차이일 가능성은 확인하지 않았다.

### mortality (test 31,339, 1년 사망 1,231; 당일 사망 포함 후 재실행)

| 모델 | C-index | AUROC@365 |
|---|---|---|
| cavalab MTLR (나이·성별) | 0.811 (0.799–0.821) | 0.814 (0.803–0.824) |
| ml4h ECG2AF 사망 head | 0.811 (0.800–0.820) | 0.814 (0.803–0.824) |
| ml4h ECG2HF 사망 head | 0.799 (0.788–0.809) | 0.802 (0.791–0.813) |
| ml4h ECG2Stroke 사망 head | 0.792 (0.782–0.802) | 0.796 (0.785–0.806) |
| cavalab DeepSurv | 0.785 (0.774–0.796) | 0.788 (0.777–0.800) |

### delineation (test 약 30,600; lead II, 기준 = 장비 측정, ms)

| 모델 | PR MAE (bias) | QRS MAE (bias) | QT MAE (bias) |
|---|---|---|---|
| HRNetV2 | 11.5 (−1.8) | 8.9 (−1.2) | 14.3 (−8.5) |
| OpenECG codec v6 | 12.0 (−0.5) | 15.2 (+9.7) | 15.7 (+3.9) |
| HeartKit TCN | 15.3 (−2.9) | 17.6 (+11.6) | 14.9 (+1.6) |
| SemiSegECG ViT-Tiny | 12.5 (−2.9) | 21.8 (+19.9) | 20.6 (+15.9) |
| SemiSegECG ResNet-18 | 15.9 (−8.9) | 27.5 (+26.3) | 26.6 (+23.1) |

CI 폭은 모두 ±0.3 ms 안팎이다. QRS·QT의 양의 bias는 학습 라벨의 경계 정의 차이로 본다(확인 안 함).
