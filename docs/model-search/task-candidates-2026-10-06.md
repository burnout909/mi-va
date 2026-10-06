# task별 후보 모델 (2026-10-06)

regression, survival, segmentation 각각 weight 공개 모델을 찾아 5개를 추천했다. AF처럼 사전 고정
프로토콜로 한 검색은 아니고, [스펙](../superpowers/specs/2026-10-06-task-draft-cards-design.md)의 카드
초안용이다. 조건: weight 공개 다운로드, 라이선스, 추론 코드, 논문(없으면 표시). MIMIC-IV-ECG 학습
모델은 목록에만 두고 쓰지 않는다. URL은 모두 HTTP 200/206으로 확인했다.

classification(LVEF)은 2026-09-21에 고른 5개(HeartWise ≤40, HeartWise <50, EchoNext-Mini,
ECGFounder, xECG)를 그대로 쓴다.

## Regression

| # | model_id | 대상 | 논문 | weights | 라이선스 | 입력 | 출력 | 학습 데이터 | 주의 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `lima-ecg-age` | 나이(년) | Nat Commun 2021, 10.1038/s41467-021-25351-7 | Zenodo 4892365 `model.zip` | MIT / CC-BY-4.0 | 12×4096, 400 Hz, 정규화 없음 | (N,1) 년 | CODE | 입력 단위가 README 안에서 모순 (1e-4 V vs mV). mV×10 추정 |
| 2 | `kardionet-k-12lead` | 칼륨 mEq/L | JACC Clin EP 2024, 10.1016/j.jacep.2024.07.023 | github ecg-net/hyperkalemia `model_12_lead.pt` | MIT | 12×5000, 500 Hz | (N,1) | Cedars-Sinai | 12-lead 전처리 미문서. 투석 fine-tune 여부 불명 |
| 3 | `ai-ntprobnp` | log NT-proBNP | Clin Chem Lab Med 2023, 10.1515/cclm-2023-0743 | Google Drive `weights.zip` (5-fold) | **없음** | 12-lead, 250 Hz, 2048 샘플, 필터 + lead별 z-score | log 값, 5 fold 평균 | Hamburg City Health Study | log 밑 미확인 |
| 4 | `vonbachmann-k` | 칼륨 (+크레아티닌) | Sci Rep 2024, 10.1038/s41598-024-65223-w | Zenodo 7456316 `regression_models/potassium/model_{0-4}` | MIT / CC-BY | **8-lead**, 400 Hz, 4096 | z-score된 값 | 스웨덴 응급실 | 역변환 평균·SD 미배포. 논문 Table 1 값(K⁺ 3.99 ± 0.50)으로 대체 |
| 5 | `singstad-ecg-age` | 나이(년) | NLDL 2022 / medRxiv 10.1101/2022.10.03.22280640 | github Bsingstad/ECG-age `.h5` | **없음** | (1000,12), 100 Hz, 원시 디지털 값 | 년 | PhysioNet 2021 | int32 캐스팅 |

- 예비: Bracke demographic-SSM age (성별 입력 필요, 출력 ×100), PKU ECG-age (ECGFounder fine-tune, UKB, 논문·라이선스 없음)
- MIMIC 학습(사용 안 함): ExECG 칼륨, QTcNet
- MIMIC에 정답이 없어 제외: 대동맥 직경, 좌심방 부피, LV mass
- 참고: PROPHECG-Age-Single(dr-you-group)은 GAN 합성 single-lead로 학습

## Survival

| # | model_id | endpoint | 논문 | weights | 라이선스 | 입력 | 출력 | 학습 데이터 | 주의 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `ml4h-ecg2af` | 신규 AF + 전체 사망 head | Circulation 2022, 10.1161/CIRCULATIONAHA.121.057480 | ml4h `ecg2af_quintuplet_v2024_01_13.keras` | GPL-3.0 | (B,5000,12), 500 Hz, ECG 전체 z-score | head 5개. 생존 head는 Dense(50) 중 앞 25개가 구간 조건부 생존확률 (AF 5년, 사망 10년) | MGH | keras 3. 사망 head는 별도 논문 없음 |
| 2 | `cavalab-deepsurv-code15` | 전체 사망 | BioData Mining 2025, arXiv 2406.17002 | Zenodo 16877773 `Code15 ResNet DeepSurv no Dem.pt` | GPL-3.0 / CC-BY-4.0 | 12×4096, 400 Hz, 가운데 7 s + padding, 저장된 lead별 z-score | log-risk 1개 | CODE-15% | baseline hazard 없음 |
| 3 | `cavalab-mtlr-code15` | 전체 사망 | 위와 같음 | `Code15 ResNet MTLR w Dem.pt` | 위와 같음 | 위와 같음 + **나이·성별** | MTLR logit 100개, 0~7.677년 | CODE-15% | 공변량 스케일 미확인 |
| 4 | `ml4h-ecg2hf` | 신규 HF (2정의) + 사망 | Circ Heart Fail 2025, 10.1161/CIRCHEARTFAILURE.125.013927 | ml4h `ecg_5000_hf_quintuplet_dropout_v2023_04_17.keras` | Broad Academic (비상업) | ECG2AF와 같음 | 생존 head 3개 (10년) + 나이·성별 | MGH | 라이선스가 출력에도 적용 |
| 5 | `ml4h-ecg2stroke` | 신규 뇌졸중 + 사망 | JACC 2026, PMID 42126358 | ml4h `ecg2stroke_dropout_2024_10_04_10_49_43.h5` | GPL-3.0 | 500 Hz, (B,5000,12) | 생존 head 2개 + 3개 | MGH | 정규화·bin 폭이 README와 코드에서 다름. 발표 점수는 공개 안 된 Cox 계수 필요. 외부 검증에 BIDMC 포함 |

- 5개 중 3개가 같은 그룹(Broad, MGH)이다. 독립된 survival 공개 모델은 이 정도가 전부다.
- 예비: AIECG-HF (MIT, 논문 없음, endpoint HFrEF라 echo LVEF 필요, 39구간), ECGSurvNet (라이선스 없음, MXNet, 데모 weight)
- MIMIC 학습: cavalab MIMICIV DeepHit 2개
- 제외: SEER (재배포 금지, 이진), AIRE (weight 없음, BIDMC 학습)

## Segmentation

| # | model_id | 논문 | weights | 라이선스 | 입력 | 출력 | 학습 데이터 | 주의 |
|---|---|---|---|---|---|---|---|---|
| 1 | `openecg-codec-v6` | **없음** | PyPI `openecg` 0.11.0 `codec_v6.pt` / `.onnx` | Apache-2.0 | single lead, 500 Hz, 5000, rank 정규화 | (1,5000,4) 샘플별 none/P/QRS/T + beat·rhythm head | LUDB, QTDB, ISP | ONNX batch 1 고정. 2~8 s만 채점 권장 |
| 2 | `hrnetv2-delineation` | **없음** | HF Space MedicalAI-DP/ECG_Delineation `weights.pth` | Apache-2.0 | (B,1,5000), 500 Hz, lead별 z-score ×0.1 (역산) | (B,3,5000) multi-label logit, logit 임계값 | LUDB | 정규화 미문서 |
| 3 | `semisegecg-resnet18` | CIKM 2025, arXiv 2507.18323 | Google Drive | Apache-2.0 vs README "All rights reserved" | (B,1,2500), 250 Hz, 필터 + z-score | (B,4,L) softmax | LUDB, QTDB, ISP, Zhejiang (+PTB-XL) | 라이선스 충돌 |
| 4 | `heartkit-seg-tcn` | **없음** | Ambiq S3 `model.keras` | BSD-3 | single lead, 100 Hz, 256 샘플 | 샘플별 4클래스 | LUDB + 합성 | 2.56 s 창, 10 ms 해상도 |
| 5 | `semisegecg-vit-tiny` | CIKM 2025 | Google Drive | 3과 같음 | 3과 같음 | 3과 같음 | 3과 같음 | 3과 같은 계열 |

- 독립된 모델 계열은 4개뿐이다. 5번은 SemiSegECG의 다른 구조로 채웠다.
- 논문 조건은 이 task에서 완화했다 (1, 2, 4번 논문 없음).
- MIMIC 학습 모델 없음.
- 예비: OpenECG v56c (250 Hz, 20 ms 격자), jjongjjong LUDB U-Net (논문 없음, 심박 단위 입력)
- 제외: ELM-Research R-U-Net (weight 미공개), UKB interval CNN (접근 제한), UMC Utrecht (웹 도구만), 규칙 기반(NeuroKit2, ECGdeli 등)

## 공통으로 드러난 것

- segmentation 5개가 모두 lead를 하나씩 받는다.
- regression은 출력을 원래 단위로 되돌리는 방식이 모델마다 다르다 (z-score 역변환, log, 곱셈).
- survival 출력 인코딩이 셋이다: 구간 조건부 생존확률, Cox log-risk(baseline 없음), MTLR.
- 공변량(나이·성별)을 입력으로 받는 모델이 있다 (cavalab MTLR, EchoNext는 7개).
