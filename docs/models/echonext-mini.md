# EchoNext-Mini (single-task, multi-label)

Columbia University의 ECG → 구조적 심질환 12-label 모델이다. 첫 번째 label이
`lvef_lte_45`다. 학습 데이터는 Columbia 단독이며 MIMIC은 쓰이지 않았다.

## 고정한 자원

- 논문: doi 10.1056/AIdbp2500516 (NEJM AI 2026), 원 EchoNext doi 10.1038/s41586-025-09227-0
- 라이선스: **저장소에 LICENSE 파일이 없다.** 게재 전 저자 확인이 필요하다.
- 코드: `PierreElias/IntroECG` `7-EchoNext Minimodel/` @ `15233e9392e9dc10c136a5624abf10199207bce9`
  → 서버 `/opt/introecg` (sparse checkout)
- 서버 weights: `/data/mi-val/models/echonext-mini/` (`SHA256SUMS` 동봉)

| 파일 | SHA-256 | 역할 |
|---|---|---|
| `weights.pt` | `aac57aad…e21cd6` | `{"model": state_dict, "validation_evaluator": …}` |
| `waveform_normalization_params.json` | `f3fc58aa…c2b1c` | lead별 clip 상·하한, mean, std |
| `tabular_transformer.joblib` | `495386a8…74b59` | sklearn StandardScaler + median impute (scikit-learn 1.1.3) |

## 실행 계약

- 아키텍처: `cradlenet.models.resnet1d_tabular.ResNet1dWithTabular(len_tabular_feature_vector=7,
  num_classes=12, filter_size=16)`, 파라미터 약 1.05M
- waveform 입력: `(batch, 1, 2500, 12)`, **250 Hz**, 10 s. `(batch, 12, 2500)`은 거부된다.
- waveform 전처리: baseline wander 제거(`parse_xml.py`) → lead별 clip → lead별 `(x-mean)/std`
  (`waveform_normalization_params.json`의 값)
- **tabular 입력 7개**: `sex(male=1)`, `age_at_ecg`, `ventricular_rate`, `atrial_rate`(결측 0),
  `pr_interval`(결측 0), `qrs_duration`, `qt_corrected`. 6개 실수 특성은 joblib 파이프라인으로 표준화한다.
- 출력: `(batch, 12)` logit, label별 sigmoid. index 0이 `lvef_lte_45`.
- 저자 제외 규칙: 18세 미만, poor quality flag, ventricular pacing, 성별·연령 결측, 계측치 전부 결측

## 검증 결과

2026-09-21, `/data/mi-val/envs/ecgfounder` (torch 2.13.0+cu130)에서 strict load 성공.
`(1, 1, 2500, 12)` + tabular `(1, 7)` forward → `(1, 12)`.

## 프레임워크 관점

- LVEF 임계값이 45%라 HeartWise(40%, 50%)와 다르다. 결과 표에 label 정의를 함께 적어야 한다.
- tabular 입력 축이 현재 프레임워크에 없다. `docs/decisions/adaptations.md` L2 보고 대상이다.
- tabular 값은 MIMIC-IV-ECG `machine_measurements.csv`(RR, PR, QRS, QT, QTc)와 person 테이블의
  연령·성별로 구성할 수 있다. `atrial_rate`의 대응은 확인이 필요하다.
