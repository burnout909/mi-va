# HeartWise DeepECG-SL LVEF (single-task)

Montreal Heart Institute의 supervised EfficientNetV2 계열 LVEF 모델 3종이다. MIMIC-IV-ECG는
학습에 쓰이지 않았고 논문에서 외부검증 세트로만 쓰였다.

## 고정한 자원

- 논문: doi 10.1093/eurheartj/ehaf1119 (Eur Heart J 2025)
- 라이선스: Apache-2.0
- Hugging Face (2026-09-21 revision):
  - `heartwise/EfficientNetV2_LVEF_equal_under_40` (`973b356566b0`) → `lvef_under_equal_40.pt`
  - `heartwise/EfficientNetV2_LVEF_under_50` (`f3d8d94e703c`) → `lvef_under_50.pt`
  - `heartwise/deepecg-sl_finetuned_LVEF_MHI` (`1ee6698a3759`) → `LVEF_MSE_SL.pt`
- 전처리 코드: `HeartWise-AI/DeepECG_Docker` @ `cabf6c06b74731c86b8c5a27ae7101f14db5ec38` → 서버 `/opt/deepecg-docker`
- 아키텍처 코드: `/opt/deepecg-docker/notebooks/EfficientNetv2.py` (`EfficientNet1DV2`)
- 서버 weights: `/data/mi-val/models/heartwise-lvef/` (`SHA256SUMS` 동봉)

| 파일 | SHA-256 | 형식 | task |
|---|---|---|---|
| `lvef_under_equal_40.pt` | `605cabad…8541e8` | torch.jit | LVEF ≤ 40% binary |
| `lvef_under_50.pt` | `01cf98fe…129d7d95` | torch.jit | LVEF < 50% binary |
| `LVEF_MSE_SL.pt` | `cc46c6e9…4926b7` | state_dict | LVEF % 회귀 |

전체 해시는 서버 `/data/mi-val/manifests/models/heartwise-lvef.json`에 있다.

## 실행 계약

- input shape: `(batch, 12, 2500)`, **250 Hz**, 10 s, layout `lead_time`
- lead order: `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`
- binary 모델 출력: logit `(batch, 1)`. 저자 wrapper가 sigmoid를 취한다.
- 회귀 모델 출력: LVEF % 값 `(batch, 1)`. `EfficientNet1DV2(num_classes=1,
  expansion_factors=[1,2,2,2,2,2,2])`로 만들고 `classifier.fc1.*` 키를 `classifier.3.*`로
  바꿔 strict load한다. 체크포인트 동봉 지표: MAE 7.55, R² 0.46 (MHI 검증 세트로 추정).

저자 파이프라인의 전처리는 세 단계다. 이 중 첫째와 셋째가 현재 프레임워크 계약에 없다.

1. **데이터셋 단위 스펙트럼 파워 스케일링**: 배치 전체의 평균 진폭 스펙트럼을 구해
   `PTBXL_POWER_RATIO = 3.003154`에 맞춘다 (`utils/ecg_signal_processor.py::scale_ecg_signals`).
   레코드 단위가 아니라 코호트 단위 정규화다.
2. peak 검출과 lead 정리 (`clean_and_process_ecg_leads`)
3. 모델 직전 `signal *= 1/0.0048` (`models/efficientnet_wrapper.py::mhi_factor`)

## 검증 결과

2026-09-21, `/data/mi-val/envs/ecgfounder` (torch 2.13.0+cu130)에서 확인했다.

- binary 2종: `torch.jit.load` 성공, `(1, 12, 2500)` forward → `(1, 1)`. `(1, 2500, 12)`
  등 다른 축 순서는 TorchScript 오류로 거부된다.
- 회귀: strict load 성공. 난수 입력에서 출력 약 55, 입력에 1/0.0048을 곱하면 약 35로
  LVEF % 범위의 값이 나온다.

## 확인할 사항

- `mhi_factor` 앞의 원 진폭 단위(mV인지 µV인지). XML 파서에서 아직 확정하지 못했다.
- `EfficientNetV2` binary 모델의 운영 threshold. Docker 저장소의 threshold 표에는 "Missing"이다.
- 논문 보충자료의 MIMIC-IV 외부검증 수치와 라벨 출처 (OUP 접근 차단으로 미확인).
