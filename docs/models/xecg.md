# xECG (foundation)

Medical University of Innsbruck의 xLSTM 기반 ECG foundation model이다 (BenchECG 논문의
baseline). 사전학습 데이터는 CODE, Chapman/Ningbo, INCART이며 MIMIC-IV-ECG는 포함되지
않는다. LVEF 과제의 두 번째 foundation model로 쓴다.

## 고정한 자원

- 논문: arXiv 2509.10151
- 라이선스: MIT
- Hugging Face: `riccardolunelli/xECG_base_model_v1`, revision `b06053253a9b`
- 코드: `dlaskalab/bench-xecg` @ `13e57523e418d8cddced73f81b27ba541413f4f3` → 서버 `/opt/xecg`
- 서버 weights: `/data/mi-val/models/xecg/` (`model.safetensors`, `config.json`, `xECG.py`,
  `downstream_models.py`, `SHA256SUMS`)
- `model.safetensors` SHA-256: `812dec69ac0fbf13f39e50bde4f35f85435a66966d8785baec65c2b5c70e722c`
- runtime: `/data/mi-val/envs/xecg`, Python 3.12.13, torch 2.8.0+cu128, xlstm 2.0.4
- 전체 기록: 서버 `/data/mi-val/manifests/models/xecg.json`

## 실행 계약

- 생성: `xECG(config["cls_type"], config)` 후 safetensors strict load
- input shape: `(batch, 1000, 12)`, layout **`time_lead`**, **100 Hz**, 10 s. `(batch, 12, 1000)`은
  padding mask 계산에서 실패한다.
- lead order: `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`. 없는 lead는 0으로 채운다.
- patch 25 샘플. 문서화된 정규화나 필터는 없다. 값이 전부 0인 구간은 padding으로 취급된다.
- 출력: pooled representation `(batch, 1024)`와 patch representation `(batch, 40, 1024)`
- downstream head: `downstream_models.xECGClassification` (분류·회귀 공용)

## sLSTM backend

xlstm의 sLSTM은 `cuda`(bf16 커스텀 커널, JIT 컴파일)와 `vanilla`(float32 torch) 두 backend가
있고 **체크포인트의 파라미터 레이아웃이 backend마다 다르다.** 공개 체크포인트는 cuda 레이아웃이다.

2026-09-21 검증:

- cuda backend를 컴파일해 forward에 성공했다. 두 번째 호출 14 ms (L4).
- vanilla backend에 저자 저장소의 변환(`_recurrent_kernel_.permute(0, 2, 1)`만)을 적용하면
  cuda 출력과 코사인 유사도 0.32로 **다른 모델이 된다.** `_bias_`도 `(heads, gates, dim)` 순서를
  `(gates, heads, dim)`으로 재배열해야 하며, 그러면 코사인 유사도 1.00000, 평균 상대 오차 0.25%로
  bf16 커널 오차 수준에서 일치한다.
- **cuda backend를 정본으로 쓴다.** 저자 코드의 vanilla 경로는 bias 순서 버그가 있다.

컴파일 조건 (서버에 이미 갖춰 둠):

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/data/mi-val/envs/xecg/bin:/usr/local/cuda-12.8/bin:$PATH
export CC=gcc-14 CXX=g++-14 NVCC_PREPEND_FLAGS="-ccbin g++-14"
export TORCH_EXTENSIONS_DIR=/data/mi-val/cache/torch_extensions TORCH_CUDA_ARCH_LIST="8.9"
```

nvcc 12.8과 cuBLAS·cuSPARSE 헤더는 NVIDIA ubuntu2404 저장소의 `cuda-nvcc-12-8`,
`cuda-libraries-dev-12-8`로 설치했다. Ubuntu 26.04 기본 gcc 15는 nvcc 12.8이 거부한다.

## 확인할 사항

- 100 Hz 입력이므로 500 Hz DICOM을 5:1 다운샘플해야 한다. 저자가 쓴 anti-aliasing 방식은
  `/opt/xecg` 전처리 코드에서 확인한다.
- LVEF 과제 성능 선례가 없다. 이 프레임워크의 결과가 첫 보고가 된다.
