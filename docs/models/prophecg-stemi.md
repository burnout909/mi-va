# PROPHECG-STEMI

## 고정한 자원

- S3: `s3://<bucket>/Users/<user>/STEMI_JKL/BestModelSaved/`
- artifact: Keras `.h5` 5개
- server path: `/data/mi-val/models/prophecg-stemi`
- runtime: Python 3.9.25, TensorFlow CPU 2.7.4, Keras 2.7.0
- 논문 DOI: <https://doi.org/10.1016/j.annemergmed.2024.06.004>

파일별 checksum은 [resource 문서](../resources/README.md)와
[`prophecg-stemi.json`](../../infra/aws/manifests/prophecg-stemi.json)에 고정했다.

## 실행 계약

- member input: `(batch, 5000, 8)`, channels-last
- lead order: `I, II, V1, V2, V3, V4, V5, V6`
- member output: 2-class softmax, STEMI positive index 1
- ensemble: 다섯 member softmax의 산술 평균
- archived PTB-XL revision notebook의 threshold: positive probability `0.0768`

8-lead 순서는 당시 dataset 생성 notebook에서 `III`, `aVR`, `aVL`, `aVF`를
제외한 순서로 확인했다. 당시 training dataset은 5000 sample 앞에 zero 120개를
붙여 5120을 사용했지만, 현재 공유된 다섯 H5의 embedded model signature는
5000 sample이다. 따라서 공유 weights에 대해서는 H5 signature인 5000을
우선하며, amplitude unit/scaling과 padding 변경 이력은 규리T에게 확인해야 한다.

## 검증 결과

2026-08-10에 다섯 member를 모두 load하고 zero tensor를 inference했다.
각 출력은 `(1, 2)`, 평균 ensemble 출력도 `(1, 2)`였고 probability sum은
1.0이었다. `0.0768`은 archived PTB-XL 분석에서 확인한 값이므로 MI-CDM
cohort의 operating point로 자동 재사용하지 않는다.
