# ECGFounder

## 고정한 자원

- Hugging Face: `PKUDigitalHealth/ECGFounder`
- model revision: `d9b1793951b2342f5f7e84f1ac03cd37f8a08724`
- artifact: `12_lead_ECGFounder.pth`
- artifact SHA-256: `ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997`
- official code commit: `68d25f25e323a4a423b9d9e8ea2e0af3f234bf22`
- server code: `/opt/ecgfounder`
- server weights: `/data/mi-val/models/ecgfounder/12_lead_ECGFounder.pth`
- runtime: Python 3.10.20, PyTorch 2.13.0+cu130

## 실행 계약

- input shape: `(batch, 12, 5000)`
- sampling: 500 Hz, 10 seconds
- lead order: `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`
- official downstream template의 normalization: 12×5000 array 전체에 대한 global z-score
- backbone feature width: 1024
- checkpoint output: 150 pretraining logits

이 checkpoint는 STEMI 확률을 직접 출력하지 않는다. STEMI label에 대한
linear probe 또는 fine-tuning head, train/validation/test split, freeze 범위를
별도로 고정해야 한다.

## 검증 결과

2026-08-10에 NVIDIA L4에서 state dictionary를 strict mode로 load했다.
zero tensor 한 건의 CUDA forward 결과는 feature `(1, 1024)`, logits
`(1, 150)`이며 모두 finite였다. 체크포인트는 임의 pickle 실행을 허용하지
않고 필요한 NumPy scalar type만 allow-list하여 읽었다.

Machine-readable record: [`../../infra/aws/manifests/ecgfounder.json`](../../infra/aws/manifests/ecgfounder.json)
