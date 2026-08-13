# 모델 문서

- [ECGFounder](ecgfounder.md)
- [PROPHECG-STEMI](prophecg-stemi.md)

두 모델은 입력 tensor와 전처리가 다르므로 동일한 preprocess recipe로 묶지
않는다. ECGFounder는 12-lead representation model이고, PROPHECG-STEMI는
8-lead binary classifier ensemble이다.

서버용 dry-run 설정은 [`../../configs/aws/`](../../configs/aws/)에 분리했다.
