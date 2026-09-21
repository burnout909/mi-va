# 모델 문서

- [ECGFounder](ecgfounder.md): foundation, STEMI·LVEF 공용
- [PROPHECG-STEMI](prophecg-stemi.md): single-task STEMI
- [xECG](xecg.md): foundation, LVEF 과제용 (2026-09-21 반입)
- [HeartWise DeepECG-SL LVEF](heartwise-lvef.md): single-task LVEF ≤40 / <50 / 회귀 (2026-09-21 반입)
- [EchoNext-Mini](echonext-mini.md): single-task 12-label, `lvef_lte_45` (2026-09-21 반입)

두 모델은 입력 tensor와 전처리가 다르므로 동일한 preprocess recipe로 묶지
않는다. ECGFounder는 12-lead representation model이고, PROPHECG-STEMI는
8-lead binary classifier ensemble이다.

서버용 dry-run 설정은 [`../../configs/aws/`](../../configs/aws/)에 분리했다.
