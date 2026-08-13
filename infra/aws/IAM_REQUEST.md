# DICOM-MIVA IAM request

## 필요한 조치

EC2 instance `<instance-id>`에 기존 instance profile
`s3readonly-role`을 연결한다.

```bash
aws ec2 associate-iam-instance-profile \
  --region ap-northeast-2 \
  --instance-id <instance-id> \
  --iam-instance-profile Name=s3readonly-role
```

현재 작업 계정 `arn:aws:iam::<aws-account>:user/<iam-user>`에는 role을
EC2에 넘기기 위한 `iam:PassRole` 권한이 없어 연결이 거부된다. 장기 access
key를 서버에 복사하는 방식은 사용하지 않는다.

## 최소 S3 read 범위

- `s3://<bucket>/Datasets/MIMIC-IV_CDM/`
- `s3://<bucket>/Datasets/MIMIC-IV_CDM/Extension/`
- `s3://<bucket>/Datasets/ECG/open_datasets/MIMIC/mimic-iv-ecg-dcm/`
- `s3://<bucket>/Users/<user>/STEMI_JKL/BestModelSaved/`

연결 후 서버에서 다음을 확인한다.

```bash
aws sts get-caller-identity
aws s3 ls s3://<bucket>/Datasets/MIMIC-IV_CDM/Extension/
```
