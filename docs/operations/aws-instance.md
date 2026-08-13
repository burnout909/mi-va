# AWS DICOM-MIVA instance

2026-08-10 기준 실제 구성과 검증 상태다.

## Instance

| 항목 | 값 |
|---|---|
| instance | `<instance-id>` |
| type | `g6.4xlarge` |
| region / AZ | `ap-northeast-2` / `ap-northeast-2d` |
| OS | Ubuntu 26.04, kernel `7.0.0-1010-aws` |
| CPU / RAM | 16 vCPU / 60 GiB |
| GPU | NVIDIA L4 23,034 MiB |
| NVIDIA driver | `595.71.05` |

## Storage

| mount | 구성 | 용도 |
|---|---|---|
| `/` | 100 GiB gp3, 96 GiB filesystem | OS only |
| `/data` | 400 GiB gp3, 393 GiB filesystem | env, models, manifests, reports |
| `/scratch` | 559 GB instance store, 549 GiB filesystem | DICOM/cache/preprocessed temp |

`mi-val-storage.service`가 부팅 시 `/data`와 `/scratch`를 준비한다. Instance
store는 stop 후 사라질 수 있으므로 재생성 가능한 파일만 `/scratch`에 둔다.
MIMIC ECG DICOM 전체 prefix는 약 91.7 GiB라 현재 `/scratch` usable 521 GiB에
preprocess 중간 산출물과 함께 수용 가능하다.

## 설치 상태

- MI-VAL source: `/opt/mi-val`, commit `62baa1f891b5313f40ef109ce720412fdd1eecbd`
- main runtime: `/data/mi-val/envs/mival`, Python 3.12.13
- ECGFounder source/runtime: `/opt/ecgfounder`, `/data/mi-val/envs/ecgfounder`
- PROPHECG runtime: `/data/mi-val/envs/prophecg`
- AWS CLI, PostgreSQL client, Git LFS, build tools, pydicom, WFDB 설치
- XFCE/TigerVNC local-only service 활성화
- OHDSI MIMIC DDL: `/opt/ohdsi-mimic/etl/ddl`, commit `d209ed37bcc533a69923dfd413fb15d53a6dad58`
- MI-CDM extension DDL: `/data/mi-val/schemas/mi-cdm-extension`

MI-VAL test는 25개 전부 통과했고 standard config dry-run도 완료했다. 두 모델의
독립 smoke test 역시 통과했다.

## 접속

SSH는 workspace에서 `./connect.sh`를 실행한다. GUI는 `./connect-vnc.sh`로
SSH tunnel을 연 상태에서 RealVNC Viewer의 `localhost:5901`로 접속한다.
VNC port는 외부에 열려 있지 않다.

현재 security group은 VNC 5901을 열지 않았지만 SSH 22/2022와 HTTP(S)를
`0.0.0.0/0`에 허용한다. 접속 IP가 고정되면 22/2022 source를 기관 또는 VPN
CIDR로 제한하는 별도 보안 변경이 권장된다.

## 아직 외부 입력이 필요한 것

1. 기존 `s3readonly-role`을 instance에 연결할 `iam:PassRole` 권한
2. MI-CDM PostgreSQL endpoint, database/schema, read-only credential
3. ATLAS에서 export한 index/outcome cohort JSON과 observation window 확정값
4. PROPHECG amplitude scaling/padding 변경 이력
5. ECGFounder STEMI downstream head 및 data split protocol

IAM 요청 내용은 [`../../infra/aws/IAM_REQUEST.md`](../../infra/aws/IAM_REQUEST.md)에 있다.
