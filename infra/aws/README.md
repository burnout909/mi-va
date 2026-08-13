# AWS DICOM-MIVA instance

## Target

- Instance type: `g6.4xlarge`
- Region/AZ: `ap-northeast-2` / `ap-northeast-2d`
- GPU: NVIDIA L4
- Root: 100 GiB gp3
- Persistent data: 400 GiB gp3 mounted at `/data`
- Ephemeral instance store: 559 GB raw / 549 GiB filesystem mounted at `/scratch`

## Storage policy

- `/data/mi-val`: environments, model weights, manifests, reports and durable workspaces.
- `/scratch/mi-val`: downloaded DICOM cache and preprocessed arrays that can be rebuilt.
- Instance store contents disappear when the EC2 instance is stopped. `mi-val-storage.service` recreates the filesystem and directories when necessary.
- Patient-level data and DICOM should not be placed on the root volume.

## Expected project paths

```text
/opt/mi-val/                         application source
/data/mi-val/envs/                   Python environments
/data/mi-val/models/                 persistent model artifacts
/data/mi-val/manifests/              checksums and dataset/run manifests
/data/mi-val/reports/                durable reports
/data/mi-val/workspaces/             durable run metadata
/scratch/mi-val/preprocessed/        rebuildable model inputs
/scratch/mi-val/dicom/               rebuildable DICOM cache
/scratch/mi-val/tmp/                 temporary files
```

## Environment

- Interactive shell defaults: `/etc/profile.d/mi-val.sh`
- Machine-readable site paths: `/etc/mi-val/site.env`
- Main Python environment: `/data/mi-val/envs/mival`
- ECGFounder environment: `/data/mi-val/envs/ecgfounder` (Python 3.10)
- PROPHECG-STEMI environment: `/data/mi-val/envs/prophecg` (Python 3.9)
- Python distributions and package caches are stored under `/data`, not the root volume.

The model environments can be reproduced with `setup-model-envs.sh`. Artifact
revisions and checksums are stored in `manifests/`.

## S3 authentication

Do not copy long-lived user access keys to the instance. Attach the existing `s3readonly-role` instance profile with read access to the four documented S3 prefixes. The current operator needs `iam:PassRole` permission for that association. See [IAM_REQUEST.md](IAM_REQUEST.md).

## Connection

The current SSH entry point is maintained by the workspace `connect.sh`.

The XFCE/TigerVNC service is enabled but listens only on `127.0.0.1:5901` and
`::1:5901`. Run `./connect-vnc.sh`, leave that terminal open, then connect
RealVNC Viewer to `localhost:5901`. The SSH key is the access control; port 5901
must not be opened in the EC2 security group.

## Verification

```bash
sudo /usr/local/sbin/verify-mi-val-instance
/data/mi-val/envs/ecgfounder/bin/python /usr/local/bin/verify-ecgfounder
/data/mi-val/envs/prophecg/bin/python /usr/local/bin/verify-prophecg
```
