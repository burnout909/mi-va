#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/mi-val.sh

echo "root_device=$(findmnt -n -o SOURCE /)"
echo "root_size=$(findmnt -n -o SIZE /)"
echo "data_mount=$(findmnt -n -o SOURCE,SIZE /data)"
echo "scratch_mount=$(findmnt -n -o SOURCE,SIZE /scratch)"
echo "storage_service=$(systemctl is-active mi-val-storage.service)"
echo "vnc_service=$(systemctl is-active mi-val-vnc.service)"
echo "vnc_listener=$(ss -H -ltn 'sport = :5901' | awk '{print $4}' | paste -sd, -)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

/data/mi-val/envs/mival/bin/python -c \
  'import torch; print("mival_cuda", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
/data/mi-val/envs/ecgfounder/bin/python -c \
  'import torch; print("ecgfounder_cuda", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
CUDA_VISIBLE_DEVICES="" /data/mi-val/envs/prophecg/bin/python -c \
  'import tensorflow as tf; print("prophecg_tensorflow", tf.__version__)'

if aws sts get-caller-identity >/dev/null 2>&1; then
  echo "instance_profile=available"
else
  echo "instance_profile=missing"
fi
