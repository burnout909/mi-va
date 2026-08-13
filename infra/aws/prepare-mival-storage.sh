#!/usr/bin/env bash
set -euo pipefail

data_device="/dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_vol0e1d7f18e761a6d0a"

wait_for_device() {
  local device_path="$1"
  local attempt_no
  for attempt_no in $(seq 1 30); do
    [[ -b "$device_path" ]] && return 0
    sleep 1
  done
  echo "Device did not appear: $device_path" >&2
  return 1
}

find_instance_store() {
  local model_path
  for model_path in /sys/block/nvme*/device/model; do
    [[ -f "$model_path" ]] || continue
    if grep -q "Amazon EC2 NVMe Instance Storage" "$model_path"; then
      printf "/dev/%s\n" "$(basename "$(dirname "$(dirname "$model_path")")")"
      return 0
    fi
  done
  return 1
}

ensure_ext4() {
  local device_path="$1"
  local filesystem_label="$2"
  if ! blkid -s TYPE -o value "$device_path" | grep -qx ext4; then
    mkfs.ext4 -F -L "$filesystem_label" "$device_path"
  fi
}

ensure_mount() {
  local device_path="$1"
  local mount_path="$2"
  mkdir -p "$mount_path"
  if ! mountpoint -q "$mount_path"; then
    mount -o defaults,noatime "$device_path" "$mount_path"
  fi
}

wait_for_device "$data_device"
ensure_ext4 "$data_device" "MI-VAL-DATA"
ensure_mount "$data_device" /data

scratch_device="$(find_instance_store)"
ensure_ext4 "$scratch_device" "MI-VAL-SCRATCH"
ensure_mount "$scratch_device" /scratch

mkdir -p \
  /data/mi-val/cache/huggingface \
  /data/mi-val/cache/torch \
  /data/mi-val/cache/uv \
  /data/mi-val/envs \
  /data/mi-val/manifests \
  /data/mi-val/models/ecgfounder \
  /data/mi-val/models/prophecg-stemi \
  /data/mi-val/reports \
  /data/mi-val/workspaces \
  /scratch/mi-val/dicom \
  /scratch/mi-val/preprocessed \
  /scratch/mi-val/tmp

chown -R ubuntu:ubuntu /data/mi-val /scratch/mi-val
chmod 0755 /data /scratch
