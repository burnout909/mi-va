#!/usr/bin/env bash
set -euo pipefail

UV_BIN=${UV_BIN:-/home/ubuntu/.local/bin/uv}
ENV_ROOT=${MIVAL_ENV_ROOT:-/data/mi-val/envs}

export UV_CACHE_DIR=${UV_CACHE_DIR:-/data/mi-val/cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/data/mi-val/python}
export UV_LINK_MODE=copy

"$UV_BIN" python install 3.10 3.9

if [[ ! -x "$ENV_ROOT/ecgfounder/bin/python" ]]; then
  "$UV_BIN" venv --python 3.10 "$ENV_ROOT/ecgfounder"
fi

"$UV_BIN" pip install --python "$ENV_ROOT/ecgfounder/bin/python" \
  numpy==2.2.6 \
  pandas==2.3.3 \
  scipy==1.15.3 \
  wfdb==4.3.1 \
  h5py==3.16.0 \
  matplotlib==3.10.9 \
  scikit-learn==1.7.2 \
  tqdm==4.70.0 \
  torch==2.13.0 \
  torchvision==0.28.0

if [[ ! -x "$ENV_ROOT/prophecg/bin/python" ]]; then
  "$UV_BIN" venv --python 3.9 "$ENV_ROOT/prophecg"
fi

"$UV_BIN" pip install --python "$ENV_ROOT/prophecg/bin/python" \
  tensorflow-cpu==2.7.4 \
  tensorflow-estimator==2.7.0 \
  keras==2.7.0 \
  h5py==3.1.0 \
  numpy==1.22.4 \
  protobuf==3.19.6 \
  wrapt==1.13.3

"$ENV_ROOT/ecgfounder/bin/python" -c \
  'import torch; print("ECGFounder:", torch.__version__, torch.cuda.is_available())'
CUDA_VISIBLE_DEVICES="" "$ENV_ROOT/prophecg/bin/python" -c \
  'import tensorflow as tf; print("PROPHECG:", tf.__version__)'
