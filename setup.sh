#!/usr/bin/env bash
#
# Create the conda env for sparse-steer-retrieval and install dependencies.
#
# GPU builds of torch/torchaudio/torchcodec come from the PyTorch CUDA index,
# so run this INSIDE a gpushell (a GPU node) the first time, so the install picks
# the CUDA wheels and can be smoke-tested against a real device.
#
#   bash setup.sh                    # env name "steer", python 3.11
#   ENV_NAME=steerable bash setup.sh # override the env name
#   PY_VERSION=3.12 bash setup.sh    # override the python version
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-steer}"
PY_VERSION="${PY_VERSION:-3.11}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# --- locate conda -----------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "!! conda not found on PATH. Load/activate your conda first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=================================================================="
echo " sparse-steer-retrieval setup"
echo "   env name    : ${ENV_NAME}"
echo "   python      : ${PY_VERSION}"
echo "   repo        : ${REPO_DIR}"
echo "=================================================================="

# --- create env (idempotent) ------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo ">> env '${ENV_NAME}' already exists — reusing it."
else
  echo ">> creating env '${ENV_NAME}' (python ${PY_VERSION}) ..."
  conda create -n "${ENV_NAME}" "python=${PY_VERSION}" -y
fi

conda activate "${ENV_NAME}"

# --- deps via uv ------------------------------------------------------------
echo ">> installing uv ..."
python -m pip install --upgrade pip
python -m pip install uv

echo ">> uv pip install -r requirements.txt ..."
uv pip install -r "${REPO_DIR}/requirements.txt"

echo "=================================================================="
echo " Done. Activate with:  conda activate ${ENV_NAME}"
echo " Smoke test:"
echo "   python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
echo "=================================================================="
