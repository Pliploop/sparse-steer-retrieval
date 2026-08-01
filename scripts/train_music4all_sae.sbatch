#!/usr/bin/env bash
#SBATCH -J ssr_m4a_sae
#SBATCH -n 1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-cpu=7500M
#SBATCH -t 24:00:00
#
# Train one BatchTopK SAE on pre-extracted MuQ-MuLan music4all audio embeddings.
# Sparsity is set by TOP_K (paper sweep: 5 10 20 50 100). Override any knob via env vars.
#
# Partition-specific resources (-p / -A / --gres / --constraint) are supplied by the
# launcher (scripts/launch_music4all_sweep.sh, PROFILE=sae|andrena), which overrides
# these #SBATCH defaults. To submit this file directly, pass them yourself, e.g.:
#
#   TOP_K=10 sbatch -p sae -A pilot_sae_gpu --gres=gpu:1 --constraint="hopper|ampere" \
#     scripts/train_music4all_sae.sbatch

set -euo pipefail

# --- knobs (all overridable via env) ---------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-}"
TOP_K="${TOP_K:-10}"
DICT_SIZE="${DICT_SIZE:-4096}"
MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRECISION="${PRECISION:-16-mixed}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ -n "${CONDA_ENV_BIN}" ]]; then
  export PATH="${CONDA_ENV_BIN}:${PATH}"
fi
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE

# CUDA (match the cluster module; adjust if unavailable on the node)
module load cuda/12.6.2-gcc-12.2.0 2>/dev/null || true
if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

cd "${REPO_DIR}"

echo "=================================================================="
echo " SSR music4all BatchTopK SAE"
echo "   host        : $(hostname)"
echo "   env         : ${CONDA_ENV_BIN}"
echo "   TOP_K (L0)  : ${TOP_K}"
echo "   dict_size   : ${DICT_SIZE}"
echo "   max_steps   : ${MAX_STEPS}"
echo "   batch_size  : ${BATCH_SIZE}"
echo "   gpu         : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=================================================================="
nvidia-smi -L || true

python steerable_retrieval/train.py \
  experiment=train_music4all_muq_batchtopk \
  model.sae_encoder.top_k="${TOP_K}" \
  model.sae_encoder.dict_size="${DICT_SIZE}" \
  model.sae_decoder.dict_size="${DICT_SIZE}" \
  data.batch_size="${BATCH_SIZE}" \
  data.num_workers="${NUM_WORKERS}" \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.devices=1 \
  trainer.strategy=auto \
  trainer.precision="${PRECISION}" \
  tags="[music4all,muq,batchtopk,L0-${TOP_K}]"

echo "Done (TOP_K=${TOP_K})."
