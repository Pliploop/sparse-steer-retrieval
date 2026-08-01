#!/usr/bin/env bash
#SBATCH -J ssr_bt
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=90G
#SBATCH -t 48:00:00
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#
# Train ONE BatchTopK SAE on pre-extracted MuQ-MuLan music4all embeddings, on a
# single andrena A100. Any extra args are forwarded verbatim as Hydra overrides,
# so you set the sparsity per job on the command line:
#
#   mkdir -p logs/slurm      # once, so SLURM can write the logs
#   sbatch scripts/train_batchtopk_andrena.sbatch model.sae_encoder.top_k=5
#   sbatch scripts/train_batchtopk_andrena.sbatch model.sae_encoder.top_k=10
#   sbatch scripts/train_batchtopk_andrena.sbatch model.sae_encoder.top_k=20
#   sbatch scripts/train_batchtopk_andrena.sbatch model.sae_encoder.top_k=50
#   sbatch scripts/train_batchtopk_andrena.sbatch model.sae_encoder.top_k=100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-}"
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
echo "   host      : $(hostname)"
echo "   env       : ${CONDA_ENV_BIN}"
echo "   overrides : $*"
echo "   gpu       : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "=================================================================="
nvidia-smi -L || true

python steerable_retrieval/train.py \
  experiment=train_music4all_muq_batchtopk \
  trainer.devices=1 \
  trainer.strategy=auto \
  logger.wandb.project=steerable-retrieval \
  "$@"

echo "Done ($*)."
