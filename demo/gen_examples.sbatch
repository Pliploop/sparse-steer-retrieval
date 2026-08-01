#!/usr/bin/env bash
#SBATCH -J ssr_gen_examples
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH -t 00:40:00
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err
#
# Generate the website's real examples.js on a GPU node (MuQ is impractical on CPU).
# Exercises the full path: load SAE + MuQ, sparse inversion, steer, retrieve.
#   mkdir -p logs/slurm && sbatch demo/gen_examples.sbatch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-}"
if [[ -n "${CONDA_ENV_BIN}" ]]; then
  export PATH="${CONDA_ENV_BIN}:${PATH}"
fi
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export SSR_DEVICE=cuda
export TOKENIZERS_PARALLELISM=false
# L0=20 checkpoint for the demo/examples (override SSR_CHECKPOINT to switch L0).
export SSR_CHECKPOINT="${SSR_CHECKPOINT:-${REPO_DIR}/logs/xps/66a0caf9/checkpoints/last.ckpt}"
export SSR_MODEL_TAG="${SSR_MODEL_TAG:-BatchTopK SAE · MuQ-MuLan · music4all · L0=20}"

module load cuda/12.6.2-gcc-12.2.0 2>/dev/null || true
cd "${REPO_DIR}/demo"

echo "host $(hostname) | gpu ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

# build the corpus if it's missing (full music4all: ~109k unique tracks, ~223MB matrix)
[ -f corpus/corpus.npz ] || python build_corpus.py --n 200000

python gen_examples.py
echo "done."
