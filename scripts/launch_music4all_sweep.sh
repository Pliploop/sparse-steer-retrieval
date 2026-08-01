#!/usr/bin/env bash
#
# Submit the BatchTopK SAE sparsity sweep on music4all (paper L0 grid).
# Each L0 is a separate single-GPU job. PROFILE selects the cluster partition.
#
#   scripts/launch_music4all_sweep.sh                      # sae partition, sweep 5 10 20 50 100
#   TOP_KS="10" scripts/launch_music4all_sweep.sh          # just the main model (L0=10)
#   PROFILE=andrena scripts/launch_music4all_sweep.sh      # andrena partition (A100)
#
# Partition-specific resources are passed on the sbatch command line, which
# overrides the #SBATCH defaults baked into train_music4all_sae.sbatch. Mirrors
# the reference launch_{sae,andrena}_instruction.sh pattern.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PROFILE="${PROFILE:-sae}"
TOP_KS="${TOP_KS:-5 10 20 50 100}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/slurm}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

case "${PROFILE}" in
  sae)
    SBATCH_RES=(-p sae -A pilot_sae_gpu --gres=gpu:1 --constraint="hopper|ampere")
    ;;
  andrena)
    # andrena is all-A100 (40GB). One GPU is plenty for a 512->4096 SAE.
    SBATCH_RES=(-p andrena -A pilot_andrena --gres=gpu:nvidia_a100-pcie-40gb:1)
    ;;
  *)
    echo "Unknown PROFILE=${PROFILE}; expected 'sae' or 'andrena'." >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

for TOP_K in ${TOP_KS}; do
  JOB_NAME="ssr_m4a_${PROFILE}_L0-${TOP_K}"
  echo "Submitting ${JOB_NAME} ..."
  sbatch \
    -J "${JOB_NAME}" \
    "${SBATCH_RES[@]}" \
    --cpus-per-gpu=12 \
    --mem-per-cpu=7500M \
    -t "${TIME_LIMIT}" \
    -o "${LOG_DIR}/%x_%j.out" \
    -e "${LOG_DIR}/%x_%j.err" \
    --export=ALL,PROFILE="${PROFILE}",TOP_K="${TOP_K}",REPO_DIR="${REPO_DIR}" \
    scripts/train_music4all_sae.sbatch
done

echo "Submitted ${PROFILE} L0 sweep: ${TOP_KS}"
echo "Watch: squeue -u \$USER   |   logs in ${LOG_DIR}"
