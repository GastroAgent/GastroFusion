#!/bin/bash
#SBATCH -J fusion_v2_sensitivity
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/slurm_out/fusion_v2_sensitivity-%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/slurm_out/fusion_v2_sensitivity-%j.err

set -eo pipefail

# =============================================================================
# Runtime
# =============================================================================
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

export LD_LIBRARY_PATH="/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}"
export PATH="/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}"

mkdir -p \
  /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/slurm_out

# =============================================================================
# Sensitivity driver
# =============================================================================
SWEEP_PY="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/fsl_mllm_fusion_tolast_rae_v2_sweep.py"
SWEEP_ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/sensitivity_v2_rae_latent_tsy_external_91_new_reference"

echo "===================================================================================================="
echo "[V2 FUSION HYPERPARAMETER SENSITIVITY]"
echo "PYTHON     = $(which python)"
echo "SWEEP_PY   = ${SWEEP_PY}"
echo "SWEEP_ROOT = ${SWEEP_ROOT}"
echo
echo "Protocol   = one-at-a-time (OAT)"
echo "Settings   = few_way + lot_way"
echo "Space      = latent / traj"
echo "===================================================================================================="

python -u "${SWEEP_PY}" \
    --settings few_way lot_way \
    --sweep_root "${SWEEP_ROOT}"

echo
echo "===================================================================================================="
echo "[DONE]"
echo "Main summary:"
echo "  ${SWEEP_ROOT}/sensitivity_macro.csv"
echo
echo "Per-dataset:"
echo "  ${SWEEP_ROOT}/sensitivity_by_dataset.csv"
echo
echo "Run status:"
echo "  ${SWEEP_ROOT}/sensitivity_runs.csv"
echo "===================================================================================================="
