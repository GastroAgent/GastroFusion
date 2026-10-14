#!/bin/bash
#SBATCH -J fusion_v2_sens_huaxi
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/slurm_out/fusion_v2_sens_huaxi-%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/slurm_out/fusion_v2_sens_huaxi-%j.err

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
# Huaxi OAT sensitivity driver
# =============================================================================
SWEEP_PY="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/fsl_mllm_fusion_tolast_rae_v2_huaxi_sweep.py"

SWEEP_ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/sensitivity_v2_rae_latent_tsy_huaxi_91_new_reference"


# =============================================================================
# Sanity check
# =============================================================================
if [[ ! -f "${SWEEP_PY}" ]]; then
    echo "[ERROR] Sweep Python does not exist:"
    echo "        ${SWEEP_PY}"
    exit 1
fi

mkdir -p "${SWEEP_ROOT}"


# =============================================================================
# Run information
# =============================================================================
echo "===================================================================================================="
echo "[HUAXI / V2 FUSION / OAT HYPERPARAMETER SENSITIVITY]"
echo
echo "PYTHON     = $(which python)"
echo "SWEEP_PY   = ${SWEEP_PY}"
echo "SWEEP_ROOT = ${SWEEP_ROOT}"
echo
echo "Protocol   = one-at-a-time (OAT)"
echo "Setting    = Huaxi"
echo "Space      = latent"
echo "FSL mode   = traj"
echo
echo "Reference:"
echo "  stomach   = alpha 0.75 / guard 0.03 / gap 0.02"
echo "  esophagus = top1 0.30 / fsl 0.15 / gap 0.02"
echo "  duodenum  = top1 0.90 / fsl 0.05"
echo "===================================================================================================="


# =============================================================================
# Execute all 8 OAT sweeps
# =============================================================================
python -u "${SWEEP_PY}" \
    --sweep_root "${SWEEP_ROOT}"


# =============================================================================
# Done
# =============================================================================
echo
echo "===================================================================================================="
echo "[DONE]"
echo
echo "Macro summary:"
echo "  ${SWEEP_ROOT}/sensitivity_huaxi_macro.csv"
echo
echo "Per-dataset:"
echo "  ${SWEEP_ROOT}/sensitivity_huaxi_by_dataset.csv"
echo
echo "Run status:"
echo "  ${SWEEP_ROOT}/sensitivity_huaxi_runs.csv"
echo "===================================================================================================="