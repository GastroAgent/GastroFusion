#!/bin/bash
#SBATCH -J fsl_mllm_v2_rae_latent_huaxi
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/slurm_out/fsl_mllm_v2_rae_latent_huaxi-%j.out

set -eo pipefail

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

PY="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/fsl_mllm_fusion_tolast_rae_v2_huaxi_decay.py"

# New MLLM batch-runner output: main JSON only, no *_fsl_cache.json required.
INPUT_ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/mllm_code/mllm_result_huaxi"

# Offline FULL-SET RAE FSL results.
RAE_FSL_RESULT_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/重复性实验/rae-fsl-result-decay-seed/huaxi"

OUTPUT_ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/result_huaxi_831_grid/v2_fusion_traj_tlast_rae_latent_tsy"

FSL_MODE="traj"
SPACE="latent"

# ============================================================
# Grid-search-selected optimal fusion parameters
#
# Selection objective:
#   maximize macro fusion accuracy under zero-degradation
#   constraint (degradation = 0).
#
# Grid-search result:
#   macro_fusion_acc = 81.3748%
#   macro_fsl_acc    = 79.9859%
#   rescue           = 1
#   degradation      = 0
#   net_rescue       = +1
# ============================================================

STOMACH_ALPHA="0.25"
STOMACH_GUARD="0.01"
STOMACH_GAP_THR="0.05"

ESOPHAGUS_TOP1_THR="0.20"
ESOPHAGUS_FSL_THR="0.10"
ESOPHAGUS_GAP_THR="0.01"

DUODENUM_TOP1_THR="0.80"
DUODENUM_FSL_THR="0.03"



EXTRA_ARGS="--verbose_details --overwrite"

if [[ ! -f "${PY}" ]]; then
    echo "[ERROR] Python script does not exist: ${PY}"
    exit 1
fi
if [[ ! -d "${INPUT_ROOT}" ]]; then
    echo "[ERROR] INPUT_ROOT does not exist: ${INPUT_ROOT}"
    exit 1
fi
if [[ ! -d "${RAE_FSL_RESULT_ROOT}" ]]; then
    echo "[ERROR] RAE_FSL_RESULT_ROOT does not exist: ${RAE_FSL_RESULT_ROOT}"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
mkdir -p /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/slurm_out

echo "===================================================================================================="
echo "[HUAXI / V2 FUSION / NEW RAE LATENT-TRAJ / MLLM MAIN-ONLY]"
echo "PY                  = ${PY}"
echo "INPUT_ROOT          = ${INPUT_ROOT}"
echo "RAE_FSL_RESULT_ROOT = ${RAE_FSL_RESULT_ROOT}"
echo "OUTPUT_ROOT         = ${OUTPUT_ROOT}"
echo "FSL                 = offline FULL-SET RAE"
echo "MLLM                = main JSON only (no cache required)"
echo "===================================================================================================="

python -u "${PY}" \
    "${INPUT_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --fsl_mode "${FSL_MODE}" \
    --space "${SPACE}" \
    --rae_fsl_result_root "${RAE_FSL_RESULT_ROOT}" \
    --stomach_alpha "${STOMACH_ALPHA}" \
    --stomach_guard "${STOMACH_GUARD}" \
    --stomach_gap_thr "${STOMACH_GAP_THR}" \
    --esophagus_top1_thr "${ESOPHAGUS_TOP1_THR}" \
    --esophagus_fsl_thr "${ESOPHAGUS_FSL_THR}" \
    --esophagus_gap_thr "${ESOPHAGUS_GAP_THR}" \
    --duodenum_top1_thr "${DUODENUM_TOP1_THR}" \
    --duodenum_fsl_thr "${DUODENUM_FSL_THR}" \
    ${EXTRA_ARGS}

SUMMARY_JSON="${OUTPUT_ROOT}/batch_fusion_v2_summary.json"
SUMMARY_CSV="${OUTPUT_ROOT}/batch_fusion_v2_summary.csv"
FAILURE_JSON="${OUTPUT_ROOT}/batch_fusion_v2_failures.json"

[[ -s "${SUMMARY_JSON}" ]] || { echo "[ERROR] Missing ${SUMMARY_JSON}"; exit 1; }
[[ -s "${SUMMARY_CSV}" ]] || { echo "[ERROR] Missing ${SUMMARY_CSV}"; exit 1; }

echo "===================================================================================================="
echo "[DONE]"
echo "SUMMARY JSON = ${SUMMARY_JSON}"
echo "SUMMARY CSV  = ${SUMMARY_CSV}"
echo "FAILURES     = ${FAILURE_JSON}"
echo "===================================================================================================="
