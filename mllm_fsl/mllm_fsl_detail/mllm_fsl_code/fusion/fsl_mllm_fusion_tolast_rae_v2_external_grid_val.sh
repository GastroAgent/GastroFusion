#!/bin/bash
#SBATCH -J fsl_mllm_fusion_v2_rae_latent_nocache
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/slurm_out/fsl_mllm_fusion_v2_rae_latent_nocache-%j.out

set -eo pipefail

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

PY="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/fsl_mllm_fusion_tolast_rae_v2_external_decay.py"

# 这里只需要指定 MLLM 结果根目录。
INPUT_ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/mllm_code/mllm_result_external9_few_way"


RAE_FSL_RESULT_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/重复性实验/rae-fsl-result-decay-seed/external_few_way"

OUTPUT_ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/mllm_fsl_few_way_subset_91_grid/v2_fusion_traj_tlast_rae_latent"

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

fi
if [[ ! -d "${RAE_FSL_RESULT_ROOT}" ]]; then
    echo "[ERROR] RAE_FSL_RESULT_ROOT does not exist: ${RAE_FSL_RESULT_ROOT}"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
mkdir -p /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/slurm_out

echo "===================================================================================================="
echo "[FSL + MLLM FUSION V2 / NO-CACHE]"
echo "PY                  = ${PY}"
echo "INPUT_ROOT          = ${INPUT_ROOT}"
echo "RAE_FSL_RESULT_ROOT = ${RAE_FSL_RESULT_ROOT}"
echo "OUTPUT_ROOT         = ${OUTPUT_ROOT}"
echo "===================================================================================================="

python -u "${PY}" \
    "${INPUT_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
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

SUMMARY_JSON="${OUTPUT_ROOT}/batch_fusion_summary.json"
SUMMARY_CSV="${OUTPUT_ROOT}/batch_fusion_summary.csv"
FAILURE_JSON="${OUTPUT_ROOT}/batch_fusion_failures.json"

if [[ ! -s "${SUMMARY_JSON}" ]]; then
    echo "[ERROR] Missing or empty summary JSON: ${SUMMARY_JSON}"
    exit 1
fi
if [[ ! -s "${SUMMARY_CSV}" ]]; then
    echo "[ERROR] Missing or empty summary CSV: ${SUMMARY_CSV}"
    exit 1
fi

echo "===================================================================================================="
echo "[DONE]"
echo "RESULT ROOT  = ${OUTPUT_ROOT}"
echo "SUMMARY JSON = ${SUMMARY_JSON}"
echo "SUMMARY CSV  = ${SUMMARY_CSV}"
echo "FAILURES     = ${FAILURE_JSON}"
echo "===================================================================================================="