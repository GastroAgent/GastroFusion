#!/bin/bash
#SBATCH -J fsl_mllm_v2_uniform_external9
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/slurm_out/fsl_mllm_v2_uniform_external9-%j.out

set -euo pipefail
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

PY="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/fsl_mllm_fusion_tolast_rae_v2_external.py"
INPUT_ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/mllm_code/mllm_result_external9_lot_way"
RAE_FSL_RESULT_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/result_831/mllm_fsl_subset_agent_lot_way_91"
OUTPUT_ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/mllm_fsl_lot_way_subset_91/v2_fusion_traj_tlast_rae_latent"

# Stomach: gap_guard_fusion
STOMACH_ALPHA="0.25"
STOMACH_GUARD="0.01"
STOMACH_GAP_THR="0.05"
# Esophagus: high_conf_override
ESOPHAGUS_TOP1_THR="0.20"
ESOPHAGUS_FSL_THR="0.10"
ESOPHAGUS_GAP_THR="0.01"
# Duodenum: very_high_conf_override
DUODENUM_TOP1_THR="0.80"
DUODENUM_FSL_THR="0.03"

[[ -f "${PY}" ]] || { echo "[ERROR] Missing Python script: ${PY}"; exit 1; }
for dir in "${INPUT_ROOT}" "${RAE_FSL_RESULT_ROOT}"; do
    [[ -d "${dir}" ]] || { echo "[ERROR] Missing input directory: ${dir}"; exit 1; }
done
mkdir -p "${OUTPUT_ROOT}"

echo "[EXTERNAL9 / UNIFORM RAE LATENT-TRAJECTORY / MLLM MAIN ONLY]"
echo "PY=${PY}"
echo "INPUT_ROOT=${INPUT_ROOT}"
echo "RAE_FSL_RESULT_ROOT=${RAE_FSL_RESULT_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

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
    --verbose_details \
    --overwrite

SUMMARY_JSON="${OUTPUT_ROOT}/batch_fusion_summary.json"
SUMMARY_CSV="${OUTPUT_ROOT}/batch_fusion_summary.csv"
FAILURE_JSON="${OUTPUT_ROOT}/batch_fusion_failures.json"
[[ -s "${SUMMARY_JSON}" ]] || { echo "[ERROR] Missing summary JSON: ${SUMMARY_JSON}"; exit 1; }
[[ -s "${SUMMARY_CSV}" ]] || { echo "[ERROR] Missing summary CSV: ${SUMMARY_CSV}"; exit 1; }
echo "[DONE] SUMMARY JSON=${SUMMARY_JSON}"
echo "[DONE] SUMMARY CSV=${SUMMARY_CSV}"
echo "[DONE] FAILURES=${FAILURE_JSON}"
