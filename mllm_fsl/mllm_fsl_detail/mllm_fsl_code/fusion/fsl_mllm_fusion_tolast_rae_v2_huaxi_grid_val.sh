
#!/bin/bash
#SBATCH -J fsl_mllm_v2_uniform_huaxi
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/slurm_out/fsl_mllm_v2_uniform_huaxi-%j.out

set -euo pipefail

# =============================================================================
# Environment
# =============================================================================
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

# =============================================================================
# Python entry point
# =============================================================================
PY="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/fsl_mllm_fusion_tolast_rae_v2_huaxi.py"

# =============================================================================
# Inputs and outputs
# =============================================================================

# GastroMLLM main JSON files only.
# Ignore legacy raw_fsl_result and *_fsl_cache.json.
INPUT_ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/mllm_code/mllm_result_huaxi"

# Offline RAE FSL results.
# Fixed protocol: uniform + tolast_mean + support-min.
RAE_FSL_RESULT_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/result_831/offline_align_rae_metric_huaxi_new"

OUTPUT_ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/result_huaxi_831/v2_fusion_traj_tlast_rae_latent_tsy"

# =============================================================================
# V2 organ-specific fusion parameters
# All numerical thresholds are explicitly supplied to Python.
# =============================================================================

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

# =============================================================================
# Validation
# =============================================================================

for file in "${PY}"; do
    if [[ ! -f "${file}" ]]; then
        echo "[ERROR] Missing Python script: ${file}"
        exit 1
    fi
done

for dir in "${INPUT_ROOT}" "${RAE_FSL_RESULT_ROOT}"; do
    if [[ ! -d "${dir}" ]]; then
        echo "[ERROR] Missing input directory: ${dir}"
        exit 1
    fi
done

mkdir -p "${OUTPUT_ROOT}"

# =============================================================================
# Configuration
# =============================================================================

echo "================================================================"
echo "[HUAXI / V2 FUSION / UNIFORM RAE LATENT-TRAJECTORY]"
echo "PY                  = ${PY}"
echo "INPUT_ROOT          = ${INPUT_ROOT}"
echo "RAE_FSL_RESULT_ROOT = ${RAE_FSL_RESULT_ROOT}"
echo "OUTPUT_ROOT         = ${OUTPUT_ROOT}"
echo "FSL                 = uniform / tolast_mean / support-min"
echo "MLLM                = main JSON only"
echo "================================================================"

# =============================================================================
# Fusion inference
# =============================================================================

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

# =============================================================================
# Output validation
# =============================================================================

SUMMARY_JSON="${OUTPUT_ROOT}/batch_fusion_v2_summary.json"
SUMMARY_CSV="${OUTPUT_ROOT}/batch_fusion_v2_summary.csv"
FAILURE_JSON="${OUTPUT_ROOT}/batch_fusion_v2_failures.json"

[[ -s "${SUMMARY_JSON}" ]] || {
    echo "[ERROR] Missing summary JSON: ${SUMMARY_JSON}"
    exit 1
}

[[ -s "${SUMMARY_CSV}" ]] || {
    echo "[ERROR] Missing summary CSV: ${SUMMARY_CSV}"
    exit 1
}

echo "================================================================"
echo "[DONE] Huaxi Uniform RAE + GastroMLLM fusion"
echo "SUMMARY JSON = ${SUMMARY_JSON}"
echo "SUMMARY CSV  = ${SUMMARY_CSV}"
echo "FAILURES     = ${FAILURE_JSON}"
echo "================================================================"
