#!/bin/bash
#SBATCH -J build_sft_rl
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:2
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/logs/build_sft_rl-%j.out

# =============================================================================
# Medical VQA SFT/RL data construction
#
# This script ONLY builds data:
#   1) split source train JSON into SFT / RL at image level
#   2) use Qwen3-VL-32B (Transformers) to generate SFT responses
#   3) rule-based leakage / format filtering
#   4) Qwen3-VL-32B visual judge
#   5) SQLite resume
#   6) export sft_train.json / rl_train.json / manifests / statistics
#
# It does NOT run SFT training.
# It does NOT run GRPO training.
# =============================================================================


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT_DIR="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift"

PY="${PROJECT_DIR}/build_sft_rl_data.py"

# Replace with your full TRAIN json.
INPUT_JSON="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/data/cold_start_huaxi/rl_train.json"

# Local Qwen3-VL-32B checkpoint used as BOTH generator and judge.
TEACHER_MODEL="/mnt/inaisfs/data/home/tansy_criait/weights/Qwen3-VL-32B-Instruct"

# All split files, resume DB, accepted/rejected logs are written here.
OUTPUT_DIR="${PROJECT_DIR}/data/cold_start_huaxi2"

# Image-level, disease-stratified split.
SFT_RATIO=0.40
SEED=42

# For each SFT sample:
#   2 candidates / attempt × max 3 attempts
# but only ONE accepted candidate is finally kept.
CANDIDATES_PER_ATTEMPT=2
MAX_ATTEMPTS=3

# Visual judge threshold.
JUDGE_THRESHOLD=0.80

GEN_MAX_NEW_TOKENS=320
GEN_TEMPERATURE=0.70
GEN_TOP_P=0.90
JUDGE_MAX_NEW_TOKENS=220

# Optional modes:
#   PREPARE_ONLY=1:
#       validate source + stable IDs + split + SQLite + manifests only;
#       Qwen3-VL-32B is NOT loaded.
#
#   EXPORT_ONLY=1:
#       only re-export JSON/statistics from the existing SQLite state.
#
# Normal full generation:
#   PREPARE_ONLY=0
#   EXPORT_ONLY=0
PREPARE_ONLY=0
EXPORT_ONLY=0


# =============================================================================
# ENV
# =============================================================================

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate agentic_crop
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/bin:${PATH:-}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# sbatch allocates 4 GPUs above. Locally you may override it before execution.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"


# =============================================================================
# SANITY CHECK
# =============================================================================

if [[ ! -f "${PY}" ]]; then
    echo "[ERROR] Python script not found:"
    echo "        ${PY}"
    exit 1
fi

if [[ ! -f "${INPUT_JSON}" ]]; then
    echo "[ERROR] Input JSON not found:"
    echo "        ${INPUT_JSON}"
    exit 1
fi

if [[ "${PREPARE_ONLY}" == "0" && "${EXPORT_ONLY}" == "0" && ! -e "${TEACHER_MODEL}" ]]; then
    echo "[ERROR] Teacher model not found:"
    echo "        ${TEACHER_MODEL}"
    exit 1
fi


# =============================================================================
# CONFIG
# =============================================================================

echo "============================================================================"
echo "[MEDICAL VQA DATA CONSTRUCTION]"
echo "HOST                    = $(hostname)"
echo "PY                      = ${PY}"
echo "INPUT_JSON              = ${INPUT_JSON}"
echo "TEACHER_MODEL           = ${TEACHER_MODEL}"
echo "OUTPUT_DIR              = ${OUTPUT_DIR}"
echo "SFT_RATIO               = ${SFT_RATIO}"
echo "SEED                    = ${SEED}"
echo "CANDIDATES/ATTEMPT      = ${CANDIDATES_PER_ATTEMPT}"
echo "MAX_ATTEMPTS            = ${MAX_ATTEMPTS}"
echo "JUDGE_THRESHOLD         = ${JUDGE_THRESHOLD}"
echo "CUDA_VISIBLE_DEVICES    = ${CUDA_VISIBLE_DEVICES}"
echo "PREPARE_ONLY            = ${PREPARE_ONLY}"
echo "EXPORT_ONLY             = ${EXPORT_ONLY}"
echo "============================================================================"


# =============================================================================
# BUILD ARGUMENTS
# =============================================================================

ARGS=(
    --input_json "${INPUT_JSON}"
    --output_dir "${OUTPUT_DIR}"
    --teacher_model "${TEACHER_MODEL}"
    --sft_ratio "${SFT_RATIO}"
    --seed "${SEED}"
    --candidates_per_attempt "${CANDIDATES_PER_ATTEMPT}"
    --max_attempts "${MAX_ATTEMPTS}"
    --judge_threshold "${JUDGE_THRESHOLD}"
    --gen_max_new_tokens "${GEN_MAX_NEW_TOKENS}"
    --gen_temperature "${GEN_TEMPERATURE}"
    --gen_top_p "${GEN_TOP_P}"
    --judge_max_new_tokens "${JUDGE_MAX_NEW_TOKENS}"
)

if [[ "${PREPARE_ONLY}" == "1" ]]; then
    ARGS+=(--prepare_only)
fi

if [[ "${EXPORT_ONLY}" == "1" ]]; then
    ARGS+=(--export_only)
fi


# =============================================================================
# RUN
# =============================================================================

python "${PY}" "${ARGS[@]}"


echo
echo "============================================================================"
echo "[DONE]"
echo "Resume DB      : ${OUTPUT_DIR}/sft_build_state.sqlite3"
echo "SFT data       : ${OUTPUT_DIR}/sft_train.json"
echo "RL data        : ${OUTPUT_DIR}/rl_train.json"
echo "Split manifest : ${OUTPUT_DIR}/split_manifest.json"
echo "Statistics     : ${OUTPUT_DIR}/statistics.json"
echo "Accepted log   : ${OUTPUT_DIR}/accepted.jsonl"
echo "Rejected log   : ${OUTPUT_DIR}/rejected.jsonl"
echo "============================================================================"
