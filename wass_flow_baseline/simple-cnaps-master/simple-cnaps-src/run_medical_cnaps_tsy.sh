#!/bin/bash

set -euo pipefail

export META_DATASET_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master/meta-dataset"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

# ============================================================
# Simple CNAPS project
# ============================================================

PROJECT_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master"
SRC_DIR="${PROJECT_ROOT}/simple-cnaps-src"
EVAL_PY="${SRC_DIR}/run_medical_cnaps_tsy.py"

# ============================================================
# Input paths
# ============================================================

RECORDS_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/records_tsy"

TEST_MODEL_PATH="${PROJECT_ROOT}/model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt"

PRETRAINED_RESNET_PATH="${PROJECT_ROOT}/model-checkpoints/pretrained_resnets/pretrained_resnet_meta_dataset.pt.tar"

# ============================================================
# Output paths
# ============================================================

LOG_ROOT="${SRC_DIR}/checkpoints/medical_tsy"
RESULT_ROOT="${PROJECT_ROOT}/results_tsy"
MODE_RESULT_DIR="${RESULT_ROOT}/test-train"

# ============================================================
# Evaluation settings
# ============================================================

GPU_ID=0
N_SHOT=5
N_QUERY=1
N_RUNS=100
SEED=12667
FEATURE_ADAPTATION="film"

# 食管运行8-way，胃和十二指肠运行7-way
TEST_TRAIN_SUBSETS=(
    "食管_8way_1query"
    "胃_7way_1query"
    "十二指肠_7way_1query"
)

# ============================================================
# Basic input checks
# ============================================================

if [[ ! -f "${EVAL_PY}" ]]; then
    echo "[ERROR] Evaluation script not found:"
    echo "  ${EVAL_PY}"
    exit 1
fi

if [[ ! -d "${RECORDS_ROOT}" ]]; then
    echo "[ERROR] Records root not found:"
    echo "  ${RECORDS_ROOT}"
    exit 1
fi

if [[ ! -f "${TEST_MODEL_PATH}" ]]; then
    echo "[ERROR] Simple CNAPS checkpoint not found:"
    echo "  ${TEST_MODEL_PATH}"
    exit 1
fi

if [[ ! -f "${PRETRAINED_RESNET_PATH}" ]]; then
    echo "[ERROR] Pretrained ResNet checkpoint not found:"
    echo "  ${PRETRAINED_RESNET_PATH}"
    exit 1
fi

mkdir -p "${LOG_ROOT}"
mkdir -p "${MODE_RESULT_DIR}"

cd "${SRC_DIR}"

# ============================================================
# Evaluation function
# ============================================================

run_evaluation() {
    local SUBSET="$1"
    local N_WAY="$2"

    local RECORD_DIR="${RECORDS_ROOT}/${SUBSET}"
    local OUTPUT_JSON

    OUTPUT_JSON="${MODE_RESULT_DIR}/${SUBSET}_SimpleCNAPS_${N_WAY}way_${N_SHOT}shot_${N_QUERY}query_${N_RUNS}ep.json"

    if [[ ! -d "${RECORD_DIR}" ]]; then
        echo "[ERROR] Record directory not found:"
        echo "  ${RECORD_DIR}"
        exit 1
    fi

    if [[ ! -f "${RECORD_DIR}/dataset_spec.json" ]]; then
        echo "[ERROR] dataset_spec.json not found:"
        echo "  ${RECORD_DIR}/dataset_spec.json"
        exit 1
    fi

    if ! find "${RECORD_DIR}" \
        -maxdepth 1 \
        -type f \
        -name "*.tfrecords" \
        -print -quit | grep -q .; then

        echo "[ERROR] No TFRecord files found:"
        echo "  ${RECORD_DIR}"
        exit 1
    fi

    echo
    echo "============================================================"
    echo "Simple CNAPS medical evaluation"
    echo "Mode       : test-train"
    echo "Subset     : ${SUBSET}"
    echo "Setting    : ${N_WAY}-way ${N_SHOT}-shot ${N_QUERY}-query"
    echo "Episodes   : ${N_RUNS}"
    echo "Seed       : ${SEED}"
    echo "Records    : ${RECORD_DIR}"
    echo "Model      : ${TEST_MODEL_PATH}"
    echo "Output JSON: ${OUTPUT_JSON}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python "${EVAL_PY}" \
        --eval_mode "test-train" \
        --data_path "${RECORDS_ROOT}" \
        --subset "${SUBSET}" \
        --way "${N_WAY}" \
        --shot "${N_SHOT}" \
        --query "${N_QUERY}" \
        --n_runs "${N_RUNS}" \
        --seed "${SEED}" \
        --feature_adaptation "${FEATURE_ADAPTATION}" \
        --test_model_path "${TEST_MODEL_PATH}" \
        --pretrained_resnet_path "${PRETRAINED_RESNET_PATH}" \
        --checkpoint_dir "${LOG_ROOT}" \
        --output_json "${OUTPUT_JSON}"

    if [[ ! -s "${OUTPUT_JSON}" ]]; then
        echo "[ERROR] Evaluation finished but result JSON was not created:"
        echo "  ${OUTPUT_JSON}"
        exit 1
    fi

    echo "[OK] Result saved:"
    echo "  ${OUTPUT_JSON}"
}

# ============================================================
# Run test-train only
# ============================================================

for SUBSET in "${TEST_TRAIN_SUBSETS[@]}"; do

    if [[ "${SUBSET}" == "食管_8way_1query" ]]; then
        CURRENT_WAY=8
    else
        CURRENT_WAY=7
    fi

    run_evaluation \
        "${SUBSET}" \
        "${CURRENT_WAY}"
done

echo
echo "============================================================"
echo "All Simple CNAPS test-train evaluations finished."
echo "Results: ${MODE_RESULT_DIR}"
echo "Logs   : ${LOG_ROOT}"
echo "============================================================"