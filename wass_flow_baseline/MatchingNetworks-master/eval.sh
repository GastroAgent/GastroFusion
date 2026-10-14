#!/usr/bin/env bash

set -e

# ============================================================
# GPU
# ============================================================

export CUDA_VISIBLE_DEVICES=1


# ============================================================
# 需要修改的配置
# ============================================================

# 主程序文件
PYTHON_FILE="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/MatchingNetworks-master/mainMiniImageNet_tsy.py"

# 数据根目录
# 目录结构应为：
# ${DATA_ROOT}/${MEDICAL_SUBSET}/train/<类别>/*.jpg
# ${DATA_ROOT}/${MEDICAL_SUBSET}/val/<类别>/*.jpg
# ${DATA_ROOT}/${MEDICAL_SUBSET}/test/<类别>/*.jpg
DATA_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/data"

# 医学数据子集
MEDICAL_SUBSET="十二指肠_7way_1query"

# 可选：
#   test-test  = Support:test，Query:test
#   test-train = Support:train，Query:test
EVAL_PROTOCOL="test-train"

# 训练好的模型权重
CHECKPOINT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/MatchingNetworks-master/checkpoints/medical/十二指肠_7way_1query/matching/best.pt"

# 输出根目录
OUTPUT_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/MatchingNetworks-master/result_tsy"

# Few-shot设置
N_WAY=7
K_SHOT=5
K_QUERY=1

# 总episode数：
# TOTAL_TEST_BATCHES × BATCH_SIZE
TOTAL_TEST_BATCHES=100
BATCH_SIZE=1

# 其他设置
CHANNELS=3
SEED=2191


# ============================================================
# 自动生成输出路径
# ============================================================

SAVE_DIR="${OUTPUT_ROOT}/${EVAL_PROTOCOL}"
LOG_DIR="${OUTPUT_ROOT}/${EVAL_PROTOCOL}/logs"

mkdir -p "${SAVE_DIR}"
mkdir -p "${LOG_DIR}"


# ============================================================
# 检查输入
# ============================================================

if [ ! -f "${PYTHON_FILE}" ]; then
    echo "[Error] Python file not found: ${PYTHON_FILE}"
    exit 1
fi

if [ ! -d "${DATA_ROOT}" ]; then
    echo "[Error] Data root not found: ${DATA_ROOT}"
    exit 1
fi

if [ ! -d "${DATA_ROOT}/${MEDICAL_SUBSET}" ]; then
    echo "[Error] Medical subset not found:"
    echo "${DATA_ROOT}/${MEDICAL_SUBSET}"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "[Error] Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

if [ "${EVAL_PROTOCOL}" != "test-test" ] && \
   [ "${EVAL_PROTOCOL}" != "test-train" ]; then
    echo "[Error] EVAL_PROTOCOL must be test-test or test-train"
    exit 1
fi


# ============================================================
# 显示本次评估配置
# ============================================================

echo "============================================================"
echo "MatchingNet evaluation"
echo "============================================================"
echo "Python file    : ${PYTHON_FILE}"
echo "Dataset        : medical"
echo "Data root      : ${DATA_ROOT}"
echo "Medical subset : ${MEDICAL_SUBSET}"
echo "Protocol       : ${EVAL_PROTOCOL}"
echo "Checkpoint     : ${CHECKPOINT}"
echo "N-way          : ${N_WAY}"
echo "K-shot         : ${K_SHOT}"
echo "K-query        : ${K_QUERY}"
echo "Episodes       : $((TOTAL_TEST_BATCHES * BATCH_SIZE))"
echo "Save directory : ${SAVE_DIR}"
echo "Log directory  : ${LOG_DIR}"
echo "GPU            : ${CUDA_VISIBLE_DEVICES}"
echo "============================================================"


# ============================================================
# 执行评估
# ============================================================

python "${PYTHON_FILE}" \
    --dataset medical \
    --dataroot "${DATA_ROOT}" \
    --medical_subset "${MEDICAL_SUBSET}" \
    --eval_protocol "${EVAL_PROTOCOL}" \
    --n_way "${N_WAY}" \
    --k_shot "${K_SHOT}" \
    --k_query "${K_QUERY}" \
    --total_test_batches "${TOTAL_TEST_BATCHES}" \
    --batch_size "${BATCH_SIZE}" \
    --channels "${CHANNELS}" \
    --seed "${SEED}" \
    --load "${CHECKPOINT}" \
    --save-dir "${SAVE_DIR}" \
    --log-dir "${LOG_DIR}" \
    --eval_only \
    --fce 


echo "============================================================"
echo "Evaluation completed"
echo "Result:"
echo "${SAVE_DIR}/${EVAL_PROTOCOL}/${MEDICAL_SUBSET}/result.json"
echo "============================================================"