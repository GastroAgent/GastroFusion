#!/bin/bash
# BPaCo Testing Script for Duodenum Dataset

DATA_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/BPaCo-main/data_tsy_827_doctor_test_split/胃_1query"

# ============================================================
# 1. Model checkpoint：保持原目录
# ============================================================
MODEL_EXP_NAME="胃_bpaco"
MODEL_DIR="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/BPaCo-main/experiments_tsy_stomach/${MODEL_EXP_NAME}"
CHECKPOINT="${MODEL_DIR}/moco_ckpt.best.pth.tar"

# ============================================================
# 2. Testing output：单独指定新目录
# Python 最终会构造：
#   root_path / dataset / mark
# 所以这里最终输出：
#   /.../test_results/gastric/十二指肠_bpaco_test
# ============================================================
OUTPUT_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/tsy/BPaCo-main/test_results"
OUTPUT_MARK="胃_bpaco_test_new7way"

OUTPUT_DIR="${OUTPUT_ROOT}/gastric/${OUTPUT_MARK}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "BPaCo Testing for Duodenum Dataset"
echo "Data root      : ${DATA_ROOT}"
echo "Model dir      : ${MODEL_DIR}"
echo "Checkpoint     : ${CHECKPOINT}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "=========================================="

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT}"
    exit 1
fi

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/BPaCo-main

python bpaco_isic.py \
    --dataset gastric \
    --arch resnet50 \
    --data "${DATA_ROOT}" \
    --root_path "${OUTPUT_ROOT}" \
    --num_classes 7 \
    --mark "${OUTPUT_MARK}" \
    --alpha 0.05 \
    --beta 1.0 \
    --gamma 1.0 \
    --moco-t 0.2 \
    --moco-k 64 \
    --aug randcls_sim \
    --randaug_m 10 \
    --randaug_n 2 \
    --workers 4 \
    --seed 1314 \
    --no-distributed \
    --world-size 1 \
    --gpu 0 \
    --batch-size 1 \
    --evaluate \
    --resume "${CHECKPOINT}"