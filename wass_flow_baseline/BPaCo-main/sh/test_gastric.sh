#!/bin/bash
# BPaCo Testing Script for Gastric (Stomach) Dataset
# Usage: bash test_gastric.sh
# NOTE: Run train_gastric.sh first to generate the model checkpoint!

DATA_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/BPaCo-main/data/十二指肠_1query"
EXP_NAME="十二指肠_bpaco_1gpu"
EXP_DIR="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/BPaCo-main/experiments/gastric/${EXP_NAME}"
CHECKPOINT="${EXP_DIR}/moco_ckpt.best.pth.tar"

echo "=========================================="
echo "BPaCo Testing for Gastric Dataset"
echo "Data root: $DATA_ROOT"
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Please run train_gastric.sh first!"
    exit 1
fi

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/BPaCo-main

python bpaco_isic.py \
    --dataset gastric \
    --arch resnet50 \
    --data "$DATA_ROOT" \
    --root_path "./experiments" \
    --num_classes 7 \
    --mark "$EXP_NAME" \
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
    --resume "$CHECKPOINT"
