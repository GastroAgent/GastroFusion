#!/bin/bash
# BPaCo Training Script for Gastric (Stomach) Dataset
# Usage: bash train_gastric.sh

DATA_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/BPaCo-main/data/十二指肠_1query"
EXP_NAME="十二指肠_bpaco_1gpu"

echo "=========================================="
echo "BPaCo Training for Gastric Dataset"
echo "Data root: $DATA_ROOT"
echo "=========================================="

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/BPaCo-main

python bpaco_isic.py \
    --dataset gastric \
    --arch resnet50 \
    --data "$DATA_ROOT" \
    --root_path "./experiments" \
    --num_classes 7 \
    --mark "$EXP_NAME" \
    --world-size 1 \
    --gpu 0 \
    --alpha 0.05 \
    --beta 1.0 \
    --gamma 1.0 \
    --lr 0.1 \
    --batch-size 1 \
    --epochs 100 \
    --moco-k 64 \
    --moco-t 0.2 \
    --moco-m 0.999 \
    --aug randcls_sim \
    --randaug_m 10 \
    --randaug_n 2 \
    --schedule 80 120 \
    --warmup_epochs 5 \
    --wd 5e-4 \
    --momentum 0.9 \
    --workers 4 \
    --seed 1314 \
    --print-freq 5
