#!/bin/bash
# BPaCo training for endoscopy data.
# Only DATA_ROOT / EXP_NAME need to be changed for a different organ.

set -e

DATA_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/BPaCo-main/data_tsy_827_doctor_test_split_test_test/食管_1query"
EXP_NAME="食管_bpaco_7way"
PROJECT_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/BPaCo-main"
OUTPUT_ROOT="./experiments_tsy"

echo "=========================================="
echo "BPaCo Training for Endoscopy Dataset"
echo "Data root : $DATA_ROOT"
echo "Experiment: $EXP_NAME"
echo "Output    : ${PROJECT_ROOT}/experiments_tsy/${EXP_NAME}"
echo "=========================================="

cd "$PROJECT_ROOT"

python bpaco_isic.py \
    --arch resnet50 \
    --data "$DATA_ROOT" \
    --root_path "$OUTPUT_ROOT" \
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
