#!/bin/bash

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/PT-MAP-master

python save_plk_tsy.py \
    --eval_mode both \
    --medical_subset "食管_8way_1query" \
    --model ResNet18 \
    --method S2M2_R \
    --gpu 1 \
    --force_reextract