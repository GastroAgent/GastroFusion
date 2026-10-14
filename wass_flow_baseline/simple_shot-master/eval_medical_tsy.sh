#!/bin/bash

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple_shot-master

CUDA_VISIBLE_DEVICES=0 python eval_medical_tsy.py \
    --eval_mode test-train \
    --data_root "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/data" \
    --csv_dir "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple_shot-master/split_tsy" \
    --subset "十二指肠_7way_1query" \
    --n_ways 7 \
    --n_shot 5 \
    --n_queries 1 \
    --n_runs 100 \
    --num_classes 7 \
    --seed 12667 \
    --output_json "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple_shot-master/results_tsy/test-train/十二指肠_result.json"