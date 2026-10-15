#!/bin/bash

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master/simple-cnaps-src

CUDA_VISIBLE_DEVICES=0 python run_medical_cnaps_tsy.py \
    --eval_mode test-test \
    --subset "十二指肠_7way_1query" \
    --way 7 \
    --shot 5 \
    --query 1 \
    --n_runs 100 \
    --data_path "../../records" \
    --test_model_path "../model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt"