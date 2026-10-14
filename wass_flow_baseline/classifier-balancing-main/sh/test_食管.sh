#!/bin/bash
# Test 食管数据 ResNet50 baseline

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/classifier-balancing-main

python main_tsy.py \
    --cfg ./config/食管_1query/baseline.yaml \
    --test \
    --model_dir ./logs_tsy/食管_1query/resnet50_baseline_7way
