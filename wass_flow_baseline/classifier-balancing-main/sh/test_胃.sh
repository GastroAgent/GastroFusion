#!/bin/bash
# Test 胃部数据 ResNet50 baseline

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/classifier-balancing-main

python main_tsy.py \
    --cfg ./config/胃_1query/baseline.yaml \
    --test \
    --model_dir ./logs_tsy/胃_1query/resnet50_baseline
