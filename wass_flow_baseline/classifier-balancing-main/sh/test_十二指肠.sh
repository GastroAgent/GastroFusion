#!/bin/bash
# Test 十二指肠数据 ResNet50 baseline

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/classifier-balancing-main

python main_tsy.py \
    --cfg ./config/十二指肠_1query/baseline.yaml \
    --test \
    --model_dir ./logs_tsy/十二指肠_1query/resnet50_baseline
