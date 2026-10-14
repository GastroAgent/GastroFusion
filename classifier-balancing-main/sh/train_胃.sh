#!/bin/bash
# Train 胃部数据 ResNet50 baseline

export CUDA_VISIBLE_DEVICES=2

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/classifier-balancing-main

python main_tsy.py --cfg ./config/胃_1query/baseline.yaml
