#!/bin/bash
# 十二指肠（duodenum）器官独立测试脚本

GPUS=0
DATA_PATH="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/LDAM-DRW-master/data_tsy_827_doctor_test_split/十二指肠_1query"
CKPT_PATH="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/LDAM-DRW-master/checkpoint/十二指肠_medical_resnet50_LDAM_DRW_ep200_bs8_lr0001/ckpt.best.pth.tar"

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/LDAM-DRW-master
python test_eval.py --data-path "${DATA_PATH}" --arch resnet50 --gpu ${GPUS} --ckpt-path ${CKPT_PATH}
