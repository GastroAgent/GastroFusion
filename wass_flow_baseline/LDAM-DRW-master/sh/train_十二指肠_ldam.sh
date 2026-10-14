#!/bin/bash
# 十二指肠（duodenum）器官 LDAM-DRW 训练脚本
# 使用十二指肠病理图像数据进行训练和测试

GPUS=0  # GPU id（单卡训练，修改为其他数字可切换 GPU）

DATA_PATH="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/LDAM-DRW-master/data_tsy_827_doctor_test_split/十二指肠_1query"
DATASET_NAME="十二指肠"

python medical_train.py \
    --data_path ${DATA_PATH} \
    --dataset_name ${DATASET_NAME} \
    --arch resnet50 \
    --loss_type LDAM \
    --train_rule DRW \
    --epochs 200 \
    --batch-size 8 \
    --lr 0.001 \
    --momentum 0.9 \
    --weight-decay 0.0001 \
    --workers 4 \
    --print-freq 10 \
    --pretrained \
    --gpu ${GPUS} \
    --img_size 224 \
    --drw_epoch 80 \
    --seed 42
