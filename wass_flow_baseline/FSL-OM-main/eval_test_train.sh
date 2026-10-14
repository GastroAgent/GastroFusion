#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/FSL-OM-main


ORGANS=(

"食管_8way_1query"

"胃_7way_1query"

"十二指肠_7way_1query"

)


for ORGAN in "${ORGANS[@]}"

do


echo "=========================================="

echo "Running FSL-OM TEST-TRAIN: ${ORGAN}"

echo "=========================================="


# 自动设置 n_way
if [[ "${ORGAN}" == "食管_8way_1query" ]]; then

    WAY=8

else

    WAY=7

fi


echo "Using ${WAY}-way"


python /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/FSL-OM-main/eval_test_test.py \
    --dataset medical \
    --medical_subset ${ORGAN} \
    --eval_mode test-train \
    --n_way ${WAY} \
    --k_shot 5 \
    --k_query 1 \
    --val_episodes 100 \
    --train_meta_save_path \
    ./results/fslom_huaxi/test-train/${ORGAN} \
    2>&1 | tee \
    ./logs/fslom_huaxi_test-train_${ORGAN}.log


done


echo "All test-train finished."