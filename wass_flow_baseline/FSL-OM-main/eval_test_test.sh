#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/FSL-OM-main


ORGANS=(
"食管_7way_1query"
"胃_7way_1query"
"十二指肠_7way_1query"
)


for ORGAN in "${ORGANS[@]}"
do

echo "=========================================="
echo "Running FSL-OM TEST-TEST: ${ORGAN}"
echo "=========================================="


python /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/FSL-OM-main/eval_test_test.py \
    --dataset medical \
    --medical_subset ${ORGAN} \
    --eval_mode test-test \
    --val_set test \
    --n_way 7 \
    --k_shot 5 \
    --k_query 1 \
    --val_episodes 100 \
    --train_meta_save_path \
    /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/FSL-OM-main/results_tsy/test-test/${ORGAN} \
    2>&1 | tee \
    ./logs/fslom_huaxi_test-test_${ORGAN}.log


done


echo "All test-test finished."