#!/bin/bash

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master

python convert_medical_data.py \
    --data_root "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/data" \
    --records_root "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/records_tsy" \
    --subset "胃_7way_1query"