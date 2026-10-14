#!/usr/bin/env bash
# End-to-end PT-MAP pipeline over the three medical subsets.
# Each subset uses its own n_ways / n_shot / n_queries (matching the
# per-subset class counts the user specified).
set -euo pipefail

ROOT=/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/PT-MAP-master
DATA=/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/data
MODEL=ResNet18
PT_EPOCH=5
S2M2_EPOCH=30
BATCH=16

cd "${ROOT}"
mkdir -p logs/ptmap

# Per-subset meta-test config: key = "subset:n_ways:n_shot:n_queries"
declare -A CFG=(
    [胃_1query]=8:5:1
    [十二指肠_1query]=7:5:1
    [食管_1query]=8:5:1
)

for sub in 胃_1query 十二指肠_1query 食管_1query; do
    IFS=':' read -r N_WAY N_SHOT N_QUERY <<< "${CFG[$sub]}"
    OUT="checkpoints/medical/${sub}/${MODEL}_S2M2_R/eval_result.json"

    echo "================================================================"
    echo "[PT-MAP] subset=${sub}  n_way=${N_WAY}  n_shot=${N_SHOT}  n_query=${N_QUERY}"
    echo "================================================================"

    echo "--- 1) filelist ---"
    python filelists/medical/write_medical_filelist.py \
        --medical_root "${DATA}" --medical_subset "${sub}"

    echo "--- 2) rotation head (${PT_EPOCH} epochs) ---"
    python train.py \
        --dataset medical --medical_subset "${sub}" \
        --model "${MODEL}" --method rotation --train_aug \
        --stop_epoch "${PT_EPOCH}" \
        --batch_size "${BATCH}" --save_freq 1 \
        2>&1 | tee "logs/ptmap/${sub}_rotation.log"

    echo "--- 3) S2M2_R (${S2M2_EPOCH} epochs) ---"
    python train.py \
        --dataset medical --medical_subset "${sub}" \
        --model "${MODEL}" --method S2M2_R --train_aug \
        --stop_epoch "${S2M2_EPOCH}" \
        --batch_size "${BATCH}" --save_freq 5 \
        2>&1 | tee "logs/ptmap/${sub}_s2m2r.log"

    echo "--- 4) extract novel features ---"
    python save_plk.py \
        --dataset medical --medical_subset "${sub}" \
        --model "${MODEL}" --method S2M2_R

    echo "--- 5) MAP test (${N_WAY}-way ${N_SHOT}-shot ${N_QUERY}-query) ---"
    python test_standard.py \
        --dataset medical \
        --medical_subset "${sub}" \
        --dsname "${sub}" \
        --model "${MODEL}" \
        --method S2M2_R \
        --n_ways "${N_WAY}" --n_shot "${N_SHOT}" --n_queries "${N_QUERY}" \
        --n_runs 1000 \
        --lam 10 --alpha 0.2 --n_iter 20 --beta 0.5 \
        --output_json "${OUT}"

    echo "[PT-MAP] done ${sub}: ${OUT}"
done

echo "All three subsets finished."