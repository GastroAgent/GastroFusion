#!/bin/bash
#SBATCH --job-name=matching_胃_1query
#SBATCH --output=logs/matching_胃_1query_%j.out
#SBATCH --error=logs/matching_胃_1query_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

ROOT=/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/MatchingNetworks-master
cd "${ROOT}"

# Activate conda env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fsl_matching

# Pin to a single GPU (the only one this job gets anyway, but keeps logs clean)
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3

mkdir -p logs

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] >>> Matching Networks medical training starting (subset=胃_1query)"

python mainMiniImageNet.py \
    --dataset medical \
    --medical_subset 胃_1query \
    --n_way 8 --k_shot 5 --k_query 1 \
    --channels 3 \
    --batch_size 32 --fce \
    --total_epochs 10 --total_train_batches 200 \
    --total_val_batches 50 --total_test_batches 100 \
    --lr 1e-3 --save_best

echo "[$(ts)] <<< Matching Networks medical training finished"