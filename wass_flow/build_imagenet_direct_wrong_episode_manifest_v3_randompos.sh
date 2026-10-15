#!/bin/bash
#SBATCH -J f1_hard_wrong_manifest
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/build_hard_wrong_manifest_%j.out

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}
# ============================================================================
# Paths: modify these to your environment
# ============================================================================
ROOT="/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow"
CODE_DIR="${ROOT}/fsl_multi_vae/rae_flow_metric_v1"

PY="${CODE_DIR}/build_imagenet_direct_wrong_episode_manifest_v3_randompos.py"
V10C_SCRIPT="${CODE_DIR}/pretrain_f1_flow_trajectory_pairwise_corrector_v10c_classjson_torchrun_ddp_batch.py"
M2_CHECKPOINT="${ROOT}/weights_metric_v1/new_m0b/metric_imagenet_ot.pt"

# Use a seed-specific filename so several independently submitted jobs do not overwrite.
SEED=${SEED:-1234555}
TRAIN_CLASS_JSON_ROOT="${ROOT}/data/data_train_pool_huaxi"
OUT_DIR="${ROOT}/data/imagenet_direct_wrong_manifest"

# TRAIN_CLASS_JSON_ROOT="${ROOT}/data/F1_data_Val_as_Train/train"
# OUT_DIR="${ROOT}/data/F1_data_Val_as_Train"

OUTPUT_MANIFEST="${OUT_DIR}/train-${SEED}.json"
OUTPUT_STATS="${OUT_DIR}/train-${SEED}_stats.json"

# ============================================================================
# Only the core mining controls are exposed here.
# ============================================================================
N_WAY=8
N_SHOT=5
NUM_HARD_EPISODES=1000

# First screen more classes, then retain the N_WAY-1 most confusing negatives.
CANDIDATE_WAY=8

# For each class, screen several concrete 5-shot realizations.
SUPPORT_REPEATS=4

DEVICE="cuda:0"

mkdir -p "${OUT_DIR}" "${ROOT}/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility"

for p in "${PY}" "${V10C_SCRIPT}" "${M2_CHECKPOINT}" "${TRAIN_CLASS_JSON_ROOT}"; do
    [[ -e "${p}" ]] || { echo "[ERROR] missing: ${p}" >&2; exit 2; }
done

cat <<EOF
====================================================================================================
[BUILD IMAGENET DIRECT-WRONG HARD EPISODE MANIFEST v2]
Seed                  = ${SEED}
Target hard episodes  = ${NUM_HARD_EPISODES}
Episode               = ${N_WAY}-way ${N_SHOT}-shot 1-query
Candidate screen       = ${CANDIDATE_WAY}-way
Support realizations   = ${SUPPORT_REPEATS} per class
Hard definition        = Direct prediction WRONG only
Distance definition    = exact M2 Dynamic-SK + Sinkhorn + K-shot soft-min
Output manifest        = ${OUTPUT_MANIFEST}
Output stats           = ${OUTPUT_STATS}
====================================================================================================
EOF

CUDA_VISIBLE_DEVICES=5 python "${PY}" \
    --v10c_script "${V10C_SCRIPT}" \
    --train_class_json_root "${TRAIN_CLASS_JSON_ROOT}" \
    --project_root "${ROOT}" \
    --m2_checkpoint "${M2_CHECKPOINT}" \
    --output_manifest "${OUTPUT_MANIFEST}" \
    --output_stats "${OUTPUT_STATS}" \
    --n_way "${N_WAY}" \
    --n_shot "${N_SHOT}" \
    --num_hard_episodes "${NUM_HARD_EPISODES}" \
    --candidate_way "${CANDIDATE_WAY}" \
    --support_repeats "${SUPPORT_REPEATS}" \
    --device "${DEVICE}" \
    --seed "${SEED}"
