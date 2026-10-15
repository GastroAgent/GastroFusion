#!/bin/bash
#SBATCH -J rae_m2_truebatch_gcls
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m2_direct/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m2_direct/%x_%j.err

# =============================================================================
# Environment
# =============================================================================
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_ROOT=${PROJECT_ROOT}/fsl_multi_vae/rae_flow_metric_v1
PY=${CODE_ROOT}/train_m2_direct_episode_visreg_globalcls_truebatch_v2_valsingle.py

M1_CHECKPOINT=${PROJECT_ROOT}/weights_metric_v1/m2_flow_aug_m0b2/metric_direct_episode.pt
OUTPUT_DIR=${PROJECT_ROOT}/weights_metric_v1/m2_visreg_main

mkdir -p "${PROJECT_ROOT}/fsl_multi_vae/logs_m2_direct"
mkdir -p "${OUTPUT_DIR}"
cd "${CODE_ROOT}"

# =============================================================================
# Data sources -- protocol unchanged
# =============================================================================
TRAIN_JSON_GLOB="${PROJECT_ROOT}/data/data_train_pool_imagenet_med"
VAL_EPISODE_MANIFEST="${PROJECT_ROOT}/fsl_multi_vae/data/total_eval.json"

# =============================================================================
# TRUE episode mini-batch
# =============================================================================
# Important:
#   EPISODE_BATCH_SIZE=4 means one GPU really processes 4 independent episodes
#   in ONE forward/backward/optimizer.step().
#
#   It is NOT:
#     - 64-way when N_WAY=16
#     - gradient accumulation
#
# Each episode remains independently 16-way 1-shot 1-query:
#   E1 query only matches E1 supports
#   E2 query only matches E2 supports
#   ...
#
# With 8-GPU DDP and EPISODE_BATCH_SIZE=4:
#   per-GPU true batch       = 4 episodes
#   global effective batch   = 8 * 4 = 32 episodes/update
#
# For 16-way 1-shot 1-query, each GPU sees:
#   4 * (16 support + 1 query) = 68 images / optimization batch.
N_WAY=${N_WAY:-64}
N_SHOT=${N_SHOT:-1}
N_QUERY=${N_QUERY:-1}
EPISODE_BATCH_SIZE=${EPISODE_BATCH_SIZE:-8}

# Number of episodes, NOT number of optimizer updates.
# At EPISODE_BATCH_SIZE=4, 10000 episodes ~= 2500 optimizer updates / rank / epoch.
EPISODES_PER_EPOCH=${EPISODES_PER_EPOCH:-2000}
VAL_EPISODES=${VAL_EPISODES:-1200}

# Frozen RAE is chunked only for memory engineering.  This does not change the
# true episode batch semantics.  64 is close to the 68 images in the default batch.
RAE_CHUNK_SIZE=${RAE_CHUNK_SIZE:-64}

# =============================================================================
# VISReg
# =============================================================================
VISREG_WEIGHT=${VISREG_WEIGHT:-1e-2}
VISREG_NUM_PROJ=${VISREG_NUM_PROJ:-512}
VISREG_PROJECTION_CHUNK=${VISREG_PROJECTION_CHUNK:-128}

# =============================================================================
# Per-image GLOBAL classification auxiliary task
# =============================================================================
# EVERY support and query image gets one global class target.
# For default 4 x (16-way 1-shot + 1 query): 68 image-level CLS targets / GPU / batch.
GLOBAL_CLS_WEIGHT=${GLOBAL_CLS_WEIGHT:-0.10}
GLOBAL_CLS_LR=${GLOBAL_CLS_LR:-1e-3}
GLOBAL_CLS_LABEL_SMOOTHING=${GLOBAL_CLS_LABEL_SMOOTHING:-0.0}

# =============================================================================
# Other training settings
# =============================================================================
EPOCHS=${EPOCHS:-50}
SEED=${SEED:-42}
MASTER_PORT=${MASTER_PORT:-29565}


echo "================================================================================================"
echo "[RAE M2 TRUE EPISODE BATCH + VISREG + PER-IMAGE GLOBAL CLS]"
echo "PY                       = ${PY}"
echo "M1_CHECKPOINT            = ${M1_CHECKPOINT}"
echo "OUTPUT_DIR               = ${OUTPUT_DIR}"
echo "TRAIN_JSON_GLOB          = ${TRAIN_JSON_GLOB}"
echo "VAL_MANIFEST             = ${VAL_EPISODE_MANIFEST}"
echo "N_WAY / SHOT / QUERY     = ${N_WAY} / ${N_SHOT} / ${N_QUERY}"
echo "EPISODE_BATCH_SIZE       = ${EPISODE_BATCH_SIZE}  [TRUE batch, not grad accumulation]"
echo "EPISODES_PER_EPOCH       = ${EPISODES_PER_EPOCH}"
echo "VAL_EPISODES             = ${VAL_EPISODES}"
echo "RAE_CHUNK_SIZE           = ${RAE_CHUNK_SIZE}"
echo "GLOBAL_CLS_WEIGHT        = ${GLOBAL_CLS_WEIGHT}"
echo "GLOBAL_CLS_LR            = ${GLOBAL_CLS_LR}"
echo "VISREG_WEIGHT            = ${VISREG_WEIGHT}"
echo "CUDA_VISIBLE_DEVICES     = ${CUDA_VISIBLE_DEVICES:-<set by Slurm>}"
echo "================================================================================================"

nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader || true

# =============================================================================
# Train
# =============================================================================
# Total objective:
#   L_total = L_episode_CE + VISREG_WEIGHT * L_VISReg
#                           + GLOBAL_CLS_WEIGHT * L_global_cls
#
# TRUE batch path on each GPU:
#   B independent episodes
#      -> concatenate all support/query images
#      -> batched frozen RAE encode
#      -> ONE MetricEncoder call for all images
#      -> batched Dynamic-SK/Sinkhorn with leading episode dimension B
#      -> independent N-way soft-min classification inside each episode
#      -> ONE backward + ONE optimizer.step
# =============================================================================
torchrun \
    --nproc_per_node=8 \
    --master_port="${MASTER_PORT}" \
    "${PY}" \
    --train_json_glob "${TRAIN_JSON_GLOB}" \
    --val_episode_manifest "${VAL_EPISODE_MANIFEST}" \
    --project_root "${PROJECT_ROOT}" \
    --init_checkpoint "${M1_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_way "${N_WAY}" \
    --n_shot "${N_SHOT}" \
    --n_query "${N_QUERY}" \
    --episode_batch_size "${EPISODE_BATCH_SIZE}" \
    --episodes_per_epoch "${EPISODES_PER_EPOCH}" \
    --val_episodes "${VAL_EPISODES}" \
    --epochs "${EPOCHS}" \
    --image_size 256 \
    --rae_chunk_size "${RAE_CHUNK_SIZE}" \
    --metric_dim 256 \
    --dropout 0.0 \
    --ot_pool_factor 1 \
    --sinkhorn_reg 0.1 \
    --sinkhorn_iter 30 \
    --sk_weight_eps 1e-3 \
    --shot_tau_init 0.10 \
    --cls_tau_init 0.50 \
    --metric_lr 1e-5 \
    --temperature_lr 1e-3 \
    --weight_decay 1e-4 \
    --grad_clip 5.0 \
    --freeze_metric_epochs 0 \
    --visreg_weight "${VISREG_WEIGHT}" \
    --visreg_num_proj "${VISREG_NUM_PROJ}" \
    --visreg_projection_chunk "${VISREG_PROJECTION_CHUNK}" \
    --global_cls_weight "${GLOBAL_CLS_WEIGHT}" \
    --global_cls_lr "${GLOBAL_CLS_LR}" \
    --global_cls_label_smoothing "${GLOBAL_CLS_LABEL_SMOOTHING}" \
    --seed "${SEED}" \
    --amp \
    --log_every 25
