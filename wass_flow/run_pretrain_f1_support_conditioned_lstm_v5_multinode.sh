#!/bin/bash


# =============================================================================
# F1-v5 ImageNet/class-JSON pretraining -- MULTI-NODE execution script
#
# This script is executed ONCE PER NODE by the SLURM submission launcher.
# Each node then starts NPROC_PER_NODE local torchrun workers.
#
# Required topology env (normally exported by the submission script):
#   NNODES, NPROC_PER_NODE, NODE_RANK, MASTER_ADDR, MASTER_PORT
# =============================================================================

NODE_RANK=${NODE_RANK:-0}
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NET_IFACE=${NET_IFACE:-auto}

# =============================================================================
# Environment
# =============================================================================
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH:-}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Multi-node NCCL defaults.  Interface may be selected by the submission runner.
export NCCL_SOCKET_FAMILY=${NCCL_SOCKET_FAMILY:-AF_INET}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}

if [[ "${NET_IFACE}" != "auto" && -n "${NET_IFACE}" ]]; then
    export NCCL_SOCKET_IFNAME="=${NET_IFACE}"
    export GLOO_SOCKET_IFNAME="${NET_IFACE}"
fi

# =============================================================================
# Paths
# =============================================================================
ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_DIR=${ROOT}/fsl_multi_vae/rae_flow_metric_v1
PY=${CODE_DIR}/pretrain_f1_support_conditioned_lstm_v5_classjson_torchrun_ddp_batch.py

TRAIN_CLASS_JSON_ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_train_pool_imagenet_med
VAL_CLASS_JSON_ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_eval_pool_imagenet

# =============================================================================
# Frozen upstream pipeline
# =============================================================================
ORGAN=${ORGAN:-Imagenet}
M2_CHECKPOINT=${M2_CHECKPOINT:-${ROOT}/weights_metric_v1/m2_flow_aug_m2_visreg/metric_direct_episode.pt}
DIT_CHECKPOINT=${DIT_CHECKPOINT:-${ROOT}/weights_dit_flow_v1/vision_hint/otcfm/otcfm_dit_L_step_400000.pt}
TRAJECTORY_SCORE_MODE=${TRAJECTORY_SCORE_MODE:-step_to_support}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/weights_metric_v1/f1_v5_support_lstm_imagenet_pretrain_ddp_batch/${ORGAN}/${TRAJECTORY_SCORE_MODE}}

# =============================================================================
# TRUE episode batch / upstream chunks
# Keep the current single-node settings unless overridden from sbatch environment.
# IMPORTANT: EPISODE_BATCH_SIZE is PER RANK.
# =============================================================================
EPISODE_BATCH_SIZE=${EPISODE_BATCH_SIZE:-32}
PAIR_CHUNK_SIZE=${PAIR_CHUNK_SIZE:-32}
RAE_CHUNK_SIZE=${RAE_CHUNK_SIZE:-32}

# Dynamic monotonic decay
GAMMA_MIN=${GAMMA_MIN:-0.70}
GAMMA_INIT=${GAMMA_INIT:-0.95}
LAMBDA_DECAY_SMOOTH=${LAMBDA_DECAY_SMOOTH:-0.02}

# Group-relative exploration
LAMBDA_RELATIVE=${LAMBDA_RELATIVE:-1.25}
RELATIVE_GROUP_SIZE=${RELATIVE_GROUP_SIZE:-8}
RELATIVE_WARMUP_EPOCHS=${RELATIVE_WARMUP_EPOCHS:-0}
RELATIVE_SIGMA_START=${RELATIVE_SIGMA_START:-1.0}
RELATIVE_SIGMA_END=${RELATIVE_SIGMA_END:-0.20}
RELATIVE_TARGET_STEP=${RELATIVE_TARGET_STEP:-0.25}
RELATIVE_ADV_CLIP=${RELATIVE_ADV_CLIP:-3.0}
RELATIVE_MIN_REWARD_STD=${RELATIVE_MIN_REWARD_STD:-1e-4}

# GLOBAL episode budgets across ALL nodes/ranks
EPISODES_PER_EPOCH=${EPISODES_PER_EPOCH:-2000}
VAL_EPISODES=${VAL_EPISODES:-200}
EPOCHS=${EPOCHS:-100}
N_QUERY=${N_QUERY:-1}
NUM_STEPS=${NUM_STEPS:-8}

# F1-v5
UTILITY_PROJ_DIM=${UTILITY_PROJ_DIM:-64}
UTILITY_HIDDEN_DIM=${UTILITY_HIDDEN_DIM:-64}
UTILITY_CONDITION_DIM=${UTILITY_CONDITION_DIM:-32}
UTILITY_HEAD_DROPOUT=${UTILITY_HEAD_DROPOUT:-0.10}
UTILITY_LR=${UTILITY_LR:-2e-4}
FEATURE_RATIO_CLIP=${FEATURE_RATIO_CLIP:-10.0}
SEED=${SEED:-42}

# Optional checkpoint resume. Leave empty for a fresh run.
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

# =============================================================================
# Sanity checks
# =============================================================================
if [[ "${NNODES}" -lt 1 || "${NPROC_PER_NODE}" -lt 1 ]]; then
    echo "[ERROR] NNODES and NPROC_PER_NODE must be >= 1" >&2
    exit 2
fi
if [[ "${NODE_RANK}" -lt 0 || "${NODE_RANK}" -ge "${NNODES}" ]]; then
    echo "[ERROR] NODE_RANK=${NODE_RANK} must satisfy 0 <= NODE_RANK < NNODES=${NNODES}" >&2
    exit 2
fi

for p in "${PY}" "${M2_CHECKPOINT}" "${DIT_CHECKPOINT}" "${TRAIN_CLASS_JSON_ROOT}" "${VAL_CLASS_JSON_ROOT}"; do
    [[ -e "${p}" ]] || { echo "[ERROR] missing: ${p}" >&2; exit 2; }
done
if [[ -n "${RESUME_CHECKPOINT}" && ! -e "${RESUME_CHECKPOINT}" ]]; then
    echo "[ERROR] RESUME_CHECKPOINT missing: ${RESUME_CHECKPOINT}" >&2
    exit 2
fi

mkdir -p "${ROOT}/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility"
mkdir -p "${OUTPUT_DIR}"
cd "${CODE_DIR}"

WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
GLOBAL_EP_BATCH=$((WORLD_SIZE * EPISODE_BATCH_SIZE))
LOCAL_TRAIN_EP=$(((EPISODES_PER_EPOCH + WORLD_SIZE - 1) / WORLD_SIZE))
LOCAL_VAL_EP=$(((VAL_EPISODES + WORLD_SIZE - 1) / WORLD_SIZE))
UPDATES_PER_EPOCH=$(((LOCAL_TRAIN_EP + EPISODE_BATCH_SIZE - 1) / EPISODE_BATCH_SIZE))

# Validate visible GPU count on this node.
if command -v python >/dev/null 2>&1; then
    VISIBLE_GPU_COUNT=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
    if [[ "${VISIBLE_GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]]; then
        echo "[ERROR] host=$(hostname): visible GPUs=${VISIBLE_GPU_COUNT} < NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
        echo "        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
        exit 2
    fi
fi

echo "===================================================================================================================="
echo "[F1-v5 MULTI-NODE EXECUTION / TRUE EPISODE BATCH]"
echo "host                        = $(hostname)"
echo "NODE_RANK                   = ${NODE_RANK}"
echo "NNODES                      = ${NNODES}"
echo "NPROC_PER_NODE              = ${NPROC_PER_NODE}"
echo "WORLD_SIZE                  = ${WORLD_SIZE}"
echo "MASTER_ADDR:PORT            = ${MASTER_ADDR}:${MASTER_PORT}"
echo "CUDA_VISIBLE_DEVICES        = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NET_IFACE                   = ${NET_IFACE}"
echo "TRAIN_CLASS_JSON_ROOT       = ${TRAIN_CLASS_JSON_ROOT}"
echo "VAL_CLASS_JSON_ROOT         = ${VAL_CLASS_JSON_ROOT}"
echo "M2_CHECKPOINT               = ${M2_CHECKPOINT}"
echo "DIT_CHECKPOINT              = ${DIT_CHECKPOINT}"
echo "TRAJECTORY_SCORE_MODE       = ${TRAJECTORY_SCORE_MODE}"
echo "EPISODES global train/val   = ${EPISODES_PER_EPOCH} / ${VAL_EPISODES}"
echo "EPISODES local train/val    ~= ${LOCAL_TRAIN_EP} / ${LOCAL_VAL_EP} per rank"
echo "EPISODE_BATCH_SIZE / rank   = ${EPISODE_BATCH_SIZE}"
echo "GLOBAL EP BATCH / update    ~= ${GLOBAL_EP_BATCH}"
echo "UPDATES / epoch             ~= ${UPDATES_PER_EPOCH}"
echo "PAIR_CHUNK / RAE_CHUNK      = ${PAIR_CHUNK_SIZE} / ${RAE_CHUNK_SIZE}"
echo "RELATIVE lambda/group       = ${LAMBDA_RELATIVE} / ${RELATIVE_GROUP_SIZE}"
echo "RELATIVE warmup/sigma       = ${RELATIVE_WARMUP_EPOCHS} / ${RELATIVE_SIGMA_START}->${RELATIVE_SIGMA_END}"
echo "OUTPUT_DIR                  = ${OUTPUT_DIR}"
echo "===================================================================================================================="

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    EXTRA_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

# =============================================================================
# IMPORTANT: multi-node torchrun.  DO NOT use --standalone here.
# Each SLURM node runs one torchrun agent with a unique --node_rank.
# =============================================================================
torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${PY}" \
    --train_class_json_root "${TRAIN_CLASS_JSON_ROOT}" \
    --val_class_json_root "${VAL_CLASS_JSON_ROOT}" \
    --project_root "${ROOT}" \
    --m2_checkpoint "${M2_CHECKPOINT}" \
    --dit_checkpoint "${DIT_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_query "${N_QUERY}" \
    --num_steps "${NUM_STEPS}" \
    --solver heun \
    --episode_batch_size "${EPISODE_BATCH_SIZE}" \
    --pair_chunk_size "${PAIR_CHUNK_SIZE}" \
    --rae_chunk_size "${RAE_CHUNK_SIZE}" \
    --episodes_per_epoch "${EPISODES_PER_EPOCH}" \
    --val_episodes "${VAL_EPISODES}" \
    --epochs "${EPOCHS}" \
    --utility_proj_dim "${UTILITY_PROJ_DIM}" \
    --utility_hidden_dim "${UTILITY_HIDDEN_DIM}" \
    --utility_condition_dim "${UTILITY_CONDITION_DIM}" \
    --utility_head_dropout "${UTILITY_HEAD_DROPOUT}" \
    --utility_lr "${UTILITY_LR}" \
    --feature_ratio_clip "${FEATURE_RATIO_CLIP}" \
    --trajectory_score_mode "${TRAJECTORY_SCORE_MODE}" \
    --gamma_min "${GAMMA_MIN}" \
    --gamma_init "${GAMMA_INIT}" \
    --lambda_decay_smooth "${LAMBDA_DECAY_SMOOTH}" \
    --lambda_relative "${LAMBDA_RELATIVE}" \
    --relative_group_size "${RELATIVE_GROUP_SIZE}" \
    --relative_warmup_epochs "${RELATIVE_WARMUP_EPOCHS}" \
    --relative_sigma_start "${RELATIVE_SIGMA_START}" \
    --relative_sigma_end "${RELATIVE_SIGMA_END}" \
    --relative_target_step "${RELATIVE_TARGET_STEP}" \
    --relative_adv_clip "${RELATIVE_ADV_CLIP}" \
    --relative_min_reward_std "${RELATIVE_MIN_REWARD_STD}" \
    --seed "${SEED}" \
    --log_cuda_memory \
    "${EXTRA_ARGS[@]}"
