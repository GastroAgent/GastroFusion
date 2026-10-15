#!/bin/bash
#SBATCH -J q2mono_grpo_ddp
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x_%j.err

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-128}

# Single-node multi-GPU DDP. Match GPUS_PER_NODE to the Slurm GPU request.
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_DIR=${ROOT}/fsl_multi_vae/rae_flow_metric_v1
PY=${CODE_DIR}/pretrain_f1_q2mono_grpo_v3_ddp_batch.py

# Directory containing many class JSON/JSONL files. data_train_pool_imagenet_med
TRAIN_JSONS_ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_eval_pool_huaxi
# Optional. Leave empty to make a deterministic per-class split from TRAIN_JSONS_ROOT. data_eval_pool_imagenet_huaxi
VAL_JSONS_ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_eval_pool_huaxi

M2_CHECKPOINT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_metric_v1/new_m2/metric_direct_episode.pt
DIT_CHECKPOINT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_dit_flow_v1/vision_hint_inclass/otcfm/otcfm_dit_L_step_500000.pt
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/weights_metric_v1/q2mono_grpo_v3_pretrain}

N_WAY=${N_WAY:-8}
N_SHOT=${N_SHOT:-5}
N_QUERY=${N_QUERY:-1}
NUM_STEPS=${NUM_STEPS:-8}
EPISODES_PER_EPOCH=${EPISODES_PER_EPOCH:-5000}
VAL_EPISODES=${VAL_EPISODES:-500}
EPOCHS=${EPOCHS:-20}

# True per-rank episode batch. Episodes in the same batch share batched RAE,
# merged Flow pair-stream, Compressor forward, and policy/GRPO forward.
EPISODE_BATCH_SIZE=${EPISODE_BATCH_SIZE:-4}
PAIR_CHUNK_SIZE=${PAIR_CHUNK_SIZE:-32}
RAE_CHUNK_SIZE=${RAE_CHUNK_SIZE:-32}

# Compressor uses a lower LR than the trajectory policy.
COMPRESSOR_LR=${COMPRESSOR_LR:-1e-5}
POLICY_LR=${POLICY_LR:-1e-4}

# Compressor-specific VISReg.
LAMBDA_VISREG=${LAMBDA_VISREG:-0.05}
VISREG_NUM_PROJECTIONS=${VISREG_NUM_PROJECTIONS:-1024}
VISREG_PROJECTION_CHUNK_SIZE=${VISREG_PROJECTION_CHUNK_SIZE:-128}

# Joint + policy losses.
LAMBDA_DIRECT_CE=${LAMBDA_DIRECT_CE:-1.0}
LAMBDA_JOINT_CE=${LAMBDA_JOINT_CE:-1.0}
LAMBDA_GRPO=${LAMBDA_GRPO:-0.25}
LAMBDA_SAFE=${LAMBDA_SAFE:-0.50}
LAMBDA_KL=${LAMBDA_KL:-0.02}
GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-8}
GRPO_WARMUP_EPOCHS=${GRPO_WARMUP_EPOCHS:-1}
GRPO_CLIP_EPS=${GRPO_CLIP_EPS:-0.20}
EMA_DECAY=${EMA_DECAY:-0.995}

mkdir -p "${OUTPUT_DIR}" "${CODE_DIR}/logs_f1_utility"
cd "${CODE_DIR}"

[[ -d "${TRAIN_JSONS_ROOT}" ]] || { echo "[ERROR] TRAIN_JSONS_ROOT missing: ${TRAIN_JSONS_ROOT}" >&2; exit 2; }
[[ -f "${M2_CHECKPOINT}" ]] || { echo "[ERROR] M2_CHECKPOINT missing: ${M2_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DIT_CHECKPOINT}" ]] || { echo "[ERROR] DIT_CHECKPOINT missing: ${DIT_CHECKPOINT}" >&2; exit 2; }
[[ -f "${PY}" ]] || { echo "[ERROR] PY missing: ${PY}" >&2; exit 2; }

VAL_ARGS=()
if [[ -n "${VAL_JSONS_ROOT}" ]]; then
    [[ -d "${VAL_JSONS_ROOT}" ]] || { echo "[ERROR] VAL_JSONS_ROOT missing: ${VAL_JSONS_ROOT}" >&2; exit 2; }
    VAL_ARGS+=(--val_jsons_root "${VAL_JSONS_ROOT}")
fi

torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" "${PY}" \
    --train_jsons_root "${TRAIN_JSONS_ROOT}" \
    "${VAL_ARGS[@]}" \
    --project_root "${ROOT}" \
    --m2_checkpoint "${M2_CHECKPOINT}" \
    --dit_checkpoint "${DIT_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_way "${N_WAY}" \
    --n_shot "${N_SHOT}" \
    --n_query "${N_QUERY}" \
    --num_steps "${NUM_STEPS}" \
    --solver heun \
    --episode_batch_size "${EPISODE_BATCH_SIZE}" \
    --pair_chunk_size "${PAIR_CHUNK_SIZE}" \
    --rae_chunk_size "${RAE_CHUNK_SIZE}" \
    --episodes_per_epoch "${EPISODES_PER_EPOCH}" \
    --val_episodes "${VAL_EPISODES}" \
    --epochs "${EPOCHS}" \
    --compressor_lr "${COMPRESSOR_LR}" \
    --policy_lr "${POLICY_LR}" \
    --lambda_visreg "${LAMBDA_VISREG}" \
    --visreg_num_projections "${VISREG_NUM_PROJECTIONS}" \
    --visreg_projection_chunk_size "${VISREG_PROJECTION_CHUNK_SIZE}" \
    --lambda_direct_ce "${LAMBDA_DIRECT_CE}" \
    --lambda_joint_ce "${LAMBDA_JOINT_CE}" \
    --lambda_grpo "${LAMBDA_GRPO}" \
    --lambda_safe "${LAMBDA_SAFE}" \
    --lambda_kl "${LAMBDA_KL}" \
    --grpo_group_size "${GRPO_GROUP_SIZE}" \
    --grpo_warmup_epochs "${GRPO_WARMUP_EPOCHS}" \
    --grpo_clip_eps "${GRPO_CLIP_EPS}" \
    --ema_decay "${EMA_DECAY}"
