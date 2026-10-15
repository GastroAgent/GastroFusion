#!/bin/bash
#SBATCH -J f1_v5_lstm_medical
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x_%j.err

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}
export PYTHONUNBUFFERED=1

ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_DIR=${ROOT}/fsl_multi_vae/rae_flow_metric_v1
PY=${CODE_DIR}/train_f1_support_conditioned_lstm_v5_medical_manifest.py

# =============================================================================
# Organ / model / data
# Run one organ-specific F1-v5 model per job. Override these from the environment
# when training 食管 / 十二指肠 or when using another organ-specific M2 checkpoint.
# =============================================================================
ORGAN=${ORGAN:-十二指肠}
M2_CHECKPOINT=${M2_CHECKPOINT:-${ROOT}/weights_metric_v1/m2_十二指肠-test-train/metric_direct_episode.pt}
DIT_CHECKPOINT=${DIT_CHECKPOINT:-${ROOT}/weights_dit_flow_v1/vision_hint/otcfm/otcfm_dit_L_step_400000.pt}

# TRAIN_EPISODE_MANIFEST=${TRAIN_EPISODE_MANIFEST:-${ROOT}/fsl_multi_vae/data/${ORGAN}/train.json}
TRAIN_EPISODE_MANIFEST=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/data/total_train.json
VAL_EPISODE_MANIFEST=${VAL_EPISODE_MANIFEST:-${ROOT}/fsl_multi_vae/data/${ORGAN}/val.json}

# Choose the learned trajectory score branch.
#   step_to_support : sum_t w_t D(z_t, support)
#   step_to_last    : sum_t w_t D(z_t, z_T)
TRAJECTORY_SCORE_MODE=${TRAJECTORY_SCORE_MODE:-step_to_support}

OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/weights_metric_v1/f1_v5_support_lstm_medical/test-train/${ORGAN}/${TRAJECTORY_SCORE_MODE}}

# Optional ImageNet/class-JSON F1 pretraining checkpoint.
# Only TemporalDecayHead is loaded; optimizer/scheduler/epoch are freshly initialized.
PRETRAIN_F1_CHECKPOINT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_metric_v1/f1_v5_support_lstm_medical/test-train/十二指肠/step_to_support/f1_dynamic_decay_best.pt
INIT_ARGS=()
if [[ -n "${PRETRAIN_F1_CHECKPOINT}" ]]; then
    [[ -e "${PRETRAIN_F1_CHECKPOINT}" ]] || { echo "[ERROR] missing PRETRAIN_F1_CHECKPOINT: ${PRETRAIN_F1_CHECKPOINT}" >&2; exit 2; }
    INIT_ARGS=(--init_decay_checkpoint "${PRETRAIN_F1_CHECKPOINT}")
fi

TRAIN_ARGS=(--train_episode_manifest "${TRAIN_EPISODE_MANIFEST}")
TRAIN_MODE="TRAIN_EPISODE_MANIFEST"

# =============================================================================
# F1-v5 architecture
# Support Metric class-set -> c0; Direct-query Metric -> h0; current Flow-step
# candidate Metric class-set -> x_t. Statistics + explicit t/T independently
# condition all LSTM gates. No simple modality concatenation is used.
# =============================================================================
# =============================================================================
# Dynamic decay
# gamma_t = GAMMA_MIN + (1-GAMMA_MIN)*sigmoid(a_t)
# u0=1; ut=prod_{i<=t} gamma_i; wt=ut/sum(u)
# => w0 >= w1 >= ... >= wT and sum(w)=1 by construction.
# =============================================================================
GAMMA_MIN=${GAMMA_MIN:-0.10}
GAMMA_INIT=${GAMMA_INIT:-0.90}
LAMBDA_DECAY_SMOOTH=${LAMBDA_DECAY_SMOOTH:-0.01}

# =============================================================================
# Group-relative exploration (training only)
# Warmup: exact differentiable CE only.
# Then use antithetic +/- perturbations around raw decay logits.
# Noise is cosine-annealed from SIGMA_START to SIGMA_END.
# =============================================================================
LAMBDA_RELATIVE=${LAMBDA_RELATIVE:-0.25}
RELATIVE_GROUP_SIZE=${RELATIVE_GROUP_SIZE:-8}
RELATIVE_WARMUP_EPOCHS=${RELATIVE_WARMUP_EPOCHS:-2}
RELATIVE_SIGMA_START=${RELATIVE_SIGMA_START:-1.0}
RELATIVE_SIGMA_END=${RELATIVE_SIGMA_END:-0.20}
RELATIVE_TARGET_STEP=${RELATIVE_TARGET_STEP:-0.25}
RELATIVE_ADV_CLIP=${RELATIVE_ADV_CLIP:-3.0}
RELATIVE_MIN_REWARD_STD=${RELATIVE_MIN_REWARD_STD:-1e-4}

# =============================================================================
# Existing training settings
# =============================================================================
NUM_STEPS=${NUM_STEPS:-8}
PAIR_CHUNK_SIZE=${PAIR_CHUNK_SIZE:-2}
EPISODES_PER_EPOCH=${EPISODES_PER_EPOCH:-500}
VAL_EPISODES=${VAL_EPISODES:-400}
EPOCHS=${EPOCHS:-20}
UTILITY_PROJ_DIM=${UTILITY_PROJ_DIM:-64}
UTILITY_HIDDEN_DIM=${UTILITY_HIDDEN_DIM:-64}
UTILITY_CONDITION_DIM=${UTILITY_CONDITION_DIM:-32}
UTILITY_HEAD_DROPOUT=${UTILITY_HEAD_DROPOUT:-0.10}
UTILITY_LR=${UTILITY_LR:-1e-3}
FEATURE_RATIO_CLIP=${FEATURE_RATIO_CLIP:-10.0}
SEED=${SEED:-42}

mkdir -p "${ROOT}/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility"
mkdir -p "${OUTPUT_DIR}"
cd "${CODE_DIR}"

for p in "${M2_CHECKPOINT}" "${DIT_CHECKPOINT}" "${TRAIN_EPISODE_MANIFEST}" "${VAL_EPISODE_MANIFEST}"; do
    [[ -e "${p}" ]] || { echo "[ERROR] missing: ${p}" >&2; exit 2; }
done

echo "===================================================================================================================="
echo "[F1-v5 SUPPORT-CONDITIONED LSTM + DYNAMIC DECAY + GROUP-RELATIVE EXPLORATION]"
echo "ORGAN                    = ${ORGAN}"
echo "TRAIN_MODE               = ${TRAIN_MODE}"
echo "TRAIN_EPISODE_MANIFEST   = ${TRAIN_EPISODE_MANIFEST:-<disabled>}"
echo "VAL_EPISODE_MANIFEST     = ${VAL_EPISODE_MANIFEST}"
echo "M2_CHECKPOINT            = ${M2_CHECKPOINT}"
echo "DIT_CHECKPOINT           = ${DIT_CHECKPOINT}"
echo "TRAJECTORY_SCORE_MODE    = ${TRAJECTORY_SCORE_MODE}"
echo "GAMMA_MIN / INIT         = ${GAMMA_MIN} / ${GAMMA_INIT}"
echo "LAMBDA_DECAY_SMOOTH      = ${LAMBDA_DECAY_SMOOTH}"
echo "RELATIVE lambda/group    = ${LAMBDA_RELATIVE} / ${RELATIVE_GROUP_SIZE}"
echo "RELATIVE warmup          = ${RELATIVE_WARMUP_EPOCHS} epoch(s)"
echo "RELATIVE sigma           = ${RELATIVE_SIGMA_START} -> ${RELATIVE_SIGMA_END}"
echo "RELATIVE target step     = ${RELATIVE_TARGET_STEP}"
echo "F1 set/hidden/cond/dropout   = ${UTILITY_PROJ_DIM} / ${UTILITY_HIDDEN_DIM} / ${UTILITY_CONDITION_DIM} / ${UTILITY_HEAD_DROPOUT}"
echo "PRETRAIN_F1_CHECKPOINT   = ${PRETRAIN_F1_CHECKPOINT:-<none>}"
echo "OUTPUT_DIR               = ${OUTPUT_DIR}"
echo "===================================================================================================================="

python "${PY}" \
    "${TRAIN_ARGS[@]}" \
    "${INIT_ARGS[@]}" \
    --val_episode_manifest "${VAL_EPISODE_MANIFEST}" \
    --project_root "${ROOT}" \
    --m2_checkpoint "${M2_CHECKPOINT}" \
    --dit_checkpoint "${DIT_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --metric_device cuda:0 \
    --flow_device cuda:0 \
    --rae_device cuda:0 \
    --num_steps "${NUM_STEPS}" \
    --solver heun \
    --pair_chunk_size "${PAIR_CHUNK_SIZE}" \
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
    --log_cuda_memory
