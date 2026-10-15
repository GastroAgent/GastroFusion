#!/bin/bash
#SBATCH -J q2l_tc_rae
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_q2l_tc_rae/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_q2l_tc_rae/%x_%j.err


# ============================================================
# Environment
# ============================================================
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}

# ============================================================
# Paths: RAE branch, NOT Exp6 Multi-VAE
# ============================================================
ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_DIR=${ROOT}/fsl_multi_vae/rae_flow_metric_v1
PY=${CODE_DIR}/pretrain_q2l_fsl_tc_from_q2l_rae_torchrun_ddp_v3.py

# metric_encoder_time.py (TimestepMetricEncoder) must sit beside the Python
# script inside CODE_DIR. The trainer tries models.metric_encoder_time first,
# then falls back to the local file in the same directory.

# Same class-pool style used by RAE M0-B.
TRAIN_JSON=${TRAIN_JSON:-"${ROOT}/data/data_train_pool_imagenet_med"}
VAL_JSON=${VAL_JSON:-"${ROOT}/data/data_eval_pool_imagenet_huaxi"}

# Trained Q2L MetricEncoder (plain, no timestep extension): produced by
# train_q2l_m0b_rae_torchrun_ddp_v2.py / metric_q2l_m0b_rae.pt. Its shared
# base MetricEncoder weights are continue-initialized into
# TimestepMetricEncoder (new time_embed + Ada-LN-Zero gates start at the
# exact Q2L copy, gate_bias=1 / scale-shift=0).
Q2L_CHECKPOINT=${Q2L_CHECKPOINT:-${ROOT}/weights_metric_v1/q2l_m0b_rae/metric_q2l_m0b_rae.pt}

# Optional TC resume written by THIS q2l_tc script.
# RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}

# Current RAE DiT V1 support-conditioned Flow.
DIT_CHECKPOINT=${DIT_CHECKPOINT:-${ROOT}/weights_dit_flow_v1/vision_hint_inclass/otcfm/otcfm_dit_L_step_265000.pt}

OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/weights_metric_v1/q2l_tc_from_q2l_rae}
LOG_DIR=${ROOT}/fsl_multi_vae/logs_q2l_tc_rae
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${CODE_DIR}"

# ============================================================
# Runtime / optimization
# ============================================================
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29642}

# TC stores and metric-encodes EVERY intermediate Flow latent: per PK batch the
# metric runs 1 (query, t=0) + num_steps (candidate, t=k/T) forward passes, so
# we start at a smaller PK than the pure Query-to-Last stage.
CLASSES_PER_BATCH=${CLASSES_PER_BATCH:-8}
SAMPLES_PER_CLASS=${SAMPLES_PER_CLASS:-4}
BATCHES_PER_EPOCH=${BATCHES_PER_EPOCH:-1000}
VAL_BATCHES=${VAL_BATCHES:-200}

EPOCHS=${EPOCHS:-10}
LR=${LR:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
TRIPLET_MARGIN=${TRIPLET_MARGIN:-0.2}

NUM_STEPS=${NUM_STEPS:-8}
SOLVER=${SOLVER:-heun}
PAIR_CHUNK_SIZE=${PAIR_CHUNK_SIZE:-8}
RAE_CHUNK_SIZE=${RAE_CHUNK_SIZE:-16}

# Timestep-conditioned extension (plain | time | q2step-mode).
METRIC_MODEL=metric_encoder_time
TIME_DIM=${TIME_DIM:-512}
GATE_BIAS=${GATE_BIAS:-1.0}

# true (default): endpoint t=1 is included in Query-to-StepMean and Q2L is a
# rollout report. false keeps an explicit two-term
# Q2-StepMean(k=1..T-1) + Q2-Last(k=T) objective.
Q2STEP_INCL_ENDPOINT=${Q2STEP_INCL_ENDPOINT:-true}

OT_POOL_FACTOR=${OT_POOL_FACTOR:-1}
SINKHORN_REG=${SINKHORN_REG:-0.1}
SINKHORN_ITER=${SINKHORN_ITER:-30}
SK_WEIGHT_EPS=${SK_WEIGHT_EPS:-1e-3}

# Pure Q2L-style default: ranking (+ optional VISReg, kept at the Q2L value).
RANK_WEIGHT=${RANK_WEIGHT:-1.0}
VISREG_WEIGHT=${VISREG_WEIGHT:-0.01}

SEED=${SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-32}

# ============================================================
# Checks
# ============================================================
[[ -f "${PY}" ]] || { echo "[ERROR] missing PY: ${PY}" >&2; exit 2; }
[[ -f "${Q2L_CHECKPOINT}" ]] || { echo "[ERROR] missing Q2L/m0b metric: ${Q2L_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DIT_CHECKPOINT}" ]] || { echo "[ERROR] missing DiT: ${DIT_CHECKPOINT}" >&2; exit 2; }
compgen -G "${TRAIN_JSON}" >/dev/null || { echo "[ERROR] no train JSON matched: ${TRAIN_JSON}" >&2; exit 2; }

EXTRA_ARGS=()
if compgen -G "${VAL_JSON}" >/dev/null; then
    EXTRA_ARGS+=(--val_json_glob "${VAL_JSON}")
else
    echo "[WARN] no explicit val JSON matched; Python will use per-class stratified split"
fi

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    EXTRA_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
else
    EXTRA_ARGS+=(--init_q2l_checkpoint "${Q2L_CHECKPOINT}")
fi

# BooleanOptionalAction must receive the full --flag / --no-flag token; a bare
# "true"/"false" value string is not accepted by argparse.
if [[ "${Q2STEP_INCL_ENDPOINT}" != "true" ]]; then
    EXTRA_ARGS+=(--no-q2step_incl_endpoint)
else
    EXTRA_ARGS+=(--q2step_incl_endpoint)
fi

# ============================================================
# Timestep-conditioned Query-to-Step / Query-to-Last pretraining
#
# query image --Frozen RAE--> z_q (t = 0)
#       +
# same-class / cross-class support image --Flow condition only
#       -> Frozen DiT FULL trajectory z_{k/T}, k = 1..num_steps (all stored)
#
# TimestepMetricEncoder(z_q, t=0) <-> TimestepMetricEncoder(z_{k/T}, t=k/T)
#       -> Dynamic-SK Sinkhorn per step
#       -> mean_k same vs cross -> relu(margin + D_same - D_cross)
#
# Base-MetricEncoder weights start at the exact Q2L metric copy
# (continue init), time conditioning extends the metric afterwards.
#
# NO Query-to-Support distance. NO F1/router/gate. NO trajectory-weight head.
# ============================================================

echo "============================================================================================================"
echo "[Q2L-TC] timestep-conditioned Query-to-Step/Query-to-Last continue-pretrain"
echo "PY                  = ${PY}"
echo "INIT                = ${RESUME_CHECKPOINT:-(Q2L continue-init)}"
echo "Q2L_METRIC          = ${Q2L_CHECKPOINT}"
echo "DIT_CHECKPOINT      = ${DIT_CHECKPOINT}"
echo "OUTPUT_DIR          = ${OUTPUT_DIR}"
echo "NPROC               = ${NPROC_PER_NODE}"
echo "P x K / GPU         = ${CLASSES_PER_BATCH} x ${SAMPLES_PER_CLASS}"
echo "Flow steps/solver   = ${NUM_STEPS}/${SOLVER}"
echo "PAIR_CHUNK_SIZE     = ${PAIR_CHUNK_SIZE}"
echo "TC metric           = ${METRIC_MODEL} (time_dim=${TIME_DIM}, gate_bias=${GATE_BIAS})"
echo "Q2-step endpoint    = ${Q2STEP_INCL_ENDPOINT}"
echo "loss                = ${RANK_WEIGHT}*Q2S/Q2L_rank + ${VISREG_WEIGHT}*VISReg"
echo "============================================================================================================"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "${PY}" \
    --train_json_glob "${TRAIN_JSON}" \
    "${EXTRA_ARGS[@]}" \
    --dit_checkpoint "${DIT_CHECKPOINT}" \
    --project_root "${ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --image_size 256 \
    --rae_chunk_size "${RAE_CHUNK_SIZE}" \
    --classes_per_batch "${CLASSES_PER_BATCH}" \
    --samples_per_class "${SAMPLES_PER_CLASS}" \
    --batches_per_epoch "${BATCHES_PER_EPOCH}" \
    --val_batches "${VAL_BATCHES}" \
    --num_workers "${NUM_WORKERS}" \
    --dit_variant L \
    --num_classes 1000 \
    --num_steps "${NUM_STEPS}" \
    --solver "${SOLVER}" \
    --pair_chunk_size "${PAIR_CHUNK_SIZE}" \
    --use_ema \
    --stochastic_hint \
    --metric_model "${METRIC_MODEL}" \
    --time_dim "${TIME_DIM}" \
    --gate_bias "${GATE_BIAS}" \
    --ot_pool_factor "${OT_POOL_FACTOR}" \
    --sinkhorn_reg "${SINKHORN_REG}" \
    --sinkhorn_iter "${SINKHORN_ITER}" \
    --sk_weight_eps "${SK_WEIGHT_EPS}" \
    --triplet_margin "${TRIPLET_MARGIN}" \
    --support_selection random \
    --val_support_selection fixed \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --rank_weight "${RANK_WEIGHT}" \
    --visreg_weight "${VISREG_WEIGHT}" \
    --grad_clip 5.0 \
    --amp \
    --seed "${SEED}" \
    --log_every 10