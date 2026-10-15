#!/bin/bash
#SBATCH -J q2l_m0b_rae
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_q2l_m0b_rae/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_q2l_m0b_rae/%x_%j.err


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
PY=${CODE_DIR}/pretrain_q2l_fsl_from_m0b_rae_torchrun_ddp_v2.py

# Same class-pool style used by RAE M0-B. Change only if your current pool differs.
TRAIN_JSON=${TRAIN_JSON:-"${ROOT}/data/data_train_pool_imagenet_med"}
VAL_JSON=${VAL_JSON:-"${ROOT}/data/data_eval_pool_imagenet_huaxi"}

# RAE M0-B checkpoint: contains checkpoint['metric_encoder'].
M0B_CHECKPOINT=${M0B_CHECKPOINT:-${ROOT}/weights_metric_v1/q2l_m0b_rae/metric_q2l_m0b_rae.pt}

# Current RAE DiT V1 support-conditioned Flow.
DIT_CHECKPOINT=${DIT_CHECKPOINT:-${ROOT}/weights_dit_flow_v1/vision_hint_inclass/otcfm/otcfm_dit_L_step_265000.pt}

OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/weights_metric_v1/q2l_m0b_rae}
LOG_DIR=${ROOT}/fsl_multi_vae/logs_q2l_m0b_rae
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
cd "${CODE_DIR}"

# ============================================================
# Runtime / optimization
# ============================================================
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29641}

# P x K is per GPU. Q2L generates 2*(P*K) frozen Flow endpoints per batch
# (same-class condition + cross-class condition), so start smaller than M0-B.
CLASSES_PER_BATCH=${CLASSES_PER_BATCH:-32}
SAMPLES_PER_CLASS=${SAMPLES_PER_CLASS:-8}
BATCHES_PER_EPOCH=${BATCHES_PER_EPOCH:-1000}
VAL_BATCHES=${VAL_BATCHES:-500}

EPOCHS=${EPOCHS:-10}
LR=${LR:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
TRIPLET_MARGIN=${TRIPLET_MARGIN:-0.2}

NUM_STEPS=${NUM_STEPS:-8}
SOLVER=${SOLVER:-heun}
PAIR_CHUNK_SIZE=${PAIR_CHUNK_SIZE:-16}
RAE_CHUNK_SIZE=${RAE_CHUNK_SIZE:-16}

OT_POOL_FACTOR=${OT_POOL_FACTOR:-1}
SINKHORN_REG=${SINKHORN_REG:-0.1}
SINKHORN_ITER=${SINKHORN_ITER:-30}
SK_WEIGHT_EPS=${SK_WEIGHT_EPS:-1e-3}

# Pure Q2L default: only ranking. VISReg is available but OFF.
RANK_WEIGHT=${RANK_WEIGHT:-1.0}
VISREG_WEIGHT=${VISREG_WEIGHT:-0.01}

SEED=${SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-32}

# ============================================================
# Checks
# ============================================================
[[ -f "${PY}" ]] || { echo "[ERROR] missing PY: ${PY}" >&2; exit 2; }
[[ -f "${M0B_CHECKPOINT}" ]] || { echo "[ERROR] missing M0-B: ${M0B_CHECKPOINT}" >&2; exit 2; }
[[ -f "${DIT_CHECKPOINT}" ]] || { echo "[ERROR] missing DiT: ${DIT_CHECKPOINT}" >&2; exit 2; }
compgen -G "${TRAIN_JSON}" >/dev/null || { echo "[ERROR] no train JSON matched: ${TRAIN_JSON}" >&2; exit 2; }

EXTRA_ARGS=()
if compgen -G "${VAL_JSON}" >/dev/null; then
    EXTRA_ARGS+=(--val_json_glob "${VAL_JSON}")
else
    echo "[WARN] no explicit val JSON matched; Python will use per-class stratified split"
fi

# ============================================================
# Pure Query-to-Last RAE pretraining
#
# query image --Frozen RAE--> z_q
#       +
# same-class / cross-class support image --Flow condition only
#       -> Frozen DiT endpoint z_T
#
# M0-B MetricEncoder(z_q) <-> M0-B MetricEncoder(z_T)
#       -> Dynamic-SK Sinkhorn
#       -> D_same / D_cross
#       -> relu(margin + D_same - D_cross)
#
# NO Query-to-Support distance.
# NO F1/router/gate.
# NO intermediate trajectory states retained.
# ============================================================

echo "============================================================================================================"
echo "[Q2L-M0B-RAE] pure Query-to-Last FSL pretraining"
echo "PY                  = ${PY}"
echo "TRAIN_JSON          = ${TRAIN_JSON}"
echo "VAL_JSON            = ${VAL_JSON}"
echo "M0B_CHECKPOINT      = ${M0B_CHECKPOINT}"
echo "DIT_CHECKPOINT      = ${DIT_CHECKPOINT}"
echo "OUTPUT_DIR          = ${OUTPUT_DIR}"
echo "NPROC               = ${NPROC_PER_NODE}"
echo "P x K / GPU         = ${CLASSES_PER_BATCH} x ${SAMPLES_PER_CLASS}"
echo "Flow steps/solver   = ${NUM_STEPS}/${SOLVER}"
echo "PAIR_CHUNK_SIZE     = ${PAIR_CHUNK_SIZE}"
echo "loss                = ${RANK_WEIGHT}*Q2L_rank + ${VISREG_WEIGHT}*VISReg"
echo "============================================================================================================"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "${PY}" \
    --train_json_glob "${TRAIN_JSON}" \
    "${EXTRA_ARGS[@]}" \
    --init_m0b_checkpoint "${M0B_CHECKPOINT}" \
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
