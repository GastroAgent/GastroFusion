#!/usr/bin/env bash
#SBATCH -J eval_f1_stomach_testtest
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# If submitted via sbatch, specify log paths using sbatch -o/-e if desired.

set -euo pipefail

# Activate your CUDA/Conda environment before running this script, for example:
#   conda activate flow
# No server-specific activation paths are hard-coded here.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# This file resides directly in GastroFusion/wass_flow/.
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${CODE_DIR}")"
ROOT="${CODE_DIR}"
PY="${CODE_DIR}/eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_m2_dit.py"

# stomach.zip extracts into stomach/<class name>/<image>.
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/demo_data/stomach}"
WEIGHT_ROOT="${WEIGHT_ROOT:-${PROJECT_DIR}/demo_weights/wass-flow-stomach}"

# Confirmed public package checkpoint paths.
F1_STOMACH="${F1_STOMACH:-${WEIGHT_ROOT}/f1_v5_support_lstm_medical/test-test/胃/step_to_support/f1_dynamic_decay_best.pt}"
M2_STOMACH="${M2_STOMACH:-${WEIGHT_ROOT}/new_m1/metric_direct_episode.pt}"
DIT_CHECKPOINT="${DIT_CHECKPOINT:-${WEIGHT_ROOT}/otcfm_dit_L_step_265000.pt}"
PROJECT_ROOT="${PROJECT_ROOT:-${ROOT}}"

# build_rae() and FrozenFlowRuntime use these existing paths under PROJECT_ROOT.
RAE_DECODER="${PROJECT_ROOT}/flow_matcher_otcfm/rae_v2/decoder.pt"
RAE_STATS="${PROJECT_ROOT}/flow_matcher_otcfm/rae_v2/stats.pt"
ENDOVIT="${PROJECT_ROOT}/flow_matcher_otcfm/EndoViT/pytorch_model.bin"

# DINOv3 loader uses DINOV3_CKPT_DIR, if set, otherwise its own default path.
# Optional example:
# export DINOV3_CKPT_DIR="/path/to/dinov3"

EPISODES="${EPISODES:-100}"
SEED="${SEED:-42}"
N_SHOT="${N_SHOT:-5}"
N_QUERY="${N_QUERY:-1}"
FIXED_DECAY="${FIXED_DECAY:-0.90}"
EPISODE_BATCH_SIZE="${EPISODE_BATCH_SIZE:-4}"
PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-16}"
RAE_CHUNK_SIZE="${RAE_CHUNK_SIZE:-32}"
TRAJECTORY_SCORE_MODE="${TRAJECTORY_SCORE_MODE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/results/f1_stomach_testtest}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/summary_seed${SEED}.json}"

for p in "${PY}" "${DATA_ROOT}" "${F1_STOMACH}" "${M2_STOMACH}" "${DIT_CHECKPOINT}" \
         "${RAE_DECODER}" "${RAE_STATS}" "${ENDOVIT}"; do
    if [[ ! -e "${p}" ]]; then
        echo "[ERROR] Missing required path: ${p}" >&2
        exit 2
    fi
done

# This script does not modify the DINOv3 loader. Ensure its pretrained ViT-L/16
# weight is available at the location required by rae/encoders/.../dinov3_loader.py.
mkdir -p "${OUTPUT_DIR}"
cd "${CODE_DIR}"

EXTRA_ARGS=()
if [[ -n "${TRAJECTORY_SCORE_MODE}" ]]; then
    EXTRA_ARGS+=(--trajectory_score_mode "${TRAJECTORY_SCORE_MODE}")
fi

printf '%s\n' '=========================================================='
printf '%s\n' '[F1-v5 / stomach / test-test / 5-shot / 1-query]'
printf 'DATA_ROOT       = %s\n' "${DATA_ROOT}"
printf 'F1_CHECKPOINT   = %s\n' "${F1_STOMACH}"
printf 'M2_CHECKPOINT   = %s\n' "${M2_STOMACH}"
printf 'DIT_CHECKPOINT  = %s\n' "${DIT_CHECKPOINT}"
printf 'PROJECT_ROOT    = %s\n' "${PROJECT_ROOT}"
printf 'EPISODES        = %s\n' "${EPISODES}"
printf 'OUTPUT_JSON     = %s\n' "${OUTPUT_JSON}"
printf '%s\n' '=========================================================='

python "${PY}" \
    --data_root "${DATA_ROOT}" \
    --f1_checkpoint "${F1_STOMACH}" \
    --m2_checkpoint "${M2_STOMACH}" \
    --project_root "${PROJECT_ROOT}" \
    --dit_checkpoint "${DIT_CHECKPOINT}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --n_shot "${N_SHOT}" \
    --n_query "${N_QUERY}" \
    --episode_batch_size "${EPISODE_BATCH_SIZE}" \
    --pair_batch_size "${PAIR_BATCH_SIZE}" \
    --rae_chunk_size "${RAE_CHUNK_SIZE}" \
    --fixed_decay "${FIXED_DECAY}" \
    --metric_device cuda:0 \
    --flow_device cuda:0 \
    --rae_device cuda:0 \
    --output_json "${OUTPUT_JSON}" \
    --no-save_episode_details \
    --log_cuda_memory \
    "${EXTRA_ARGS[@]}"
