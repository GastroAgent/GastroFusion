#!/bin/bash
#SBATCH -J f1_v10c_testtrain
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x_%j.err

set -euo pipefail

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}
export PYTHONUNBUFFERED=1

ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_DIR=${ROOT}/fsl_multi_vae/rae_flow_metric_v1
PY=${CODE_DIR}/test_f1_v10c_testtrain.py
PRETRAIN_PY=${CODE_DIR}/pretrain_f1_flow_trajectory_pairwise_corrector_v10c_classjson_torchrun_ddp_batch.py

# =============================================================================
# 你通常只需要修改这两项：
#   QUERY_ROOT   : TEST/query pool
#   SUPPORT_ROOT : TRAIN/support pool
# 也支持环境变量覆盖后直接 bash。
# =============================================================================
QUERY_ROOT=${QUERY_ROOT:-/path/to/test_query_root}
SUPPORT_ROOT=${SUPPORT_ROOT:-/path/to/train_support_root}

# =============================================================================
# Model weights: exactly follow the corresponding training script.
# =============================================================================
F1_CHECKPOINT=${F1_CHECKPOINT:-${ROOT}/weights_metric_v1/f1_v10c_flow_trajectory_pairwise_corrector/Imagenet/f1_direct_first_pairwise_trajectory_corrector_best.pt}
M2_CHECKPOINT=${M2_CHECKPOINT:-${ROOT}/weights_metric_v1/m2_flow_aug_m2_visreg/metric_direct_episode.pt}
DIT_CHECKPOINT=${DIT_CHECKPOINT:-${ROOT}/weights_dit_flow_v1/vision_hint/otcfm/otcfm_dit_L_step_440000.pt}

# =============================================================================
# Simplified test-train protocol.
# n_way=0: preserve all classes shared by query and support roots.
# =============================================================================
N_WAY=${N_WAY:-0}
N_SHOT=${N_SHOT:-5}
EPISODES=${EPISODES:-100}
EPISODE_BATCH_SIZE=${EPISODE_BATCH_SIZE:-1}
SEED=${SEED:-42}
GPU_ID=${GPU_ID:-0}

QUERY_NAME=${QUERY_NAME:-$(basename "${QUERY_ROOT%/}")}
SUPPORT_NAME=${SUPPORT_NAME:-$(basename "${SUPPORT_ROOT%/}")}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/result_f1_v10c/test_train/${QUERY_NAME}__support_${SUPPORT_NAME}}
OUTPUT_JSON=${OUTPUT_JSON:-${OUTPUT_DIR}/result.json}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
mkdir -p "${ROOT}/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility" "${OUTPUT_DIR}"

for p in "${PY}" "${PRETRAIN_PY}" "${F1_CHECKPOINT}" "${M2_CHECKPOINT}" "${DIT_CHECKPOINT}" "${QUERY_ROOT}" "${SUPPORT_ROOT}"; do
    [[ -e "${p}" ]] || { echo "[ERROR] missing: ${p}" >&2; exit 2; }
done

cat <<EOF2
====================================================================================================
[F1-v10c SIMPLIFIED TEST-TRAIN]
Query root         = ${QUERY_ROOT}
Support root       = ${SUPPORT_ROOT}
F1 checkpoint      = ${F1_CHECKPOINT}
M2 checkpoint      = ${M2_CHECKPOINT}
DiT checkpoint     = ${DIT_CHECKPOINT}
Protocol           = query=test; support=train; candidate classes=query/support intersection
Episode            = dynamic-way(${N_WAY}) ${N_SHOT}-shot 1-query
Episodes           = ${EPISODES}
GPU                = physical ${GPU_ID} -> runtime cuda:0
Output             = ${OUTPUT_JSON}
====================================================================================================
EOF2

cd "${CODE_DIR}"
python "${PY}" \
    --pretrain_py "${PRETRAIN_PY}" \
    --f1_checkpoint "${F1_CHECKPOINT}" \
    --test_class_json_root "${QUERY_ROOT}" \
    --train_class_json_root "${SUPPORT_ROOT}" \
    --output_json "${OUTPUT_JSON}" \
    --project_root "${ROOT}" \
    --m2_checkpoint "${M2_CHECKPOINT}" \
    --dit_checkpoint "${DIT_CHECKPOINT}" \
    --n_way "${N_WAY}" \
    --n_shot "${N_SHOT}" \
    --episodes "${EPISODES}" \
    --episode_batch_size "${EPISODE_BATCH_SIZE}" \
    --seed "${SEED}" \
    --metric_device cuda:0 \
    --flow_device cuda:0 \
    --rae_device cuda:0 \
    --save_episode_details
