#!/bin/bash
#SBATCH -J m2_f1_router_testtrain
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x_%j.err

export CUDA_VISIBLE_DEVICES=1
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}
export PYTHONUNBUFFERED=1

ROOT="/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow"
CODE_DIR="${ROOT}/fsl_multi_vae/rae_flow_metric_v1"
PY="${CODE_DIR}/eval_m2_f1_flow_router_testtrain.py"

QUERY_BASE="${ROOT}/data/indomain_support_76"
SUPPORT_ESOPHAGUS="${ROOT}/data/食管_1query/train"
SUPPORT_STOMACH="${ROOT}/data/胃_1query/train"
SUPPORT_DUODENUM="${ROOT}/data/十二指肠_1query/train"
DIT_CHECKPOINT="${DIT_CHECKPOINT:-${ROOT}/weights_dit_flow_v1/vision_hint/otcfm/otcfm_dit_L_step_400000.pt}"

F1_ESOPHAGUS="${ROOT}/weights_metric_v1/f1_v5_support_lstm_medical/test-test/食管/step_to_support/f1_dynamic_decay_best.pt"
F1_STOMACH="${ROOT}/weights_metric_v1/f1_v5_support_lstm_medical/test-test/胃/step_to_support/f1_dynamic_decay_best.pt"
F1_DUODENUM="${ROOT}/weights_metric_v1/f1_v5_support_lstm_medical/test-test/十二指肠/step_to_support/f1_dynamic_decay_best.pt"

M2_ESOPHAGUS="${ROOT}/weights_metric_v1/m2_食管-test-train/metric_direct_episode.pt"
M2_STOMACH="${ROOT}/weights_metric_v1/m2_胃-test-train/metric_direct_episode.pt"
M2_DUODENUM="${ROOT}/weights_metric_v1/m2_十二指肠-test-train/metric_direct_episode.pt"

OUTPUT_DIR="${CODE_DIR}/m2_f1_flow_router/test-train"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-42}"
N_SHOT="${N_SHOT:-5}"
EPISODE_BATCH_SIZE="${EPISODE_BATCH_SIZE:-4}"
PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-8}"
RAE_CHUNK_SIZE="${RAE_CHUNK_SIZE:-32}"

mkdir -p "${OUTPUT_DIR}" "${ROOT}/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility"
cd "${CODE_DIR}"

for FILE in \
  "${PY}" "${DIT_CHECKPOINT}" \
  "${F1_ESOPHAGUS}" "${F1_STOMACH}" "${F1_DUODENUM}" \
  "${M2_ESOPHAGUS}" "${M2_STOMACH}" "${M2_DUODENUM}"; do
  [[ -f "${FILE}" ]] || { echo "[ERROR] file not found: ${FILE}" >&2; exit 1; }
done

for DIR in "${QUERY_BASE}" "${SUPPORT_ESOPHAGUS}" "${SUPPORT_STOMACH}" "${SUPPORT_DUODENUM}"; do
  [[ -d "${DIR}" ]] || { echo "[ERROR] directory not found: ${DIR}" >&2; exit 1; }
done

python -u "${PY}" \
  --query_base "${QUERY_BASE}" \
  --hospitals 华西 自贡 苍溪 南充 \
  --expected_hospitals 4 \
  --organs 食管 胃 十二指肠 \
  --expected_organs 3 \
  --organ_support \
    "食管=${SUPPORT_ESOPHAGUS}" \
    "胃=${SUPPORT_STOMACH}" \
    "十二指肠=${SUPPORT_DUODENUM}" \
  --organ_f1_checkpoint \
    "食管=${F1_ESOPHAGUS}" \
    "胃=${F1_STOMACH}" \
    "十二指肠=${F1_DUODENUM}" \
  --organ_m2_checkpoint \
    "食管=${M2_ESOPHAGUS}" \
    "胃=${M2_STOMACH}" \
    "十二指肠=${M2_DUODENUM}" \
  --n_shot "${N_SHOT}" \
  --episodes "${EPISODES}" \
  --seed "${SEED}" \
  --project_root "${ROOT}" \
  --dit_checkpoint "${DIT_CHECKPOINT}" \
  --output_json "${OUTPUT_DIR}/summary_seed${SEED}.json" \
  --metric_device cuda:0 \
  --flow_device cuda:0 \
  --rae_device cuda:0 \
  --episode_batch_size "${EPISODE_BATCH_SIZE}" \
  --pair_batch_size "${PAIR_BATCH_SIZE}" \
  --rae_chunk_size "${RAE_CHUNK_SIZE}" \
  --save_episode_details \
  --log_cuda_memory
