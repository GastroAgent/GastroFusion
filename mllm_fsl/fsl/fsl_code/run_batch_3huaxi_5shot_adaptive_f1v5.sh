#!/bin/bash
#SBATCH -J tsy_f1v5_huaxi
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/f1v5_huaxi-%j.out

# Huaxi 3-organ flat-JSON evaluation.
# Keep the existing JSON/query/support structure but use the adaptive(F1-v5)
# model and TEST-TRAIN weight loading policy from the new eval files.

set -o pipefail

DATA_ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/data_tsy_deepemd"
SUPPORT_ROOT="${DATA_ROOT}/huaxi_mllm_fsl_agent/support_831_val"
RESULT_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/result_831_val"
OUTPUT_BASE="${RESULT_ROOT}/offline_align_rae_metric_huaxi_adaptive_f1v5_new"

PROJECT_ROOT="/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow"
CODE_DIR="${PROJECT_ROOT}/fsl_multi_vae/rae_flow_metric_v1"
FLOW_SCRIPT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/offline_align_f1v5_adaptive_test_train.py"
DIT_CHECKPOINT="${PROJECT_ROOT}/weights_dit_flow_v1/vision_hint_inclass/otcfm_dit_L_step_265000.pt"

ORGANS=("十二指肠" "食管" "胃")
N_SHOT=${N_SHOT:-5}
FIXED_DECAY=${FIXED_DECAY:-0.90}
SEED=${SEED:-42}
EPISODE_BATCH_SIZE=${EPISODE_BATCH_SIZE:-4}
PAIR_BATCH_SIZE=${PAIR_BATCH_SIZE:-16}
RAE_CHUNK_SIZE=${RAE_CHUNK_SIZE:-32}

GPU="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${GPU}" ]]; then
  if [[ "${USE_AUTO_GPU:-1}" == "1" ]]; then
    GPU="$(nvidia-smi --query-gpu=memory.free,index --format=csv,noheader,nounits \
      | awk -F, '{printf "%d %s\n",$1,$2}' | sort -nr | head -1 | awk '{print $2}')"
  else
    GPU=0
  fi
fi

echo "[INFO] Using GPU index: ${GPU}"
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}
export PYTHONUNBUFFERED=1
cd "${CODE_DIR}"

[[ -d "${DATA_ROOT}" ]] || { echo "[ERROR] DATA_ROOT 不存在：${DATA_ROOT}"; exit 1; }
[[ -d "${SUPPORT_ROOT}" ]] || { echo "[ERROR] SUPPORT_ROOT 不存在：${SUPPORT_ROOT}"; exit 1; }
[[ -f "${FLOW_SCRIPT}" ]] || { echo "[ERROR] FLOW_SCRIPT 不存在：${FLOW_SCRIPT}"; exit 1; }
[[ -f "${DIT_CHECKPOINT}" ]] || { echo "[ERROR] DIT 不存在：${DIT_CHECKPOINT}"; exit 1; }

LOG_DIR="${RESULT_ROOT}/_logs/adaptive_f1v5_huaxi_testtrain"
mkdir -p "${LOG_DIR}" "${OUTPUT_BASE}"

for O in "${ORGANS[@]}"; do
  mapfile -t MATCHES < <(find "${SUPPORT_ROOT}" -maxdepth 1 -type f -name "*${O}*.json" | sort)
  N_MATCH=${#MATCHES[@]}
  if [[ ${N_MATCH} -eq 0 ]]; then
    echo "[ERROR] ${O}: 没找到 JSON in ${SUPPORT_ROOT}"
    exit 1
  fi
  if [[ ${N_MATCH} -gt 1 ]]; then
    echo "[ERROR] ${O}: 找到多个 JSON："
    printf '  %s\n' "${MATCHES[@]}"
    exit 1
  fi

  JSON="${MATCHES[0]}"
  F1_CHECKPOINT="${PROJECT_ROOT}/weights_metric_v1/f1_v5_support_lstm_medical/test-test/${O}/step_to_support/f1_dynamic_decay_best.pt"
  M2_CHECKPOINT="${PROJECT_ROOT}/weights_metric_v1/m2_${O}-test-train/metric_direct_episode.pt"

  [[ -f "${F1_CHECKPOINT}" ]] || { echo "[ERROR] missing F1: ${F1_CHECKPOINT}"; exit 2; }
  [[ -f "${M2_CHECKPOINT}" ]] || { echo "[ERROR] missing M2: ${M2_CHECKPOINT}"; exit 2; }

  OUT_FLOW="${OUTPUT_BASE}/${O}"
  mkdir -p "${OUT_FLOW}"
  if [[ -f "${OUT_FLOW}/summary_adaptive_f1v5.json" ]]; then
    echo "[ADAPTIVE DONE] ${O} -> ${OUT_FLOW}/summary_adaptive_f1v5.json"
    continue
  fi

  echo "===================================================================="
  echo "[ADAPTIVE F1-v5 START] ${O}"
  echo "Input       : ${JSON}"
  echo "Output      : ${OUT_FLOW}"
  echo "F1          : ${F1_CHECKPOINT}"
  echo "M2          : ${M2_CHECKPOINT}"
  echo "Protocol    : TEST-TRAIN / ${N_SHOT}-shot"
  echo "Primary     : adaptive(F1-v5)"
  echo "Fixed decay : ${FIXED_DECAY} (diagnostic only)"
  echo "GPU         : ${GPU}"
  echo "===================================================================="

  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u "${FLOW_SCRIPT}" \
    --data_path "${JSON}" \
    --output_dir "${OUT_FLOW}" \
    --eval_code_dir "${CODE_DIR}" \
    --project_root "${PROJECT_ROOT}" \
    --f1_checkpoint "${F1_CHECKPOINT}" \
    --m2_checkpoint "${M2_CHECKPOINT}" \
    --dit_checkpoint "${DIT_CHECKPOINT}" \
    --n_shot "${N_SHOT}" \
    --seed "${SEED}" \
    --fixed_decay "${FIXED_DECAY}" \
    --episode_batch_size "${EPISODE_BATCH_SIZE}" \
    --pair_batch_size "${PAIR_BATCH_SIZE}" \
    --rae_chunk_size "${RAE_CHUNK_SIZE}" \
    --metric_device cuda:0 \
    --flow_device cuda:0 \
    --rae_device cuda:0 \
    --log_every 25 \
    2>&1 | tee "${LOG_DIR}/adaptive_${O}.log" | tail -60

  STATUS=${PIPESTATUS[0]}
  if [[ ${STATUS} -eq 0 ]]; then
    echo "[ADAPTIVE F1-v5 END] ${O} SUCCESS $(date)"
  else
    echo "[ADAPTIVE F1-v5 ERROR] ${O} exit_code=${STATUS}"
    exit ${STATUS}
  fi
done

echo "===================================================================="
echo "ALL 3 HUAXI ADAPTIVE F1-v5 JOBS DONE at $(date)"
echo "RESULT ROOT = ${OUTPUT_BASE}"
echo "===================================================================="
