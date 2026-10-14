#!/bin/bash
#SBATCH -J tsy_f1v5_waiyuan_fewway
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/slurm_out/f1v5_waiyuan_fewway-%j.out

# 9 external centre-organ datasets, few-way / TEST-TEST protocol.
# Existing flat JSON candidate/support structure is retained; FSL inference is
# replaced by adaptive(F1-v5) from the new test-test eval implementation.

set -o pipefail

ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/data_tsy_deepemd/mllm_fsl_subset_agent_few_way_91"
RESULT_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/result_831_6"

PROJECT_ROOT="/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow"
CODE_DIR="${PROJECT_ROOT}/fsl_multi_vae/rae_flow_metric_v1"
FLOW_SCRIPT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/offline_align_f1v5_adaptive_test_test.py"
DIT_CHECKPOINT="${PROJECT_ROOT}/weights_dit_flow_v1/vision_hint_inclass/otcfm_dit_L_step_265000.pt"

HOSPITALS=(苍溪 南充 自贡)
ORGANS=(十二指肠 食管 胃)

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

LOG_DIR="${RESULT_ROOT}/_logs/adaptive_f1v5_fewway_testtest"
mkdir -p "${LOG_DIR}"

[[ -f "${FLOW_SCRIPT}" ]] || { echo "[ERROR] missing FLOW_SCRIPT: ${FLOW_SCRIPT}"; exit 2; }
[[ -f "${DIT_CHECKPOINT}" ]] || { echo "[ERROR] missing DiT: ${DIT_CHECKPOINT}"; exit 2; }

for H in "${HOSPITALS[@]}"; do
  for O in "${ORGANS[@]}"; do
    JSON="${ROOT}/${H}/${O}/final_doctor_exam_${H}_${O}_huaxi_train_support_flat.json"
    [[ -f "${JSON}" ]] || { echo "[SKIP] data not found: ${JSON}"; continue; }

    # EXACTLY follow the new TEST-TEST eval weights.
    F1_CHECKPOINT="${PROJECT_ROOT}/weights_metric_v1/f1_v5_support_lstm_medical/test-test/${O}/step_to_support/f1_dynamic_decay_best.pt"
    M2_CHECKPOINT="${PROJECT_ROOT}/weights_metric_v1/new_m2_${O}-test-test/metric_direct_episode.pt"

    [[ -f "${F1_CHECKPOINT}" ]] || { echo "[ERROR] missing F1: ${F1_CHECKPOINT}"; exit 2; }
    [[ -f "${M2_CHECKPOINT}" ]] || { echo "[ERROR] missing M2: ${M2_CHECKPOINT}"; exit 2; }

    OUT_FLOW="${RESULT_ROOT}/mllm_fsl_subset_agent_few_way_91_adaptive_f1v5/${H}/${O}"
    mkdir -p "${OUT_FLOW}"

    if [[ -f "${OUT_FLOW}/summary_adaptive_f1v5.json" ]]; then
      echo "[ADAPTIVE DONE] ${H}/${O}"
      continue
    fi

    echo "===================================================================="
    echo "[ADAPTIVE F1-v5 START] ${H}/${O}"
    echo "Input       : ${JSON}"
    echo "Output      : ${OUT_FLOW}"
    echo "F1          : ${F1_CHECKPOINT}"
    echo "M2          : ${M2_CHECKPOINT}"
    echo "Protocol    : TEST-TEST / existing flat JSON / ${N_SHOT}-shot"
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
      2>&1 | tee "${LOG_DIR}/adaptive_${H}_${O}.log" | tail -60

    STATUS=${PIPESTATUS[0]}
    if [[ ${STATUS} -eq 0 ]]; then
      echo "[ADAPTIVE F1-v5 END] ${H}/${O} SUCCESS $(date)"
    else
      echo "[ADAPTIVE F1-v5 ERROR] ${H}/${O} exit_code=${STATUS} $(date)"
      exit ${STATUS}
    fi
  done
done

echo "===================================================================="
echo "ALL FEW-WAY ADAPTIVE F1-v5 JOBS DONE at $(date)"
echo "===================================================================="
