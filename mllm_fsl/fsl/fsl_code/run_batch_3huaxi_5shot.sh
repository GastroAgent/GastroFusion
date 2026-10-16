#!/bin/bash
#SBATCH -J tsy_rae_metric_huaxi
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/rae_metric_huaxi-%j.out

# ============================================================
# Batch run RAE Metric + Flow FSL on Huaxi-support 3 JSONs
#
# 只需要指定：
#   /mnt/.../data_tsy_deepemd
#
# 自动寻找：
#   data_tsy_deepemd/
#     huaxi_mllm_fsl_agent/
#       support/
#         *十二指肠*.json
#         *食管*.json
#         *胃*.json
#
# Protocol:
#   - 使用 JSON 中 ALL support pairs
#   - No N-shot cap
#   - No episode resampling
#   - Class-level support aggregation = min
# ============================================================

set -o pipefail


# ============================================================
# 1. 你只需要指定到这里
# ============================================================

DATA_ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/data_tsy_deepemd"


# ============================================================
# 2. 自动拼接输入目录
# ============================================================

SUPPORT_ROOT="${DATA_ROOT}/huaxi_mllm_fsl_agent/support_831_val"


# ============================================================
# 3. 输出路径
# ============================================================

RESULT_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/result_831_val"

OUTPUT_BASE="${RESULT_ROOT}/offline_align_rae_metric_huaxi_new"


# ============================================================
# 4. RAE Flow Python
# ============================================================

FLOW_SCRIPT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/offline_align_rae_metric_flow_service_new_test_train.py"


# ============================================================
# 5. 三个器官
# ============================================================

ORGANS=(
  "十二指肠"
  "食管"
  "胃"
)


# ============================================================
# 6. GPU selection
# ============================================================

GPU="${CUDA_VISIBLE_DEVICES:-}"

if [[ -z "${GPU}" ]]; then

  if [[ "${USE_AUTO_GPU:-1}" == "1" ]]; then

    GPU="$(
      nvidia-smi \
        --query-gpu=memory.free,index \
        --format=csv,noheader,nounits \
      | awk -F, '{printf "%d %s\n",$1,$2}' \
      | sort -nr \
      | head -1 \
      | awk '{print $2}'
    )"

  else
    GPU=0
  fi

fi

echo "[INFO] Using GPU index: ${GPU}"


# ============================================================
# 7. Environment
# ============================================================

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:$LD_LIBRARY_PATH
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:$PATH

cd /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1


# ============================================================
# 8. Check
# ============================================================

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] DATA_ROOT 不存在：${DATA_ROOT}"
  exit 1
fi

if [[ ! -d "${SUPPORT_ROOT}" ]]; then
  echo "[ERROR] support 目录不存在：${SUPPORT_ROOT}"
  exit 1
fi

if [[ ! -f "${FLOW_SCRIPT}" ]]; then
  echo "[ERROR] Python 不存在：${FLOW_SCRIPT}"
  exit 1
fi


# ============================================================
# 9. Log
# ============================================================

LOG_DIR="${RESULT_ROOT}/_logs/offline_align_rae_metric_flow_service_819_huaxi_support"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_BASE}"


echo "===================================================================="
echo "[INFO] Huaxi support 3-JSON RAE evaluation"
echo "[INFO] DATA_ROOT    = ${DATA_ROOT}"
echo "[INFO] SUPPORT_ROOT = ${SUPPORT_ROOT}"
echo "[INFO] OUTPUT_BASE  = ${OUTPUT_BASE}"
echo "[INFO] FLOW_SCRIPT  = ${FLOW_SCRIPT}"
echo "[INFO] GPU          = ${GPU}"
echo "[INFO] START        = $(date)"
echo "===================================================================="


# ============================================================
# 10. 逐器官自动寻找 JSON
# ============================================================

for O in "${ORGANS[@]}"; do

  echo
  echo "===================================================================="
  echo "[SEARCH] Organ = ${O}"
  echo "===================================================================="

  # ----------------------------------------------------------
  # 在 support/ 第一层寻找：
  #
  #   *十二指肠*.json
  #   *食管*.json
  #   *胃*.json
  #
  # 每个器官必须严格只有 1 个 JSON。
  # ----------------------------------------------------------

  mapfile -t MATCHES < <(
    find "${SUPPORT_ROOT}" \
      -maxdepth 1 \
      -type f \
      -name "*${O}*.json" \
      | sort
  )

  N_MATCH=${#MATCHES[@]}

  if [[ ${N_MATCH} -eq 0 ]]; then
    echo "[ERROR] ${O}: 没找到 JSON"
    echo "[ERROR] 搜索目录：${SUPPORT_ROOT}"
    exit 1
  fi

  if [[ ${N_MATCH} -gt 1 ]]; then
    echo "[ERROR] ${O}: 找到多个 JSON，无法确定使用哪一个："

    for P in "${MATCHES[@]}"; do
      echo "        ${P}"
    done

    exit 1
  fi


  JSON="${MATCHES[0]}"

  echo "[MATCH] ${O}"
  echo "        ${JSON}"


  # ==========================================================
  # Output
  # ==========================================================

  OUT_FLOW="${OUTPUT_BASE}/${O}"

  mkdir -p "${OUT_FLOW}"


  # ==========================================================
  # 已完成则跳过
  # ==========================================================

  if [[ -f "${OUT_FLOW}/summary.json" ]]; then

    echo "[FLOW DONE] ${O}"
    echo "[RESULT] ${OUT_FLOW}/summary.json"
    continue

  fi


  # ==========================================================
  # Run
  # ==========================================================

  echo
  echo "--------------------------------------------------------------------"
  echo "[FLOW START]"
  echo "Organ       : ${O}"
  echo "Time        : $(date)"
  echo "Input       : ${JSON}"
  echo "Output      : ${OUT_FLOW}"
  echo "Support     : ALL supports from JSON"
  echo "Support agg : min"
  echo "GPU         : ${GPU}"
  echo "--------------------------------------------------------------------"


  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u "${FLOW_SCRIPT}" \
    --data_path "${JSON}" \
    --output_dir "${OUT_FLOW}" \
    --device cuda:0 \
    --flow_device same \
    --log_every 25 \
    2>&1 \
    | tee "${LOG_DIR}/flow_${O}.log" \
    | tail -40


  STATUS=${PIPESTATUS[0]}


  if [[ ${STATUS} -eq 0 ]]; then

    echo "[FLOW END] ${O} SUCCESS"
    echo "[RESULT]   ${OUT_FLOW}"
    echo "[TIME]     $(date)"

  else

    echo "[FLOW ERROR] ${O}"
    echo "[EXIT CODE] ${STATUS}"
    exit ${STATUS}

  fi

done


echo
echo "===================================================================="
echo "ALL 3 HUAXI-SUPPORT FLOW JOBS DONE"
echo "END = $(date)"
echo "===================================================================="

echo
echo "[RESULT ROOT]"
echo "${OUTPUT_BASE}"
echo