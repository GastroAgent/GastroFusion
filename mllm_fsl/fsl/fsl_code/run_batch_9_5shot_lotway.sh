#!/bin/bash
#SBATCH -J tsy_rae_metric_waiyuan
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/slurm_out/rae_metric_waiyuan_lot_way-%j.out

# Batch run RAE Metric + Flow FSL on 9 (医院, 部位) combinations.
#
# Protocol:
#   - Use ALL support pairs provided by the existing flat JSON
#   - No N-shot cap
#   - No episode resampling
#   - Primary class-level support aggregation = min
#
# Directory:
# result/offline_align_rae_metric_flow_service_allsupport/
#   mllm_fsl_subset_agent_lot_way/
#     医院/
#       部位/

set -o pipefail


# ============================================================
# Paths
# ============================================================

ROOT="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/data_tsy_deepemd/mllm_fsl_subset_agent_lot_way_91"

RESULT_ROOT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/result_831_6"

# 修改为新的 all-support RAE Python
FLOW_SCRIPT="/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/offline_align_rae_metric_flow_service_new_test_train.py"


# ============================================================
# Experiment settings
# ============================================================

HOSPITALS=(苍溪 南充 自贡)
ORGANS=(十二指肠 食管 胃)

# Support protocol:
# JSON 中每个 query × candidate class 有多少 support，
# 就全部使用，不再进行 1-shot / 5-shot 截断。
#
# Class-level aggregation:
# min support distance


# ============================================================
# GPU selection
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

echo "Using GPU index: ${GPU}"


# ============================================================
# Environment
# ============================================================

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:$LD_LIBRARY_PATH
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:$PATH


cd /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1


# ============================================================
# Log directory
# ============================================================

LOG_DIR="${RESULT_ROOT}/_logs/offline_align_rae_metric_flow_service_allsupport"
mkdir -p "${LOG_DIR}"


# ============================================================
# Batch evaluation
# ============================================================

for H in "${HOSPITALS[@]}"; do

  for O in "${ORGANS[@]}"; do

    JSON="${ROOT}/${H}/${O}/final_doctor_exam_${H}_${O}_huaxi_train_support_flat.json"

    if [[ ! -f "${JSON}" ]]; then
      echo "[SKIP] data not found: ${JSON}"
      continue
    fi


    OUT_FLOW="${RESULT_ROOT}/mllm_fsl_subset_agent_lot_way_91/${H}/${O}"

    mkdir -p "${OUT_FLOW}"


    # --------------------------------------------------------
    # Skip completed datasets
    # --------------------------------------------------------

    if [[ -f "${OUT_FLOW}/summary.json" ]]; then

      echo "[FLOW DONE] ${H}/${O}"

    else

      echo "===================================================================="
      echo "[FLOW START] ${H}/${O}"
      echo "Time        : $(date)"
      echo "Input       : ${JSON}"
      echo "Output      : ${OUT_FLOW}"
      echo "Support     : ALL supports from JSON"
      echo "Support agg : min"
      echo "GPU         : ${GPU}"
      echo "===================================================================="


      CUDA_VISIBLE_DEVICES="${GPU}" \
      python "${FLOW_SCRIPT}" \
        --data_path "${JSON}" \
        --output_dir "${OUT_FLOW}" \
        --device cuda:0 \
        --flow_device same \
        --log_every 25 \
        2>&1 \
        | tee "${LOG_DIR}/flow_${H}_${O}.log" \
        | tail -40


      STATUS=${PIPESTATUS[0]}

      if [[ ${STATUS} -eq 0 ]]; then
        echo "[FLOW END] ${H}/${O} SUCCESS  $(date)"
      else
        echo "[FLOW ERROR] ${H}/${O} exit_code=${STATUS}  $(date)"
      fi

    fi

  done

done


echo "===================================================================="
echo "ALL FLOW JOBS DONE at $(date)"
echo "===================================================================="