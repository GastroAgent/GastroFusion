#!/bin/bash
#SBATCH -J merge
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/logs/%x-%j.out

set -e

# =============================================================================
# [配置区] -- 按需修改
# =============================================================================
# export CUDA_VISIBLE_DEVICES=1

# 基座模型路径
BASE_MODEL="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/weights/Qwen3-VL-8B-Instruct-RL1-last/Qwen3-VL-8B-Instruct"

# LoRA 权重目录（checkpoint-xxx，即 swift 训练产物目录）
LORA_CKPT="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/output/medical-vqa-grpo-qwen3vl8b-colocate-172219/last"

# 融合后输出目录（必填，需为空或不存在）
OUTPUT_DIR="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/weights/Qwen3-VL-8B-Instruct-RL2-last/Qwen3-VL-8B-Instruct"

# 融合精度: bfloat16 / float16 / float32（默认与训练一致用 bfloat16）
DTYPE="float32"

# =============================================================================
# 命令行参数覆盖配置区:  <base_model> <lora_ckpt> <output_dir>
# 传 "" 表示保留配置区默认值
# =============================================================================
if [[ -n "${1:-}" ]]; then BASE_MODEL="$1"; fi
if [[ -n "${2:-}" ]]; then LORA_CKPT="$2"; fi
if [[ -n "${3:-}" ]]; then OUTPUT_DIR="$3"; fi

# =============================================================================
# ENV -- 复用 sft.sh 的环境初始化
# =============================================================================

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate agentic_crop
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/bin:${PATH:-}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 融合只需要一张卡，避免占用其它卡的显存
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# =============================================================================
# SANITY CHECK
# =============================================================================

echo "========================================================================"
echo "[MERGE LORA CONFIG]"
echo
echo "BASE_MODEL   = ${BASE_MODEL}"
echo "LORA_CKPT    = ${LORA_CKPT}"
echo "OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "DTYPE        = ${DTYPE}"
echo "GPU          = ${CUDA_VISIBLE_DEVICES}"
echo "========================================================================"

# 必填检查
if [[ -z "${BASE_MODEL}" ]]; then
    echo "[ERROR] BASE_MODEL 为空，请配置或在命令行传入。" >&2
    exit 1
fi
if [[ -z "${LORA_CKPT}" ]]; then
    echo "[ERROR] LORA_CKPT 为空，请配置或在命令行传入。" >&2
    exit 1
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "[ERROR] OUTPUT_DIR 为空，请配置或在命令行传入。" >&2
    exit 1
fi

# 路径存在性检查
for p in "${BASE_MODEL}" "${LORA_CKPT}"; do
    if [[ ! -d "${p}" ]]; then
        echo "[ERROR] 路径不存在或不是目录:" >&2
        echo "        ${p}" >&2
        exit 1
    fi
done

if [[ -e "${OUTPUT_DIR}" && -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
    echo "[ERROR] OUTPUT_DIR 已存在且非空，为避免覆盖请换一个目录:" >&2
    echo "        ${OUTPUT_DIR}" >&2
    exit 1
fi

# =============================================================================
# RUN
# =============================================================================

echo
echo "[INFO] 开始融合 LoRA 权重 ..."
echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S')"

python /mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/merge_lora.py \
    --model "${BASE_MODEL}" \
    --lora_ckpt "${LORA_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --dtype "${DTYPE}"

EXIT_CODE=$?

# =============================================================================
# POST CHECK
# =============================================================================

if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "[ERROR] 融合失败，退出码 ${EXIT_CODE}" >&2
    exit ${EXIT_CODE}
fi

echo
echo "========================================================================"
echo "[POST CHECK] 输出目录内容:"
echo "========================================================================"
ls -lh "${OUTPUT_DIR}"

if [[ -f "${OUTPUT_DIR}/config.json" ]] && \
   ls "${OUTPUT_DIR}"/model*.safetensors >/dev/null 2>&1; then
    echo
    echo "✅ 融合成功！"
    echo "   输出目录: ${OUTPUT_DIR}"
    echo "   该目录可直接作为 --model 用于后续 swift rlhf / 推理。"
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo
    echo "[ERROR] 校验未通过：output 目录缺少 config.json 或 model*.safetensors" >&2
    exit 1
fi