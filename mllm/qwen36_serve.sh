#!/bin/bash
#SBATCH --job-name=Qwen3.6-27B_vllm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/inaisfs/data/home/tansy_criait/jmf/wgw/agentic_swift/openai_serve/logs/%x-%j.out

set -euo pipefail

export SGLANG_DISABLE_CUDNN_CHECK=1

# ====== 1) 环境准备 ======
cd /mnt/inaisfs/data/home/tansy_criait/jmf/wgw/agentic_swift/openai_serve
mkdir -p logs

# 如果你用 conda
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate openai_serve
# source /mnt/inaisfs/data/home/tansy_criait/jmf/wgw/big_paper/openai_serve/cuda124.sh



# 可选：更稳的 NCCL 配置（集群常见）
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

# ====== 2) 关键参数 ======
MODEL="Qwen3.6-27B"
TP=8
PORT=8811
MODEL_PATH="/mnt/inaisfs/data/home/tansy_criait/weights/${MODEL}"

# 绑定到本机可路由 IP（推荐）
HOST="$(hostname -I | awk '{print $1}')"

echo "Node: $(hostname)"
echo "Host IP: ${HOST}"
echo "Model: ${MODEL}"
echo "TP: ${TP}"
echo "Port: ${PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# ====== 3) 启动服务 ======
srun --cpu-bind=none \
vllm serve ${MODEL_PATH} \
    --trust-remote-code \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8811 \
    --limit-mm-per-prompt.video 0 \
    --served-model-name ${MODEL} \
    -tp 8 \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --mm_processor_cache_type shm 

  
