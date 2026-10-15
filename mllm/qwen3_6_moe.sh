#!/bin/bash
#SBATCH --job-name=vllm_qwen3.6_moe_serve
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=5-00:00:00
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/wgw/think_with_image_zh_swift/logs/%x-%j.out

VLLM_PORT=9911

CURRENT_IP=$(hostname -I | awk '{print $1}')
echo "CURRENT_IP: $CURRENT_IP"
echo "VLLM_PORT: $VLLM_PORT"

source /mnt/inaisfs/data/apps/cuda12.4.sh

source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate big_paper

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
unset http_proxy
unset https_proxy
unset all_proxy

export NO_PROXY=localhost,127.0.0.1,172.16.0.89,172.21.0.89,172.16.,172.21.,gpu33,gpu90
export no_proxy=$NO_PROXY

vllm serve /mnt/inaisfs/data/home/tansy_criait/weights/Qwen3.6-35B-A3B\
    --trust-remote-code \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --served-model-name Qwen3.6-35B-A3B \
    -tp 8 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --reasoning-parser qwen3 \
    --mm-encoder-tp-mode data