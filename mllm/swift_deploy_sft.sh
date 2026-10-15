#!/bin/bash
#SBATCH --job-name=sft-swift-deploy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=16
#SBATCH --time=5-00:00:00
#SBATCH --output=/mnt/inaisfs/data/home/tansy_criait/jmf/wgw/think_with_image_zh_swift/logs/%x-%j.out

cd /mnt/inaisfs/data/home/tansy_criait/jmf/wgw/think_with_image_zh_swift

export NO_PROXY=localhost,127.0.0.1,::1,172.21.0.0/16
export no_proxy="$NO_PROXY"
export NCCL_SOCKET_IFNAME=ibp20s0
export GLOO_SOCKET_IFNAME=ibp20s0

source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate agentic_crop

export CUDA_VISIBLE_DEVICES=0,1

HOST="$(hostname -I | awk '{print $1}')"

export VLLM_SERVER_GROUP_HOST=${HOST}
export VLLM_SERVER_GROUP_PORT=51316


VLLM_HOST=0.0.0.0
VLLM_PORT=8888
MODEL=Qwen3-VL-8B-Instruct
TP=2

HOST="$(hostname -I | awk '{print $1}')"

echo "Node: $(hostname)"
echo "Host IP: ${HOST}"
echo "Model: ${MODEL}"
echo "Port: ${VLLM_PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "tensor parallel size: ${TP}"

srun --cpu-bind=none \
swift rollout \
    --model /mnt/inaisfs/data/home/tansy_criait/jmf/wgw/think_with_image_zh_swift/output/Qwen3-VL-8B-Instruct-3epoch-sft/v0-20260401-104311/checkpoint-1128-merged \
    --host ${VLLM_HOST} \
    --port ${VLLM_PORT} \
    --external_plugins /mnt/inaisfs/data/home/tansy_criait/jmf/wgw/think_with_image_zh_swift/agentic_crop_plugin_async.py \
    --vllm_use_async_engine true \
    --multi_turn_scheduler doc_crop_scheduler \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_limit_mm_per_prompt '{"image": 6, "video": 0}' \
    --max_turns 5 \
    --vllm_tensor_parallel_size ${TP} \
    --served_model_name ${MODEL} \
    --system /mnt/inaisfs/data/home/tansy_criait/jmf/wgw/think_with_image_zh_swift/data/sys_prompt.txt \
    --padding_free true \
    --attn_impl flash_attn \
    --vllm_enable_prefix_caching true 
