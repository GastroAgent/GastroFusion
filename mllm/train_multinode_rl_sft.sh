#!/bin/bash
#SBATCH --job-name=rl-Qwen3-VL-8B-multinode-sft-lr1e-5-a32-r8-t1.0-05041230
#SBATCH --partition=gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=16
#SBATCH --time=5-00:00:00
#SBATCH --output=/mnt/inaisfs/data/home/tansy_criait/jmf/wgw/think_with_image_zh_swift/logs/%x-%j.out

cd /mnt/inaisfs/data/home/tansy_criait/jmf/wgw/think_with_image_zh_swift

# 修改1：统一设置总节点数和每节点进程数
nnodes="${SLURM_NNODES}"
nproc_per_node=8
export NPROC_PER_NODE=8
export NCCL_SOCKET_IFNAME=ibp20s0
export GLOO_SOCKET_IFNAME=ibp20s0
export NCCL_SOCKET_FAMILY=AF_INET

# 修改2：自动解析主节点地址
MASTER_ADDR=$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=29500

echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${nnodes}"
echo "NPROC_PER_NODE=${nproc_per_node}"

export VLLM_SERVER_HOST="172.21.0.93 172.21.0.93 172.21.0.93 172.21.0.93 172.21.0.95 172.21.0.95 172.21.0.95 172.21.0.95"
export VLLM_SERVER_PORT="8811 8822 8833 8844 8855 8866 8877 8888"


# ===== W&B 配置 =====
export WANDB_API_KEY="939c13ebbba31a8d087ac8e9c47ed158d6e9e160"
export WANDB_PROJECT="agentic-swift-gspo-qwen3-vl"
export WANDB_DIR="/mnt/inaisfs/data/home/tansy_criait/jmf/wgw/think_with_image_zh_swift/wandb"
# export WANDB_RUN_ID=pqi05o03
export WANDB_RESUME=allow
# export WANDB_NAME="rl-Qwen3-VL-8B-multinode-124633"
export WANDB_NAME="${SLURM_JOB_NAME}"

# export SWANLAB_API_KEY="tZVlXFDUaQqfVHnrL6nUc"
export SAVE_CKPT_PATH="./output/$WANDB_NAME"
# export SAVE_CKPT_PATH="/mnt/inaisfs/data/home/tansy_criait/jmf/wgw/think_with_image_zh_swift/output/rl-Qwen3-VL-8B-multinode-124633/v0-20260402-101333"

# 修改3：所有节点执行同一个 swift sft 文件，只是 NODE_RANK 不同
srun \
  --nodes="${nnodes}" \
  --ntasks="${nnodes}" \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  bash -c '
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export NNODES='"${nnodes}"'
    export NODE_RANK=${SLURM_NODEID}
    export MASTER_ADDR='"${MASTER_ADDR}"'
    export MASTER_PORT='"${MASTER_PORT}"'
    export LOG_REWARD_COMPONENTS=1
    export REWARD_LOG_INTERVAL=1

    cd /mnt/inaisfs/data/home/tansy_criait/jmf/wgw/think_with_image_zh_swift
    source /mnt/inaisfs/data/apps/cuda12.4.sh
    source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate agentic_crop

    bash ./rl_sft.sh
  '