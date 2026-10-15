#!/bin/bash
#SBATCH -J sft
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=gpu:4
#SBATCH --output=/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/logs/%x-%j.out

cd /mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift

# 修改1：统一设置总节点数和每节点进程数
nnodes="${SLURM_NNODES}"
nproc_per_node=4
export NCCL_SOCKET_IFNAME=ibp20s0
export GLOO_SOCKET_IFNAME=ibp20s0

# 修改2：自动解析主节点地址
MASTER_ADDR=$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=29500

echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${nnodes}"
echo "NPROC_PER_NODE=${nproc_per_node}"

# 修改3：所有节点执行同一个 swift sft 文件，只是 NODE_RANK 不同
srun \
  --nodes="${nnodes}" \
  --ntasks="${nnodes}" \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  bash -c '
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export NNODES='"${nnodes}"'
    export NODE_RANK=${SLURM_NODEID}
    export MASTER_ADDR='"${MASTER_ADDR}"'
    export MASTER_PORT='"${MASTER_PORT}"'
    export NPROC_PER_NODE='"${nproc_per_node}"'

    bash /mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/sft.sh
  '