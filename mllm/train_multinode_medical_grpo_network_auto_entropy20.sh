#!/bin/bash
#SBATCH -J RL
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --gres=gpu:8
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/logs/%x-%j.out


# =============================================================================
# USER CONFIG
# =============================================================================
PROJECT_DIR="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift"

GPUS_PER_NODE=8

# 网络接口：
#   auto  -> 不强制指定 NCCL/Gloo 网卡，使用框架自动选择（推荐默认）
#   ib0   -> 只有确认每个节点都存在该接口且有 IPv4 时才手工指定
#   eth0  -> 同上
NET_IFACE="auto"

CUDA_ENV_SCRIPT="/mnt/inaisfs/data/apps/cuda12.4.sh"
CONDA_ACTIVATE="/mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate"
CONDA_ENV="agentic_crop"

RUN_SCRIPT="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/rl_medical_grpo_colocate_fixed_v2_entropy20.sh"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# =============================================================================
# SLURM / DISTRIBUTED
# =============================================================================
NNODES="${SLURM_NNODES}"
NPROC_PER_NODE="${GPUS_PER_NODE}"

MASTER_ADDR="$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

export PROJECT_DIR
export GPUS_PER_NODE
export NET_IFACE
export CUDA_ENV_SCRIPT
export CONDA_ACTIVATE
export CONDA_ENV
export RUN_SCRIPT

export NNODES
export NPROC_PER_NODE
export MASTER_ADDR
export MASTER_PORT

export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "========================================================================"
echo "[SLURM]"
echo "JOB_ID          = ${SLURM_JOB_ID}"
echo "JOB_NAME        = ${SLURM_JOB_NAME}"
echo "NODELIST        = ${SLURM_JOB_NODELIST}"
echo "NNODES          = ${NNODES}"
echo "GPUS_PER_NODE   = ${GPUS_PER_NODE}"
echo "WORLD_SIZE      = $((NNODES * GPUS_PER_NODE))"
echo "MASTER_ADDR     = ${MASTER_ADDR}"
echo "MASTER_PORT     = ${MASTER_PORT}"
echo "NET_IFACE       = ${NET_IFACE}"
echo "RUN_SCRIPT      = ${RUN_SCRIPT}"
echo "========================================================================"

if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "[ERROR] RUN_SCRIPT does not exist:"
    echo "        ${RUN_SCRIPT}"
    exit 1
fi

# =============================================================================
# Build a node-local launcher script on the shared filesystem.
# This avoids fragile nested bash -lc quoting.
# =============================================================================
NODE_RUNNER="${PROJECT_DIR}/logs/.grpo_node_runner_${SLURM_JOB_ID}.sh"

cat > "${NODE_RUNNER}" <<'NODE_EOF'
#!/bin/bash


NODE_RANK="${SLURM_NODEID:-0}"
export NODE_RANK

# Expose exactly NPROC_PER_NODE local GPUs: 0,1,...,N-1
GPU_LIST=""
for ((i=0; i<NPROC_PER_NODE; i++)); do
    if [[ -n "${GPU_LIST}" ]]; then
        GPU_LIST+=","
    fi
    GPU_LIST+="${i}"
done
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

source "${CUDA_ENV_SCRIPT}"
source "${CONDA_ACTIVATE}" "${CONDA_ENV}"

export LD_LIBRARY_PATH="/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/lib:${LD_LIBRARY_PATH:-}"
export PATH="/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/agentic_crop/bin:${PATH:-}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# -----------------------------------------------------------------------------
# Network interface selection
# -----------------------------------------------------------------------------
if [[ -z "${NET_IFACE:-}" || "${NET_IFACE}" == "auto" ]]; then
    # Let NCCL/Gloo perform their own interface selection.
    unset NCCL_SOCKET_IFNAME || true
    unset GLOO_SOCKET_IFNAME || true
    EFFECTIVE_NET_IFACE="<auto>"
else
    if ! ip link show dev "${NET_IFACE}" >/dev/null 2>&1; then
        echo "[ERROR] Requested NET_IFACE does not exist on $(hostname): ${NET_IFACE}"
        echo "[NETWORK] Available interfaces:"
        ip -o link show 2>/dev/null || true
        exit 1
    fi

    if ! ip -o -4 addr show dev "${NET_IFACE}" scope global 2>/dev/null | grep -q "inet "; then
        echo "[ERROR] Requested NET_IFACE has no global IPv4 address on $(hostname): ${NET_IFACE}"
        echo "[NETWORK] IPv4 addresses:"
        ip -o -4 addr show 2>/dev/null || true
        exit 1
    fi

    # '=' means exact interface match for NCCL.
    export NCCL_SOCKET_IFNAME="=${NET_IFACE}"
    export GLOO_SOCKET_IFNAME="${NET_IFACE}"
    EFFECTIVE_NET_IFACE="${NET_IFACE}"
fi

echo "------------------------------------------------------------------------"
echo "[NODE ${NODE_RANK}] host=$(hostname)"
echo "NODE_RANK=${NODE_RANK}"
echo "NNODES=${NNODES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "WORLD_SIZE=$((NNODES * NPROC_PER_NODE))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NET_IFACE=${EFFECTIVE_NET_IFACE}"
echo "------------------------------------------------------------------------"
echo "[NETWORK] IPv4 addresses:"
ip -o -4 addr show 2>/dev/null || true
echo "------------------------------------------------------------------------"

cd "${PROJECT_DIR}"
bash "${RUN_SCRIPT}"
NODE_EOF

chmod +x "${NODE_RUNNER}"

cleanup() {
    rm -f "${NODE_RUNNER}" || true
}
trap cleanup EXIT

# One SLURM task per node. The inner swift launcher uses NPROC_PER_NODE GPUs.
srun \
    --nodes="${NNODES}" \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    --kill-on-bad-exit=1 \
    bash "${NODE_RUNNER}"
