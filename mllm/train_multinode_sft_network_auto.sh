#!/bin/bash
#SBATCH -J sft
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --gres=gpu:8
#SBATCH --output=/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/logs/%x-%j.out

PROJECT_DIR="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift"
GPUS_PER_NODE=8

# auto = do not force NCCL/Gloo to a guessed interface.
# Only set an explicit interface after verifying it exists on every node.
NET_IFACE="auto"

SFT_SCRIPT="${PROJECT_DIR}/sft_network_auto.sh"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

NNODES="${SLURM_NNODES}"
NPROC_PER_NODE="${GPUS_PER_NODE}"
MASTER_ADDR="$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

export PROJECT_DIR GPUS_PER_NODE NET_IFACE SFT_SCRIPT
export NNODES NPROC_PER_NODE MASTER_ADDR MASTER_PORT
export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "========================================================================"
echo "[SFT SLURM]"
echo "JOB_ID          = ${SLURM_JOB_ID}"
echo "JOB_NAME        = ${SLURM_JOB_NAME}"
echo "NODELIST        = ${SLURM_JOB_NODELIST}"
echo "NNODES          = ${NNODES}"
echo "GPUS_PER_NODE   = ${GPUS_PER_NODE}"
echo "WORLD_SIZE      = $((NNODES * GPUS_PER_NODE))"
echo "MASTER_ADDR     = ${MASTER_ADDR}"
echo "MASTER_PORT     = ${MASTER_PORT}"
echo "NET_IFACE       = ${NET_IFACE}"
echo "SFT_SCRIPT      = ${SFT_SCRIPT}"
echo "========================================================================"

if [[ ! -f "${SFT_SCRIPT}" ]]; then
    echo "[ERROR] SFT script not found: ${SFT_SCRIPT}"
    exit 1
fi

NODE_RUNNER="${PROJECT_DIR}/logs/.sft_node_runner_${SLURM_JOB_ID}.sh"

cat > "${NODE_RUNNER}" <<'NODE_EOF'
#!/bin/bash


export NODE_RANK="${SLURM_NODEID:-0}"

GPU_LIST=""
for ((i=0; i<NPROC_PER_NODE; i++)); do
    [[ -n "${GPU_LIST}" ]] && GPU_LIST+=","
    GPU_LIST+="${i}"
done
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"

export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

if [[ -z "${NET_IFACE:-}" || "${NET_IFACE}" == "auto" ]]; then
    unset NCCL_SOCKET_IFNAME || true
    unset GLOO_SOCKET_IFNAME || true
    EFFECTIVE_NET_IFACE="<auto>"
else
    if ! ip link show dev "${NET_IFACE}" >/dev/null 2>&1; then
        echo "[ERROR] NET_IFACE=${NET_IFACE} does not exist on $(hostname)"
        echo "[NETWORK] Available interfaces:"
        ip -o link show 2>/dev/null || true
        exit 1
    fi

    if ! ip -o -4 addr show dev "${NET_IFACE}" scope global 2>/dev/null | grep -q "inet "; then
        echo "[ERROR] NET_IFACE=${NET_IFACE} has no global IPv4 on $(hostname)"
        echo "[NETWORK] IPv4 addresses:"
        ip -o -4 addr show 2>/dev/null || true
        exit 1
    fi

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
echo "[NETWORK] IPv4 addresses on $(hostname):"
ip -o -4 addr show 2>/dev/null || true
echo "------------------------------------------------------------------------"

cd "${PROJECT_DIR}"
bash "${SFT_SCRIPT}"
NODE_EOF

chmod +x "${NODE_RUNNER}"

cleanup() {
    rm -f "${NODE_RUNNER}" || true
}
trap cleanup EXIT

srun \
    --nodes="${NNODES}" \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    --kill-on-bad-exit=1 \
    bash "${NODE_RUNNER}"
