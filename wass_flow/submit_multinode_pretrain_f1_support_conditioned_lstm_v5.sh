#!/bin/bash
#SBATCH -J f1_v5_imgnet_mn
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --gres=gpu:8
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x-%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_f1_utility/%x-%j.err



# =============================================================================
# USER CONFIG
# Same launcher pattern as train_multinode_medical_grpo_network_auto_entropy20.sh:
#   sbatch -> srun one node-runner per node -> one torchrun agent per node
# =============================================================================
ROOT="/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow"
CODE_DIR="${ROOT}/fsl_multi_vae/rae_flow_metric_v1"

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NET_IFACE=${NET_IFACE:-auto}

CUDA_ENV_SCRIPT="/mnt/inaisfs/data/apps/cuda12.4.sh"
CONDA_ACTIVATE="/mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate"
CONDA_ENV="flow"

RUN_SCRIPT="${CODE_DIR}/run_pretrain_f1_support_conditioned_lstm_v5_multinode.sh"

mkdir -p "${CODE_DIR}/logs_f1_utility"
cd "${CODE_DIR}"

# =============================================================================
# SLURM / DISTRIBUTED TOPOLOGY
# =============================================================================
NNODES="${SLURM_NNODES}"
NPROC_PER_NODE="${GPUS_PER_NODE}"

MASTER_ADDR="$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

export ROOT CODE_DIR GPUS_PER_NODE NET_IFACE CUDA_ENV_SCRIPT CONDA_ACTIVATE CONDA_ENV RUN_SCRIPT
export NNODES NPROC_PER_NODE MASTER_ADDR MASTER_PORT

export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Optional: pass any F1 training overrides from the submit environment through
# to every node's execution script. Examples:
#   EPISODE_BATCH_SIZE=4 sbatch submit_multinode_pretrain_f1_support_conditioned_lstm_v5.sh
#   PAIR_CHUNK_SIZE=8 EPISODE_BATCH_SIZE=2 sbatch ...
for name in \
    ORGAN M2_CHECKPOINT DIT_CHECKPOINT TRAJECTORY_SCORE_MODE OUTPUT_DIR \
    EPISODE_BATCH_SIZE PAIR_CHUNK_SIZE RAE_CHUNK_SIZE \
    GAMMA_MIN GAMMA_INIT LAMBDA_DECAY_SMOOTH \
    LAMBDA_RELATIVE RELATIVE_GROUP_SIZE RELATIVE_WARMUP_EPOCHS \
    RELATIVE_SIGMA_START RELATIVE_SIGMA_END RELATIVE_TARGET_STEP \
    RELATIVE_ADV_CLIP RELATIVE_MIN_REWARD_STD \
    EPISODES_PER_EPOCH VAL_EPISODES EPOCHS N_QUERY NUM_STEPS \
    UTILITY_PROJ_DIM UTILITY_HIDDEN_DIM UTILITY_CONDITION_DIM \
    UTILITY_HEAD_DROPOUT UTILITY_LR FEATURE_RATIO_CLIP SEED \
    RESUME_CHECKPOINT OMP_NUM_THREADS PYTORCH_CUDA_ALLOC_CONF
 do
    if [[ -v "${name}" ]]; then
        export "${name}"
    fi
done

echo "========================================================================"
echo "[F1-v5 MULTI-NODE SLURM SUBMISSION]"
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
# Build a tiny node-local launcher on the shared filesystem.
# One SLURM task is launched per node.  That task starts ONE torchrun agent,
# and torchrun then launches NPROC_PER_NODE GPU workers on that node.
# =============================================================================
NODE_RUNNER="${CODE_DIR}/logs_f1_utility/.f1_v5_node_runner_${SLURM_JOB_ID}.sh"

cat > "${NODE_RUNNER}" <<'NODE_EOF'
#!/bin/bash


NODE_RANK="${SLURM_NODEID:-0}"
export NODE_RANK

# Expose exactly the local GPUs reserved for this node.
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

export LD_LIBRARY_PATH="/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}"
export PATH="/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH:-}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# -----------------------------------------------------------------------------
# Network interface selection -- same policy as the GRPO multi-node launcher.
# -----------------------------------------------------------------------------
if [[ -z "${NET_IFACE:-}" || "${NET_IFACE}" == "auto" ]]; then
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
    export NCCL_SOCKET_IFNAME="=${NET_IFACE}"
    export GLOO_SOCKET_IFNAME="${NET_IFACE}"
    EFFECTIVE_NET_IFACE="${NET_IFACE}"
fi

echo "------------------------------------------------------------------------"
echo "[F1-v5 NODE ${NODE_RANK}] host=$(hostname)"
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

cd "${CODE_DIR}"
bash "${RUN_SCRIPT}"
NODE_EOF

chmod +x "${NODE_RUNNER}"

cleanup() {
    rm -f "${NODE_RUNNER}" || true
}
trap cleanup EXIT

# One SLURM task per node.  Each node task launches its own torchrun agent.
srun \
    --nodes="${NNODES}" \
    --ntasks="${NNODES}" \
    --ntasks-per-node=1 \
    --kill-on-bad-exit=1 \
    bash "${NODE_RUNNER}"
