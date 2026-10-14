#!/bin/bash
#SBATCH -J tsy_gastro_candidate
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/mllm_code/slurm_out/gastro_mllm_candidate-%j.out

set -eo pipefail


# =============================================================================
# Runtime
# =============================================================================
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate tsy_vllm_env


# =============================================================================
# Paths
# =============================================================================
SERVER="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/mllm_code/gastro_mllm_qwenvl3_server.py"

MODEL="/mnt/inaisfs/data/home/tansy_criait/jmf/work/think_with_image_zh_swift/weights/Qwen3-VL-8B-Instruct-3epoch-best/Qwen3-VL-8B-Instruct"


# =============================================================================
# Service
# =============================================================================
HOST="0.0.0.0"
PORT="8030"


# =============================================================================
# Sanity checks
# =============================================================================
if [[ ! -f "${SERVER}" ]]; then
    echo "[ERROR] Server script not found:"
    echo "        ${SERVER}"
    exit 1
fi

if [[ ! -d "${MODEL}" ]]; then
    echo "[ERROR] Model directory not found:"
    echo "        ${MODEL}"
    exit 1
fi

mkdir -p \
    /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/slurm


# =============================================================================
# Information
# =============================================================================
echo "===================================================================================================="
echo "[GastroMLLM Candidate Service]"
echo
echo "HOSTNAME      = $(hostname)"
echo "PYTHON        = $(which python)"
echo "CUDA_VISIBLE  = ${CUDA_VISIBLE_DEVICES:-<managed by slurm>}"
echo
echo "SERVER        = ${SERVER}"
echo "MODEL         = ${MODEL}"
echo "HOST          = ${HOST}"
echo "PORT          = ${PORT}"
echo
echo "Candidate API = /v1/vision/candidate_diseases"
echo "Score API     = /v1/vision/score_candidates"
echo "===================================================================================================="


# =============================================================================
# Start service
# =============================================================================
python -u "${SERVER}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --model_path "${MODEL}"