#!/bin/bash
#SBATCH -J mllm_batch
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8

#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/mllm_fsl_result/mllm_code/slurm_out/mllm_batch-%j.out

set -eo pipefail

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate tsy_vllm_env


# =============================================================================
# 禁止 requests 访问内部 MLLM service 时走系统代理
# =============================================================================
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
unset http_proxy
unset https_proxy
unset all_proxy

export NO_PROXY="127.0.0.1,localhost,gpu35"
export no_proxy="127.0.0.1,localhost,gpu35"


# =============================================================================
# Test service
# =============================================================================
echo "Testing MLLM service..."
curl --noproxy "*" --fail http://gpu35:8030/

echo
echo "MLLM service OK."


# =============================================================================
# Run batch
# =============================================================================
python -u \
/mnt/inaisfs/data/home/tansy_criait/GastroFusion/mllm_fsl/mllm_code/semantic_score_mllm_external.py