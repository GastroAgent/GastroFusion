#!/bin/bash
#SBATCH -J wass_vs_sk_tr
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/eval_m2_direct_episode/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/eval_m2_direct_episode/%x_%j.err

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate qdeepemd
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/qdeepemd/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/qdeepemd/bin:${PATH}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=7
ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_DIR=${ROOT}/fsl_multi_vae/rae_flow_metric_v1
PY=${CODE_DIR}/benchmark_exact_wass_vs_dynamic_sk_testtrain_reduced.py
cd "${CODE_DIR}"
[[ -f "${PY}" ]] || { echo "[ERROR] missing ${PY}" >&2; exit 2; }

export EPISODES_PER_TASK=${EPISODES_PER_TASK:-100}
python "${PY}"
