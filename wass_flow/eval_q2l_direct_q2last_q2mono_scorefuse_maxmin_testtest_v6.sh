#!/bin/bash
#SBATCH -J eval_q2l_q2s_testtest
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/eval_m2_direct_episode/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/eval_m2_direct_episode/%x_%j.err

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}

ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_DIR=${ROOT}/fsl_multi_vae/rae_flow_metric_v1
cd "${CODE_DIR}"

python "${CODE_DIR}/eval_q2l_direct_q2last_q2mono_scorefuse_maxmin_testtest_v6.py"
