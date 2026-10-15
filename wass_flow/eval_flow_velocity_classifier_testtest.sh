#!/bin/bash
#SBATCH -J flow_velcls_testtest
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/%x_%j.err


source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}
export PYTHONUNBUFFERED=1

ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_DIR=${ROOT}/fsl_multi_vae/rae_flow_metric_v1
PY=${CODE_DIR}/eval_flow_velocity_classifier.py

DIT_CHECKPOINT=${DIT_CHECKPOINT:-${ROOT}/weights_dit_flow_v1/vision_hint_inclass/otcfm/otcfm_dit_L_step_500000.pt}

# export CUDA_VISIBLE_DEVICES='4'

python "${PY}" \
  --eval-setting test_test \
  --dit-checkpoint "${DIT_CHECKPOINT}" \
  --n-shot 5 \
  --n-query 1 \
  --episodes 100 \
  --num-trials 16 \
  --time-sampling even \
  --flow-sigma 0.10 \
  --error-type l2 \
  --hint-mode shared_noise \
  --shot-aggs mean min softmin \
  --shot-tau 0.10 \
  --pair-chunk-size 8
