#!/bin/bash
#SBATCH -J test_m3_causal_visual
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m3_visual/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m3_visual/%x_%j.err

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}
export PYTHONUNBUFFERED=1

PROJECT_ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_ROOT=${PROJECT_ROOT}/fsl_multi_vae/rae_flow_metric_v1

PY=${CODE_ROOT}/test_m3_causal_visualize_v2.py
M3_SOURCE=${CODE_ROOT}/train_m3_causal_delta_world_model.py

M3_CHECKPOINT=${PROJECT_ROOT}/weights_world_model_v1/m3_causal_delta_rae1024/world_model_epoch_024.pt
FLOW_CHECKPOINT=${PROJECT_ROOT}/weights_dit_flow_v1/vision_hint_inclass/otcfm/otcfm_dit_L_step_500000.pt

VAL_JSON_GLOB=${PROJECT_ROOT}/data/data_eval_pool_imagenet_huaxi

OUTPUT_DIR=${PROJECT_ROOT}/weights_world_model_v1/m3_causal_visual_results

mkdir -p ${PROJECT_ROOT}/fsl_multi_vae/logs_m3_visual
mkdir -p ${OUTPUT_DIR}

cd ${CODE_ROOT}

python ${PY} \
    --m3_source ${M3_SOURCE} \
    --project_root ${PROJECT_ROOT} \
    --m3_checkpoint ${M3_CHECKPOINT} \
    --flow_checkpoint ${FLOW_CHECKPOINT} \
    --val_json_glob ${VAL_JSON_GLOB} \
    --output_dir ${OUTPUT_DIR} \
    --num_per_class 5 \
    --seed 42 \
    --device cuda:0
