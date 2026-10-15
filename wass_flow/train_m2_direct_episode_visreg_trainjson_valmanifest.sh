#!/bin/bash
#SBATCH -J rae_m2_visreg
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m2_direct/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m2_direct/%x_%j.err


# =============================================================================
# Environment
# =============================================================================
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Normally let Slurm set CUDA_VISIBLE_DEVICES automatically.
# If your cluster specifically requires a physical GPU index, uncomment:
# export CUDA_VISIBLE_DEVICES=6


# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_ROOT=${PROJECT_ROOT}/fsl_multi_vae/rae_flow_metric_v1

PY=${CODE_ROOT}/train_m2_direct_episode_visreg_trainjson_valmanifest.py

# M2 must initialize from M1.
M1_CHECKPOINT=${PROJECT_ROOT}/weights_metric_v1/m2_flow_aug_m0b2/metric_direct_episode.pt

# Keep VISReg experiment separate from the original M2 baseline.
OUTPUT_DIR=${PROJECT_ROOT}/weights_metric_v1/m2_visreg_main

mkdir -p "${PROJECT_ROOT}/fsl_multi_vae/logs_m2_direct"
mkdir -p "${OUTPUT_DIR}"

cd "${CODE_ROOT}"


# =============================================================================
# Data sources
# =============================================================================
# Training: class-pool JSON glob. Keep the wildcard quoted so Python expands it.
# One JSON is treated as one class pool (top-level label/class_name overrides filename).
TRAIN_JSON_GLOB="${PROJECT_ROOT}/data/data_train_pool_imagenet_med" 

# Validation: keep the existing fixed episode-manifest protocol unchanged.
VAL_EPISODE_MANIFEST="${PROJECT_ROOT}/fsl_multi_vae/data/total_eval.json"


# =============================================================================
# VISReg
# Same defaults as train_m1_medical_metric.py
# =============================================================================
VISREG_WEIGHT=1e-2
VISREG_NUM_PROJ=512
VISREG_PROJECTION_CHUNK=128


echo "================================================================================================"
echo "[RAE M2 DIRECT EPISODIC + VISREG]"
echo "PY                     = ${PY}"
echo "M1_CHECKPOINT          = ${M1_CHECKPOINT}"
echo "OUTPUT_DIR             = ${OUTPUT_DIR}"
echo "TRAIN_JSON_GLOB        = ${TRAIN_JSON_GLOB}"
echo "VAL_MANIFEST           = ${VAL_EPISODE_MANIFEST}"
echo "VISREG_WEIGHT          = ${VISREG_WEIGHT}"
echo "VISREG_NUM_PROJ        = ${VISREG_NUM_PROJ}"
echo "VISREG_PROJECTION_CHUNK= ${VISREG_PROJECTION_CHUNK}"
echo "CUDA_VISIBLE_DEVICES   = ${CUDA_VISIBLE_DEVICES:-<set by Slurm>}"
echo "================================================================================================"

nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader || true


# =============================================================================
# Train
#
# Total objective:
#   L_total = L_episode_CE + VISREG_WEIGHT * L_VISReg
#
# VISReg is applied to:
#   concat(support_cloud, query_cloud)
# =============================================================================
torchrun \
    --nproc_per_node=8 \
    --master_port=29565 \
    "${PY}" \
    --train_json_glob "${TRAIN_JSON_GLOB}" \
    --val_episode_manifest "${VAL_EPISODE_MANIFEST}" \
    --project_root "${PROJECT_ROOT}" \
    --init_checkpoint "${M1_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_way 99 \
    --n_shot 1 \
    --n_query 1 \
    --episodes_per_epoch 10000 \
    --val_episodes 1000 \
    --epochs 50 \
    --image_size 256 \
    --rae_chunk_size 50 \
    --metric_dim 256 \
    --dropout 0.0 \
    --ot_pool_factor 1 \
    --sinkhorn_reg 0.1 \
    --sinkhorn_iter 30 \
    --sk_weight_eps 1e-3 \
    --shot_tau_init 0.10 \
    --cls_tau_init 0.50 \
    --metric_lr 1e-5 \
    --temperature_lr 1e-3 \
    --weight_decay 1e-4 \
    --grad_clip 5.0 \
    --freeze_metric_epochs 0 \
    --visreg_weight "${VISREG_WEIGHT}" \
    --visreg_num_proj "${VISREG_NUM_PROJ}" \
    --visreg_projection_chunk "${VISREG_PROJECTION_CHUNK}" \
    --seed 42 \
    --amp \
    --log_every 25
