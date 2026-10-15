#!/bin/bash
#SBATCH -J rae_m3_delta_wm
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m3_world/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m3_world/%x_%j.err


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

# Slurm --gres=gpu:8 分配的 8 张 GPU 依次映射为 local cuda:0 .. cuda:7。
# torchrun launches one process per visible GPU.
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_ROOT=${PROJECT_ROOT}/fsl_multi_vae/rae_flow_metric_v1
PY=${CODE_ROOT}/train_m3_causal_delta_world_model.py

TRAIN_JSON_GLOB=${PROJECT_ROOT}/data/data_train_pool_imagenet_med
VAL_JSON_GLOB=${PROJECT_ROOT}/data/data_eval_pool_imagenet_huaxi

FLOW_CHECKPOINT=${PROJECT_ROOT}/weights_dit_flow_v1/vision_hint_inclass/otcfm/otcfm_dit_L_step_500000.pt
OUTPUT_DIR=${PROJECT_ROOT}/weights_world_model_v1/m3_causal_delta_rae1024

mkdir -p "${PROJECT_ROOT}/fsl_multi_vae/logs_m3_world"
mkdir -p "${OUTPUT_DIR}"
cd "${CODE_ROOT}"

# =============================================================================
# DDP semantics
# =============================================================================
# Each GPU/rank owns a COMPLETE local pipeline:
#
#   GPU/rank 0:
#       frozen RAE + frozen Flow/EndoViT + trainable World Model replica
#
#   GPU/rank 1:
#       frozen RAE + frozen Flow/EndoViT + trainable World Model replica
#
# World-model gradients are synchronized by DistributedDataParallel.
# Frozen RAE / Flow are local inference modules and are NOT synchronized.
#
# IMPORTANT:
#   --batch_size is PER GPU.
#   2 GPUs x batch_size=4 => global batch=8.
#   2 GPUs x batch_size=8 => global batch=16.
#
# DDP replicates model/optimizer memory; it does NOT shard them.
# Because Flow is now also present on every GPU, start conservatively with
# batch_size=4. If memory is sufficient, change it to 8 for global batch=16.
# =============================================================================

# =============================================================================
# Task semantics
# =============================================================================
# Encoder: t=0..3 always observed, 8 blocks.
# Decoder: t=4..8, 8 blocks; t=4..7 random MASK, t=8 always MASK.
# Prediction/loss: only masked decoder states.
# Loss: masked SmoothL1 + visreg_weight * masked VISReg.
#
# Flow hint exactly aligned to deterministic weak grayscale protocol:
#   gray = grayscale(support)
#   hint = 0.1 * gray
#   stochastic noise = OFF
#   hint dropout = 0
# =============================================================================

PER_GPU_BATCH=48
NPROC_PER_NODE=8

echo "================================================================================================"
echo "[M3 CAUSAL MASKED DELTA WORLD MODEL | 8-GPU DDP]"
echo "PY                  = ${PY}"
echo "TRAIN_JSON_GLOB     = ${TRAIN_JSON_GLOB}"
echo "VAL_JSON_GLOB       = ${VAL_JSON_GLOB}"
echo "FLOW_CHECKPOINT     = ${FLOW_CHECKPOINT}"
echo "OUTPUT_DIR          = ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES}"
echo "NPROC_PER_NODE      = ${NPROC_PER_NODE}"
echo "PER_GPU_BATCH       = ${PER_GPU_BATCH}"
echo "GLOBAL_BATCH        = $((PER_GPU_BATCH * NPROC_PER_NODE))"
echo "Each GPU            = frozen RAE + frozen Flow/EndoViT + DDP world model"
echo "Encoder / Decoder   = 8 / 8"
echo "Flow hint           = deterministic 0.1 x grayscale"
echo "================================================================================================"

nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader || true

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=${NPROC_PER_NODE} \
    "${PY}" \
    --train_json_glob "${TRAIN_JSON_GLOB}" \
    --val_json_glob "${VAL_JSON_GLOB}" \
    --project_root "${PROJECT_ROOT}" \
    --flow_checkpoint "${FLOW_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --image_size 256 \
    --rae_chunk_size 8 \
    --flow_num_steps 8 \
    --flow_solver heun \
    --flow_dit_variant L \
    --flow_hint_signal_ratio 0.1 \
    --flow_hint_dropout 0.0 \
    --no-flow_stochastic_hint \
    --batch_size ${PER_GPU_BATCH} \
    --steps_per_epoch 5000 \
    --val_steps 200 \
    --tail_start 4 \
    --tail_end 7 \
    --tail_mask_prob 0.15 \
    --ensure_extra_tail_mask \
    --model_dim 1024 \
    --depth 16 \
    --encoder_depth 8 \
    --encoder_time 4 \
    --num_heads 16 \
    --ffn_hidden_dim 2048 \
    --dropout 0.0 \
    --attn_dropout 0.0 \
    --gradient_checkpointing \
    --smooth_l1_beta 1.0 \
    --visreg_weight 0 \
    --visreg_num_proj 512 \
    --visreg_projection_chunk 128 \
    --epochs 100 \
    --lr 2e-4 \
    --weight_decay 0.05 \
    --grad_clip 1.0 \
    --warmup_steps 500 \
    --min_lr_ratio 0.05 \
    --seed 42 \
    --amp \
    --log_every 25 \
    --save_every 1 \
    --resume_checkpoint /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_world_model_v1/m3_causal_delta_rae1024/world_model_best.pt