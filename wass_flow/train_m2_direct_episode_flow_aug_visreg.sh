#!/bin/bash
#SBATCH -J rae_m2_flow_aug
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
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

# Do not manually set CUDA_VISIBLE_DEVICES.
# Slurm exposes the two allocated GPUs as cuda:0 and cuda:1 to this job.
# export CUDA_VISIBLE_DEVICES=0

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
CODE_ROOT=${PROJECT_ROOT}/fsl_multi_vae/rae_flow_metric_v1
PY=${CODE_ROOT}/train_m2_direct_episode_flow_aug_visreg.py

M1_CHECKPOINT=${PROJECT_ROOT}/weights_metric_v1/new_m2_食管-test-train/metric_direct_episode.pt

FLOW_CHECKPOINT=${PROJECT_ROOT}/weights_dit_flow_v1/vision_hint_inclass/otcfm/otcfm_dit_L_step_215000.pt
OUTPUT_DIR=${PROJECT_ROOT}/weights_metric_v1/new_m2_食管-val-train
### ### 胃：53-200、胃122-406、十二指肠 397-460
TRAIN_EPISODE_MANIFEST="/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/data/食管/train.json"
VAL_EPISODE_MANIFEST="/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/data/食管/val.json"

mkdir -p "${PROJECT_ROOT}/fsl_multi_vae/logs_m2_direct"
mkdir -p "${OUTPUT_DIR}"
cd "${CODE_ROOT}"

echo "================================================================================================"
echo "[RAE M2 DIRECT + TRAINING-ONLY FLOW LATENT AUGMENTATION]"
echo "PY                  = ${PY}"
echo "M1_CHECKPOINT       = ${M1_CHECKPOINT}"
echo "FLOW_CHECKPOINT     = ${FLOW_CHECKPOINT}"
echo "OUTPUT_DIR          = ${OUTPUT_DIR}"
echo "TRAIN_MANIFEST      = ${TRAIN_EPISODE_MANIFEST}"
echo "VAL_MANIFEST        = ${VAL_EPISODE_MANIFEST}"
echo "primary device      = cuda:0  (RAE + Metric + Sinkhorn)"
echo "Flow device         = cuda:0  (frozen DiT + EndoViT)"
echo "================================================================================================"

nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader || true

# One Python process, two GPUs used only for explicit module placement:
#   cuda:0 -> RAE + trainable MetricEncoder + Sinkhorn
#   cuda:1 -> frozen Flow + EndoViT for augmentation
# No torchrun / NCCL / DDP is used.
#
# Flow augmentation semantics:
#   near_source: early trajectory states, close to the original query
#   near_target: late trajectory states, moving toward an EXTRA same-class image
#                that is excluded from the current query/K-shot supports
#   both: one early + one late synthetic sample per augmented real query
# Validation is always real-image Direct M2 without Flow augmentation.
# VISReg defaults to REAL support + REAL query clouds only.
# Add --visreg_include_flow_aug only for an explicit ablation on synthetic clouds.

python "${PY}" \
    --train_episode_manifest "${TRAIN_EPISODE_MANIFEST}" \
    --val_episode_manifest "${VAL_EPISODE_MANIFEST}" \
    --project_root "${PROJECT_ROOT}" \
    --init_checkpoint "${M1_CHECKPOINT}" \
    --output_dir "${OUTPUT_DIR}" \
    --n_way 7 \
    --n_shot 5 \
    --n_query 1 \
    --episodes_per_epoch 137 \
    --val_episodes 193 \
    --epochs 30 \
    --image_size 256 \
    --rae_chunk_size 8 \
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
    --visreg_weight 1e-3 \
    --visreg_num_proj 1024 \
    --visreg_projection_chunk 128 \
    --flow_aug \
    --flow_aug_checkpoint "${FLOW_CHECKPOINT}" \
    --flow_aug_device cuda:0 \
    --flow_aug_dit_variant L \
    --flow_aug_num_steps 8 \
    --flow_aug_solver heun \
    --flow_aug_mode both \
    --flow_aug_prob 0.50 \
    --flow_aug_samples_per_query 1 \
    --flow_aug_early_max_step 2 \
    --flow_aug_late_min_step 6 \
    --flow_aug_loss_weight 0.50 \
    --flow_aug_stochastic_hint \
    --seed 42 \
    --amp \
    --log_every 25
