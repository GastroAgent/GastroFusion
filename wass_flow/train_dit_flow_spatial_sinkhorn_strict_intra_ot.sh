#!/bin/bash

#SBATCH -J dit_strict_intra_ot
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --gres=gpu:8
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/logs_dit/%x-%j.out

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow

export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:${LD_LIBRARY_PATH:-}
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:${PATH:-}

# 不手工覆盖 CUDA_VISIBLE_DEVICES。
# Slurm 会把 --gres=gpu:2 分配到的两张 GPU 暴露给 torchrun。
cd /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1

PY=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/train_dit_flow_spatial_sinkhorn_strict_intra_ot.py
# export CUDA_VISIBLE_DEVICES='4,5'

torchrun \
  --nproc_per_node=8 \
  --master_port=29515 \
  "${PY}" \
  --model otcfm \
  --dit_variant L \
  --batch_size 14 \
  --total_steps 500000 \
  --image_size 256 \
  --caption_mode vision_only \
  --no-use_class_condition \
  --no-train_text_encoder \
  --train_vision_encoder \
  --use_cross_attention \
  --hint_signal_ratio 0.10 \
  --hint_dropout 0.10 \
  --op_match_batch \
  --data_glob "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_train_pool_imagenet_med/*.json" \
  --sampler_seed 42 \
  --min_class_size 2 \
  --suboptimal_eps 1e-6 \
  --ot_reg_weight 0.01 \
  --ot_reg 0.1 \
  --ot_reg_iters 20 \
  --ot_max_tokens 256 \
  --ot_reg_start_step 0 \
  --output_dir /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_dit_flow_v1/vision_hint_inclass \
  --log_dir /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/logs_dit_flow_v1/vision_hint_inclass
