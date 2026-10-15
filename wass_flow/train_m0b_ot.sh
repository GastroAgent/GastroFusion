#!/bin/bash
#SBATCH -J train_m0a_cls
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m0a_cls/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m0a_cls/%x_%j.err

cd "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1"

# ── 环境激活 ──
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:$LD_LIBRARY_PATH
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:$PATH

# export CUDA_VISIBLE_DEVICES='6,7' # data_train_pool_imagenet_med

torchrun --nproc_per_node=8 --master_port=29523 \
    /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/train_m0b_ot.py \
    --train_json_glob "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_train_pool_imagenet_med" \
    --val_json_glob "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_eval_pool_imagenet_huaxi" \
    --project_root /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow \
    --init_checkpoint /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_metric_v1/new_m0b/metric_imagenet_ot.pt \
    --output_dir /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_metric_v1/new_m0b \
    --classes_per_batch 50 \
    --samples_per_class 8 \
    --batches_per_epoch 2000 \
    --val_batches 1000 \
    --epochs 5 \
    --ot_pool_factor 1