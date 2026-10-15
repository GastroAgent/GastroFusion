#!/bin/bash
#SBATCH -J train_m0a_cls
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m0a_cls/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m0a_cls/%x_%j.err

cd "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1"

# ── 环境激活 ──
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:$LD_LIBRARY_PATH
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:$PATH

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
    # --train_root /mnt/inaisfs/data/home/tansy_criait/jmf/work/mm_vae/data/data_imagenet/train \
    # --val_root /mnt/inaisfs/data/home/tansy_criait/jmf/work/mm_vae/data/data_imagenet/test \
    # --train_json_glob /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_train_pool_imagenet \
    # --val_json_glob /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_eval_pool_imagenet \

torchrun --nproc_per_node=8  --master_port=29514 \
    /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/train_m0a_cls.py \
    --train_json_glob "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_train_pool_imagenet_med" \
    --val_json_glob "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_eval_pool_imagenet" \
    --project_root /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow \
    --output_dir /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_metric_v1/m0a \
    --batch_size 512 \
    --rae_chunk_size 512 \
    --epochs 10