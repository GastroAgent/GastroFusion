#!/bin/bash
#SBATCH -J train_m0a_cls
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m0a_cls/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/logs_m0a_cls/%x_%j.err

cd "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1"

# ── 环境激活 ──
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
export LD_LIBRARY_PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/lib:$LD_LIBRARY_PATH
export PATH=/mnt/inaisfs/data/home/tansy_criait/miniconda3/envs/flow/bin:$PATH

torchrun --nproc_per_node=8 \
    /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/train_m1_medical_metric.py \
    --train_json_glob "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_train_pool_huaxi" \
    --val_json_glob "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_eval_pool_imagenet_huaxi" \
    --project_root /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow \
    --init_checkpoint /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_metric_v1/m0b/metric_imagenet_ot.pt \
    --output_dir /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_metric_v1/m1 \
    --classes_per_batch 16 \
    --samples_per_class 8 \
    --epochs 15 \
    --lr 3e-5