#!/bin/bash
#SBATCH -J wass_vs_sk_tt
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/logs/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/logs/%x_%j.err

# export CUDA_VISIBLE_DEVICES=7

python /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/main.py \
    --dataset miniImageNet \
    --folder /mnt/inaisfs/data/home/tansy_criait/Datasets/mini-imagenet-folder \
    --gpu_id 7 \
    --N_ways 8 \
    --K_shots_for_support 5 \
    --iter 50000 \
    --outer_lr1 0.0001 \
    --outer_lr2 0.0001 \
    --batch_size 8 \
    --use-cuda \
    --GAP \
    --resume /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/checkpoints/miniImageNet/4conv_5way_5shot/GAP/approx=False/GAP.outer_lr1=0.0001.outer_lr2=0.0001.batch_size=8.flags=best.th