#!/bin/bash
#SBATCH -J wass_vs_sk_tt
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/logs/%x_%j.out
#SBATCH -e /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/logs/%x_%j.err

export CUDA_VISIBLE_DEVICES=6

python /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/main.py \
    --dataset_for_source miniImageNet \
    --dataset_for_target miniImageNet \
    --gpu_id 0 --N_ways 7 --test_way 7 \
    --K_shots_for_support 5 \
    --use-cuda --GAP --test \
    --folder /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/data/data_tsy_827_doctor_test_split/indomain_support_76/华西/十二指肠 \
    --checkpoint /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/checkpoints/miniImageNet/4conv_5way_5shot/GAP/approx=False/GAP.outer_lr1=0.0001.outer_lr2=0.0001.batch_size=8.flags=best.th
    