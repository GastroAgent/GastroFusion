#!/bin/bash
#SBATCH -J tsy_deepemd_shiguan
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/slurm_out/%x-%j.out

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate qdeepemd

cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD

python /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/train_meta1_shiguan_tsy.py \
    -gpu 0 \
    -save_path /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/checkpoint_tsy_new/食管/fcn/食管-5shot-8way