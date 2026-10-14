#!/bin/bash
#SBATCH -J tsy_deepemd
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/logs_eval/%x-%j.out

# export CUDA_VISIBLE_DEVICES=5

source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate qdeepemd

# export CUDA_VISIBLE_DEVICES=7
python /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/eval_1_batch_dynamic_way_tsy_test_test.py
