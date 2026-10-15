#!/bin/bash
#SBATCH -J GAP_test_train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

python /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/test-train.py \
 --support_dir /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/data/data_tsy_827_doctor_test_split/食管_1query/train \
 --test_dir /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/data/data_tsy_827_doctor_test_split/indomain_support_76/自贡/食管 \
 --checkpoint /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/checkpoints/miniImageNet/4conv_8way_5shot/GAP/approx=False/GAP.outer_lr1=0.0001.outer_lr2=0.0001.batch_size=8.flags=best.th \
 --episode_way 8 --shots 5 --query_shots 1 --episodes 100 \
 --GAP --use_cuda