#!/bin/bash


python /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/test-test.py \
 --test_dir /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/data/data_tsy_827_doctor_test_split/indomain_support_76/南充/食管 \
 --checkpoint /mnt/inaisfs/data/home/tansy_criait/jmf/work/GAP/checkpoints/miniImageNet/4conv_8way_5shot/GAP/approx=False/GAP.outer_lr1=0.0001.outer_lr2=0.0001.batch_size=8.flags=best.th \
 --episode_way 8 --shots 15 --query_shots 1 --episodes 100 \
 --GAP --use_cuda