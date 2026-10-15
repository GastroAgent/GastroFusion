#!/bin/bash
# Replace paths before use. Each stage is intentionally independent.
set -euo pipefail

PROJECT_ROOT=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
WEIGHTS=/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_rae_flow_metric_v1

# M0-A
# python train_m0a_cls.py --train_root /path/imagenet/train --val_root /path/imagenet/val \
#   --project_root "$PROJECT_ROOT" --output_dir "$WEIGHTS/m0a"

# M0-B
# python train_m0b_ot.py --data_root /path/imagenet/train --project_root "$PROJECT_ROOT" \
#   --init_checkpoint "$WEIGHTS/m0a/metric_cls_pretrain.pt" --output_dir "$WEIGHTS/m0b"

# M1
# python train_m1_medical_metric.py --json_glob "$PROJECT_ROOT/data/data_pool_imagenet_med/*.json" \
#   --project_root "$PROJECT_ROOT" --init_checkpoint "$WEIGHTS/m0b/metric_imagenet_ot.pt" \
#   --output_dir "$WEIGHTS/m1"

# M2
# python train_m2_direct_episode.py --json_glob '/path/medical_train/*.json' \
#   --val_json_glob '/path/medical_val/*.json' --project_root "$PROJECT_ROOT" \
#   --init_checkpoint "$WEIGHTS/m1/metric_medical_adapt.pt" --output_dir "$WEIGHTS/m2" \
#   --n_way 7 --n_shot 5 --n_query 1

# F1
# python train_f1_flow_episode.py --json_glob '/path/medical_train/*.json' \
#   --project_root "$PROJECT_ROOT" --flow_checkpoint /path/to/v1_dit.pt \
#   --metric_checkpoint "$WEIGHTS/m2/metric_direct_episode.pt" --output_dir "$WEIGHTS/f1"
