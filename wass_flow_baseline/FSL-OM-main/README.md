# Transductive Few-Shot Classification on the Oblique Manifold

## Introduction
The folders contain the code for  paper [Transductive Few-Shot Classification on the Oblique Manifold](https://arxiv.org/abs/2108.04009).


## Environment

    numpy==1.18.5
    torch==1.7.0
    Pillow==7.2.0
    torchvision==0.8.0
    tqdm==4.46.0
    
## Train

For example:\
`5-way 5-shot` with `resnet18` in `mini-ImageNet`:\
``python train.py --n_way 5 --k_shot 5 --k_query 15 --skip False --dataset mini --backbone resnet18 --gpu 0,1,2``\

`5-way 1-shot` with `WRN` in `tiered-ImageNet`:\
``python train.py --n_way 5 --k_shot 1 --k_query 15 --skip False --dataset tiered --backbone wideres --gpu 0,1,2``\

## Use the Pretrained Models

Move the  models to folder ``checkpoint``, and change the argument `skip`  with value `True`:\
For example:\
`5-way 5-shot` with `resnet18` in `mini-ImageNet`:\
``python train.py --n_way 5 --k_shot 5 --k_query 15 --skip True --dataset mini --backbone resnet18 --gpu 0,1,2``\

`5-way 1-shot` with `WRN` in `tiered-ImageNet`:\
``python train.py --n_way 5 --k_shot 1 --k_query 15 --skip True --dataset tiered --backbone wideres --gpu 0,1,2``\

## Reference
* [TIM](https://github.com/mboudiaf/TIM)
* [Geoopt](https://github.com/geoopt/geoopt)
ps://download.pytorch.org/whl/torch_stable.html
```

### 3. Install remaining dependencies

```bash
pip install numpy==1.18.5
pip install Pillow==7.2.0
pip install tqdm==4.46.0
pip install tensorboard
```

### 4. Verify CUDA availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

## Dataset Preparation

The data should be located at:

```
/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/data/
```

No symlink is needed — the code automatically resolves paths relative to the project root.

## Running Experiments

### Quick Start (Recommended: Skip Pretrain, Use ImageNet Pretrained Weights)

The code automatically loads ImageNet pretrained ResNet18 weights for medical data when no local pretrain model exists. This is the recommended approach for small medical datasets.

**On a single GPU:**

```bash
cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/FSL-OM-main
python train.py \
    --dataset medical \
    --medical_subset 十二指肠_1query \
    --n_way 5 \
    --k_shot 5 \
    --k_query 5 \
    --backbone resnet18 \
    --skip_pretrain True \
    --gpu 0
```

**On multiple GPUs:**

```bash
cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/FSL-OM-main

# 3 GPUs (e.g., GPU 0,1,2)
python train.py \
    --dataset medical \
    --medical_subset 胃_1query \
    --n_way 5 \
    --k_shot 5 \
    --k_query 5 \
    --backbone resnet18 \
    --skip_pretrain True \
    --gpu 0,1,2

# 2 GPUs
python train.py \
    --dataset medical \
    --medical_subset 食管_1query \
    --n_way 5 \
    --k_shot 1 \
    --k_query 5 \
    --backbone resnet18 \
    --skip_pretrain True \
    --gpu 0,1
```

### With Local Pretraining

First pretrain the backbone on the training set, then fine-tune on few-shot episodes:

```bash
cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/FSL-OM-main

# Phase 1: Pretrain
python train.py \
    --dataset medical \
    --medical_subset 十二指肠_1query \
    --backbone resnet18 \
    --skip_pretrain False \
    --train_pretrain_epochs 30 \
    --batch_size 32 \
    --gpu 0,1,2

# Phase 2: Fine-tune (will auto-load the pretrain model)
python train.py \
    --dataset medical \
    --medical_subset 十二指肠_1query \
    --n_way 5 \
    --k_shot 5 \
    --k_query 5 \
    --backbone resnet18 \
    --skip_pretrain True \
    --gpu 0,1,2
```

### Full Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `mini` | Dataset name. Use `medical` for your data. |
| `--medical_subset` | `十二指肠_1query` | Subset: `十二指肠_1query`, `胃_1query`, `食管_1query` |
| `--backbone` | `resnet18` | Backbone: `resnet18`, `wideres` |
| `--n_way` | `5` | N-way classification (auto-limited to max classes) |
| `--k_shot` | `5` | K-shot (support images per class) |
| `--k_query` | `5` | Query images per class per episode |
| `--gpu` | `4,5,6` | GPU device IDs |
| `--skip_pretrain` | `True` | Skip pretrain (uses ImageNet weights) |
| `--train_pretrain_epochs` | `30` | Pretrain epochs (for medical dataset) |
| `--train_meta_epochs` | `20` | Fine-tuning meta epochs |
| `--train_episodes` | `200` | Training episodes per epoch |
| `--val_episodes` | `200` | Validation episodes |
| `--batch_size` | `32` | Batch size (auto-set for medical) |
| `--T` | `14` | Number of anchors - 1 (transductive) |
| `--kernels` | `11` | Number of RSMA kernels |
| `--seed` | `12667` | Random seed |

## Experiment Matrix

Recommended experiments for each medical dataset:

| Subset | Classes | Recommended n_way | Recommended k_shot | k_query |
|--------|---------|-------------------|-------------------|---------|
| 十二指肠_1query | 7 | 5 or 7 | 1, 5 | 5 |
| 胃_1query | 8 | 5 or 8 | 1, 5 | 5 |
| 食管_1query | 8 | 5 or 8 | 1, 5 | 5 |

### All-in-one experiment script

```bash
#!/bin/bash
# Run all experiments for all three medical subsets

SUBSETS=("十二指肠_1query" "胃_1query" "食管_1query")
SHOTS=(1 5)
GPU="0"

for subset in "${SUBSETS[@]}"; do
    for shot in "${SHOTS[@]}"; do
        echo "=========================================="
        echo "Running: subset=$subset, k_shot=$shot"
        echo "=========================================="
        python train.py \
            --dataset medical \
            --medical_subset "$subset" \
            --n_way 5 \
            --k_shot $shot \
            --k_query 5 \
            --backbone resnet18 \
            --skip_pretrain True \
            --gpu $GPU \
            --seed 12667
    done
done
```

## Expected Output

The model outputs accuracy on the test set after fine-tuning:

```
val_acc: 85.234, pm: 2.156
```

Where:
- `val_acc`: mean accuracy across all validation episodes (%)
- `pm`: 95% confidence interval

## Output Files

Checkpoint and log files are saved to:

```
FSL-OM-main/checkpoints/medical/{subset}/softmax/{backbone}/
├── model_best.pth.tar          # Best pretrain model
└── models/                     # Periodic checkpoints
    └── {epoch}_model.pkl
```
