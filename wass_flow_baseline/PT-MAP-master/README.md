# Leveraging the Feature Distribution in Transfer-based Few-Shot Learning (patched for PyTorch ≥ 1.10)

Official PyTorch implementation of [Leveraging the Feature Distribution in Transfer-based Few-Shot Learning](https://arxiv.org/abs/2004.05674),
patched so it runs on modern PyTorch (1.10+) and supports the medical
image few-shot benchmark that lives alongside this repo.

## What changed vs. the upstream repo

This fork is *only* about bringing the 2020 code up to modern PyTorch
and adding medical-subset support; algorithm-wise it is unchanged.

| Fix | File(s) |
|-----|---------|
| `from torch.autograd import Variable` removed | `train.py`, `train_cifar.py`, `wrn_mixup_model.py` |
| `loss.data.item()` → `loss.item()`; `.data` access deprecation | `train.py`, `train_cifar.py` |
| `Variable(...)` → `torch.stack(...)` | `train.py`, `train_cifar.py` |
| `transforms.Scale` removed → `Resize` | `data/datamgr.py` |
| `Bottleneck` `NameError` in `zero_init_residual` | `res_mixup_model.py` |
| `distLinear` in-place weight rewrite wrapped in `torch.no_grad()` | `wrn_mixup_model.py`, `res_mixup_model.py` |
| `tqdm.notebook` unavailable in headless env → `tqdm.auto` | `test_standard.py` |
| Hard-coded `.cuda()` and `inputs.cuda()` | `save_plk.py` |
| `torch.save(_randStates, …)` fails on numpy random state in torch ≥ 2 | `FSLTask.py` (now uses `pickle`) |
| `np.random` state compatibility with newer NumPy | `FSLTask.py` |
| `data_dir['medical']` and `--dataset medical` plumbing | `configs.py`, `io_utils.py`, `train.py`, `save_plk.py`, `FSLTask.py`, `test_standard.py` |
| JSON filelist writer for `<subset>/{train,val,test}` directories | `filelists/medical/write_medical_filelist.py` |
| MAP test CLI flags (`--n_shot`, `--n_ways`, `--n_queries`, `--n_runs`, `--lam`, `--alpha`, …) | `test_standard.py` |
| Save result as JSON | `test_standard.py` (`--output_json`) |

## End-to-end pipeline

1. **(optional) build the medical filelist JSONs** —
   reads `<medical_root>/<subset>/{train,val,test}/<class>/<img>`
   and writes `filelists/medical/<subset>/{base,val,novel}.json`.
2. **train rotation head** (`--method rotation`) — produces
   `checkpoints/<dataset>/[<subset>/]<model>_rotation/<epoch>.tar`.
3. **train S2M2_R** (`--method S2M2_R`) — loads the rotation head's
   weights as initialization and adds mixup; produces
   `checkpoints/<dataset>/[<subset>/]<model>_S2M2_R/<epoch>.tar`.
4. **extract features** (`save_plk.py`) — runs the trained backbone
   over `novel.json` and dumps a class-indexed pickle.
5. **MAP test** (`test_standard.py`) — runs the MAP classifier over
   those features for `n_runs` episodes.

## Environment

```bash
cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/PT-MAP-master
bash setup_env.sh
```

The default is `torch 2.1.0 + cu118`. Adjust via env vars
(`CUDA_TAG=cu110 TORCH_VER=1.7.0 bash setup_env.sh`) if needed.

## Step 1 — generate filelists (medical)

```bash
conda activate fsl_ptmap
python filelists/medical/write_medical_filelist.py \
    --medical_root /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/data \
    --medical_subset 胃_1query
```

This writes:
```
filelists/medical/胃_1query/base.json
filelists/medical/胃_1query/val.json
filelists/medical/胃_1query/novel.json
```

## Step 2 — train rotation head (ResNet18, 5 epochs)

```bash
python train.py \
    --dataset medical \
    --medical_subset 胃_1query \
    --model ResNet18 \
    --method rotation \
    --train_aug \
    --stop_epoch 5 \
    --batch_size 16 \
    --save_freq 1
```

## Step 3 — train S2M2_R (loads rotation head init)

```bash
python train.py \
    --dataset medical \
    --medical_subset 胃_1query \
    --model ResNet18 \
    --method S2M2_R \
    --train_aug \
    --stop_epoch 30 \
    --batch_size 16 \
    --save_freq 5
```

For WRN-28-10 (the paper's strongest backbone):
```bash
python train.py --dataset medical --medical_subset 胃_1query \
    --model WideResNet28_10 --method rotation --train_aug \
    --stop_epoch 5 --batch_size 16 --save_freq 1

python train.py --dataset medical --medical_subset 胃_1query \
    --model WideResNet28_10 --method S2M2_R --train_aug \
    --stop_epoch 30 --batch_size 16 --save_freq 5
```

## Step 4 — extract novel-split features

```bash
python save_plk.py \
    --dataset medical \
    --medical_subset 胃_1query \
    --model ResNet18 \
    --method S2M2_R
```

Output:
```
checkpoints/medical/胃_1query/ResNet18_S2M2_R/last/output.plk
```

## Step 5 — MAP evaluation

```bash
python test_standard.py \
    --dataset medical \
    --medical_subset 胃_1query \
    --dsname 胃_1query \
    --model ResNet18 \
    --method S2M2_R \
    --n_ways 8 --n_shot 5 --n_queries 1 \
    --n_runs 1000 \
    --lam 10 --alpha 0.2 --n_iter 20 --beta 0.5 \
    --output_json checkpoints/medical/胃_1query/ResNet18_S2M2_R/eval_result.json
```

The output JSON records the test accuracy with 95% confidence interval.

## Per-subset meta-test configurations

Each medical subset has its own (n_way, n_shot, n_query) because the
class count differs across subsets:

| Subset | n_way | n_shot | n_queries | reason |
|--------|-------|--------|-----------|--------|
| 胃_1query      | 8 | 5 | 1 | 8 classes available |
| 十二指肠_1query | 7 | 5 | 1 | 7 classes available |
| 食管_1query    | 8 | 5 | 1 | 8 classes available |

## One-liner for three subsets (uses per-subset config above)

```bash
bash run_all_medical.sh
```

Equivalent inline commands (for reference):

```bash
# 胃_1query -- 8-way 5-shot 1-query
python filelists/medical/write_medical_filelist.py \
    --medical_root /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/data \
    --medical_subset 胃_1query

python train.py --dataset medical --medical_subset 胃_1query \
    --model ResNet18 --method rotation --train_aug \
    --stop_epoch 5 --batch_size 16 --save_freq 1

python train.py --dataset medical --medical_subset 胃_1query \
    --model ResNet18 --method S2M2_R --train_aug \
    --stop_epoch 30 --batch_size 16 --save_freq 5

python save_plk.py --dataset medical --medical_subset 胃_1query \
    --model ResNet18 --method S2M2_R

python test_standard.py \
    --dataset medical --medical_subset 胃_1query \
    --dsname 胃_1query --model ResNet18 --method S2M2_R \
    --n_ways 8 --n_shot 5 --n_queries 1 --n_runs 1000 \
    --output_json checkpoints/medical/胃_1query/ResNet18_S2M2_R/eval_result.json


# 十二指肠_1query -- 7-way 5-shot 1-query
python filelists/medical/write_medical_filelist.py \
    --medical_root /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/data \
    --medical_subset 十二指肠_1query

python train.py --dataset medical --medical_subset 十二指肠_1query \
    --model ResNet18 --method rotation --train_aug \
    --stop_epoch 5 --batch_size 16 --save_freq 1

python train.py --dataset medical --medical_subset 十二指肠_1query \
    --model ResNet18 --method S2M2_R --train_aug \
    --stop_epoch 30 --batch_size 16 --save_freq 5

python save_plk.py --dataset medical --medical_subset 十二指肠_1query \
    --model ResNet18 --method S2M2_R

python test_standard.py \
    --dataset medical --medical_subset 十二指肠_1query \
    --dsname 十二指肠_1query --model ResNet18 --method S2M2_R \
    --n_ways 7 --n_shot 5 --n_queries 1 --n_runs 1000 \
    --output_json checkpoints/medical/十二指肠_1query/ResNet18_S2M2_R/eval_result.json


# 食管_1query -- 8-way 5-shot 1-query
python filelists/medical/write_medical_filelist.py \
    --medical_root /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/data \
    --medical_subset 食管_1query

python train.py --dataset medical --medical_subset 食管_1query \
    --model ResNet18 --method rotation --train_aug \
    --stop_epoch 5 --batch_size 16 --save_freq 1

python train.py --dataset medical --medical_subset 食管_1query \
    --model ResNet18 --method S2M2_R --train_aug \
    --stop_epoch 30 --batch_size 16 --save_freq 5

python save_plk.py --dataset medical --medical_subset 食管_1query \
    --model ResNet18 --method S2M2_R

python test_standard.py \
    --dataset medical --medical_subset 食管_1query \
    --dsname 食管_1query --model ResNet18 --method S2M2_R \
    --n_ways 8 --n_shot 5 --n_queries 1 --n_runs 1000 \
    --output_json checkpoints/medical/食管_1query/ResNet18_S2M2_R/eval_result.json
```

## Original miniImageNet / CUB / CIFAR-FS

```bash
# miniImageNet
python train.py --dataset miniImagenet --method S2M2_R --model WideResNet28_10 --train_aug
python train.py --dataset miniImagenet --method S2M2_R --model ResNet18 --train_aug
# Rotation head needs to be trained first
python train.py --dataset miniImagenet --method rotation --model WideResNet28_10 --train_aug
python train.py --dataset miniImagenet --method rotation --model ResNet18 --train_aug

# Same pattern for CUB / cifar (use train_cifar.py for cifar).
```

## References

* Hu, Y., Gripon, V., & Lefebvre, S. (2020). Leveraging the Feature Distribution in Transfer-based Few-Shot Learning.
* Original repo: <https://github.com/yaoyao-liu/PT-MAP>.

## Acknowledgements

Patches applied by the FSL-scoreboard integration; original code by
the authors above (MIT-style license).