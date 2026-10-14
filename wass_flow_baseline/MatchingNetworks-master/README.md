# Matching Networks for One Shot Learning (patched for PyTorch ≥ 1.7)

A PyTorch implementation of [Matching Networks for One Shot Learning](http://papers.nips.cc/paper/6385-matching-networks-for-one-shot-learning.pdf),
patched so it runs on modern PyTorch and supports a custom medical-image
few-shot dataset alongside the original Omniglot / miniImageNet benchmarks.

## What changed vs. the original repo

This fork rewrites the deprecated parts of the 2017 code so it runs on
PyTorch ≥ 1.7 (verified against `torch 1.7.0+cu110`, which is what your
`fsl_om` environment uses) and on PyTorch ≥ 2.x.

| Fix | Files |
|-----|-------|
| `dict.keys()[i]` → `list(...)` (Py3-only) | `datasets/miniImagenetOneShot.py`, `datasets/omniglotNShot.py` |
| `Variable(..., volatile=True)` removed | `experiments/OneShotMiniImageNet.py`, `experiments/OneShotBuilder.py` |
| `.data[0]` removed → `.item()` | same |
| `nn.Softmax()` (deprecated) → `torch.softmax(..., dim=0)` | `models/AttentionalClassify.py` |
| `init.xavier_uniform` → `init.xavier_uniform_` | `models/Classifier.py` |
| Hard-coded `.cuda()` in `BidirectionalLSTM` | `models/BidirectionalLSTM.py` |
| `tensorboard_logger` no-op fallback | `logger.py` |
| Medical subset loader + configurable n-way / k-shot / k-query | `datasets/miniImagenetOneShot.py`, `option.py`, `mainMiniImageNet.py` |
| Save best model + JSON test result | `experiments/OneShotMiniImageNetBuilder.py` |

## Environment setup

The repo ships a helper script that creates a dedicated `fsl_matching`
conda environment pinned to the same CUDA build your `fsl_om` env uses.

```bash
cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/MatchingNetworks-master
bash setup_env.sh
```

Or manually:

```bash
conda create -n fsl_matching python=3.10
conda activate fsl_matching
python -m pip install "torch==1.7.0+cu110" "torchvision==0.8.1+cu110" \
    --extra-index-url https://download.pytorch.org/whl/cu110
python -m pip install numpy tqdm Pillow tensorboard_logger
```

> If you want PyTorch ≥ 2.x instead, the script defaults are
> `torch 2.1.0 + cu118`. Both work with the patched code.

## Data layout for the medical dataset

Each subset under `data/` must follow:

```
<project_root>/data/<medical_subset>/{train,val,test}/<class>/*.jpg|png|...
```

which is exactly the layout you already have:

```
fsl_scoreboard/data/
├── 胃_1query/
│   ├── train/慢性非萎缩性胃窦炎伴痘疹/01.jpg ...
│   ├── val/...
│   └── test/...
├── 十二指肠_1query/...
└── 食管_1query/...
```

`--dataroot` defaults to `<this repo>/data` whenever `--dataset medical`
is selected, so you don't need to pass it explicitly.

## Run

```bash
conda activate fsl_matching
cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/MatchingNetworks-master
```

### Train + test on 胃_1query (5-way 1-shot, 5 query / class)

```bash
python mainMiniImageNet.py \
    --dataset medical \
    --medical_subset 胃_1query \
    --n_way 5 --k_shot 1 --k_query 5 \
    --channels 3 \
    --batch_size 32 --fce \
    --total_epochs 100 --total_train_batches 200 \
    --total_val_batches 50 --total_test_batches 100 \
    --lr 1e-3 --save_best
```

### Test only (load a saved checkpoint)

```bash
python mainMiniImageNet.py \
    --dataset medical \
    --medical_subset 胃_1query \
    --n_way 5 --k_shot 1 --k_query 5 \
    --channels 3 \
    --eval_only \
    --load ./checkpoints/medical/胃_1query/matching/best.pt \
    --total_test_batches 100
```

### Other subsets

Just change `--medical_subset`. The save folder is namespaced per
subset under `checkpoints/<dataset>/<medical_subset>/matching/`, so the
three (胃 / 十二指肠 / 食管) runs never overwrite each other:

```
MatchingNetworks-master/checkpoints/
└── medical/
    ├── 胃_1query/matching/best.pt + eval_result.json
    ├── 十二指肠_1query/matching/best.pt + eval_result.json
    └── 食管_1query/matching/best.pt + eval_result.json
```

## Omniglot (unchanged)

```bash
python mainOmniglot.py --dataroot /tmp/omniglot
```

The Omniglot dataset is auto-downloaded on first run.

## Acknowledgements

Original code by Albert Berenguel (CVC, UAB). 2017-era Python-2
implementation; see [LICENSE.md](LICENSE.md).