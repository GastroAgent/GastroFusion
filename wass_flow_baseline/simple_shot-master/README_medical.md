# SimpleShot Medical Evaluation — README

This directory contains a **drop-in reproduction pipeline** that adapts
`simple_shot-master` (https://arxiv.org/abs/1911.04623) to the medical
image datasets under `@fsl_scoreboard/data`.

## What this pipeline does

1. Converts PT-MAP’s JSON filelists (absolute paths) into SimpleShot’s CSV
   split format (relative paths).
2. Loads a **PT-MAP pretrained ResNet-18** backbone and extracts ImageNet-style
   features for every train / test image.
3. Runs the **SimpleShot few-shot evaluation** (UN / L2N / CL2N
   nearest-centroid classifiers) over **N-way K-shot Q-query** episodes,
   with the same task sampling protocol as the reference result.

Reference PT-MAP result for 食管_1query:

```json
{
  "dataset": "食管_1query",
  "n_ways": 8,
  "n_shot": 5,
  "n_queries": 1,
  "n_runs": 10000,
  "test_acc": 27.60,
  "pm": 0.26,
  "lam": 10.0,
  "alpha": 0.2,
  "beta": 0.5
}
```

## Directory layout

```
simple_shot-master/
├── generate_medical_csv.py        # Step 1: JSON → CSV conversion
├── eval_medical.py                # Step 2+3: extract features + evaluate
├── configs/
│   └── medical/
│       └── resnet18_食管_1query.config
├── split/
│   ├── 食管_1query/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── 胃_1query/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── 十二指肠_1query/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── README_medical.md              # This file
```

## Environment requirements

### Conda environment (recommended)

A ready-made environment is already available:

```bash
conda activate fsl_ptmap
# python 3.10, torch 2.1.0+cu118, torchvision 0.16.0, numpy<2, matplotlib, tqdm, Pillow
```

If you need to recreate it, see `@fsl_scoreboard/PT-MAP-master/setup_env.sh`.

### Manual pip install

```bash
pip install torch>=1.10 torchvision numpy<2 matplotlib tqdm Pillow
```

A GPU is **strongly recommended** (evaluation runs 10 000 episodes).

## Quick start: reproduce the 食管_1query result

All commands below assume your **current working directory is
`simple_shot-master/`**.

### Step 1 — generate CSV splits (already done)

```bash
python generate_medical_csv.py \
    --subset 食管_1query \
    --filelist_dir ../PT-MAP-master/filelists/medical/ \
    --data_dir   ../PT-MAP-master/data/ \
    --out_dir    ./split/
```

CSV files are now in `split/食管_1query/{train,val,test}.csv`.

### Step 2 — extract features and evaluate

Full pipeline (extract features + run SimpleShot evaluation):

```bash
python eval_medical.py \
    --subset     食管_1query \
    --model      ResNet18 \
    --method     S2M2_R \
    --n_ways     8 \
    --n_shot     5 \
    --n_queries  1 \
    --n_runs     10000 \
    --num_classes 8
```

For a **quick sanity check** (fewer episodes, no progress bars):

```bash
python eval_medical.py \
    --subset        食管_1query \
    --n_runs        200 \
    --disable_tqdm \
    --seed          42
```

### Step 3 — check the output

The script prints UN / L2N / CL2N accuracies with 95 % confidence intervals.
Use `--output_json eval_result.json` to save results for comparison.

```bash
python eval_medical.py \
    --subset        食管_1query \
    --n_runs        10000 \
    --output_json   ./eval_result.json \
    --seed          0
```

## Other medical subsets

The same commands work for 胃_1query and 十二指肠_1query; just change
`--subset` and the episode parameters:

| Subset | n_ways | n_shot | n_queries |
|--------|--------|--------|-----------|
| 胃_1query | 8 | 5 | 1 |
| 十二指肠_1query | 7 | 5 | 1 |
| 食管_1query | 8 | 5 | 1 |

```bash
# Example: 胃_1query
python eval_medical.py \
    --subset     胃_1query \
    --n_ways     8 \
    --n_shot     5 \
    --n_queries  1 \
    --n_runs     10000 \
    --num_classes 8
```

## How it works

| Component | File | Role |
|-----------|------|------|
| `generate_medical_csv.py` | Converts PT-MAP JSON filelists → SimpleShot CSV |
| `eval_medical.py` | End-to-end: loads PT-MAP pretrained weights, extracts features, runs 10 000 episodes of N-way K-shot evaluation |

### Feature extraction

`eval_medical.py` loads the **latest PT-MAP checkpoint** (e.g. `69.tar`)
from:

```
../PT-MAP-master/checkpoints/medical/<subset>/<model>_<method>/
```

It extracts the penultimate feature vector (512-d for ResNet-18) for every
image in `train.csv` and `test.csv`.  Results are cached as pickle files so
repeated runs don’t need re-extraction.

### SimpleShot evaluation

The core loop is in `eval_medical.py::meta_evaluate()`:

1. **Sample an episode**: randomly pick `n_ways` classes, draw `n_shot` support
   + `n_query` query images per class.
2. **Build class prototype**: mean of support features.
3. **Classify queries**: nearest-centroid under three feature transforms:
   - **UN**  — raw features
   - **L2N** — L2-normalised features
   - **CL2N** — centre-cancelled + L2-normalised features
4. **Aggregate** over `n_runs` episodes → mean accuracy ± 95 % CI.

## Notes and caveats

- **Feature caching**: features are cached in `../PT-MAP-master/checkpoints/medical/<subset>/<model>_<method>/last/`.  Delete the `*_features.pkl` files to force re-extraction.
- **GPU memory**: evaluation is CPU-friendly once features are cached;
  feature extraction needs a GPU with ~4 GB VRAM (ResNet-18).
- **Reproducibility**: set `--seed` for deterministic task sampling; the
  `setRandomStates()` logic mirrors PT-MAP’s original implementation.
