# GastroFusion

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

[![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange.svg)](https://pytorch.org/)

**Geometry-aware multimodal AI improves few-shot gastrointestinal diagnosis under limited data**

English | [简体中文](README_CN.md)

---

## Contents

- [Overview](#overview)

- [Key Features](#key-features)

- [Changelog](#changelog)

- [System Requirements](#system-requirements)

- [Installation](#installation)

- [Model Weights and Demo Data](#model-weights-and-demo-data)

- [Quick Reproduction Demos](#quick-reproduction-demos)

  - [Demo A — Wasserstein-GastroFlow Inference and Evaluation](#demo-a--wasserstein-gastroflow-inference-and-evaluation)

  - [Demo B — GastroFusion Geometric–Semantic Fusion Inference](#demo-b--gastrofusion-geometricsemantic-fusion-inference)

  - [Baseline Evaluation — Few-Shot and Long-Tail Recognition](#baseline-evaluation--few-shot-and-long-tail-recognition)

- [Training](#training)

- [Project Structure](#project-structure)

- [Visualizations](#visualizations)

- [Citation](#citation)

- [Acknowledgements](#acknowledgements)

- [Data Availability](#data-availability)

- [License](#license)

- [Contact](#contact)

---

## Overview

**GastroFusion** is a multimodal AI framework for upper gastrointestinal endoscopic image diagnosis. It addresses challenges arising from long-tailed disease distributions, limited annotations for rare lesions, and cross-centre data shifts. The framework combines few-shot trajectory-based geometric matching from **Wasserstein-GastroFlow** with medical semantic evidence from **GastroMLLM (Med-Qwen3-VL)**, integrating their predictions through organ-specific fusion rules.

| Core component | Main function |

|---|---|

| **Wasserstein-GastroFlow** | Few-shot matching using support-conditioned latent-space trajectories and Sinkhorn-approximated distributional transport costs. |

| **GastroMLLM (Med-Qwen3-VL)** | A Qwen3-VL-8B-based medically adapted multimodal model providing semantic scores for candidate diagnoses. |

| **GastroFusion fusion module** | Integrates geometric matching and semantic prediction evidence using organ-specific rules. |

**Main experimental findings:**

- **Few-shot baseline comparison:** Across six internal organ–support-protocol settings, all three trajectory scoring schemes achieved higher mean accuracy than eight FSL baselines. The best-performing variant in each setting exceeded the strongest FSL baseline by **10.86–42.00 percentage points** and the strongest long-tail recognition baseline by **1.25–61.18 percentage points** (the latter is a contextual, non-episodic comparison).

- **External validation:** Three independent external centres contributed **352 images across nine centre–organ cohorts**. Relative to DeepEMD, the three trajectory schemes improved macro accuracy by **8.70–10.14 percentage points** with target-domain support and **13.91–19.80 percentage points** with source-domain support.

- **Physician comparison:** On a fixed **166-case** internal WCH reader-study cohort, GastroFusion achieved **86.00%** organ-macro accuracy, compared with **64.85%** for senior endoscopists.

- **Multimodal evaluation:** GastroMLLM achieved **94.99%** accuracy on **OmniMedVQA-GI (1,537 questions)**.

- **Fusion complementarity:** External fusion yielded numerical gains of **1.24** and **0.75 percentage points** under cohort-aligned and full-organ label spaces, respectively. A branch-selection oracle revealed complementary diagnostic coverage between geometric and semantic predictions.

![GastroFusion Workflow](assets/figure/overview-ill.png)

---

## Key Features

- **Few-shot and long-tail diagnosis:** Matches candidate diseases using limited labelled support examples.

- **Trajectory-based geometric matching:** Combines RAE representations, conditional DiT Flow, and Sinkhorn transport costs.

- **Medical semantic modelling:** Med-Qwen3-VL provides complementary semantic prediction evidence.

- **Organ-specific fusion:** Selectively combines geometric and semantic scores.

- **Multicentre validation:** Includes independent external centres, different candidate-label spaces, and physician reader comparisons.

- **Modular implementation:** Separates geometric modelling, semantic modelling, fusion, and baseline evaluation.

---

## Changelog

- **2026-10-15:** Standardized the GastroFusion naming; updated MLLM and Wasserstein-GastroFlow training/evaluation entry points; adopted the current fusion pipeline; and documented baseline training, adaptation, and testing.

---

## System Requirements

Different modules use **separate environments**.

| Module | Environment |

|---|---|

| **GastroMLLM training** | Python 3.11 and CUDA 12.4; the recorded training environment uses ms-swift 4.0.3, PyTorch 2.8.0, and related packages. See [`mllm/README_CN.md`](mllm/README_CN.md). |

| **Wasserstein-GastroFlow** | Uses the `flow` environment. See [`wass_flow/README_CN.md`](wass_flow/README_CN.md) for GPU/CUDA requirements and script settings. |

| **GastroFusion fusion** | Runs the MLLM service, FSL inference, and fusion scripts separately; see the scripts under `mllm_fsl/`. |

| **Baselines** | Dependencies and checkpoints vary by method; see [`wass_flow_baseline/README_CN.md`](wass_flow_baseline/README_CN.md). |

Linux with NVIDIA GPUs is recommended. Required GPU count, memory, and runtime depend on the model and experiment configuration. Multi-GPU training configurations should not be interpreted as minimum requirements for single-GPU inference.

---

## Installation

### 1) Clone the repository

```bash

git clone https://github.com/GastroAgent/GastroFusion.git

cd GastroFusion

```

### 2) Set up module-specific environments

**Example: GastroMLLM training environment** (refer to its module README for the full setup):

```bash

conda create -n gastrofusion python=3.11 -y

conda activate gastrofusion

cd mllm

pip install -r requirements.txt

cd ..

```

**Wasserstein-GastroFlow:** Configure the `flow` environment and dependencies as described in [`wass_flow/README_CN.md`](wass_flow/README_CN.md). Training, checkpoint loading, and evaluation require compatible CUDA and PyTorch versions.

---

## Model Weights and Demo Data

The released archives are available from [Hugging Face — GastroFusion](https://huggingface.co/GastroAgent/GastroFusion).

| Archive | Contents and use |
|---|---|
| `demo_data.zip` | Stomach endoscopy images for Demo A, organized by disease class for episodic test-test evaluation. |
| `demo-wass-flow-stomach.zip` | Wasserstein-GastroFlow stomach evaluation checkpoints, including F1/Decay, M2, DiT Flow and related checkpoints. |
| `Med-Qwen3-VL-8B-Instruct.zip` | GastroMLLM-related model files. |

Download and extract `demo_data.zip` and `demo-wass-flow-stomach.zip` for Demo A. The F1 evaluation script accepts the extracted **class-image directory** and **checkpoint directory** through `DATA_ROOT` and `WEIGHT_ROOT`; the directory names are not hard-coded.

RAE Decoder, RAE normalization statistics, EndoViT, and DINOv3 ViT-L/16 are additional runtime dependencies. Place them at the paths expected by the unchanged Python loader (see Demo A below and [`wass_flow/README_CN.md`](wass_flow/README_CN.md)).

---

## Quick Reproduction Demos

Below, `<PROJECT_ROOT>` denotes the repository root. Configure model, data, and output paths for your environment. **Check the paths and checkpoints defined inside the scripts before execution.**

### Demo A — Wasserstein-GastroFlow Inference and Evaluation

Demo A provides two stomach test-test few-shot evaluation entry points using the released stomach endoscopy images organized by disease class:

- **F1 / Decay:** Support-conditioned trajectory evaluation with the F1/Decay, M2, and DiT checkpoints.
- **Q2L / Q2S-Retained:** Evaluation of endpoint matching (`Q2Last` in the code), retained-step aggregation (`Q2Mono` in the code), and their score-level fusion using the stomach Q2L and DiT checkpoints.

The default evaluation configuration is **5-shot, 1-query, 100 episodes, seed 42**.

#### Evaluation protocol: test-pool support (test–test)

This demo follows the internal test-pool support setting described in the paper.
The released gastric images are organized by disease category and used to
construct both support and query sets during inference. No separate
training-pool support images need to be downloaded.

Each episode samples one query image from this pool and constructs a
5-shot support set for each candidate class. The query image is excluded
from its corresponding support candidates, preventing the same image
from appearing in both the query and support sets within an episode.

The demo loads pretrained checkpoints and performs inference and evaluation
without updating model parameters. This setting evaluates classification
performance when a small number of locally labelled reference images
are available.


**1. Download and extract the Demo A archives:**

```bash
# Run from the GastroFusion repository root.
mkdir -p demo_data demo_weights
unzip /path/to/demo_data.zip -d demo_data
unzip /path/to/demo-wass-flow-stomach.zip -d demo_weights
```

The evaluation scripts expect the following default directory structure:

```text
GastroFusion/
├── demo_data/
│   └── stomach/
│       ├── <disease class A>/
│       └── <disease class B>/
└── demo_weights/
    └── wass-flow-stomach/
        ├── f1_v5_support_lstm_medical/
        ├── new_m1/
        ├── stomach-test-test/
        └── otcfm_dit_L_step_265000.pt
```

If your extracted directories differ, set `DATA_ROOT` and `WEIGHT_ROOT` to their actual locations.

**2. Run the evaluation scripts:**

```bash
conda activate flow
cd /path/to/GastroFusion/wass_flow

# F1 / Decay
bash eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_m2_dit.sh

# Q2L / Q2S-Retained / score-level fusion
bash eval_q2l_q2last_q2mono_scorefuse_maxmin_testtest.sh
```

Both evaluation scripts support `DATA_ROOT` and `WEIGHT_ROOT` environment variables for custom data and checkpoint paths.

For additional training, evaluation, and configuration details, see [`wass_flow/README_CN.md`](wass_flow/README_CN.md).

### Demo B — GastroFusion Geometric–Semantic Fusion Inference

This pipeline generates **GastroMLLM semantic scores** and **Wasserstein-GastroFlow geometric scores** separately, selects fusion parameters on independent validation data, and then performs **GastroFusion fusion prediction** using fixed parameters. The two scoring branches can run independently; neither must precede the other. Run the following commands from the repository root, selecting the scripts appropriate to your dataset.

**1) GastroMLLM semantic prediction**

```bash
# Start the MLLM model service first.
bash mllm_fsl/mllm/mllm_code/gastro-mllm-server.sh

# Internal WCH cohorts
bash mllm_fsl/mllm/mllm_code/run_batch_mllm_huaxi.sh
```

Result directory: `mllm_fsl/mllm/mllm_result/`.

**2) Wasserstein-GastroFlow geometric prediction**

```bash
# Internal WCH cohorts
bash mllm_fsl/fsl/fsl_code/run_batch_3huaxi_5shot.sh
```

Result directory: `mllm_fsl/fsl/fsl_result/`. Offline geometric scoring typically produces pair-level `result.json` and query-aggregated `query_results.json` files.

**3) Parameter selection on an independent validation set (`val_grid/`)**

`mllm_fsl/mllm_fsl_detail/mllm_fsl_code/val_grid/val_grid_search.py` compares candidate organ-specific threshold combinations using validation-set MLLM and FSL results. Once selected, numerical thresholds are explicitly specified in the final fusion shell script. The final test set is not used to reselect parameters.

**4) GastroFusion fusion prediction**

```bash
# Internal WCH fusion
bash mllm_fsl/mllm_fsl_detail/mllm_fsl_code/fusion/fsl_mllm_fusion_tolast_rae_v2_huaxi_grid_val.sh
```

Result directory: `mllm_fsl/mllm_fsl_detail/mllm_fsl_result/`. The fusion Python program fixes the three organ-specific strategies: `gap_guard_fusion` for the stomach, `high_conf_override` for the esophagus, and `very_high_conf_override` for the duodenum. **All eight numerical thresholds are explicitly supplied by the shell script.** Only the main MLLM JSON files and offline RAE FSL results are needed. Before running fusion, verify that both branches refer to **the same evaluation cohort and candidate-label space**.

**5) Parameter sensitivity analysis (optional)**

`sensitivity/` examines the stability of the results by perturbing parameters around the fixed fusion configuration; it is **not used to reselect optimal parameters on the test set**.

### Baseline Evaluation — Few-Shot and Long-Tail Recognition

`wass_flow_baseline/` provides training and evaluation code for **eight few-shot learning (FSL) baselines** and **three long-tail recognition baselines** used in the manuscript. Methods are trained, fine-tuned, adapted, or evaluated with appropriately trained weights and the corresponding medical data splits. **Not all methods are trained from scratch.** Some methods directly load existing medical checkpoints; SimpleShot reuses pretrained PT-MAP weights, and FSL-OM and PT-MAP can load their corresponding medical-experiment checkpoints.

- **FSL baselines:** Evaluated using the corresponding N-way K-shot episodic protocols, distinguishing test-pool support from training-pool support.
- **Long-tail baselines:** Evaluated as contextual comparators using non-episodic classification; their evaluation protocol is not identical to episodic FSL.
- **Data separation:** Test-pool support examples are supplied at inference time; this does not imply optimizing model parameters using test-query images.

For training/testing entry points, checkpoints, and data configurations, see [`wass_flow_baseline/README_CN.md`](wass_flow_baseline/README_CN.md).

---

## Training

GastroMLLM and Wasserstein-GastroFlow are trained independently. Within Wasserstein-GastroFlow, metric learning and DiT Flow Matching can proceed in parallel, followed by F1/Decay trajectory classification training. The resulting predictions are then combined during GastroFusion fusion evaluation.

```text

GastroMLLM                          Wasserstein-GastroFlow

    SFT → GRPO                      ├─ Metric: M0-A → M0-B → M1 → M2

         │                          └─ DiT Flow Matching (in parallel)

   Med-Qwen3-VL                                    │

         │                              F1 / Decay trajectory training

         │                                        │

   Semantic scores                          Geometric scores

         └──────────────────┬─────────────────────┘

                            ↓

                   GastroFusion fusion

                            ↓

                     Final prediction

```

### 1) GastroMLLM (Med-Qwen3-VL)

Built on **Qwen3-VL-8B**, GastroMLLM uses ms-swift for data construction, single-image/multi-image **supervised fine-tuning (SFT)**, inter-stage LoRA merging, and **GRPO** reinforcement learning to produce medically adapted semantic predictions.

Training data are constructed by `build_sft_rl_data.py`, including teacher-model generation, visual judging, and resumable data processing, and are stored under `mllm/data/pretrain/`. `merge_lora.py` / `merge_lora.sh` merge weights between iterative SFT rounds. GRPO uses `medical_vqa_reward.py` for option-accuracy and final-answer-format rewards. With `--top_entropy_quantile 0.20`, only approximately the highest-entropy 20% of completion tokens in each batch contribute to the policy-gradient loss.

Training entry points:

```bash
cd mllm

# SFT
bash sft_network_auto.sh
# Multi-node SLURM: sbatch train_multinode_sft_network_auto.sh

# GRPO
bash rl_medical_grpo_colocate_fixed_v2_entropy20.sh
# Multi-node SLURM: sbatch train_multinode_medical_grpo_network_auto_entropy20.sh
```

For the current configuration, SFT uses LoRA rank 64; GRPO uses colocated vLLM and `top_entropy_quantile=0.20`, with multi-node SLURM launchers available. See [`mllm/README.md`](mllm/README.md) for data construction, LoRA merging, dependencies, hyperparameters, and checkpoints.

### 2) Wasserstein-GastroFlow

Using RAE, conditional DiT Flow Matching, and SK-Wasserstein metric matching, the training sequence consists of:

1. **M0-A → M0-B → M1 → M2:** Classification pretraining, SK-Wasserstein metric pretraining, medical-domain adaptation, and Direct episodic calibration.

2. **DiT Flow Matching:** Independent training of the support-conditioned trajectory generator, which may run alongside M0–M2.

3. **F1 / Decay:** Training a trajectory LSTM decay head with RAE, M2 MetricEncoder, and DiT Flow frozen. Direct, Q2L, and Q2S-Retained remain available as alternative geometric scoring schemes.

Training entry points (the commented Python examples require the mandatory arguments documented in the module README):

```bash

cd wass_flow

# Metric learning: run in sequence

# torchrun --nproc_per_node=8 train_m0a_cls.py <ARGS>

# torchrun --nproc_per_node=8 train_m0b_ot.py <ARGS>

# torchrun --nproc_per_node=8 train_m1_medical_metric.py <ARGS>

# torchrun --nproc_per_node=8 train_m2_direct_episode_visreg.py <ARGS>

# DiT Flow Matching

sbatch train_dit_flow_spatial_sinkhorn_strict_intra_ot.sh

# F1 pretraining

sbatch submit_multinode_pretrain_f1_support_conditioned_lstm_v5.sh

```

Medical-domain F1 adaptation uses the `train_f1_support_conditioned_lstm_v5_medical_manifest.*` entry points. For full parameters and evaluation protocols, see [`wass_flow/README_CN.md`](wass_flow/README_CN.md).

### 3) GastroFusion Fusion

This stage combines separately generated MLLM semantic scores and offline FSL geometric scores; the backbone models are not retrained by the fusion scripts. `val_grid/` selects parameters on independent validation data; `fusion/` applies the eight fixed numerical thresholds supplied by the shell script for organ-specific fusion; and `sensitivity/` evaluates the effects of parameter perturbations. Test labels are not used to recalibrate fusion parameters during evaluation. See [Demo B](#demo-b--gastrofusion-geometricsemantic-fusion-inference) for execution entry points.

---

## Project Structure

The main code directories are organized as follows:

```text
GastroFusion/
├── assets/
│   └── figures/                  # README figures and result visualizations
├── mllm/                         # Med-Qwen3-VL data, SFT, LoRA merging, GRPO training
│   └── README.md
├── wass_flow/                    # RAE + DiT Flow + Metric + F1 training/evaluation
│   └── README.md
├── mllm_fsl/                     # Geometric–semantic inference and fusion
│   ├── data/                     # MLLM inputs and FSL query–support pair JSON files
│   ├── mllm/                     # Semantic serving, batch scoring, and results
│   ├── fsl/                      # Offline RAE/Metric/Flow geometric scoring and results
│   ├── mllm_fsl_detail/          # val_grid, fusion, sensitivity, and results
│   └── README_CN.md
├── wass_flow_baseline/           # Eight FSL and three long-tail baselines
│   ├── BPaCo-main/
│   ├── classifier-balancing-main/
│   ├── DeepEMD/
│   ├── FSL-OM-main/
│   ├── GAP/
│   ├── LDAM-DRW-master/
│   ├── pmf_cvpr22/
│   ├── prototypical-network-pytorch/
│   ├── PT-MAP-master/
│   ├── simple_shot-master/
│   ├── simple-cnaps-master/
│   └── README_CN.md
├── requirements.txt             # Root dependencies, if provided; not universal
├── README.md
└── README_CN.md
```

---

## Visualizations

![Long-tail performance](assets/figure/kshot-case.png)

---

## Citation

If this project contributes to your research, please cite the corresponding manuscript.

```bibtex

@misc{GastroFusion2026,

  title = {Geometry-aware multimodal AI improves few-shot gastrointestinal diagnosis under limited data},

  author = {Shuyue Tan and Nuoya Zhou and Mingfei Jin and Xingwu Liu and Xianglei Yuan and Yi Mou and Linzhuang Sun and Yahui Zhang and Sizhe Qiu and Wei Liu and Yuhang Zhang and Yuyan Zhang and Qi Luo and Ruide Liu and Huan Jiang and Pinghua Wen and Shunyao Wang and Hengduo Jiang and Xuan Liang and Meiling Shi and Zhilie Jiang and Dongbo Bu and Bing Hu and Ming Li and Shuaicheng Li},

  year = {2026},

  note = {Manuscript under preparation or review}

}

```

---

## Acknowledgements

We thank the collaborating clinical centres, participating endoscopists, and the open-source community. This project was developed using [PyTorch](https://pytorch.org/) and other open-source software.

---

## Data Availability

This study uses an internal clinical cohort from WCH and data from three independent external centres. To support reproducibility, the project provides a subset of inference/evaluation data from the internal stomach experiments and the corresponding code. Other clinical images are not publicly released because of patient privacy, ethics, and data-use restrictions. Refer to the public repository for available files and to the manuscript's Data Availability statement for access conditions.

---

## License

This project is released under the [Apache-2.0 License](LICENSE) (subject to the actual `LICENSE` file in the repository).

**Research use only:** The framework is not intended for direct clinical diagnosis or as a substitute for professional medical judgement.

---

## Contact

- Issues: [GitHub Issues](https://github.com/GastroAgent/GastroFusion/issues)

- Email: shuyuetan0@gmail.com

---

## Star History

If you find this project useful, please consider starring the repository.

[![Star History Chart](https://api.star-history.com/svg?repos=GastroAgent/GastroFusion&type=Date)](https://star-history.com/#GastroAgent/GastroFusion&Date)
