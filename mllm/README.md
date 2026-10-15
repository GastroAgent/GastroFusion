# mllm — Qwen3-VL Medical VQA (SFT + GRPO)

A medical multimodal large language model training pipeline based on **ms-swift**:

```text
Data construction (build_sft_rl_data.py)
→ SFT cold start (sft_network_auto.sh series)
→ GRPO reinforcement learning (top-entropy token filtering)
```

> **Scope:** This directory covers **training only**. Deployment and inference are handled in `mllm_fsl/` and are outside the scope of this README.

## 1. Directory Structure

```text
data/                       Data and data-building scripts (Huaxi, Gastrovision, RL JSON, system prompts, etc.)
weights/                    Checkpoints from SFT/RL stages (Qwen3-VL-8B-Instruct-3epoch-*, RL1-best, etc.)
logs/  output/              Training logs and outputs
build_sft_rl_data.py        SFT cold-start and held-out RL data construction (teacher generation, visual judge, SQLite resumability)
medical_vqa_reward.py       GRPO reward plugin (weighted accuracy + format)
merge_lora.py / merge_lora.sh  LoRA weight merging (between iterative SFT rounds)
sft_network_auto.sh         SFT training (swift sft; shared by single-node and multi-node runs)
train_multinode_sft_network_auto.sh  Multi-node SLURM launcher for SFT
rl_medical_grpo_colocate_fixed_v2_entropy20.sh           GRPO training (colocate mode)
train_multinode_medical_grpo_network_auto_entropy20.sh   Multi-node SLURM launcher for GRPO
```

## 2. Environment

- CUDA 12.4 (`/mnt/inaisfs/data/apps/cuda12.4.sh`)
- Conda environment `agentic_crop` (training and vLLM serving), Python 3.11:

```bash
conda create -n agentic_crop python=3.11 -y
conda activate agentic_crop
pip install -r requirements.txt
```

Core dependencies are listed in [`requirements.txt`](requirements.txt), including ms-swift 4.0.3, vLLM 0.11.0, PyTorch 2.8.0, and Transformers 4.57.6. These versions correspond to the tested training environment.

## 3. Data

Data sources include Huaxi (WCH) and the publicly available Gastrovision dataset. Training data are stored in `data/pretrain/` and organized into four stages. Statistics can be recomputed with [`data/stat_data_scale.py`](data/stat_data_scale.py).

| Stage | File | Number of records |
| --- | --- | ---: |
| Single-image SFT | `pretrain/Single_Image/train_sft.json` | 744,766 |
| | `pretrain/Single_Image/train_sft2.json` | 623,712 |
| Multi-image SFT | `pretrain/Multi_Image/train_sft_f_8_clean1.json` (used by the current script) | 66,505 |
| | `pretrain/Multi_Image/train_sft_f_12_clean1.json` | 91,873 |
| | `pretrain/Multi_Image/train_sft_f_18_clean1.json` | 105,988 |
| RL round 1 | `pretrain/RL_1/train.json` | 13,396 |
| RL round 2 | `pretrain/RL_2/train.json`; `filtered_train.json` (used by the GRPO script) | 15,532; 5,354 |

In addition, top-level `data/*.json` files and `cold_start_huaxi*/` directories are outputs of `build_sft_rl_data.py` (teacher generation, visual judging, and data splitting). For example, `cold_start_huaxi` contains 6,814 SFT records and 10,233 RL records.

## 4. Supervised Fine-Tuning (SFT)

| Script | Purpose |
| --- | --- |
| `train_multinode_sft_network_auto.sh` | Multi-node SLURM launcher (one task per node) |
| `sft_network_auto.sh` | Runs `swift sft` on each node |

**Current configuration:** Base model `weights/Qwen3-VL-8B-Instruct-917/Qwen3-VL-8B-Instruct` (merged output from the preceding round); LoRA rank 64; learning rate `1e-4`; 3 epochs; training data `pretrain/Multi_Image/train_sft_f_8_clean1.json`; output `output/Qwen3-VL-8B-Instruct-918`.

## 5. GRPO Reinforcement Learning

**Entry point:** `rl_medical_grpo_colocate_fixed_v2_entropy20.sh`. In colocate mode, vLLM rollouts and training share GPUs (`vllm_mode colocate`, tensor parallelism `TP=2`, GPU memory utilization `0.40`). Multi-node runs are launched through `train_multinode_medical_grpo_network_auto_entropy20.sh`.

**Meaning of `entropy20`:** As verified against the local ms-swift 4.0.3 source, `--top_entropy_quantile 0.20` computes the `1-q` quantile threshold of completion-token entropy within each batch. **Only the highest-entropy 20% of tokens contribute to the policy-gradient loss**; lower-entropy (more certain) tokens are filtered out. The default value `1.0` disables filtering. This concentrates gradient updates on uncertain tokens and is intended to mitigate entropy collapse. The script also enables `--log_entropy true` to track entropy dynamics.

**Reward:** `medical_vqa_reward.py` implements:

```text
accuracy_weight * exact_option_match + format_weight * final_answer_format
```

**Key hyperparameters** (values from the training script):

| Parameter | Value |
| --- | --- |
| Initialization | `weights/Qwen3-VL-8B-Instruct-RL1-last/Qwen3-VL-8B-Instruct` |
| LoRA | rank 64 / alpha 32 / all-linear; ViT and aligner frozen |
| Learning rate / epochs | `1e-5` / `3` |
| `num_generations` | 16 during training; 2 during evaluation |
| `beta` (KL) / clipping | `0.04`; `epsilon=0.20`, `epsilon_high=0.28` |
| `importance_sampling_level` | `sequence` |
| `max_completion_length` / `max_length` | `512` / `7168` |
| Temperature | `1.0` |
| Other settings | DeepSpeed ZeRO-2; `dynamic_sample=true`; `max_resample_times=1` |

**Output:** `output/medical-vqa-grpo-qwen3vl8b-colocate-<job_id>`.
