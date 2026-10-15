# mllm — Qwen3-VL 医学 VQA（SFT + GRPO）

基于 **ms-swift** 的医疗大模型训练流水线：

```text
数据构建（build_sft_rl_data.py）
→ SFT 冷启动（sft_network_auto.sh 系列）
→ GRPO 强化学习（top-entropy token 过滤）
```

> **范围说明**：本目录只覆盖**训练**；部署与使用在 `mllm_fsl/`，不在本 README 范围。

## 1. 目录结构

```text
data/                       数据与数据构建脚本（huaxi、Gastrovision、RL json、系统提示词等）
weights/                    SFT / RL 各阶段 checkpoint（Qwen3-VL-8B-Instruct-3epoch-*、RL1-best 等）
logs/  output/              训练日志与输出
build_sft_rl_data.py        SFT 冷启动 + RL held-out 数据构建（teacher 生成 + 视觉裁判 + SQLite 断点续跑）
medical_vqa_reward.py       GRPO 奖励插件（accuracy + format 加权）
merge_lora.py / merge_lora.sh  LoRA 权重合并（SFT 多轮迭代时衔接用）
sft_network_auto.sh         SFT 训练（swift sft，单节点或多节点共用）
train_multinode_sft_network_auto.sh  SFT 多机 SLURM 启动器
rl_medical_grpo_colocate_fixed_v2_entropy20.sh           GRPO 训练（colocate 模式）
train_multinode_medical_grpo_network_auto_entropy20.sh   GRPO 多机 SLURM 启动器
```

## 2. 环境

- CUDA 12.4（`/mnt/inaisfs/data/apps/cuda12.4.sh`）
- conda 环境 `agentic_crop`（训练、vLLM 服务），Python 3.11：

```bash
conda create -n agentic_crop python=3.11 -y
conda activate agentic_crop
pip install -r requirements.txt
```

核心依赖见 [`requirements.txt`](requirements.txt)（ms-swift 4.0.3、vllm 0.11.0、torch 2.8.0、transformers 4.57.6 等，版本为训练时实测）。

## 3. 数据

来源：华西（huaxi）数据 + 开源 Gastrovision。训练数据在 `data/pretrain/`，按 4 个阶段组织（可用 [`data/stat_data_scale.py`](data/stat_data_scale.py) 重新统计）：

| 阶段 | 文件 | 规模 |
| --- | --- | --- |
| SFT 单图 | `pretrain/Single_Image/train_sft.json` | 744 766 条 |
| | `pretrain/Single_Image/train_sft2.json` | 623 712 条 |
| SFT 多图 | `pretrain/Multi_Image/train_sft_f_8_clean1.json`（当前脚本使用） | 66 505 条 |
| | `pretrain/Multi_Image/train_sft_f_12_clean1.json` | 91 873 条 |
| | `pretrain/Multi_Image/train_sft_f_18_clean1.json` | 105 988 条 |
| RL 第 1 轮 | `pretrain/RL_1/train.json` | 13 396 条 |
| RL 第 2 轮 | `pretrain/RL_2/train.json`；`filtered_train.json`（GRPO 脚本实际使用） | 15 532 条；5 354 条 |

另外，顶层 `data/*.json` 与 `cold_start_huaxi*/` 是 `build_sft_rl_data.py`（teacher 生成 + 视觉裁判 + 划分）的产物（如 cold_start_huaxi：SFT 6 814 条 / RL 10 233 条）。

## 4. SFT

| 脚本 | 用途 |
| --- | --- |
| `train_multinode_sft_network_auto.sh` | SLURM 多机启动器（每节点一个 task） |
| `sft_network_auto.sh` | 每节点执行的 `swift sft` |

当前配置：基座 `weights/Qwen3-VL-8B-Instruct-917/Qwen3-VL-8B-Instruct`（上一轮合并产物），LoRA r64、lr 1e-4、3 epochs，数据 `pretrain/Multi_Image/train_sft_f_8_clean1.json`，产出 `output/Qwen3-VL-8B-Instruct-918`。

## 5. GRPO 强化学习

入口：`rl_medical_grpo_colocate_fixed_v2_entropy20.sh`（colocate：vLLM rollout 与训练同卡混部，`vllm_mode colocate`，TP=2，显存占比 0.40）；多机经 `train_multinode_medical_grpo_network_auto_entropy20.sh` 启动。

**entropy20 的含义**（已对照本地 ms-swift 4.0.3 源码确认）：即 `--top_entropy_quantile 0.20` —— 每个 batch 对所有 completion token 的熵计算 `1-q` 分位阈值，**只有熵位于最高 20% 的 token 参与策略梯度 loss**，低熵（近乎确定）token 被过滤；默认 1.0 表示不过滤。梯度集中在模型不确定的 token 上，用于缓解熵坍缩。脚本同时开了 `--log_entropy true` 记录熵动态。

奖励：`medical_vqa_reward.py` —— `accuracy_weight * 选项精确命中 + format_weight * final_answer_format`。

关键超参（脚本实值）：

| 参数 | 值 |
| --- | --- |
| 初始化 | `weights/Qwen3-VL-8B-Instruct-RL1-last/Qwen3-VL-8B-Instruct` |
| LoRA | r64 / α32 / all-linear，freeze ViT + aligner |
| 学习率 / epochs | 1e-5 / 3 |
| num_generations | 训练 16，评估 2 |
| beta（KL）/ clip | 0.04；ε 0.20，ε_high 0.28 |
| importance_sampling_level | sequence |
| max_completion_length / max_length | 512 / 7168 |
| 温度 | 1.0 |
| 其他 | deepspeed zero2、dynamic_sample true、max_resample_times 1 |

产出：`output/medical-vqa-grpo-qwen3vl8b-colocate-<job_id>`。
