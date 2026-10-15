# GastroFusion

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange.svg)](https://pytorch.org/)

**几何感知多模态人工智能提升有限数据条件下的胃肠道疾病少样本诊断性能**  
*Geometry-aware multimodal AI improves few-shot gastrointestinal diagnosis under limited data*

[English](README_EN.md) | 简体中文

---

## 目录

- [简介](#简介)
- [主要特性](#主要特性)
- [更新日志](#更新日志)
- [系统要求](#系统要求)
- [安装指南](#安装指南)
- [模型权重与演示数据](#模型权重与演示数据)
- [快速复现 Demo](#快速复现-demo)
  - [Demo A — Wasserstein-GastroFlow 推理与评估](#demo-a--wasserstein-gastroflow-推理与评估)
  - [Demo B — GastroFusion 几何—语义融合推理](#demo-b--gastrofusion-几何语义融合推理)
  - [Baseline Evaluation — 少样本与长尾识别基线](#baseline-evaluation--少样本与长尾识别基线)
- [训练说明](#训练说明)
- [项目结构](#项目结构)
- [可视化结果](#可视化结果)
- [引用](#引用)
- [致谢](#致谢)
- [数据开放说明](#数据开放说明)
- [许可证](#许可证)
- [联系方式](#联系方式)

---

## 简介

**GastroFusion** 是面向上消化道内镜图像诊断的多模态 AI 框架，旨在应对胃肠道疾病长尾分布、罕见病变标注不足和跨中心数据差异等挑战。框架结合 **Wasserstein-GastroFlow** 的少样本轨迹几何匹配与 **GastroMLLM（Med-Qwen3-VL）** 的医学语义证据，通过器官特异性策略融合预测。

| 核心模块 | 主要功能 |
|---|---|
| **Wasserstein-GastroFlow** | 通过支持条件潜在空间轨迹及 Sinkhorn 近似的分布传输代价进行少样本匹配。 |
| **GastroMLLM（Med-Qwen3-VL）** | 基于 Qwen3-VL-8B 医学适配的多模态模型，提供候选诊断语义评分。 |
| **GastroFusion 融合模块** | 根据器官特异性规则，整合几何匹配和语义预测证据。 |

**主要实验结果：**

- **少样本基线比较**：在六种内部器官—支持集设置中，三种轨迹评分方案的平均准确率均优于八种 FSL 基线；每种设置中的最佳变体较最强 FSL 基线提升 **10.86–42.00 个百分点**，较最强长尾识别基线提升 **1.25–61.18 个百分点**（后者为非 episodic 上下文比较）。
- **外部验证**：三个独立外部中心共 **352 张图像、九个中心—器官队列**。三种轨迹方案相较 DeepEMD 的宏平均提升分别为 **8.70–10.14 个百分点**（target-domain support）和 **13.91–19.80 个百分点**（source-domain support）。
- **医生对照**：固定 **166 例**内部 WCH 阅片队列中，GastroFusion 的器官宏平均准确率为 **86.00%**，高级医生组为 **64.85%**。
- **多模态评估**：GastroMLLM 在 **OmniMedVQA-GI（1,537 题）**上达到 **94.99%** 准确率。
- **融合互补性**：外部 cohort-aligned 与 full-organ 标签空间下的融合增益分别为 **1.24** 与 **0.75 个百分点**；分支选择 Oracle 分析显示几何与语义预测具有互补诊断覆盖。

![GastroFusion Workflow](assets/figures/overview-ill.jpg)

---

## 主要特性

- **少样本与长尾诊断**：利用有限支持样本进行候选疾病匹配。
- **轨迹几何匹配**：结合 RAE 表征、条件 DiT Flow 与 Sinkhorn 分布代价。
- **医学语义建模**：Med-Qwen3-VL 提供补充语义预测证据。
- **器官特异性融合**：选择性整合几何分数与语义分数。
- **多中心验证**：涵盖独立外部中心、不同候选标签空间及医生阅片对照。
- **模块化实现**：几何分支、语义分支、融合与 Baseline 评估分别组织。

---

## 更新日志

- **2026-10-15**：统一 GastroFusion 命名；更新 MLLM 与 Wasserstein-GastroFlow 的训练和评估入口；改为当前的融合流水线；补充 Baseline 训练/适配及测试说明。

---

## 系统要求

各模块采用**独立环境**。

| 模块 | 环境说明 |
|---|---|
| **GastroMLLM 训练** | Python 3.11，CUDA 12.4；训练记录中使用 ms-swift 4.0.3、PyTorch 2.8.0 等；详见 [`mllm/README.md`](mllm/README.md)。 |
| **Wasserstein-GastroFlow** | 使用 `flow` 环境，相关 GPU/CUDA 和脚本配置详见 [`wass_flow/README_CN.md`](wass_flow/README_CN.md)。 |
| **GastroFusion 融合** | MLLM 服务、FSL 推理和融合脚本分别运行；详见 `mllm_fsl/` 中各脚本。 |
| **Baseline** | 不同方法依赖与 checkpoint 各异；详见 [`wass_flow_baseline/README_CN.md`](wass_flow_baseline/README_CN.md)。 |

推荐使用 Linux + NVIDIA GPU；GPU 数量、显存和运行耗时取决于模型及实验配置。训练脚本中涉及的多卡数量不是单卡推理的最低要求。

---

## 安装指南

### 1) 克隆仓库

```bash
git clone https://github.com/GastroAgent/GastroFusion.git
cd GastroFusion
```

### 2) 按模块配置环境

**GastroMLLM 训练环境示例**（参考其子目录文档）：

```bash
conda create -n gastrofusion python=3.11 -y
conda activate gastrofusion
cd mllm
pip install -r requirements.txt
cd ..
```

**Wasserstein-GastroFlow**：按照 [`wass_flow/README_CN.md`](wass_flow/README_CN.md) 配置 `flow` 环境及对应依赖；其训练、权重和评估脚本需使用匹配的 CUDA/PyTorch 环境。


---

## 模型权重与演示数据

模型文件及已公开演示数据请查看：[Hugging Face — GastroFusion](https://huggingface.co/GastroAgent/GastroFusion)。

| 文件或目录 | 用途 |
|---|---|
| `Qwen3-VL-8B-Instruct-3epoch-best.tar.gz` | GastroMLLM 相关权重（使用前核对所需推理阶段与版本）。 |
| `wass-flow-stomach.zip` | 胃部 Wasserstein-GastroFlow 推理评估所需权重包。 |
| `stomach.zip` | Demo A 的 episodic 少样本测试输入；以仓库实际公开文件及相应脚本配置为准。 |

下载示例（需要先安装 Hugging Face CLI，并替换存储目录）：

```bash
hf download GastroAgent/GastroFusion --local-dir <WEIGHT_DIR>
```

获取文件后请检查压缩包中的 checkpoint 实际名称，并根据 [`wass_flow/README_CN.md`](wass_flow/README_CN.md) 配置 RAE、MetricEncoder、DiT Flow 和 Decay 权重路径。

---

## 快速复现 Demo

在下文中，`<PROJECT_ROOT>` 表示仓库根目录；模型、数据及结果路径应按本地环境配置。**执行前必须检查脚本内部的路径和 checkpoint 配置**。

### Demo A — Wasserstein-GastroFlow 推理与评估

使用**胃部内部 episodic few-shot 数据**演示 test-test 协议下的少样本推理与评估，不再使用旧版 Kvasir 双类别 Demo。

1. 下载并解压胃部权重及可公开的演示数据；在脚本中配置数据集、模型 checkpoint 和结果目录。
2. 进入 `wass_flow/`，激活该模块的环境：

```bash
cd wass_flow
conda activate flow

# F1 / Decay 轨迹评分与诊断对比
bash eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_m2_dit.sh

# Q2L / Q2S-Retained 轨迹评分
bash eval_q2l_q2last_q2mono_scorefuse_maxmin_testtest.sh
```

脚本名中的 `Q2Last` 对应论文的 **Q2L**，`Q2Mono` 对应论文的 **Q2S-Retained**。评估结果写入脚本配置的 `BASE_RESULT_DIR`（含预测和统计结果）。

完整配置、训练和评估协议见 [`wass_flow/README_CN.md`](wass_flow/README_CN.md)；若仓库只提供 `wass_flow/README.md`，请以实际文件名为准。

### Demo B — GastroFusion 几何—语义融合推理

依次完成 **GastroMLLM 语义评分 → Wasserstein-GastroFlow 几何评分 → GastroFusion 融合预测**。以下命令均从仓库根目录执行；不同数据集可选择对应脚本。

**1）GastroMLLM 语义预测**

```bash
# 先启动 MLLM 模型服务
bash mllm_fsl/mllm/mllm_code/gastro-mllm-server.sh

# WCH 内部数据
bash mllm_fsl/mllm/mllm_code/run_batch_mllm_huaxi.sh

# 外部多中心数据
bash mllm_fsl/mllm/mllm_code/run_batch_mllm_external.sh
```

结果目录：`mllm_fsl/mllm/mllm_result/`。

**2）Wasserstein-GastroFlow 几何预测**

```bash
# WCH 内部数据
bash mllm_fsl/fsl/fsl_code/run_batch_3huaxi_5shot_adaptive_f1v5.sh

# 外部数据：cohort-aligned 标签空间
bash mllm_fsl/fsl/fsl_code/run_batch_9_5shot_fewway_adaptive_f1v5.sh

# 外部数据：full-organ 标签空间
bash mllm_fsl/fsl/fsl_code/run_batch_9_5shot_lotway_adaptive_f1v5.sh
```

结果目录：`mllm_fsl/fsl/fsl_result/`。

**3）GastroFusion 融合预测**

```bash
# WCH 内部融合
bash mllm_fsl/mllm_fsl_detail/mllm_fsl_code/fusion/fsl_mllm_fusion_tolast_rae_v2_huaxi_grid_val.sh

# 外部多中心融合
bash mllm_fsl/mllm_fsl_detail/mllm_fsl_code/fusion/fsl_mllm_fusion_tolast_rae_v2_external_grid_val.sh
```

结果目录：`mllm_fsl/mllm_fsl_detail/mllm_fsl_result/`。运行融合前，确保 MLLM 与 FSL 输出文件属于**同一数据队列与标签空间**，并与融合脚本配置一致。

### Baseline Evaluation — 少样本与长尾识别基线

`wass_flow_baseline/` 提供论文中 **8 种 Few-Shot Learning 基线**与 **3 种 Long-Tail Recognition 基线**的训练及评估代码。各方法基于本研究相应的医学数据划分进行训练、微调或加载经训练的权重后测试；**并非所有方法都从头训练**。对于已有医学数据 checkpoint 的方法，可直接加载权重评估；SimpleShot 使用 PT-MAP 的预训练权重。

- **FSL 基线**：按相应 N-way K-shot episodic 协议进行评估，区分 test-pool 与 training-pool support。
- **长尾基线**：使用非 episodic 分类评估，作为上下文方法比较，并非与 FSL 完全相同的 episodic 协议。
- **数据隔离**：test-pool support 是推理时的参考样本，不代表用测试查询样本优化模型参数。

所有方法的 Train/Test 脚本、checkpoint 和数据配置详见 [`wass_flow_baseline/README_CN.md`](wass_flow_baseline/README_CN.md)。

---

## 训练说明

GastroMLLM 与 Wasserstein-GastroFlow 独立训练；Wasserstein-GastroFlow 内部的度量学习与 DiT Flow Matching 可并行训练，随后进行 F1/Decay 轨迹分类训练。最后将两分支输出用于 GastroFusion 融合评估。

```text
GastroMLLM                          Wasserstein-GastroFlow
  SFT → GRPO                         ├─ Metric: M0-A → M0-B → M1 → M2
      │                              └─ DiT Flow Matching（独立并行）
 Med-Qwen3-VL                                  │
      │                              F1 / Decay 轨迹分类训练
      │                                        │
  语义评分                                   几何评分
      └─────────────────┬──────────────────────┘
                        ↓
               GastroFusion 融合
                        ↓
                    最终预测
```

### 1) GastroMLLM（Med-Qwen3-VL）

基于 **Qwen3-VL-8B**，通过 ms-swift 进行单图/多图数据 **SFT** 与 **GRPO** 强化学习，得到医学适配的语义预测模型。

训练入口：

```bash
cd mllm

# SFT
bash sft_network_auto.sh
# 多机 SLURM：sbatch train_multinode_sft_network_auto.sh

# GRPO
bash rl_medical_grpo_colocate_fixed_v2_entropy20.sh
# 多机 SLURM：sbatch train_multinode_medical_grpo_network_auto_entropy20.sh
```

完整数据构建、LoRA 合并、依赖、超参数和权重说明见 [`mllm/README.md`](mllm/README.md)；推理与融合见 `mllm_fsl/`。

### 2) Wasserstein-GastroFlow

基于 RAE、条件 DiT Flow Matching 和 SK-Wasserstein 度量匹配，采用以下阶段：

1. **M0-A → M0-B → M1 → M2**：分类预训练、SK-Wasserstein 度量预训练、医学域适配、Direct episodic 校准。
2. **DiT Flow Matching**：独立训练支持条件轨迹生成器，可与 M0–M2 并行。
3. **F1 / Decay**：冻结 RAE、M2 MetricEncoder 与 DiT Flow，训练轨迹 LSTM 衰减头。Direct、Q2L 和 Q2S-Retained 作为其他几何评分方案保留。

训练入口（下列 Python 命令需补齐子目录 README 中的必需参数）：

```bash
cd wass_flow

# 度量学习，按顺序运行
# torchrun --nproc_per_node=8 train_m0a_cls.py <参数>
# torchrun --nproc_per_node=8 train_m0b_ot.py <参数>
# torchrun --nproc_per_node=8 train_m1_medical_metric.py <参数>
# torchrun --nproc_per_node=8 train_m2_direct_episode_visreg.py <参数>

# DiT Flow Matching
sbatch train_dit_flow_spatial_sinkhorn_strict_intra_ot.sh

# F1 预训练
sbatch submit_multinode_pretrain_f1_support_conditioned_lstm_v5.sh
```

F1 医学域适配使用 `train_f1_support_conditioned_lstm_v5_medical_manifest.*` 系列入口。完整训练参数与验证协议见 [`wass_flow/README_CN.md`](wass_flow/README_CN.md)。

### 3) GastroFusion 融合

加载分别完成训练的 MLLM 与 FSL 分支，利用**预先确定**的器官特异性规则和参数组合语义、几何证据；评估时不使用测试标签重新校准。运行入口见 [Demo B](#demo-b--gastrofusion-几何语义融合推理)。

---

## 项目结构

以下列出主要代码目录；请根据最终提交仓库的实际文件树核对路径与文件是否齐全。

```text
GastroFusion/
├── assets/
│   └── figures/                  # README 图片与结果图
├── mllm/                         # Med-Qwen3-VL：SFT / GRPO 训练
│   └── README.md
├── wass_flow/                    # RAE + DiT Flow + Metric + F1 训练评估
│   └── README_CN.md
├── mllm_fsl/                     # 几何—语义推理及融合
│   ├── mllm/                     # 语义评分脚本/结果
│   ├── fsl/                      # 几何评分脚本/结果
│   └── mllm_fsl_detail/          # 融合脚本/结果
├── wass_flow_baseline/           # 8 种 FSL + 3 种长尾识别方法
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
├── requirements.txt             # 根目录依赖（如提供；非所有模块通用）
├── README.md
└── README_CN.md
```


---

## 可视化结果

![Benchmark comparison](assets/figures/doctor-dataset.jpg)

![Long-tail performance](assets/figures/kshot-case.jpg)

![Workflow result](assets/figures/workflow-result-ill.jpg)

---

## 引用

如本项目对您的研究有帮助，请引用相应论文。

```bibtex
@misc{GastroFusion2026,
  title = {Geometry-aware multimodal AI improves few-shot gastrointestinal diagnosis under limited data},
  author = {Shuyue Tan and Nuoya Zhou and Mingfei Jin and Xingwu Liu and Xianglei Yuan and Yi Mou and Linzhuang Sun and Yahui Zhang and Sizhe Qiu and Wei Liu and Yuhang Zhang and Yuyan Zhang and Qi Luo and Ruide Liu and Huan Jiang and Pinghua Wen and Shunyao Wang and Hengduo Jiang and Xuan Liang and Meiling Shi and Zhilie Jiang and Dongbo Bu and Bing Hu and Ming Li and Shuaicheng Li},
  year = {2026},
  note = {Manuscript under preparation or review}
}
```


---

## 致谢

感谢各临床合作中心、参与评估的内镜医师及开源社区。本项目基于 [PyTorch](https://pytorch.org/) 等开源软件开发。

---

## 数据开放说明

本研究使用 WCH 内部临床队列及三个独立外部中心的数据。为支持研究复现，项目提供胃部内部实验的部分推理评估数据与相应代码；其余临床数据受患者隐私、伦理及数据使用条件限制，暂不公开原始图像。实际可用文件以公开仓库为准，数据获取条件请参见论文的 Data Availability 声明。

---

## 许可证

本项目采用 [Apache-2.0 License](LICENSE)（请以仓库中实际 `LICENSE` 文件为准）。

**注意**：本项目仅用于科学研究，不得直接用于临床诊断或替代专业医生判断。

---

## 联系方式

- 问题反馈：[GitHub Issues](https://github.com/GastroAgent/GastroFusion/issues)
- 邮箱：shuyuetan0@gmail.com

---

## Star History

欢迎 Star 支持项目。

[![Star History Chart](https://api.star-history.com/svg?repos=GastroAgent/GastroFusion&type=Date)](https://star-history.com/#GastroAgent/GastroFusion&Date)