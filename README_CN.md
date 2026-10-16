# GastroFusion

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange.svg)](https://pytorch.org/)

**面向有限数据条件下胃肠道少样本诊断的几何感知多模态人工智能框架**

[English](README_EN.md) | 简体中文

---

## 目录

- [项目概述](#项目概述)
- [主要特点](#主要特点)
- [更新日志](#更新日志)
- [系统要求](#系统要求)
- [安装](#安装)
- [模型权重与演示数据](#模型权重与演示数据)
- [快速复现实验](#快速复现实验)
  - [Demo A — Wasserstein-GastroFlow 推理与评估](#demo-a--wasserstein-gastroflow-推理与评估)
  - [Demo B — GastroFusion 几何–语义融合推理](#demo-b--gastrofusion-几何语义融合推理)
  - [基线评估 — 少样本与长尾识别](#基线评估--少样本与长尾识别)
- [模型训练](#模型训练)
- [项目结构](#项目结构)
- [结果可视化](#结果可视化)
- [引用](#引用)
- [致谢](#致谢)
- [数据可用性](#数据可用性)
- [许可证](#许可证)
- [联系方式](#联系方式)

---

## 项目概述

**GastroFusion** 是一个面向上消化道内镜图像诊断的多模态人工智能框架，旨在应对疾病类别长尾分布、罕见病变标注数据有限以及跨中心数据分布偏移等挑战。该框架结合 **Wasserstein-GastroFlow** 基于少样本轨迹的几何匹配能力与 **GastroMLLM（Med-Qwen3-VL）** 提供的医学语义证据，并通过器官特异性融合规则整合预测结果。

| 核心组件 | 主要功能 |
|---|---|
| **Wasserstein-GastroFlow** | 利用支持集条件化的潜在空间轨迹和 Sinkhorn 近似分布传输代价进行少样本匹配。 |
| **GastroMLLM（Med-Qwen3-VL）** | 基于 Qwen3-VL-8B 的医学适配多模态模型，为候选诊断类别提供语义评分。 |
| **GastroFusion 融合模块** | 通过器官特异性规则整合几何匹配证据与语义预测证据。 |

**主要实验结果：**

- **少样本基线比较：** 在六种内部器官–支持集协议组合中，三种轨迹评分方案的平均准确率均超过所评估的八种少样本学习（FSL）基线。各设置中的最佳方案比最强 FSL 基线高 **10.86–42.00 个百分点**，比最强长尾识别基线高 **1.25–61.18 个百分点**；后者属于非 episodic 的参考性比较。
- **外部验证：** 三家独立外部中心共提供 **352 张图像，组成九个中心–器官队列**。与 DeepEMD 相比，三种轨迹方案在 target-domain support 条件下的宏平均准确率提高 **8.70–10.14 个百分点**，在 source-domain support 条件下提高 **13.91–19.80 个百分点**。
- **医生对比：** 在内部 WCH 固定的 **166 例**医生阅片队列上，GastroFusion 的器官宏平均准确率达到 **86.00%**，高级内镜医生为 **64.85%**。
- **多模态评估：** GastroMLLM 在 **OmniMedVQA-GI（1,537 道问题）**上达到 **94.99%** 的准确率。
- **融合互补性：** 外部融合在 cohort-aligned 和 full-organ 标签空间下分别取得 **1.24** 和 **0.75 个百分点**的数值提升。分支选择 Oracle 揭示了几何预测与语义预测之间的互补诊断覆盖能力。

![GastroFusion 工作流程](assets/figure/overview-ill.png)

---

## 主要特点

- **少样本与长尾疾病诊断：** 使用有限的已标注支持样本匹配候选疾病。
- **基于轨迹的几何匹配：** 结合 RAE 表征、条件 DiT Flow 与 Sinkhorn 传输代价。
- **医学语义建模：** Med-Qwen3-VL 提供互补的语义预测证据。
- **器官特异性融合：** 有选择地整合几何评分与语义评分。
- **多中心验证：** 涵盖独立外部中心、不同候选标签空间以及医生阅片对比。
- **模块化实现：** 将几何建模、语义建模、融合与基线评估分开组织。

---

## 更新日志

- **2026-10-15：** 统一 GastroFusion 命名；更新 MLLM 与 Wasserstein-GastroFlow 的训练及评估入口；采用当前融合流程；补充基线模型的训练、适配与测试说明。

---

## 系统要求

不同模块使用**独立的运行环境**。

| 模块 | 环境说明 |
|---|---|
| **GastroMLLM 训练** | Python 3.11、CUDA 12.4；记录的训练环境使用 ms-swift 4.0.3、PyTorch 2.8.0 及相关依赖。详见 [`mllm/README_CN.md`](mllm/README_CN.md)。 |
| **Wasserstein-GastroFlow** | 使用 `flow` 环境。GPU/CUDA 要求及脚本配置见 [`wass_flow/README_CN.md`](wass_flow/README_CN.md)。 |
| **GastroFusion 融合** | 分别运行 MLLM 服务、FSL 推理与融合脚本，见 `mllm_fsl/` 下的相关脚本。 |
| **基线模型** | 不同方法的依赖和 checkpoint 各不相同，详见 [`wass_flow_baseline/README_CN.md`](wass_flow_baseline/README_CN.md)。 |

推荐使用配备 NVIDIA GPU 的 Linux 系统。所需 GPU 数量、显存和运行时间取决于模型与实验配置。多 GPU 训练配置不应被理解为单 GPU 推理的最低要求。

---

## 安装

### 1）克隆仓库

```bash
git clone https://github.com/GastroAgent/GastroFusion.git
cd GastroFusion
```

### 2）配置各模块的独立环境

**GastroMLLM 训练环境示例**（完整配置请参阅对应模块 README）：

```bash
conda create -n gastrofusion python=3.11 -y
conda activate gastrofusion
cd mllm
pip install -r requirements.txt
cd ..
```

**Wasserstein-GastroFlow：** 请按照 [`wass_flow/README_CN.md`](wass_flow/README_CN.md) 配置 `flow` 环境及相关依赖。训练、checkpoint 加载和评估均需要兼容的 CUDA 与 PyTorch 版本。

---

## 模型权重与演示数据

已发布的压缩包可从 [Hugging Face — GastroFusion](https://huggingface.co/GastroAgent/GastroFusion) 获取。

| 压缩包 | 内容与用途 |
|---|---|
| `demo_data.zip` | Demo A 使用的胃部内镜图像，按疾病类别组织，用于 episodic test-test 评估。 |
| `demo-wass-flow-stomach.zip` | Wasserstein-GastroFlow 胃部评估 checkpoint，包括 F1/Decay、M2、DiT Flow 及相关权重。 |
| `Med-Qwen3-VL-8B-Instruct.zip` | GastroMLLM 相关模型文件。 |

运行 Demo A 时，下载并解压 `demo_data.zip` 与 `demo-wass-flow-stomach.zip`。评估脚本通过 `DATA_ROOT` 和 `WEIGHT_ROOT` 指定类别图像目录及 checkpoint 目录。

其他运行依赖包括 RAE Decoder、RAE 归一化统计量、EndoViT 和 DINOv3 ViT-L/16。配置详情见 [`wass_flow/README_CN.md`](wass_flow/README_CN.md)。

---

## 快速复现实验

下文中 `<PROJECT_ROOT>` 表示仓库根目录。请根据实际运行环境配置模型、数据及输出路径。**执行前务必检查脚本内部定义的路径和 checkpoint。**

### Demo A — Wasserstein-GastroFlow 推理与评估

Demo A 提供两个胃部 test-test 少样本评估入口，使用已发布、按类别组织的胃部内镜图像：

- **F1 / Decay：** 使用 F1/Decay、M2 和 DiT checkpoint 进行支持集条件化的轨迹评估。
- **Q2L / Q2S-Retained：** 评估末步匹配（脚本中称 `Q2Last`）、保留轨迹步骤聚合（脚本中称 `Q2Mono`）以及两者的分数级融合，使用胃部 Q2L 和 DiT checkpoint。

默认评估设置为 **5-shot、1-query、100 episodes、随机种子 42**。

**1. 下载并解压 Demo A 压缩包：**

```bash
# 在 GastroFusion 仓库根目录执行
mkdir -p demo_data demo_weights
unzip /path/to/demo_data.zip -d demo_data
unzip /path/to/demo-wass-flow-stomach.zip -d demo_weights
```

评估脚本默认预期的目录结构如下：

```text
GastroFusion/
├── demo_data/
│   └── stomach/
│       ├── <疾病类别 A>/
│       └── <疾病类别 B>/
└── demo_weights/
    └── wass-flow-stomach/
        ├── f1_v5_support_lstm_medical/
        ├── new_m1/
        ├── stomach-test-test/
        └── otcfm_dit_L_step_265000.pt
```

如果实际解压目录不同，请将 `DATA_ROOT` 和 `WEIGHT_ROOT` 设置为对应的真实路径。

**2. 运行评估脚本：**

```bash
conda activate flow
cd /path/to/GastroFusion/wass_flow

# F1 / Decay
bash eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_m2_dit.sh

# Q2L / Q2S-Retained / 分数级融合
bash eval_q2l_q2last_q2mono_scorefuse_maxmin_testtest.sh
```

两个评估脚本均支持通过 `DATA_ROOT` 和 `WEIGHT_ROOT` 环境变量自定义数据及 checkpoint 路径。

训练、评估和配置的更多说明见 [`wass_flow/README_CN.md`](wass_flow/README_CN.md)。

### Demo B — GastroFusion 几何–语义融合推理

该流程分别生成 **GastroMLLM 语义评分**和 **Wasserstein-GastroFlow 几何评分**，随后在独立验证数据上选择融合参数，再使用固定参数执行 **GastroFusion 融合预测**。两路评分可分别计算，不要求按固定先后顺序运行。以下命令从仓库根目录运行，请根据数据集选择对应脚本。

**1）GastroMLLM 语义预测**

```bash
# 首先启动 MLLM 模型服务
bash mllm_fsl/mllm/mllm_code/gastro-mllm-server.sh

# 内部 WCH 队列
bash mllm_fsl/mllm/mllm_code/run_batch_mllm_huaxi.sh

```

结果目录：`mllm_fsl/mllm/mllm_result/`。

**2）Wasserstein-GastroFlow 几何预测**

```bash
# 内部 WCH 队列
bash mllm_fsl/fsl/fsl_code/run_batch_3huaxi_5shot.sh

```

结果目录：`mllm_fsl/fsl/fsl_result/`。离线几何评分通常产生逐配对的 `result.json` 和逐查询聚合的 `query_results.json`。

**3）独立验证集参数选择（`val_grid/`）**

`mllm_fsl/mllm_fsl_detail/mllm_fsl_code/val_grid/val_grid_search.py` 根据验证集 MLLM 与 FSL 结果比较器官特异的候选阈值组合。选定后，将数值阈值明确写入最终融合 Shell；最终测试集不参与参数重选。

**4）GastroFusion 融合预测**

```bash
# 内部 WCH 融合
bash mllm_fsl/mllm_fsl_detail/mllm_fsl_code/fusion/fsl_mllm_fusion_tolast_rae_v2_huaxi_grid_val.sh

```

结果目录：`mllm_fsl/mllm_fsl_detail/mllm_fsl_result/`。融合 Python 固定胃部 `gap_guard_fusion`、食管 `high_conf_override`、十二指肠 `very_high_conf_override` 三种策略；**全部八个数值阈值由 Shell 显式传入**。只需主 MLLM JSON 和离线 RAE FSL 结果。执行前，须确保两路结果对应**相同的评估队列和候选标签空间**。

**5）参数敏感性分析（可选）**

`sensitivity/` 使用固定融合配置附近的参数扰动评估结果稳定性，**不用于在测试集上重新选择最优参数**。


### 基线评估 — 少样本与长尾识别

`wass_flow_baseline/` 提供论文中所使用的**八种少样本学习（FSL）基线**和**三种长尾识别基线**的训练与评估代码。各方法依据相应医学数据划分，使用适当训练得到的权重进行训练、微调、适配或评估。**并非所有方法均从头训练。**部分方法可直接加载已有医学数据 checkpoint；SimpleShot 使用预训练的 PT-MAP 权重；FSL-OM 与 PT-MAP 可直接加载对应医学实验 checkpoint。

- **FSL 基线：** 按照相应 N-way K-shot episodic 协议评估，并区分测试池支持样本与训练池支持样本。
- **长尾基线：** 采用非 episodic 分类方式作为参考性比较，其评估协议与 episodic FSL 并不完全相同。
- **数据划分：** 测试池支持样本在推理阶段提供，不意味着使用测试查询样本优化模型参数。

各方法的训练/测试脚本、checkpoint 和数据配置见 [`wass_flow_baseline/README_CN.md`](wass_flow_baseline/README_CN.md)。

---

## 模型训练

GastroMLLM 与 Wasserstein-GastroFlow 分别独立训练。在 Wasserstein-GastroFlow 内部，度量学习与 DiT Flow Matching 可以并行进行，之后训练 F1/Decay 轨迹分类模块。最终在 GastroFusion 融合评估阶段组合两路预测结果。

```text
GastroMLLM                          Wasserstein-GastroFlow
    SFT → GRPO                      ├─ 度量学习：M0-A → M0-B → M1 → M2
         │                          └─ DiT Flow Matching（并行）
   Med-Qwen3-VL                                    │
         │                              F1 / Decay 轨迹训练
         │                                        │
     语义评分                                   几何评分
         └──────────────────┬─────────────────────┘
                            ↓
                    GastroFusion 融合
                            ↓
                         最终预测
```

### 1）GastroMLLM（Med-Qwen3-VL）

该模型基于 **Qwen3-VL-8B**，使用 ms-swift 完成数据构建、单图像/多图像**监督微调（SFT）**、阶段间 LoRA 合并与 **GRPO** 强化学习，以获得医学适配后的语义预测能力。

训练数据由 `build_sft_rl_data.py` 构建，涵盖教师模型生成、视觉判别和可恢复的数据处理；训练数据保存在 `mllm/data/pretrain/`。`merge_lora.py` / `merge_lora.sh` 用于迭代 SFT 阶段间的权重合并。GRPO 使用 `medical_vqa_reward.py` 计算选项准确性与最终答案格式奖励，并以 `--top_entropy_quantile 0.20` 仅让每批完成序列中熵最高约 20% 的 token 参与策略梯度损失。

训练入口：

```bash
cd mllm

# SFT
bash sft_network_auto.sh
# 多节点 SLURM：sbatch train_multinode_sft_network_auto.sh

# GRPO
bash rl_medical_grpo_colocate_fixed_v2_entropy20.sh
# 多节点 SLURM：sbatch train_multinode_medical_grpo_network_auto_entropy20.sh
```

当前训练配置可参见子模块 README：SFT 使用 LoRA rank 64；GRPO 使用 colocate vLLM、`top_entropy_quantile=0.20`，并可通过多节点 SLURM 启动。数据构建、LoRA 合并、依赖、超参数及 checkpoint 详见 [`mllm/README.md`](mllm/README.md)。

### 2）Wasserstein-GastroFlow

该模块结合 RAE、条件 DiT Flow Matching 和 SK-Wasserstein 度量匹配，训练顺序如下：

1. **M0-A → M0-B → M1 → M2：** 依次进行分类预训练、SK-Wasserstein 度量预训练、医学领域适配和 Direct episodic 校准。
2. **DiT Flow Matching：** 独立训练支持集条件化的轨迹生成器，可与 M0–M2 并行执行。
3. **F1 / Decay：** 在冻结 RAE、M2 MetricEncoder 和 DiT Flow 的条件下，训练轨迹 LSTM 衰减头。Direct、Q2L 和 Q2S-Retained 仍可作为替代的几何评分方案。

训练入口（注释中的 Python 示例需要补充模块 README 所列的必需参数）：

```bash
cd wass_flow

# 度量学习：按顺序执行
# torchrun --nproc_per_node=8 train_m0a_cls.py <ARGS>
# torchrun --nproc_per_node=8 train_m0b_ot.py <ARGS>
# torchrun --nproc_per_node=8 train_m1_medical_metric.py <ARGS>
# torchrun --nproc_per_node=8 train_m2_direct_episode_visreg.py <ARGS>

# DiT Flow Matching
sbatch train_dit_flow_spatial_sinkhorn_strict_intra_ot.sh

# F1 预训练
sbatch submit_multinode_pretrain_f1_support_conditioned_lstm_v5.sh
```

医学领域 F1 适配使用 `train_f1_support_conditioned_lstm_v5_medical_manifest.*` 入口。完整参数及评估协议见 [`wass_flow/README_CN.md`](wass_flow/README_CN.md)。

### 3）GastroFusion 融合

该阶段使用分别生成的 MLLM 语义评分和 FSL 离线几何评分，而不是在融合脚本中重新训练骨干模型。`val_grid/` 在独立验证数据上选择参数；`fusion/` 使用 Shell 中固定的八个数值阈值执行三器官特异融合；`sensitivity/` 分析参数扰动的影响。评估时不使用测试标签重新校准融合参数。运行入口见 [Demo B](#demo-b--gastrofusion-几何语义融合推理)。

---

## 项目结构

主要代码目录如下：

```text
GastroFusion/
├── assets/
│   └── figures/                  # README 图片与结果可视化
├── mllm/                         # Med-Qwen3-VL 数据构建、SFT、LoRA 合并、GRPO 训练
│   └── README.md
├── wass_flow/                    # RAE + DiT Flow + Metric + F1 训练/评估
│   └── README.md
├── mllm_fsl/                     # 几何–语义推理与融合
│   ├── data/                     # MLLM 输入与 FSL query–support 配对 JSON
│   ├── mllm/                     # 语义推理服务、批量评分与结果
│   ├── fsl/                      # 离线 RAE/Metric/Flow 几何评分与结果
│   └── mllm_fsl_detail/          # val_grid、fusion、sensitivity 与结果
│   └── README_CN.md
├── wass_flow_baseline/           # 八种 FSL + 三种长尾识别基线
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
├── requirements.txt             # 根目录依赖（如有）；不适用于所有模块
├── README.md
└── README_CN.md
```

---

## 结果可视化

![长尾类别性能](assets/figure/kshot-case.png)

---

## 引用

如果本项目对你的研究有所帮助，请引用相应论文。

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

感谢合作临床中心、参与研究的内镜医生以及开源社区。本项目使用 [PyTorch](https://pytorch.org/) 和其他开源软件开发。

---

## 数据可用性

本研究包含来自华西医院（West China Hospital，WCH）及三家独立外部中心的临床内镜图像。出于患者隐私、伦理要求和数据使用限制，完整临床数据集未公开发布。

为促进研究复现，我们提供部分内部胃部内镜数据用于演示和评估，同时提供对应的 Wasserstein-GastroFlow 推理代码及模型 checkpoint。

已发布的演示子集不应被视为论文所报告的完整内部或外部评估队列。临床数据可用性的更多信息见论文中的 Data Availability 声明。

---

## 许可证

本项目采用 [Apache-2.0 License](LICENSE) 开源。

**仅供科研使用：** 本框架不适用于直接临床诊断，也不能替代专业医务人员的医学判断。

---

## 联系方式

- Issues：[GitHub Issues](https://github.com/GastroAgent/GastroFusion/issues)
- 邮箱：shuyuetan0@gmail.com

---

## Star History

如果本项目对你有所帮助，欢迎为仓库点亮 Star。

[![Star History Chart](https://api.star-history.com/svg?repos=GastroAgent/GastroFusion&type=Date)](https://star-history.com/#GastroAgent/GastroFusion&Date)
