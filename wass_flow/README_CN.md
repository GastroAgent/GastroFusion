# wass_flow — RAE–Flow–SK-Wasserstein 少样本分类

[English](README_EN.md) | 简体中文

本实现将训练流程解耦为多个阶段。最终任务为 **N-way K-shot episodic 少样本分类**：每个 episode 从候选类别中抽取支持样本（support）和查询样本（query）。

## 1. 整体流程

```text
query / K-shot support
→ 冻结的 RAE（将内镜图像编码为空间特征图，供后续 OT 使用）
→ 以候选类别 support 为条件的冻结 DiT Flow（生成轨迹 z_0 … z_T）
→ 每个预测状态由同一个共享 Compressor 映射至度量 token 空间
→ 对 Direct pair 计算一次 SK 节点质量，并在整条轨迹上复用
  （F1；Q2L 评估则在每一步动态计算节点质量）
→ 与 cosine 距离等价的 ground cost + log-domain Sinkhorn
→ F1 LSTM 衰减头 → 单调不增的状态权重 w_t → 轨迹加权分数
→ K-shot soft-min → N-way logits
```

**Dynamic SK（动态 SK）：** 节点质量由数据预测，而不是统一分配。Ground cost 为 L2 归一化 token 之间的平方欧氏距离（与 cosine 距离单调等价）。熵正则 Sinkhorn 在 log 域中计算，以提高数值稳定性；返回的是**正则化传输代价，而非精确的 W2 距离**（见 `models/sk_sinkhorn.py`）。

## 2. 训练阶段

度量学习阶段依次执行 **M0-A → M0-B → M1 → M2**。DiT Flow 与度量学习阶段**相互独立，可以并行训练**，两条训练分支互不依赖。F1（使用冻结 Flow 的轨迹分类）是当前主要训练流程。

### M0-A：分类预热（Classification Warm-Up）

使用常规分类 batch，不使用 SK、OT、Flow、episode 或 VISReg。

```bash
torchrun --nproc_per_node=8 train_m0a_cls.py \
  --train_json_glob "/path/to/data_train_pool_imagenet_med/*.json" \
  --val_json_glob "/path/to/data_eval_pool_imagenet/*.json" \
  --project_root /path/to/wass_flow \
  --output_dir /path/to/weights_metric_v1/m0a \
  --batch_size 512 --rae_chunk_size 512 --epochs 10
```

也可以通过 `--train_root / --val_root` 直接读取 ImageNet 目录，而不使用 `--train_json_glob`。

输出：

```text
metric_cls_pretrain.pt
```

### M0-B：ImageNet SK-Wasserstein 度量预训练

使用 P×K 类别均衡 batch，为每个 anchor 分别采样一个 positive 和一个 negative：

```bash
torchrun --nproc_per_node=8 train_m0b_ot.py \
  --train_json_glob "/path/to/data_train_pool_imagenet_med/*.json" \
  --val_json_glob "/path/to/data_eval_pool_imagenet_huaxi/*.json" \
  --project_root /path/to/wass_flow \
  --init_checkpoint /path/to/weights_metric_v1/m0a/metric_cls_pretrain.pt \
  --output_dir /path/to/weights_metric_v1/m0b \
  --classes_per_batch 50 --samples_per_class 8 \
  --batches_per_epoch 2000 --epochs 5 \
  --ot_pool_factor 1
```

损失：

```text
OT ranking + decaying CE anchor + VISReg
```

输出：

```text
metric_imagenet_ot.pt
```

M0-B 和 M1 使用 P×K 类别均衡采样器。RAE 编码通过 `--rae_chunk_size` 分块执行（默认每次编码 8 张图像），避免较大的 P×K batch 耗尽 GPU 显存。

### M1：医学领域度量适配

沿用 M0-B 的训练框架，但将数据切换为医学类别池：

```bash
torchrun --nproc_per_node=8 train_m1_medical_metric.py \
  --train_json_glob "/path/to/data_train_pool_huaxi/*.json" \
  --val_json_glob "/path/to/data_eval_pool_imagenet_huaxi/*.json" \
  --project_root /path/to/wass_flow \
  --init_checkpoint /path/to/weights_metric_v1/m0b/metric_imagenet_ot.pt \
  --output_dir /path/to/weights_metric_v1/m1 \
  --classes_per_batch 16 --samples_per_class 8 \
  --epochs 15 --lr 3e-5
```

输出：

```text
metric_medical_adapt.pt
```

JSON 数据加载器支持以下常见字段：

```text
path / image_path / x1_path / x0_path
label / label_B / class_name / category / caption
class_id / label_id / label_B_id
```

如果 JSON 结构不同，可以修改 `data/class_pool.py` 中的 `_iter_json_items()` 或候选字段列表。

### M2：Direct Episodic 校准（首次引入 Episode）

此阶段首次引入 N-way K-shot episode，**不使用 Flow**。当前入口为 `train_m2_direct_episode_visreg.py`（episode + VISReg）。Episode 可以通过 `--train_json_glob` 从类别池在线采样，也可以通过 episode manifest JSON 直接提供（当前 `.sh` 文件采用后一种方式）：

```bash
torchrun --nproc_per_node=8 train_m2_direct_episode_visreg.py \
  --train_episode_manifest /path/to/fsl_multi_vae/data/食管/train.json \
  --val_episode_manifest /path/to/fsl_multi_vae/data/食管/val-val.json \
  --project_root /path/to/wass_flow \
  --init_checkpoint /path/to/weights_metric_v1/m1/<m1_checkpoint>.pt \
  --output_dir /path/to/weights_metric_v1/m2_食管-test-test \
  --n_way 8 --n_shot 3 --n_query 1 \
  --episodes_per_epoch 137 --val_episodes 193 --epochs 50 \
  --visreg_weight 1e-3 --amp
```

M2 **必须从 M1 checkpoint 初始化**。器官专用流程从对应器官的 M1 checkpoint 初始化。输出为 `metric_direct_episode.pt` 和 `_last.pt`。

其他入口包括：`train_m2_direct_episode_visreg_trainjson_valmanifest.py`（训练类别池 + 验证 manifest）、`train_m2_direct_episode_visreg_globalcls_truebatch_v2_valsingle.py` 和 `train_m2_direct_episode_flow_aug_visreg.py`（Flow 轨迹增强）。

### DiT Flow（与度量学习并行）

DiT Flow 训练独立于 M0–M2，可与其并行进行。训练脚本为 `train_dit_flow_spatial_sinkhorn_strict_intra_ot.py`，SLURM 入口为 `train_dit_flow_spatial_sinkhorn_strict_intra_ot.sh`。**训练使用 8 张 GPU，验证除外。**

条件输入配置如下：timestep 和 target/support 图像的 CLS 特征进入 AdaLN；patch tokens 进入 Cross-Attention；弱噪声灰度提示（weak noisy grayscale hint）通过 Spatial Addition 注入。默认关闭 caption 和 class conditioning。Hint 的默认设置为 `alpha = 0.10`、`hint_dropout = 0.10`。图像分辨率统一设为 256，与轨迹生成和度量学习流程保持一致。

```bash
torchrun --nproc_per_node=8 --master_port=29515 \
  train_dit_flow_spatial_sinkhorn_strict_intra_ot.py \
  --model otcfm \
  --dit_variant L \
  --batch_size 14 \
  --total_steps 500000 \
  --image_size 256 \
  --caption_mode vision_only \
  --no-use_class_condition \
  --no-train_text_encoder \
  --train_vision_encoder \
  --use_cross_attention \
  --hint_signal_ratio 0.10 \
  --hint_dropout 0.10 \
  --op_match_batch \
  --data_glob "/path/to/data_train_pool_imagenet_med/*.json" \
  --ot_reg_weight 0.01 --ot_reg 0.1 --ot_reg_iters 20 --ot_max_tokens 256 \
  --output_dir /path/to/weights_dit_flow_v1/vision_hint_inclass
```

脚本还保留 Flow Matching、OT pairing、可选 OT regularization、GAN、DDP、EMA 和 checkpoint 清理逻辑。完整的 OT 正则化参数见对应 `.sh` 文件。

### F1：冻结 Flow 的轨迹分类（当前主线）

F1 是**使用 Flow 轨迹**的分类分支。在冻结的 RAE、M2 MetricEncoder 和 DiT Flow 之上，仅训练轻量的 support-conditioned causal LSTM 衰减头（`TemporalDecayHead`）：

- **轨迹：** 为每个 query–candidate pair 生成候选条件轨迹 `z_0 … z_T` 以及每一步的距离。
- **衰减头：** LSTM 为每次状态转移预测信息保留系数 `γ_t = γ_min + (1-γ_min)·σ(a_t)`。状态权重通过累积保留计算：`u_0=1, u_t=∏γ_i, w_t=u_t/Σu_j`。因此 `w_0 ≥ w_1 ≥ … ≥ w_T > 0` 且 `Σ w_t = 1`。这种设计保证权重**单调不增**，编码“轨迹后段状态的判别信号较弱”的先验，同时允许衰减率沿轨迹自适应变化。
- **轨迹分数：** `step_to_support`（`Σ w_t D(z_t, support)`）或 `step_to_last`（`Σ w_t D(z_t, z_T)`），在冻结 M2 的 K-shot soft-min 之前计算。Direct 基线始终为 `D(z_0, support)`。
- **优化：** 主要目标为可微的最终分类交叉熵（CE），并辅以 antithetic group-relative 扰动进行对比探索。

训练分为两步（torchrun DDP）：

1. **ImageNet/class-JSON 预训练：** `pretrain_f1_support_conditioned_lstm_v5_classjson_torchrun_ddp_batch.py`。使用 `sbatch` 提交 `submit_multinode_pretrain_f1_support_conditioned_lstm_v5.sh`；随后由 `srun` 在各节点执行 `run_pretrain_f1_support_conditioned_lstm_v5_multinode.sh`（默认每节点 8 个进程，多机 torchrun 通过 `node_rank` 组网）。
2. **器官专用医学训练：** 使用 `train_f1_support_conditioned_lstm_v5_medical_manifest.*`。通过 `--m2_checkpoint` 初始化对应器官的 M2 checkpoint，再通过 `PRETRAIN_F1_CHECKPOINT` / `--init_decay_checkpoint` 加载第一步预训练得到的 decay head。使用 `--trajectory_score_mode step_to_support / step_to_last` 选择评分分支。

早期的 F1 V1（轨迹能量评分）属于历史实现，不再作为当前入口列出。

F1 的轨迹生成计算成本较高：每个进程需要持有完整的冻结 DiT、EndoViT 和 RAE。Query–candidate pairs 采用分块处理，而不是一次性把所有轨迹放到 GPU 中。

## 3. 快速开始：评估

Wasserstein-GastroFlow 使用 Conda `flow` 环境。运行评估脚本前先激活环境：

```bash
conda activate flow
cd /path/to/GastroFusion/wass_flow
```

评估框架支持不同器官的实验。**目前公开的 Demo A 默认采用胃部数据，并使用 test-test 协议。** 评估其他器官需要相应的数据集、checkpoint 和配置调整。

### 公开 Demo A：胃部 test-test 评估

从 [GastroFusion Hugging Face 仓库](https://huggingface.co/GastroAgent/GastroFusion) 下载 `demo_data.zip` 和 `demo-wass-flow-stomach.zip`，并解压到评估脚本能够访问的位置。相对于 **GastroFusion 仓库根目录**，默认目录结构为：

```text
demo_data/stomach/<disease-class>/*.jpg
demo_weights/wass-flow-stomach/
```

`demo_weights/wass-flow-stomach/` 必须包含评估脚本预期路径下的 F1/Decay、M2、Q2L 和 DiT checkpoints。这里列出的是**压缩包解压后的目录路径，而不是压缩包文件名**。除非显式设置 `DATA_ROOT` 和 `WEIGHT_ROOT`，否则脚本使用这些默认路径。

原始模型加载器还依赖其他预训练组件，包括 RAE decoder 和归一化统计量、EndoViT 以及 DINOv3 ViT-L/16。请确保这些组件位于配置的加载路径；预期位置见仓库根目录 README。

公开的胃部评估提供两个运行入口：

| 评估流程 | 脚本 | 说明 |
|---|---|---|
| F1 / Decay（test-test） | `eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_m2_dit.sh` | 使用学习得到的衰减权重进行轨迹评估，并输出相关诊断评分。 |
| Q2L / Q2S-Retained（test-test） | `eval_q2l_q2last_q2mono_scorefuse_maxmin_testtest.sh` | 轨迹终点匹配（代码中为 `Q2Last`）、保留步聚合（代码中为 `Q2Mono`），以及分数级融合。 |

```bash
# 激活 flow 环境后，从 GastroFusion/wass_flow 目录执行
bash eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_m2_dit.sh
bash eval_q2l_q2last_q2mono_scorefuse_maxmin_testtest.sh
```

如果解压位置不同，可在执行前设置以下两个变量：

```bash
export DATA_ROOT="/absolute/path/to/stomach-class-root"
export WEIGHT_ROOT="/absolute/path/to/checkpoint-root"
```

默认输出位置：

```text
wass_flow/results/f1_stomach_testtest/summary_seed42.json
wass_flow/results/q2l_stomach_testtest/fusion_q2last_q2mono/summary.csv
wass_flow/results/q2l_stomach_testtest/fusion_q2last_q2mono/summary.json
```

公开 Demo A 的默认评估配置为 **5-shot、1-query、100 episodes、随机种子 42**。详细参数选项和输出含义请查阅对应评估脚本。

### 其他研究评估协议

本模块还包含针对其他器官及 test-train 实验的评估脚本。这些实验的数据布局、checkpoint 路径和配置可能不同于公开的胃部 Demo A。运行前应检查对应 Shell 脚本和 Python 配置；不能将其直接等同于上述两个公开 Demo A 入口。

几何评分定义和评估协议见第 4 节。

## 4. 评估协议

### 两种协议

- **test-test：** Query 和 support 均来自同一测试切分。
- **reduced test-train：** 候选类别限定为 query 侧和 support 侧的交集（排除仅出现在 support 侧的类别）。对于共享类别中不足 K 张图像的情况，采用有放回采样构建支持集。

### 几何评分（使用相同 Episode、MetricEncoder / SK 几何与 K-shot 聚合方式）

Q2L 评估报告两种分数（`Q2S-Retained` 为论文中使用的名称，在代码中对应 `Q2Mono`）：

- **Q2Last：** 使用 Query-to-Step Dynamic-SK 计算 query 与轨迹终点 `z_T` 的匹配距离。
- **Q2S-Retained（保留步均值）：** 沿轨迹仅保留严格大于此前已保留距离的距离值（running maximum），对保留值取平均，其余距离被丢弃。

`scorefuse_maxmin` 变体在预测前融合 Q2Last 和 Q2S-Retained 的类别级距离向量，提供三种融合规则：

```text
dist_sum      逐类别对距离求和，再取 argmin
soft_vote     对每组分数使用 softmax(-d/τ)，将软投票求和后取 argmax
max_then_min  对每个类别取不同分数中的最差（最大）距离；
              选择最差距离最小的类别（max-min）
```

## 5. 关键文件

```text
models/metric_encoder.py       共享 token projector
models/sk_sinkhorn.py          动态 SK 节点质量 + cosine Sinkhorn
                               （fixed-mass API 支持“整条轨迹复用固定 SK 质量”）
models/trajectory_scorer.py    D0/DT/area/monotonic energy + shot soft-min
models/v1_model.py             Direct 与完整 Flow 的核心接口
losss/visreg.py                VISReg
losss/geometry.py              保持 RAE/metric 特征随时间变化的 profile
flow_adapter.py                冻结 vision+hint DiT 的 Heun/Euler adapter
train_dit_flow_spatial_sinkhorn_strict_intra_ot.py   DiT Flow 训练脚本
```
