# wass_flow — RAE–Flow–SK-Wasserstein few-shot 分类

这是一套按阶段解耦的实现：最终任务是 N-way K-shot 的 episodic few-shot classification（每个 episode 从候选类中采样 support 与 query）。

## 1. 主线

```text
query / K-shot support
→ frozen RAE（内镜图像编码为空间特征图，供后续 OT 使用）
→ candidate support-conditioned frozen DiT Flow（生成轨迹 z_0 … z_T）
→ 每个预测状态经同一个共享 Compressor 映射到度量 token 空间
→ Direct pair 计算一次 SK 节点质量，整条轨迹复用同一组质量（F1；Q2L 评测为逐步动态质量）
→ cosine 等价代价 + log-domain Sinkhorn
→ F1 LSTM 衰减头 → 单调状态权重 w_t → 轨迹加权分数
→ K-shot soft-min → N-way logits
```

"动态 SK"：节点质量由数据预测、区别于均匀质量；ground cost 取 L2-normalized tokens 上的平方欧氏距离（与 cosine 距离单调等价），entropic Sinkhorn 在 log 域稳定求解，返回的是正则化传输代价而非精确 W2（实现见 `models/sk_sinkhorn.py`）。

## 2. 训练阶段

训练主线：度量阶段 M0-A → M0-B → M1 → M2 依次训练；DiT Flow 与度量阶段**并行独立**、互不依赖；F1（冻结 Flow 的轨迹分类）为当前训练主线。

### M0-A：分类 warm-up

普通分类 batch，不使用 SK、OT、Flow、episode、VISReg。

```bash
torchrun --nproc_per_node=8 train_m0a_cls.py \
  --train_json_glob "/path/to/data_train_pool_imagenet_med/*.json" \
  --val_json_glob "/path/to/data_eval_pool_imagenet/*.json" \
  --project_root /path/to/wass_flow \
  --output_dir /path/to/weights_metric_v1/m0a \
  --batch_size 512 --rae_chunk_size 512 --epochs 10
```

也支持 `--train_root / --val_root` 直接读 ImageNet 目录（与 `--train_json_glob` 二选一）。

输出：

```text
metric_cls_pretrain.pt
```

### M0-B：ImageNet SK-Wasserstein 度量预训练

使用 P×K class-balanced batch，每个 anchor 采一个 positive 和一个 negative：

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

M0-B/M1 使用 P×K class-balanced sampler；RAE 编码通过 `--rae_chunk_size` 分块，默认每次编码 8 张，避免大 PK batch 直接压满显存。

### M1：医学 metric adaptation

与 M0-B 同一训练器，但切换到医学类池：

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

JSON loader 支持常见字段：

```text
path / image_path / x1_path / x0_path
label / label_B / class_name / category / caption
class_id / label_id / label_B_id
```

若你的 JSON 结构不同，只需修改 `data/class_pool.py` 的 `_iter_json_items()` 或字段候选列表。

### M2：Direct episode 校准（首次引入 episode）

首次进入 N-way K-shot episode，不使用 Flow。当前入口为 `train_m2_direct_episode_visreg.py`（episode + VISReg），episode 既可从 `--train_json_glob` 类池在线采样，也可直接给定 episode manifest JSON（当前 `.sh` 的用法）：

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

M2 必须从 M1 checkpoint 初始化；器官专用管线按器官从对应 M1 checkpoint 初始化。输出为 `metric_direct_episode.pt`（及 `_last.pt`）。变体入口：`train_m2_direct_episode_visreg_trainjson_valmanifest.py`（train 类池 + val manifest）、`train_m2_direct_episode_visreg_globalcls_truebatch_v2_valsingle.py`、`train_m2_direct_episode_flow_aug_visreg.py`（Flow 轨迹增强）。

### DiT Flow（与度量阶段并行）

DiT Flow 训练与 M0–M2 并行独立进行，互不影响。训练脚本为 `train_dit_flow_spatial_sinkhorn_strict_intra_ot.py`（SLURM 入口 `train_dit_flow_spatial_sinkhorn_strict_intra_ot.sh`；训练一律 8 卡，只有验证例外）。条件配置：timestep 与 target/support image CLS 走 AdaLN，patch tokens 走 Cross-Attention，weak noisy grayscale hint 走 Spatial Addition；caption 与 class condition 默认关闭，hint 默认 `alpha = 0.10`、`hint_dropout = 0.10`；图像分辨率统一为 256，与轨迹生成和 metric pipeline 对齐。

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

脚本同时保留 Flow Matching、OT pairing、可选 OT regularization、GAN、DDP、EMA 和 checkpoint 清理逻辑；OT 正则的完整参数见 `.sh`。

### F1：冻结 Flow 的轨迹分类（当前主线）

F1 即"使用 Flow 轨迹"的分类分支。F1 在冻结的 RAE / M2 MetricEncoder / DiT Flow 之上，只训练一个很小的 support-conditioned causal LSTM 衰减头（TemporalDecayHead）：

- 轨迹：对每个 query–candidate pair 生成候选条件轨迹 `z_0 … z_T` 及逐步距离。
- 衰减头：LSTM 对每个转移预测信息保留系数 `γ_t = γ_min + (1-γ_min)·σ(a_t)`；状态权重由累积保留得到 `u_0=1, u_t=∏γ_i, w_t=u_t/Σu_j`，因此 `w_0 ≥ w_1 ≥ … ≥ w_T > 0` 且 `Σ w_t = 1`——严格单调，编码"轨迹后段判别信号更弱"的先验，同时衰减率沿轨迹自适应。
- 轨迹分数：`step_to_support`（`Σ w_t D(z_t, support)`）或 `step_to_last`（`Σ w_t D(z_t, z_T)`），在冻结 M2 的 K-shot soft-min 之前形成；Direct 基线始终为 `D(z_0, support)`。
- 优化：主目标是最终分类 CE（可微），辅以 antithetic group-relative 扰动做对比探索。

训练分两步（torchrun DDP）：

1. **ImageNet/class-JSON 预训练**：`pretrain_f1_support_conditioned_lstm_v5_classjson_torchrun_ddp_batch.py`。`submit_multinode_pretrain_f1_support_conditioned_lstm_v5.sh` 以 sbatch 提交，srun 在每个节点执行 `run_pretrain_f1_support_conditioned_lstm_v5_multinode.sh`（默认每节点 8 进程，多机 torchrun 按 node_rank 组网）。
2. **具体部位的医学训练**：`train_f1_support_conditioned_lstm_v5_medical_manifest.*`，从对应器官 M2 checkpoint 初始化（`--m2_checkpoint`），并经 `PRETRAIN_F1_CHECKPOINT` / `--init_decay_checkpoint` 载入第 1 步的预训练 decay 头；`--trajectory_score_mode step_to_support / step_to_last` 选择分数分支。

早期 V1 版 F1（轨迹能量打分）为历史实现，入口不再列出。

F1 生成轨迹开销很高：单进程持有完整冻结 DiT、EndoViT 和 RAE；pair 通过 chunk 分块生成，不把全部轨迹同时放 GPU。

## 3. Quick Start：验证

环境（wass_flow 全程使用 conda `flow`）：

```bash
source /mnt/inaisfs/data/apps/cuda12.4.sh
source /mnt/inaisfs/data/home/tansy_criait/miniconda3/bin/activate flow
```

日常验证两族链路（F1 与 Q2L / Q2S-Retained，各含 test-test / test-train 两个协议；`.sh` 既是 SLURM 脚本也可直接 `bash` 运行，评测设置在 `.py` 的 USER CONFIG 块或 `.sh` 的环境变量里改）：

| 链路 | 脚本 | 内容 |
| --- | --- | --- |
| F1（test-test） | `eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_m2_dit.sh` | learned 衰减加权主分数 + Direct / Endpoint / Uniform / Fixed-decay 诊断 |
| F1（test-train） | `eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_testtrain_m2_dit.sh` | 同上，test-train 协议 |
| Q2L / Q2S-Retained（test-test） | `eval_q2l_q2last_q2mono_scorefuse_maxmin_testtest.sh` | Q2Last / Q2S-Retained + scorefuse（dist_sum / soft_vote / max_then_min） |
| Q2L / Q2S-Retained（reduced test-train） | `eval_q2l_q2last_q2mono_scorefuse_maxmin_testtrain.sh` | 同上，reduced test-train 协议 |

```bash
bash eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_m2_dit.sh
bash eval_q2l_q2last_q2mono_scorefuse_maxmin_testtest.sh
```

- F1 链路按器官加载权重：`.sh` 内 `F1_*` / `M2_*` 环境变量给出食管 / 胃 / 十二指肠各自的 F1 与 M2 checkpoint（`--organ_f1_checkpoint` / `--organ_m2_checkpoint`）；test-train 版另用 `--organ_support` 指定各器官的 support 池。
- Q2L 链路的 `.sh` 只是无参调用对应 `.py`，改 `.py` 顶部 `USER CONFIG`（`N_WAY` / `ORGAN_N_WAY` / `Q2L_CHECKPOINTS` / `SHOT_AGG` 等）即可。

运行前检查：

- **路径**：脚本内 `ROOT` 仍硬编码旧仓库路径 `/mnt/.../jmf/work/wass_flow`，`DATA_BASE`、checkpoint 路径按需用环境变量覆盖或改脚本。
- **数据布局**：`DATA_BASE / hospital / organ / class / image`（如 `data/indomain_support_76`），query 与 support 采自同一 hospital-organ 池，query 图像本身从 support 中排除。
- **关键权重**（已写在各 `.sh` / `.py` 默认值中，可用环境变量覆盖）：DiT 在 `weights_dit_flow_v1/` 下（如 `vision_hint_inclass/otcfm_dit_L_step_265000.pt`）；M2 与 F1 按器官/协议存于 `weights_metric_v1/<organ>-<protocol>/`。
- **输出**：写入各脚本 `BASE_RESULT_DIR`（CSV/JSON，含 accuracy、fuse、rescue/degradation 等）。

协议与指标含义见 §4。

## 4. 评测协议

### 两套协议

- **test-test**：query 与 support 来自同一测试切分。
- **reduced test-train**：candidate classes = query 侧 ∩ support 侧的共享类（剔除仅 support 侧的类）；共享类中不足 K 张的类有放回采样支持集。

### 几何分数（同一 episode、同一 MetricEncoder / SK 几何 / K-shot 聚合）

Q2L 评测报告两个分数（Q2S-Retained 为论文中的名称，对应代码里的 Q2Mono）：

- **Q2Last**：query 到轨迹末状态 `z_T` 的 Query-to-Step Dynamic-SK。
- **Q2S-Retained（保留步均值）**：沿轨迹只保留严格大于此前保留步的距离（running maximum，不保留则丢弃）取均值。

`scorefuse_maxmin` 变体把 Q2Last / Q2S-Retained 两个类级距离向量融合后预测（三种融合规则）：

```text
dist_sum      类间求和后 argmin
soft_vote     每个分数 softmax(-d/τ) 软投票求和后 argmax
max_then_min  每类取跨分数最坏距离，选最坏距离最小者（即 maxmin）
```

## 5. 关键文件

```text
models/metric_encoder.py       共享 token projector
models/sk_sinkhorn.py          SK 动态质量 + cosine Sinkhorn（fixed-mass API 支持"fixed SK across trajectory"）
models/trajectory_scorer.py    D0/DT/area/monotonic energy + shot soft-min
models/v1_model.py             Direct 与完整 Flow 核心接口
losss/visreg.py                VISReg
losss/geometry.py              RAE/metric 时间变化 profile 保持
flow_adapter.py                冻结 vision+hint DiT Heun/Euler adapter
train_dit_flow_spatial_sinkhorn_strict_intra_ot.py   DiT Flow 训练脚本
```
