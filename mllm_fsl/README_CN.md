# GastroFusion — 几何–语义推理、融合与评估

[English](README_EN.md) | 简体中文

`mllm_fsl/` 负责将已训练的 **GastroMLLM（Med-Qwen3-VL）** 与 **Wasserstein-GastroFlow** 应用于内部和外部胃肠道内镜队列，生成两分支预测，通过验证数据确定器官特异性融合参数，执行最终融合，并分析参数敏感性。本目录**不负责训练两个基础模型**；模型训练分别参见仓库根目录 `mllm/` 和 `wass_flow/`。

> 本说明依据最新仓库中的代码、数据及已发布结果整理。脚本保留了服务器的绝对路径，执行前需要修改。提供 JSON 并不等于提供 JSON 所引用的临床原始图像或模型权重。

## 1. Overall pipeline

```text
               内部 WCH / 外部中心–器官队列
                              │
                ┌─────────────┴─────────────┐
                │                           │
          GastroMLLM                 Wasserstein-GastroFlow
          语义证据分支                     几何证据分支
                │                           │
     data/mllm_data/ JSON          data/fsl_data/ flat JSON
                │                           │
      MLLM inference server           RAE → Metric → Flow
                │                           │
     semantic_score_mllm_*.py       offline_align_rae_metric_*
                │                           │
       mllm/mllm_result/               fsl/fsl_result/
                │                           │
                └─────────────┬─────────────┘
                              │
                   val_grid/ 验证集参数搜索
                              │
                   预先确定并固定融合参数
                              │
                   fusion/ 内部/外部融合
                              │
              mllm_fsl_detail/mllm_fsl_result/
                              │
                  sensitivity/ 参数敏感性分析
```

这里的 `val_grid/`、`fusion/` 与 `sensitivity/` 是**不同用途**的代码：前者在独立验证数据上比较候选参数，中者以选定配置产生逐病例融合预测，后者评估结果对参数扰动的敏感程度。最终测试数据不用于重新选择参数。

## 2. Directory structure

```text
mllm_fsl/
├── data/
│   ├── mllm_data/
│   │   ├── external_few_way/              # 外院 cohort-aligned 输入（9 JSON）
│   │   └── external_lot_way/              # 外院 full-organ 输入（9 JSON）
│   └── fsl_data/
│       ├── huaxi/support_831_val/         # 华西三器官 flat JSON
│       └── external/
│           ├── mllm_fsl_subset_agent_few_way_91/  # 3 中心 × 3 器官
│           └── mllm_fsl_subset_agent_lot_way_91/  # 3 中心 × 3 器官
├── mllm/
│   ├── mllm_code/                         # 模型服务和语义评分脚本
│   └── mllm_result/                       # 内部及外部语义结果
├── fsl/
│   ├── fsl_code/                          # 离线 RAE/Metric/Flow 推理脚本
│   └── fsl_result/                        # 仓库当前附带的几何评估结果
└── mllm_fsl_detail/
    ├── mllm_fsl_code/
    │   ├── val_grid/                      # 参数网格搜索
    │   ├── fusion/                        # 内部和外部融合
    │   └── sensitivity/                   # 参数敏感性分析
    └── mllm_fsl_result/                   # 仓库附带的最终融合结果
```

**数据命名：** `huaxi` 表示华西医院（WCH）；外部中心包括南充、自贡、苍溪，各包含食管、胃和十二指肠。`few_way` 指 external cohort-aligned（以该队列的疾病标签为候选空间），`lot_way` 指 full-organ（以该器官的完整疾病标签为候选空间）。这些是实验的**标签空间设置**，不是不同模型版本。

## 3. Input data (`data/`)

### 3.1 GastroMLLM input (`data/mllm_data/`)

公开目录包含外部两组输入，每组九个 `<中心><器官>_agent.json`：

```text
data/mllm_data/
├── external_few_way/南充胃_agent.json
└── external_lot_way/南充胃_agent.json
```

MLLM 输入描述待评估病例、图像路径及相应候选信息，由 `semantic_score_mllm_external.py` 解析，送往 MLLM 服务获取开放式诊断及各候选疾病的语义评分。

### 3.2 Wasserstein-GastroFlow input (`data/fsl_data/`)

```text
data/fsl_data/
├── huaxi/support_831_val/
│   ├── 食管_final_doctor_exam_flat.json
│   ├── 胃_final_doctor_exam_flat.json
│   └── 十二指肠_final_doctor_exam_flat.json
└── external/
    ├── mllm_fsl_subset_agent_few_way_91/<中心>/<器官>/final_doctor_exam_*_flat.json
    └── mllm_fsl_subset_agent_lot_way_91/<中心>/<器官>/final_doctor_exam_*_flat.json
```

每个 flat JSON 是用于构造 query–support 比较的**配对数据**。实际记录包含 `data_id`、`question_id`、`x0`、`x1`、`label_A`、`label_B`、`support_dir`、`support_selection` 等字段；`x0`、`x1` 等路径必须能够被本地文件系统访问。对同一 query 可以存在多个候选类别及多个 support pair。


## 4. GastroMLLM semantic branch (`mllm/`)

| 实际文件 | 作用 |
|---|---|
| `mllm_code/gastro_mllm_qwenvl3_server.py` | 加载医学适配 Qwen3-VL 并提供语义推理/评分服务 |
| `mllm_code/gastro-mllm-server.sh` | 配置环境、checkpoint 与服务端口并启动服务 |
| `mllm_code/semantic_score_mllm_huaxi.py` | 华西数据逐病例调用服务并整理语义分数 |
| `mllm_code/semantic_score_mllm_external.py` | 外部九个中心–器官队列的批量推理及语义评分 |
| `mllm_code/run_batch_mllm_huaxi.sh` | 华西批处理启动入口 |
| `mllm_code/run_batch_mllm_external.sh` | 外部批处理启动入口 |

流程是先启动服务，再运行对应批处理脚本。服务与客户端可能位于不同节点，运行前必须统一主机、端口、网络访问方式和模型权重路径。

```bash
# 从仓库根目录执行；先修正脚本中的路径和服务地址
bash mllm_fsl/mllm/mllm_code/gastro-mllm-server.sh

# 在另一终端或作业中，选择需要的队列
bash mllm_fsl/mllm/mllm_code/run_batch_mllm_huaxi.sh
bash mllm_fsl/mllm/mllm_code/run_batch_mllm_external.sh
```

**结果目录（当前仓库均有相应文件）：**

```text
mllm/mllm_result/
├── mllm_result_huaxi/<器官>/Med-qwen3-vl-RL_latent_<器官>.json
├── mllm_result_external9_few_way/<中心>/<器官>/Med-qwen3-vl-RL_latent_<中心>_<器官>.json
└── mllm_result_external9_lot_way/<中心>/<器官>/Med-qwen3-vl-RL_latent_<中心>_<器官>.json
```

每个结果文件具有 `results` 列表及部分运行配置/汇总信息。逐病例字段通常包括 `image_id`、`target_image`、`ground_truth`、`mllm_diag`、`mllm_score_map`、`candidate_diseases` 等。融合使用候选标签上的**评分证据**（及相关置信度）；病例参考标签用于离线评估。

## 5. Wasserstein-GastroFlow geometric branch (`fsl/`)

| 实际文件 | 作用 |
|---|---|
| `fsl_code/offline_align_rae_metric_flow_service_new_test_test.py` | 离线 RAE + Metric + Flow 几何距离计算，用于对应 test-test 设置 |
| `fsl_code/offline_align_rae_metric_flow_service_new_test_train.py` | 离线几何距离计算，用于对应 test-train 设置 |
| `fsl_code/run_batch_3huaxi_5shot.sh` | 华西三个器官的固定配对批量评估 |
| `fsl_code/run_batch_9_5shot_fewway.sh` | 外部九个 cohort-aligned 队列的批量评估 |
| `fsl_code/run_batch_9_5shot_lotway.sh` | 外部九个 full-organ 队列的批量评估 |

该分支按照输入的 query–support 配对提取图像表征、生成候选条件轨迹并计算几何匹配距离，再按类别聚合，得到候选疾病距离、类别排序与最终预测。离线评估会保存逐配对结果和逐 query 结果，以便下游直接读取类别级距离。

```bash
# 从仓库根目录执行，须先修正脚本中的路径和模型权重
bash mllm_fsl/fsl/fsl_code/run_batch_3huaxi_5shot.sh
bash mllm_fsl/fsl/fsl_code/run_batch_9_5shot_fewway.sh
bash mllm_fsl/fsl/fsl_code/run_batch_9_5shot_lotway.sh
```

**FSL 常见输出文件：**

| 文件 | 用途 |
|---|---|
| `result.json` | 原始 query–support 配对及几何距离信息 |
| `query_results.json` | 按 query 聚合的类别级结果，供最终融合程序优先读取 |


**本次压缩包实际随仓库提供的 FSL 结果只有：**

```text
fsl/fsl_result/huaxi/食管/query_results.json
```

其他器官及外部队列的几何结果需要按相应脚本生成或另行准备。

## 6. Validation-grid parameter selection (`val_grid/`)

`mllm_fsl_detail/mllm_fsl_code/val_grid/val_grid_search.py` 针对验证集的 MLLM 和 FSL 预测，枚举器官特异性参数组合，调用内部融合 Python，汇总每组结果并按验证表现比较。

**输入：** 验证集 MLLM 结果根目录 `INPUT_ROOT`、对应几何结果根目录 `RAE_ROOT`、参数网格 `GRID`。**输出：** `OUTPUT_ROOT` 下每一组实验的预测结果和汇总/排名文件，供选择固定参数使用。脚本内的 `PY`、`INPUT_ROOT`、`RAE_ROOT` 和 `OUTPUT_ROOT` 当前为研究服务器路径，必须按实际验证集修改。

参数范围涉及胃的分数融合权重/保护阈值、食管和十二指肠的语义覆盖阈值等。由验证得到的参数再写入对应 Fusion Shell 配置。

## 7. Final geometric–semantic fusion (`fusion/`)

| 实际文件 | 作用 |
|---|---|
| `fsl_mllm_fusion_tolast_rae_v2_huaxi.py` | 华西内部逐病例融合及队列统计 |
| `fsl_mllm_fusion_tolast_rae_v2_external.py` | 外部中心–器官队列融合及统计 |
| `fsl_mllm_fusion_tolast_rae_v2_huaxi_grid_val.sh` | 华西融合的 Shell / SLURM 启动器 |
| `fsl_mllm_fusion_tolast_rae_v2_external_grid_val.sh` | 外部融合的 Shell / SLURM 启动器 |

**数据连接：**

- `INPUT_ROOT`：上游 `mllm_result_huaxi/` 或 `mllm_result_external9_{few,lot}_way/` 中的主结果 JSON；每个病例具有语义预测及疾病评分。
- `RAE_FSL_RESULT_ROOT`（传入 Python 的 `--rae_fsl_result_root`）：离线几何结果的根目录；程序按器官或中心–器官定位 `query_results.json`，必要时回退读取 `result.json`。
- 同一病例的图像标识、候选诊断名称及候选标签空间必须在两分支之间对齐；避免混用 cohort-aligned 与 full-organ 结果。

**融合规则：** 胃使用结合几何间隔保护的分数融合；食管以几何预测为基础并允许高置信语义覆盖；十二指肠采用更严格的语义覆盖条件。具体参数在两个 `.sh` 中显式设置，由验证集确定后固定用于相应测试数据。

**输出内容：** 逐病例融合 JSON 通常包含原始诊断信息、FSL 类级几何距离与排序、两分支预测、融合预测、正确性标记及覆盖诊断；队列级 JSON/CSV 汇总分支与融合准确率，以及纠错、退化、匹配失败等统计。内部已发布的输出路径为：

```text
mllm_fsl_detail/mllm_fsl_result/
└── result_huaxi_831_grid/v2_fusion_traj_tlast_rae_latent_tsy/
    ├── 十二指肠/Med-qwen3-vl-RL_latent_十二指肠_fusion_v2_traj.json
    ├── 胃/Med-qwen3-vl-RL_latent_胃_fusion_v2_traj.json
    └── 食管/Med-qwen3-vl-RL_latent_食管_fusion_v2_traj.json
```

本次压缩包的 `mllm_fsl_result/` **只包含以上三个器官的逐病例融合 JSON**，未附带外部融合结果及批量 summary/failure 汇总文件；但运行对应脚本可以在配置的输出目录生成相应结果。

```bash
# 从仓库根目录执行；须先修正 PY、输入根目录、输出根目录和环境
bash mllm_fsl/mllm_fsl_detail/mllm_fsl_code/fusion/fsl_mllm_fusion_tolast_rae_v2_huaxi_grid_val.sh
bash mllm_fsl/mllm_fsl_detail/mllm_fsl_code/fusion/fsl_mllm_fusion_tolast_rae_v2_external_grid_val.sh
```

## 8. Parameter sensitivity (`sensitivity/`)

| 实际文件 | 作用 |
|---|---|
| `fsl_mllm_fusion_tolast_rae_v2_huaxi_sweep.py` / `.sh` | 华西内部敏感性实验 |
| `fsl_mllm_fusion_tolast_rae_v2_sweep.py` / `.sh` | 外部队列敏感性实验 |

敏感性分析使用已确定的融合配置，在其附近改变选定参数，检查模型准确率、纠错与退化指标随阈值/权重变化的情况，辅助评估结果稳健性；它不承担独立测试集重新选参的功能。运行前应检查 `.sh` 中的 `SWEEP_PY`、输入结果目录和输出根目录，并确认分析程序依赖的融合代码版本与当前正式融合代码一致。

## 9. Reproduction workflows

### 从输入数据重新生成

1. 准备使用的图像、MLLM 输入 JSON、FSL flat 配对 JSON，以及匹配的模型权重。
2. 启动 GastroMLLM 服务，执行内部或外部语义批处理，生成 `mllm_result/`。
3. 按对应数据协议运行离线几何批处理，生成各队列 `query_results.json`（以及必要的 `result.json`）。
4. 在独立验证数据上运行 `val_grid/`，确定器官特异性参数。
5. 将参数固定到 `fusion/` Shell 配置；对内部或外部匹配队列执行融合。
6. 需要时运行 `sensitivity/` 分析，检查参数附近的稳定性与误差变化。


模型训练和底层实现见仓库根目录的 `mllm/README.md` 与 `wass_flow/README_CN.md`；本文件重点说明 **`mllm_fsl/` 的数据组织、推理、参数选择、融合**。
