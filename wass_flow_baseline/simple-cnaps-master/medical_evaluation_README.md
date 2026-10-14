# Simple CNAPS 在医疗数据集上的复现指南

本目录新增/修改的文件用于在 `fsl_scoreboard/data/` 的三个医疗内窥镜数据集
（`胃_1query`、`十二指肠_1query`、`食管_1query`）上复现 Simple CNAPS 的测试结果。

## 1. 环境要求

| 组件 | 版本 | 备注 |
| --- | --- | --- |
| Python | 3.10 | |
| PyTorch | 2.4.0 + CUDA 12.4 | 至少 1 张 16GB+ GPU |
| TensorFlow | 2.11.0 | Simple CNAPS 仍依赖 Meta-Dataset 的 TF 流水线 |
| gin-config | 0.5.0 | Meta-Dataset 配置 |
| Pillow | 10.x | |
| numpy | 1.26.x | |
| simclr | **空 stub** | 见下 |

`openthinkimg` 环境已验证满足全部依赖（`conda activate openthinkimg` 即可）。

### 已打补丁 / 需要 stub 的依赖

1. **simclr**：原仓库 `https://github.com/google-research/simclr` 没有 `setup.py`，无法 `pip install`。
   在仓库中已通过 `simclr_episode_fraction=0.0` 关闭 SimCLR 增强，但 `meta_dataset.data.pipeline`
   仍会 `import simclr.data_util`。本仓库已自带最小 stub：

   ```
   fsl_scoreboard/simple-cnaps-master/_env_stubs/simclr/
       __init__.py     # 空
       data_util.py    # preprocess_for_train 抛 NotImplementedError
   ```

   运行时只要把它加进 `PYTHONPATH` 即可（无需自行创建）。

2. **`meta-dataset/meta_dataset/data/dataset_spec.py`** 顶部已经打过补丁，
   把 `collections.Mapping` 等 `collections.abc.Mapping` 的兼容逻辑补回 `collections`。
   若重新拉取 meta-dataset 子仓库，需要重新应用该补丁（参见 `dataset_spec.py` 第 23 行附近）。

## 2. 数据准备（一次性）

把 `fsl_scoreboard/data/{subset}/{train,val,test}/*.jpg` 转成 Meta-Dataset TFRecord：

```bash
conda activate openthinkimg
export PYTHONPATH=/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master/_env_stubs:$PYTHONPATH
cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master
python convert_medical_data.py \
    --data_root /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/data \
    --records_root /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/records
```

成功后会看到三个数据集的 classes 数和 TFRecord 数量。

## 3. 评估

```bash
conda activate openthinkimg
export PYTHONPATH=/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master/_env_stubs:$PYTHONPATH
export META_DATASET_ROOT=/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master/meta-dataset
cd /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/simple-cnaps-master/simple-cnaps-src

# 食管 8-way 5-shot
python run_medical_cnaps.py \
    --data_path /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/records \
    --subset 食管_1query \
    --n_runs 600 \
    --test_model_path ../model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt \
    --feature_adaptation film \
    --output_json ../eval_result_食管.json

# 胃 8-way 5-shot
python run_medical_cnaps.py \
    --data_path /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/records \
    --subset 胃_1query \
    --n_runs 600 \
    --test_model_path ../model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt \
    --feature_adaptation film \
    --output_json ../eval_result_胃.json

# 十二指肠 7-way 5-shot
python run_medical_cnaps.py \
    --data_path /mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/fsl_scoreboard/records \
    --subset 十二指肠_1query \
    --way 7 \
    --n_runs 600 \
    --test_model_path ../model-checkpoints/meta-dataset-checkpoints/best_simple_cnaps.pt \
    --feature_adaptation film \
    --output_json ../eval_result_十二指肠.json
```

参数：
- `--subset` 必须为 `胃_1query` / `十二指肠_1query` / `食管_1query` 之一
- `--way` 默认按子集自动选择（胃/食管 8，十二指肠 7）
- `--shot` 默认 5，`--query` 默认 1
- `--feature_adaptation` 可选 `film` / `film+ar`，与原始 Simple CNAPS 一致
- `--n_runs` 是测试 episode 数；Meta-Dataset 论文用 600，可以根据需要改成 10000 之类

输出格式与 `simple_shot-master/eval_result.json` 风格类似：

```json
{
  "dataset": "食管_1query",
  "method": "SimpleCNAPS",
  "feature_adaptation": "film",
  "n_ways": 8,
  "n_shot": 5,
  "n_queries": 1,
  "n_runs": 600,
  "test_acc": 41.66,
  "pm": 2.01,
  "test_model_path": "..."
}
```

## 4. 已知注意点

- Simple CNAPS 默认是 8-way 训练，但 Meta-Dataset 训练时还会采样更多类别；
  在医疗数据集这种小分类数场景下也能工作。
- 评估脚本固定 way/shot/query，使用 `SingleDatasetReader` 的固定配置，
  保证每个 episode 都符合 `--way --shot --query` 的设定。
- 如果需要重新训练 Simple CNAPS（含 feature adaptation），仍然可以用官方
  `run_simple_cnaps.py` 训练，把 `data_path` 指向 `records/`，并把
  `init_data()` 中 train/test_set 列表替换成医学数据集名称。
  训练本身会显著更慢（需要在 8 张 GPU 上跑 1-3 天），所以推荐直接使用
  `best_simple_cnaps.pt` 提供的 Meta-Dataset 预训练权重做评估。
- Simple AR-CNAPS（`film+ar`）因为需要 2 张 GPU，且权重 98 MB，与本项目
  的 8-way 5-shot 1-query 配置可以共存；用相同命令外加 `--feature_adaptation film+ar`
  即可。