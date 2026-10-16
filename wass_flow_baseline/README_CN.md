# Wasserstein-GastroFlow Baselines

本目录汇总使用的少样本学习（Few-Shot Learning, FSL）与长尾识别（Long-Tail Recognition）对比方法，提供各方法的训练、测试入口及评估说明。

## 1. 实验说明

各 Baseline 根据其原始方法及适配策略，使用本研究对应的医学数据划分进行训练、微调或加载已训练权重后评估，而不是直接使用未经适配的模型进行比较。

- **FSL 基线**：在胃、食管和十二指肠数据上，按论文规定的 N-way K-shot episodic 协议评估；区分 test-pool support 与 training-pool support。
- **长尾识别基线**：按各方法对应的非 episodic 分类协议进行训练和评估，作为上下文比较，而非与 FSL 完全相同的 episodic 实验。
- **数据隔离**：测试查询图像不应用于模型参数优化；test-pool support 是推理时提供的参考样本，不能等同于用整个测试池训练模型。

不同方法可能使用不同的训练实现、初始化权重和 checkpoint；具体以各子目录脚本与配置为准。

## 2. 训练与测试入口

下表列出当前的脚本；部分脚本以胃或十二指肠为例，其他器官的脚本与路径应在相应方法目录中配置。

| 类型 | 方法 | Train 入口 | Test 入口 |
|---|---|---|---|
| FSL | DeepEMD | `DeepEMD/train_meta1_stomach_tsy.py` | `DeepEMD/eval_1_batch_dynamic_way_tsy_test_test.sh` |
| FSL | FSL-OM（Oblique Manifold） | `FSL-OM-main/train.py` | `FSL-OM-main/eval_test_test.sh` |
| FSL | PMF | `pmf_cvpr22/main_胃.py` | `pmf_cvpr22/test_5shot_our_new_data_test_test.py` |
| FSL | Prototypical Networks | `prototypical-network-pytorch/train_our.py` | `prototypical-network-pytorch/test_our_test_test.py` |
| FSL | PT-MAP | `PT-MAP-master/train.py` | `PT-MAP-master/test_standard_tsy.py` |
| FSL | SimpleShot | 不单独训练；使用 PT-MAP 预训练权重 | `simple_shot-master/eval_medical_tsy.sh` |
| FSL | Simple-CNAPS | `simple-cnaps-master/simple-cnaps-src/run_simple_cnaps.py` | `simple-cnaps-master/simple-cnaps-src/run_medical_cnaps_tsy.sh` |
| FSL | GAP | `GAP/main.sh` | `GAP/test-test_effectiveway_fix_12datasets.sh` |
| Long-tail | BPaCo | `BPaCo-main/sh/train_gastric_tsy.sh` | `BPaCo-main/sh/test_gastric_tsy_胃.sh` |
| Long-tail | Decoupling | `classifier-balancing-main/main.py`（或对应器官训练脚本） | `classifier-balancing-main/sh/test_胃.sh`（示例） |
| Long-tail | LDAM-DRW | `LDAM-DRW-master/sh/train_十二指肠_ldam.sh` | `LDAM-DRW-master/sh/test_十二指肠_ldam.sh` |



## 3. 训练权重与评估

- **FSL-OM、PT-MAP**：已在对应医学数据上完成训练，评估时加载各自目录下的医学实验 checkpoint（例如 `FSL-OM-main/checkpoints/medical/`）；不必为了运行已有评估结果而重新训练。
- **SimpleShot**：复用 PT-MAP 预训练权重，不执行独立模型训练。
- **其余方法**：根据表中的训练入口生成或准备所需 checkpoint，再使用对应测试入口评估。




> 本文档提供入口索引，不保证全部脚本在未修改路径与环境配置时可直接运行。更完整的环境、参数及器官特定命令应以各方法的实现文件为准。
