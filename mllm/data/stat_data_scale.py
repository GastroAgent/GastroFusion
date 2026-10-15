#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计 data/ 下各数据集规模。

- pretrain/ 四阶段训练数据（Single_Image / Multi_Image / RL_1 / RL_2）
- cold_start_huaxi* 划分（SFT 冷启动 / RL）

用法：
    python stat_data_scale.py
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

PRETRAIN_JSONS = [
    "pretrain/Single_Image/train_sft.json",
    "pretrain/Single_Image/train_sft2.json",
    "pretrain/Multi_Image/train_sft_f_8_clean1.json",
    "pretrain/Multi_Image/train_sft_f_12_clean1.json",
    "pretrain/Multi_Image/train_sft_f_18_clean1.json",
    "pretrain/RL_1/train.json",
    "pretrain/RL_2/train.json",
    "pretrain/RL_2/filtered_train.json",
]
COLD_START_DIRS = ["cold_start_huaxi", "cold_start_huaxi2"]


def count_json(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return len(d) if isinstance(d, list) else len(d)


def main():
    print("== pretrain 四阶段训练数据 ==")
    for name in PRETRAIN_JSONS:
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            print(f"{name:50s} {count_json(p):>8d} 条")
        else:
            print(f"{name:50s} 缺失")

    print("\n== cold_start 划分（SFT 冷启动 / RL）==")
    for d in COLD_START_DIRS:
        sp = os.path.join(HERE, d, "statistics.json")
        if not os.path.exists(sp):
            print(f"{d}: 缺少 statistics.json")
            continue
        with open(sp, "r", encoding="utf-8") as f:
            s = json.load(f)
        print(
            f"{d}: SFT {s.get('exported_sft_samples')} 条, "
            f"RL {s.get('exported_rl_samples')} 条, "
            f"裁判均分 {s.get('accepted_judge_score_mean', 0):.3f}"
        )


if __name__ == "__main__":
    main()
