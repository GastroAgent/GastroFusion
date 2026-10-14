#!/usr/bin/env python3
"""查看 gastric 数据集的类别映射及样本统计"""

import os

# ── 十二指肠数据集（7类）────────────────────────────────────────
DUODENAL_CLASSES = {
    0: "十二指肠副乳头",        # Duodenal minor papilla
    1: "十二指肠憩室",          # Duodenal diverticulum
    2: "十二指肠正常",          # Duodenum normal
    3: "十二指肠溃疡",          # Duodenal ulcer
    4: "十二指肠炎",            # Duodenitis
    5: "十二指肠球部息肉",      # Duodenal bulb polyp
    6: "十二指肠钩虫",          # Duodenal hookworm
}

# ── 胃部数据集（8类）────────────────────────────────────────────
STOMACH_CLASSES = {
    0: "慢性非萎缩性胃窦炎伴痘疹",
    1: "正常胃窦",
    2: "残胃炎伴胆汁反流",
    3: "皮革胃",
    4: "胃MALT淋巴瘤",
    5: "胃神经内分泌肿瘤",
    6: "胃窦溃疡（A1期）",
    7: "胃窦溃疡（S1期）",
}

DATA_ROOT_DUO  = "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/BPaCo-main/data/十二指肠_1query"
DATA_ROOT_GAST = "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/BPaCo-main/data/胃_1query"

splits_duo  = {"train": "十二指肠_train.txt", "test": "十二指肠_test.txt",  "val": "十二指肠_val.txt"}
splits_gast = {"train": "胃_train.txt",       "test": "胃_test.txt",        "val": "胃_val.txt"}


def show_split(txt_path, class_names, split_name, dataset_name):
    print(f"\n{'='*55}")
    print(f"  {dataset_name}  |  Split: {split_name}")
    print(f"{'='*55}")
    print(f"{'Class ID':<12} {'类别名称':<28} {'样本数':<8} {'占比':<8}")
    print("-" * 55)

    counts = {}
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rsplit(None, 1)
            if len(parts) >= 2:
                label = int(parts[1])
                counts[label] = counts.get(label, 0) + 1

    total = sum(counts.values())
    for cid in sorted(class_names):
        n = counts.get(cid, 0)
        pct = n / total * 100 if total else 0
        name = class_names.get(cid, f"未知类别({cid})")
        print(f"{cid:<12} {name:<28} {n:<8} {pct:.1f}%")
    print("-" * 55)
    print(f"{'Total':<12} {'':<28} {total:<8}")


# 十二指肠
print("\n" + "#" * 60)
print("  十二指肠数据集（Duodenum）—— 7 类")
print("#" * 60)
for split_name, txt_name in splits_duo.items():
    show_split(os.path.join(DATA_ROOT_DUO, txt_name),
               DUODENAL_CLASSES, split_name, "十二指肠")

# 胃部
print("\n" + "#" * 60)
print("  胃部数据集（Gastric / Stomach）—— 8 类")
print("#" * 60)
for split_name, txt_name in splits_gast.items():
    show_split(os.path.join(DATA_ROOT_GAST, txt_name),
               STOMACH_CLASSES, split_name, "胃部")
