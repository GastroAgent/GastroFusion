#!/usr/bin/env python3
"""
根据 data/{dataset}/train|test|val/{class_name}/*.jpg 的目录结构，
自动生成 train.txt / val.txt / test.txt 列表文件。

支持三个项目：
  1. BPaCo          -> data/{dataset}/   (e.g. 胃_train.txt，相对路径)
  2. classifier-bal -> dataset/{dataset}/ (e.g. 胃_1query_train.txt，相对路径)
  3. LDAM-DRW       -> data/{dataset}/   (e.g. 胃_1query_train.txt，绝对/相对路径)

class_idx 按子文件夹名字母顺序从 0 开始编号。

用法:
    python generate_filelist.py                       # 默认：生成 BPaCo + classifier-balancing
    python generate_filelist.py --gen-bpaco          # 仅 BPaCo
    python generate_filelist.py --gen-classifier     # 仅 classifier-balancing
    python generate_filelist.py --gen-ldam --abs     # 仅 LDAM-DRW，绝对路径
    python generate_filelist.py --gen-all            # 全部项目
    python generate_filelist.py 胃_1query            # 指定数据集（配合上述 --gen-* 使用）
"""

import os
import sys
import argparse

SUPPORTED_DATASETS = ["胃_1query", "食管_1query", "十二指肠_1query"]
# BPaCo txt 文件名去掉 "_1query" 后缀
BPaCo_NAME = {"胃_1query": "胃", "食管_1query": "食管", "十二指肠_1query": "十二指肠"}
SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


def get_class_order(data_root: str) -> list[str]:
    train_dir = os.path.join(data_root, "train")
    folders = sorted([
        f for f in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, f))
    ])
    return folders


def scan_images(data_root: str, split: str, class_order: list[str]):
    """扫描单个 split 目录，返回 (img_path, class_idx) 元组列表"""
    split_dir = os.path.join(data_root, split)
    entries = []
    for cls_idx, cls_name in enumerate(class_order):
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for img_name in sorted(os.listdir(cls_dir)):
            ext = os.path.splitext(img_name)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            img_path = os.path.join(split, cls_name, img_name)
            entries.append((img_path, cls_idx))
    return entries


def write_txt(entries: list, output_file: str, use_abs: bool, data_root: str):
    lines = []
    for img_path, cls_idx in entries:
        if use_abs:
            img_path = os.path.join(data_root, img_path)
        lines.append(f"{img_path} {cls_idx}")
    lines.sort()
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_dataset(data_root: str, class_order: list[str], output_dir: str,
                     filename_prefix: str, use_abs: bool):
    """生成三个 split 的 txt 文件"""
    print(f"  输出目录: {output_dir}")
    total = 0
    for split in SPLITS:
        entries = scan_images(data_root, split, class_order)
        output_file = os.path.join(output_dir, f"{filename_prefix}_{split}.txt")
        write_txt(entries, output_file, use_abs, data_root)
        print(f"    {split}.txt -> {len(entries)} images")
        total += len(entries)
    print(f"  共 {total} images")
    return total


def main():
    parser = argparse.ArgumentParser(description="生成数据集列表文件")
    parser.add_argument("datasets", nargs="*", default=SUPPORTED_DATASETS,
                        help=f"数据集名称，可选: {SUPPORTED_DATASETS}，默认全部")
    parser.add_argument("--abs", action="store_true",
                        help="LDAM-DRW 使用绝对路径（默认相对路径）")
    parser.add_argument("--gen-bpaco", action="store_true",
                        help="生成 BPaCo 格式（相对路径，文件名如 胃_train.txt）")
    parser.add_argument("--gen-classifier", action="store_true",
                        help="生成 classifier-balancing 格式（相对路径，文件名如 胃_1query_train.txt）")
    parser.add_argument("--gen-ldam", action="store_true",
                        help="生成 LDAM-DRW 格式（文件名如 胃_1query_train.txt）")
    parser.add_argument("--gen-all", action="store_true",
                        help="生成全部三个项目（BPaCo + classifier-balancing + LDAM-DRW）")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ws_root = os.path.dirname(script_dir.rstrip("/"))

    datasets = [ds for ds in args.datasets if ds in SUPPORTED_DATASETS]
    invalid = [ds for ds in args.datasets if ds not in SUPPORTED_DATASETS]
    for ds in invalid:
        print(f"[ERROR] 不支持的数据集 '{ds}'，跳过。可用: {SUPPORTED_DATASETS}", file=sys.stderr)

    if not datasets:
        print("[ERROR] 没有可处理的数据集，退出。", file=sys.stderr)
        sys.exit(1)

    # 默认：生成 BPaCo + classifier-balancing
    gen_bpaco = args.gen_bpaco or args.gen_all
    gen_classifier = args.gen_classifier or args.gen_all
    gen_ldam = args.gen_ldam or False  # LDAM-DRW 不在默认范围内

    if not gen_bpaco and not gen_classifier and not gen_ldam:
        gen_bpaco = True
        gen_classifier = True

    for ds in datasets:
        bpaco_name = BPaCo_NAME[ds]
        print(f"\n{'='*60}")
        print(f"数据集: {ds}  (BPaCo 文件名: {bpaco_name}_*.txt)")

        # ---- BPaCo ----
        if gen_bpaco:
            bpaco_root = os.path.join(ws_root, "BPaCo-main", "data", ds)
            if os.path.isdir(bpaco_root):
                print(f"\n[BPaCo] 相对路径模式")
                class_order = get_class_order(bpaco_root)
                print(f"  共 {len(class_order)} 个类别:")
                for i, c in enumerate(class_order):
                    print(f"    {i}: {c}")
                generate_dataset(bpaco_root, class_order, bpaco_root,
                                 filename_prefix=bpaco_name, use_abs=False)
            else:
                print(f"  [WARN] BPaCo 目录不存在: {bpaco_root}，跳过")

        # ---- classifier-balancing ----
        if gen_classifier:
            cb_root = os.path.join(ws_root, "classifier-balancing-main", "dataset", ds)
            if os.path.isdir(cb_root):
                print(f"\n[classifier-balancing] 相对路径模式")
                class_order = get_class_order(cb_root)
                print(f"  共 {len(class_order)} 个类别:")
                for i, c in enumerate(class_order):
                    print(f"    {i}: {c}")
                generate_dataset(cb_root, class_order, cb_root,
                                 filename_prefix=ds, use_abs=False)
            else:
                print(f"  [WARN] classifier-balancing 目录不存在: {cb_root}，跳过")

        # ---- LDAM-DRW ----
        if gen_ldam:
            ldam_root = os.path.join(ws_root, "LDAM-DRW-master", "data", ds)
            if os.path.isdir(ldam_root):
                mode = "绝对路径" if args.abs else "相对路径"
                print(f"\n[LDAM-DRW] {mode}模式")
                class_order = get_class_order(ldam_root)
                print(f"  共 {len(class_order)} 个类别:")
                for i, c in enumerate(class_order):
                    print(f"    {i}: {c}")
                generate_dataset(ldam_root, class_order, ldam_root,
                                 filename_prefix=ds, use_abs=args.abs)
            else:
                print(f"  [WARN] LDAM-DRW 目录不存在: {ldam_root}，跳过")

    print(f"\n{'='*60}")
    print("全部完成！")


if __name__ == "__main__":
    main()
