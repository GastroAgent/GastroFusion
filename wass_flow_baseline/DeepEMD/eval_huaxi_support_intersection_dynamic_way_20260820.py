#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepEMD 跨中心 few-shot 批量评估：华西 support -> 外院 query
============================================================

一次运行 3 个外院 × 3 个部位：
    医院：自贡、苍溪、南充
    部位：食管、胃、十二指肠

协议：
    query   = 外院数据
    support = 华西本院数据
    candidate pool = query/support 同名类别交集
    GT pool        = candidate pool

    不保留仅存在于华西 support 的 support-only distractor 类。

目录兼容：
    Query 既支持：
        QUERY_BASE/医院/部位/test/类别/图像
    也支持：
        QUERY_BASE/医院/部位/类别/图像

    Support 既支持：
        SUPPORT_BASE/部位_1query/train/类别/图像
    也支持：
        SUPPORT_BASE/部位_1query/类别/图像

重要：
    1) SH/命令行只需要给到 QUERY_BASE，不需要手工把 /test 写进每个数据集路径。
    2) 每个部位使用独立 DeepEMD 权重：食管 / 胃 / 十二指肠分别加载。
    3) 同一个部位的模型只加载一次，然后连续评估 3 家医院，避免 9 次重复加载。
    4) support 按 shot-major(round-robin) 排列，保持与 get_sfc() 的 view(shot, way) 约定一致。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats
import torch
import torch.nn as nn
import tqdm

from Models.models.Network import DeepEMD
from Models.utils import *
from Models.dataloader.data_utils import *


# =============================================================================
# 默认配置：只需要优先检查这一区域
# =============================================================================
ROOT = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow"

# 外院 query 的共同根目录：下面是 医院/部位/[test/]类别/图像
DEFAULT_QUERY_BASE = f"/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/data/data_tsy_827_doctor_test_split/indomain_support_76_test"

# 华西 support 的共同根目录：默认按 {部位}_1query/[train/]类别/图像 解析
DEFAULT_SUPPORT_BASE = f"/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/data/data_tsy_827_doctor_test_split"

# 输出：自动保存到 OUTPUT_BASE/医院/部位/
DEFAULT_OUTPUT_BASE = f"/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/tsy/deepemd/test-train"

HOSPITALS = ["华西", "自贡", "苍溪", "南充"]
# HOSPITALS = ["华西"]
ORGANS = ["食管", "胃", "十二指肠"]

# 每个部位使用独立模型；way 作为最大 N-way 上限。
# 实际 N-way = min(配置 way, 当前医院-部位 query/support 共同类别数)。
ORGAN_CONFIG = {
    "食管": {
        "way": 8,
        
        #"model_dir": "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/checkpoint/食管/fcn/食管-5shot-8way/max_acc.pth",
        "model_dir": "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/checkpoint_tsy_new/食管/fcn/食管-5shot-8way/max_acc.pth",
        "support_name": "食管_1query",
    },
    "胃": {
        # "way": 8,
        "way": 7,
        #"model_dir": "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/checkpoint/cifar_fs/胃-5shot-8way/max_acc.pth",
        "model_dir": "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/checkpoint_tsy_new/胃/fcn/胃-5shot-7way/max_acc.pth",
        "support_name": "胃_1query",
    },
    "十二指肠": {
        "way": 7,
        #"model_dir": "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/checkpoint/十二指肠/fcn/十二指肠-5shot-7way/max_acc.pth",
        "model_dir": "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/checkpoint_tsy_new/十二指肠/fcn/十二指肠-5shot-7way/max_acc.pth",
        "support_name": "十二指肠_1query",
    },
}
# ORGAN_CONFIG = {
#     "食管": {
#         "way": 8,
#         "model_dir": "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/deepemd_trained_model/miniimagenet/fcn/max_acc.pth",
#         "support_name": "食管_1query",
#     },
#     "胃": {
#         "way": 8,
#         "model_dir": "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/deepemd_trained_model/miniimagenet/fcn/max_acc.pth",
#         "support_name": "胃_1query",
#     },
#     "十二指肠": {
#         "way": 7,
#         "model_dir": "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD/deepemd_trained_model/miniimagenet/fcn/max_acc.pth",
#         "support_name": "十二指肠_1query",
#     },
# }

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepEMD: 华西 support -> 3 外院 × 3 部位批量 cross-center few-shot evaluation"
    )

    # 路径只给共同根目录；test/train 由代码自动判断。
    parser.add_argument("-query_base", type=str, default=DEFAULT_QUERY_BASE)
    parser.add_argument("-support_base", type=str, default=DEFAULT_SUPPORT_BASE)
    parser.add_argument("-output_base", type=str, default=DEFAULT_OUTPUT_BASE)

    parser.add_argument("-hospitals", nargs="+", default=HOSPITALS)
    parser.add_argument("-organs", nargs="+", default=ORGANS)

    # few-shot protocol
    parser.add_argument("-shot", type=int, default=5)
    parser.add_argument("-query", type=int, default=1, help="每个 GT 类每 episode 的 query 数")
    parser.add_argument("-test_episode", type=int, default=100)
    parser.add_argument("-seed", type=int, default=1)

    # Dataset/DeepEMD settings，保持原脚本默认值。
    parser.add_argument(
        "-dataset", type=str, default="cifar_fs",
        choices=["miniimagenet", "cub", "tieredimagenet", "fc100", "tieredimagenet_yao", "cifar_fs"],
    )
    parser.add_argument("-set", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("-temperature", type=float, default=12.5)
    parser.add_argument("-metric", type=str, default="cosine", choices=["cosine"])
    parser.add_argument("-norm", type=str, default="center", choices=["center"])
    parser.add_argument("-deepemd", type=str, default="fcn", choices=["fcn", "grid", "sampling"])
    parser.add_argument("-feature_pyramid", type=str, default=None)
    parser.add_argument("-num_patch", type=int, default=9)
    parser.add_argument("-patch_list", type=str, default="2,3")
    parser.add_argument("-patch_ratio", type=float, default=2)
    parser.add_argument("-solver", type=str, default="opencv", choices=["opencv"])

    # SFC
    parser.add_argument("-sfc_lr", type=float, default=100)
    parser.add_argument("-sfc_wd", type=float, default=0)
    parser.add_argument("-sfc_update_step", type=float, default=10)
    parser.add_argument("-sfc_bs", type=int, default=4)

    # support 聚合模式：true=使用 SFC(get_sfc)；false=不使用 SFC，改用均值(mean)模式。
    parser.add_argument(
        "-use_sfc",
        type=str,
        default="true",
        choices=["true", "false"],
        help="true=使用 SFC (get_sfc)；false=不使用 SFC，改用 support 特征均值(mean)模式。",
    )

    # GPU
    parser.add_argument("-gpu", default="0")

    # 可选覆盖：一般不需要；若传入则对所有部位统一覆盖 N-way。
    parser.add_argument(
        "-way", type=int, default=None,
        help="默认使用 ORGAN_CONFIG 中每个部位自己的 way；传入后统一覆盖。",
    )

    # 兼容 set_up_datasets / Dataset 所需字段；运行中动态更新。
    parser.add_argument("-data_dir", type=str, default=DEFAULT_QUERY_BASE)
    parser.add_argument("-model_dir", type=str, default="")
    parser.add_argument("-save_dir", type=str, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("-support_dir", type=str, default=DEFAULT_SUPPORT_BASE)
    parser.add_argument("-support_set", type=str, default="")

    args = parser.parse_args()
    args.use_sfc = args.use_sfc.lower() == "true"
    if args.feature_pyramid is not None:
        args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(",")]
    args.patch_list = [int(x) for x in args.patch_list.split(",")]
    return args


# =============================================================================
# 路径与数据辅助
# =============================================================================
def list_class_images(root: str | Path) -> Dict[str, List[str]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")

    out: Dict[str, List[str]] = {}
    for cdir in sorted(root.iterdir(), key=lambda p: p.name):
        if not cdir.is_dir() or cdir.name.startswith("."):
            continue
        imgs = sorted(
            str(p) for p in cdir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        if imgs:
            out[cdir.name] = imgs
    return out


def _looks_like_class_root(root: Path) -> bool:
    """root 下是否看起来已经直接是 类别/图像。"""
    if not root.is_dir():
        return False
    for cdir in root.iterdir():
        if not cdir.is_dir() or cdir.name.startswith("."):
            continue
        for p in cdir.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                return True
    return False


def resolve_split_root(base: str | Path, preferred_split: str) -> Path:
    """
    自动兼容：
      base/test/类别/图像  或 base/类别/图像
      base/train/类别/图像 或 base/类别/图像

    优先使用 preferred_split；如果 base 本身已经是类别根目录，则直接使用 base。
    """
    base = Path(base)
    if not base.is_dir():
        raise FileNotFoundError(f"基础目录不存在: {base}")

    # 如果调用方本身已经传到了 .../test 或 .../train，直接接受。
    if base.name == preferred_split and _looks_like_class_root(base):
        return base

    split_root = base / preferred_split
    if split_root.is_dir() and _looks_like_class_root(split_root):
        return split_root

    if _looks_like_class_root(base):
        return base

    raise RuntimeError(
        f"无法解析类别目录: base={base}, preferred_split={preferred_split}\n"
        f"既没有找到 {split_root}/类别/图像，也没有发现 {base}/类别/图像。"
    )


def resolve_query_paths(query_base: str, hospital: str, organ: str, split: str) -> Tuple[Path, Path]:
    """
    返回：
      dataset_base: 交给 Dataset(set, args) 的根目录，通常是 .../医院/部位
      class_root:    手工 episode 扫描实际使用的 .../[test]/类别 根目录
    """
    organ_base = Path(query_base) / hospital / organ
    class_root = resolve_split_root(organ_base, split)

    # Dataset(args.set, args) 一般期望 data_dir 指向 split 的上一级。
    # 若 class_root == organ_base/test，则 probe 用 organ_base，避免 test/test。
    dataset_base = organ_base if class_root.parent == organ_base else class_root
    return dataset_base, class_root


def resolve_support_root(support_base: str, organ: str, support_name: str) -> Path:
    base = Path(support_base) / support_name
    return resolve_split_root(base, "train")


def _sample(cands: List[str], n: int) -> List[str]:
    if not cands:
        return []
    return random.sample(cands, n) if len(cands) >= n else random.choices(cands, k=n)


def _safe_norm(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except Exception:
        return os.path.abspath(path)


# =============================================================================
# 单个医院-部位评估
# =============================================================================
def evaluate_one_dataset(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    num_gpu: int,
    Dataset,
    hospital: str,
    organ: str,
    way_requested: int,
    query_dataset_base: Path,
    query_root: Path,
    support_root: Path,
    save_dir: Path,
    dataset_seed: int,
) -> dict:
    random.seed(dataset_seed)
    np.random.seed(dataset_seed)
    torch.manual_seed(dataset_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(dataset_seed)

    save_dir.mkdir(parents=True, exist_ok=True)

    query_files = list_class_images(query_root)
    support_files = list_class_images(support_root)

    # 用 split 上一级作为 Dataset 的 data_dir，仅借其 eval transform。
    old_data_dir = args.data_dir
    args.data_dir = str(query_dataset_base)
    try:
        probe_set = Dataset(args.set, args)
        img_transform = getattr(probe_set, "transform", None)
    finally:
        args.data_dir = old_data_dir

    if img_transform is None:
        raise RuntimeError(
            "无法从 Dataset 取到 .transform。请确认 Dataset 类暴露 self.transform，"
            "这样才能保证与 DeepEMD 训练/原评估的 resize/normalize 完全一致。"
        )

    # Reduced candidate space:
    # candidate = external query classes ∩ Huaxi support classes.
    # Huaxi-only support classes are NOT retained as distractors.
    query_classes = [
        c for c in sorted(query_files.keys())
        if c in support_files and query_files[c] and support_files[c]
    ]
    candidate_pool = list(query_classes)

    if not candidate_pool:
        raise RuntimeError(
            f"query 与 support 没有同名交集类别！\n"
            f"query={query_root}\nsupport={support_root}\n"
            f"query_classes={sorted(query_files.keys())}\n"
            f"support_classes={sorted(support_files.keys())}"
        )

    way_eff = min(int(way_requested), len(candidate_pool))
    if way_eff < 2:
        raise RuntimeError(
            f"有效 way 太小: requested={way_requested}, shared_classes={len(candidate_pool)}"
        )

    # ------------------------------------------------------------------
    # CRITICAL: DeepEMD/SFC reads self.args.way at runtime.
    # After reducing the candidate set, synchronize the effective task way
    # into BOTH the shared argparse object and the loaded model.
    # The checkpoint may have been trained as 8-way/7-way; DeepEMD's meta
    # matcher itself supports a dynamic episode way, but its runtime args
    # must match the current support tensor shape exactly.
    # ------------------------------------------------------------------
    args.way = int(way_eff)
    args.use_sfc = bool(args.use_sfc)
    core_model = model.module if isinstance(model, nn.DataParallel) else model

    if hasattr(core_model, "args"):
        core_model.args.way = int(way_eff)
        core_model.args.shot = int(args.shot)
        core_model.args.use_sfc = bool(args.use_sfc)

    # Compatibility with forks that cache way/shot as direct attributes.
    if hasattr(core_model, "way"):
        core_model.way = int(way_eff)
    if hasattr(core_model, "shot"):
        core_model.shot = int(args.shot)

    print(
        f"[TASK SHAPE] {hospital}-{organ}: "
        f"runtime way={way_eff}, shot={args.shot}"
    )

    print("\n" + "=" * 100)
    print(f"[DATASET] {hospital}-{organ}")
    print(f"query_dataset_base = {query_dataset_base}")
    print(f"query_root         = {query_root}")
    print(f"support_root       = {support_root}")
    print(f"model              = {args.model_dir}")
    print(f"query classes      = {len(query_files)}: {sorted(query_files.keys())}")
    print(f"support classes    = {len(support_files)}: {sorted(support_files.keys())}")
    print(f"shared candidates  = {len(candidate_pool)}: {candidate_pool}")
    print(f"protocol           = {way_eff}-way {args.shot}-shot, query/class={args.query}, episodes={args.test_episode}")
    print("=" * 100)

    if way_requested > len(candidate_pool):
        print(
            f"[INFO] requested way={way_requested}, shared candidates={len(candidate_pool)}; "
            f"automatically reduce to {way_eff}-way"
        )
    elif way_eff < len(candidate_pool):
        print(
            f"[INFO] shared candidates={len(candidate_pool)} > way={way_eff}; "
            f"each episode samples {way_eff} shared classes"
        )

    global_class_to_idx = {c: i for i, c in enumerate(candidate_pool)}
    n_classes = len(candidate_pool)

    def _load(path: str):
        return img_transform(Image.open(path).convert("RGB"))

    def build_episode():
        # Candidate pool has already been reduced to query/support shared classes.
        # Therefore every episode class is a valid GT/query class.
        # No Huaxi-only support distractor is added.
        if len(candidate_pool) > way_eff:
            gt_in_episode = random.sample(candidate_pool, way_eff)
        else:
            gt_in_episode = list(candidate_pool)

        episode_classes = list(gt_in_episode)
        random.shuffle(episode_classes)

        # 先抽 query，再在同一路径池的极端情况下从 support 中排除这些 query 路径。
        query_paths_by_class: Dict[str, List[str]] = {}
        for c in gt_in_episode:
            query_paths_by_class[c] = _sample(query_files[c], args.query)

        per_class_support: Dict[str, List[str]] = {}
        for c in episode_classes:
            support_candidates = list(support_files[c])
            if c in query_paths_by_class:
                query_norm = {_safe_norm(p) for p in query_paths_by_class[c]}
                filtered = [p for p in support_candidates if _safe_norm(p) not in query_norm]
                # 无论是否 cross-center，只要路径发生重叠，就严格排除 query。
                # 数量不足 shot 时允许从剩余 support 中有放回采样，但绝不把 query 放回 support。
                if len(filtered) < len(support_candidates):
                    if not filtered:
                        raise RuntimeError(
                            f"排除 query 后 class={c} 没有可用 support；无法保证 support/query 不重叠。"
                        )
                    support_candidates = filtered
            per_class_support[c] = _sample(support_candidates, args.shot)
            if len(per_class_support[c]) != args.shot:
                raise RuntimeError(f"support 抽样失败: class={c}, need={args.shot}")

        # shot-major，匹配 get_sfc() 内部的 view(shot, way)。
        shot_imgs = []
        for s in range(args.shot):
            for c in episode_classes:
                shot_imgs.append(_load(per_class_support[c][s]))

        query_imgs = []
        query_labels = []
        for ci, c in enumerate(episode_classes):
            if c not in gt_in_episode:
                continue
            for p in query_paths_by_class[c]:
                query_imgs.append(_load(p))
                query_labels.append(ci)

        data = torch.stack(shot_imgs + query_imgs, dim=0)
        labels = torch.tensor(query_labels, dtype=torch.long)
        return data, labels, episode_classes

    support_aggregation_mode = "sfc" if args.use_sfc else "mean"

    confusion_matrix_accum = np.zeros((n_classes, n_classes), dtype=np.int64)
    intra_distances: List[float] = []
    inter_distances: List[float] = []
    latency_per_episode: List[float] = []
    test_acc_record = np.zeros((args.test_episode,), dtype=np.float64)
    ave_acc = Averager()

    tqdm_gen = tqdm.tqdm(range(1, args.test_episode + 1), desc=f"{hospital}-{organ}")
    with torch.no_grad():
        for i in tqdm_gen:
            data, label_local, episode_classes = build_episode()
            data = data.cuda(non_blocking=True)
            label_local = label_local.cuda(non_blocking=True)
            k = len(episode_classes) * args.shot

            core_model = model.module if isinstance(model, nn.DataParallel) else model
            core_model.mode = "encoder"
            data = model(data)
            data_shot, data_query = data[:k], data[k:]

            # Fail early with a readable error instead of a CUDA index assertion.
            runtime_way = int(len(episode_classes))
            expected_support = runtime_way * int(args.shot)

            core_model = model.module if isinstance(model, nn.DataParallel) else model
            model_way = int(getattr(getattr(core_model, "args", args), "way", args.way))

            if int(data_shot.shape[0]) != expected_support:
                raise RuntimeError(
                    f"SFC support count mismatch before get_sfc: "
                    f"got={int(data_shot.shape[0])}, "
                    f"expected={expected_support} "
                    f"({runtime_way}-way x {args.shot}-shot)"
                )
            if model_way != runtime_way:
                raise RuntimeError(
                    f"DeepEMD runtime way mismatch before get_sfc: "
                    f"model.args.way={model_way}, episode_way={runtime_way}"
                )

            if args.shot > 1:
                if args.use_sfc:
                    data_shot = core_model.get_sfc(data_shot)
                    support_aggregation_mode = "sfc"
                else:
                    data_shot = data_shot.mean(0)
                    support_aggregation_mode = "mean"

            t0 = time.perf_counter()
            core_model.mode = "meta"
            # Re-assert task shape before meta matching.
            if hasattr(core_model, "args"):
                core_model.args.way = int(len(episode_classes))
                core_model.args.shot = int(args.shot)

            logits = model((data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1), data_query))
            torch.cuda.synchronize()
            latency_per_episode.append(time.perf_counter() - t0)

            acc = count_acc(logits, label_local) * 100
            ave_acc.add(acc)
            test_acc_record[i - 1] = acc

            preds_local = logits.argmax(dim=1).cpu().numpy()
            true_local = label_local.cpu().numpy()
            for tl, pl in zip(true_local, preds_local):
                gt_g = global_class_to_idx[episode_classes[int(tl)]]
                pred_g = global_class_to_idx[episode_classes[int(pl)]]
                confusion_matrix_accum[gt_g, pred_g] += 1

            dist_matrix = (-logits / args.temperature).cpu().numpy()
            for q_idx in range(dist_matrix.shape[0]):
                true_cls = int(true_local[q_idx])
                for cls_idx in range(dist_matrix.shape[1]):
                    d = float(dist_matrix[q_idx, cls_idx])
                    (intra_distances if cls_idx == true_cls else inter_distances).append(d)

            m_now, pm_now = compute_confidence_interval(test_acc_record[:i])
            tqdm_gen.set_description(
                f"{hospital}-{organ} ep={i} this={acc:.2f} avg={m_now:.4f}+{pm_now:.4f}"
            )

    m, pm = compute_confidence_interval(test_acc_record)

    # ---------------- confusion matrix ----------------
    cm_norm = confusion_matrix_accum.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = cm_norm / (row_sums + 1e-8)

    per_class_acc = {}
    for cls_idx, cls_name in enumerate(candidate_pool):
        if row_sums[cls_idx, 0] > 0:
            per_class_acc[cls_name] = float(cm_norm[cls_idx, cls_idx] * 100.0)

    fig, ax = plt.subplots(figsize=(max(6, n_classes), max(5, n_classes * 0.9)))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(candidate_pool, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(candidate_pool, fontsize=7)
    ax.set_title(f"{hospital}-{organ}: Normalised Confusion Matrix")
    for r in range(n_classes):
        for c in range(n_classes):
            ax.text(
                c, r, f"{cm_norm[r, c]:.2f}", ha="center", va="center",
                color="white" if cm_norm[r, c] > 0.6 else "black", fontsize=7,
            )
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # ---------------- distance quality ----------------
    intra = np.asarray(intra_distances, dtype=np.float64)
    inter = np.asarray(inter_distances, dtype=np.float64)
    mu_intra, std_intra = float(intra.mean()), float(intra.std())
    mu_inter, std_inter = float(inter.mean()), float(inter.std())
    fisher_ratio = float(
        (mu_inter - mu_intra) ** 2 / (std_intra ** 2 + std_inter ** 2 + 1e-8)
    )

    all_d = np.concatenate([intra, inter])
    if float(all_d.max()) > float(all_d.min()):
        bins = np.linspace(all_d.min(), all_d.max(), 60)
        p_intra, _ = np.histogram(intra, bins=bins, density=True)
        p_inter, _ = np.histogram(inter, bins=bins, density=True)
        p_intra = p_intra + 1e-8
        p_inter = p_inter + 1e-8
        p_intra /= p_intra.sum()
        p_inter /= p_inter.sum()
        kl_div = float(stats.entropy(p_inter, p_intra))
    else:
        kl_div = 0.0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(intra, bins=50, alpha=0.6, label="Intra-class", density=True)
    ax.hist(inter, bins=50, alpha=0.6, label="Inter-class", density=True)
    ax.axvline(mu_intra, linestyle="--", linewidth=1.5, label=f"mu_intra={mu_intra:.3f}")
    ax.axvline(mu_inter, linestyle="--", linewidth=1.5, label=f"mu_inter={mu_inter:.3f}")
    ax.set_xlabel("Distance (-logit / temperature)")
    ax.set_ylabel("Density")
    ax.set_title(f"{hospital}-{organ}: Fisher Ratio={fisher_ratio:.3f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "distance_distribution.png", dpi=150)
    plt.close()

    latencies = np.asarray(latency_per_episode, dtype=np.float64)
    mean_lat = float(latencies.mean())
    std_lat = float(latencies.std())
    median_lat = float(np.median(latencies))

    result = {
        "hospital": hospital,
        "organ": organ,
        "model_dir": str(args.model_dir),
        "support_aggregation_mode": support_aggregation_mode,
        "query_dataset_base": str(query_dataset_base),
        "query_root": str(query_root),
        "support_root": str(support_root),
        "way_requested": int(way_requested),
        "way_effective": int(way_eff),
        "shot": int(args.shot),
        "query_per_gt_class": int(args.query),
        "episodes": int(args.test_episode),
        "seed": int(dataset_seed),
        "query_class_count": len(query_files),
        "support_class_count": len(support_files),
        "shared_gt_class_count": len(query_classes),
        "candidate_class_count": len(candidate_pool),
        "candidate_space_rule": "query_support_intersection",
        "deepemd_runtime_way_synced": True,
        "query_classes": sorted(query_files.keys()),
        "support_classes": sorted(support_files.keys()),
        "candidate_classes": candidate_pool,
        "shared_gt_classes": query_classes,
        "accuracy_percent": float(m),
        "ci95_percent": float(pm),
        "averager_accuracy_percent": float(ave_acc.item()),
        "fisher_ratio": fisher_ratio,
        "kl_divergence_inter_intra": kl_div,
        "latency_mean_ms": mean_lat * 1000.0,
        "latency_std_ms": std_lat * 1000.0,
        "latency_median_ms": median_lat * 1000.0,
        "per_class_accuracy_percent": per_class_acc,
        "save_dir": str(save_dir),
    }

    (save_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[RESULT] {hospital}-{organ}: Acc={m:.4f} +/- {pm:.4f}% | "
        f"Fisher={fisher_ratio:.4f} | KL={kl_div:.4f} | Lat={mean_lat * 1000.0:.2f} ms"
    )
    return result


# =============================================================================
# 主流程：每个部位加载一次模型 -> 连续跑 3 家医院
# =============================================================================
def main() -> None:
    args = parse_args()
    print(vars(args))

    set_seed(args.seed)
    num_gpu = set_gpu(args)
    Dataset = set_up_datasets(args)

    # 检查输入配置。
    for organ in args.organs:
        if organ not in ORGAN_CONFIG:
            raise KeyError(f"未知部位 {organ!r}; 可选={list(ORGAN_CONFIG.keys())}")
        model_path = Path(ORGAN_CONFIG[organ]["model_dir"])
        if not model_path.is_file():
            raise FileNotFoundError(f"{organ} 模型不存在: {model_path}")

    Path(args.output_base).mkdir(parents=True, exist_ok=True)
    all_results: List[dict] = []
    # 保存 CLI 的统一 way 覆盖值；不能直接复用循环中动态修改后的 args.way。
    global_way_override = args.way

    print("\n" + "#" * 110)
    print("[BATCH PROTOCOL] Huaxi support -> external-hospital query")
    print(f"Support aggregation mode = {'SFC (get_sfc)' if args.use_sfc else 'Mean (no SFC)'}")
    print(f"Hospitals = {args.hospitals}")
    print(f"Organs    = {args.organs}")
    print(f"QueryBase = {args.query_base}")
    print(f"SupportBase = {args.support_base}")
    print("Query path rule   : prefer 医院/部位/test/类别, fallback 医院/部位/类别")
    print("Support path rule : prefer 部位_1query/train/类别, fallback 部位_1query/类别")
    print("#" * 110 + "\n")

    for organ_idx, organ in enumerate(args.organs):
        cfg = ORGAN_CONFIG[organ]
        way_requested = int(global_way_override) if global_way_override is not None else int(cfg["way"])
        args.way = way_requested
        args.model_dir = str(cfg["model_dir"])

        support_root = resolve_support_root(
            args.support_base, organ=organ, support_name=str(cfg["support_name"])
        )

        print("\n" + "=" * 110)
        print(f"[LOAD MODEL] organ={organ}, way={way_requested}")
        print(f"checkpoint={args.model_dir}")
        print(f"support_root={support_root}")
        print("=" * 110)

        model = DeepEMD(args)
        model = load_model(model, args.model_dir)
        model = nn.DataParallel(model, list(range(num_gpu)))
        model = model.cuda()
        model.eval()

        try:
            for hospital_idx, hospital in enumerate(args.hospitals):
                query_dataset_base, query_root = resolve_query_paths(
                    args.query_base, hospital=hospital, organ=organ, split=args.set
                )
                save_dir = Path(args.output_base) / hospital / organ
                dataset_seed = int(args.seed) + organ_idx * 10000 + hospital_idx * 1009

                one = evaluate_one_dataset(
                    args=args,
                    model=model,
                    num_gpu=num_gpu,
                    Dataset=Dataset,
                    hospital=hospital,
                    organ=organ,
                    way_requested=way_requested,
                    query_dataset_base=query_dataset_base,
                    query_root=query_root,
                    support_root=support_root,
                    save_dir=save_dir,
                    dataset_seed=dataset_seed,
                )
                all_results.append(one)
        finally:
            del model
            torch.cuda.empty_cache()

    # ---------------- batch summary ----------------
    summary_json = Path(args.output_base) / "summary_3hospital_3organ.json"
    summary_csv = Path(args.output_base) / "summary_3hospital_3organ.csv"

    macro_acc = float(np.mean([r["accuracy_percent"] for r in all_results])) if all_results else float("nan")
    summary = {
        "protocol": "Huaxi-support -> external-query; candidates=query/support intersection",
        "hospitals": list(args.hospitals),
        "organs": list(args.organs),
        "dataset_count": len(all_results),
        "shot": int(args.shot),
        "query_per_gt_class": int(args.query),
        "episodes_per_dataset": int(args.test_episode),
        "macro_accuracy_percent": macro_acc,
        "datasets": all_results,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_fields = [
        "hospital", "organ", "way_effective", "shot", "episodes",
        "shared_gt_class_count", "accuracy_percent", "ci95_percent",
        "fisher_ratio", "kl_divergence_inter_intra", "latency_mean_ms",
        "query_root", "support_root", "model_dir",
    ]
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k) for k in csv_fields})

    print("\n" + "#" * 110)
    print("[FINAL SUMMARY]")
    for r in all_results:
        print(
            f"{r['hospital']}-{r['organ']:<6s} | {r['way_effective']}-way | "
            f"Acc={r['accuracy_percent']:.2f} +/- {r['ci95_percent']:.2f}% | "
            f"shared={r['shared_gt_class_count']}"
        )
    print(f"MACRO ACC = {macro_acc:.4f}%")
    print(f"JSON -> {summary_json}")
    print(f"CSV  -> {summary_csv}")
    print("#" * 110)


if __name__ == "__main__":
    main()
