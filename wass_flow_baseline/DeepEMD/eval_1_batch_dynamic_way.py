#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepEMD test-test batch evaluation: hospital x organ, dynamic way.

Key protocol
------------
1. Evaluate 3 hospitals x 3 organs in one run.
2. Test-test: support and query are sampled from the SAME hospital-organ test pool.
   Query images are excluded from support; undersized support pools are sampled with replacement.
3. way is NOT hard-coded by organ. For every hospital-organ dataset:
       way = actual number of query classes in that Dataset.
4. Support adaptation is switchable:
       default: SFC (DeepEMD get_sfc)
       --no_sfc: class-wise mean of K-shot support features
   The no-SFC mean preserves one prototype per class; it never averages all ways together.
5. Keep the original extra metrics:
       accuracy + 95% CI, confusion, Fisher ratio, KL divergence, latency.

Examples
--------
# Default: use SFC
python eval_1_batch_dynamic_way.py -gpu 0

# Disable SFC: use class-wise K-shot mean
python eval_1_batch_dynamic_way.py -gpu 0 --no_sfc
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
from scipy import stats
from torch.utils.data import DataLoader

from Models.models.Network import DeepEMD
from Models.utils import *
from Models.dataloader.data_utils import *


# =============================================================================
# Default configuration
# =============================================================================

ROOT = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow"
DEEPEMD_ROOT = "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD"

# Directory layout:
# DATA_BASE / hospital / organ / [test/] / class / image
DEFAULT_DATA_BASE = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/data/data_tsy_827_doctor_test_split/indomain_support_76_test"
DEFAULT_OUTPUT_BASE = "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/tsy/deepemd/test-test"

HOSPITALS = ["自贡", "苍溪", "南充"]
# HOSPITALS = ["华西"]
ORGANS = ["食管", "胃", "十二指肠"]


# Checkpoint depends on organ; runtime/model way follows the actual query classes for each dataset.
ORGAN_MODEL_DIR = {
    "食管": f"{DEEPEMD_ROOT}/checkpoint_tsy_new/食管/fcn/食管-5shot-8way/max_acc.pth",
    "胃": f"{DEEPEMD_ROOT}/checkpoint_tsy_new/胃/fcn/胃-5shot-7way/max_acc.pth",
    "十二指肠": f"{DEEPEMD_ROOT}/checkpoint_tsy_new/十二指肠/fcn/十二指肠-5shot-7way/max_acc.pth",
}
# ORGAN_MODEL_DIR = {
#     "食管": f"{DEEPEMD_ROOT}/deepemd_trained_model/miniimagenet/fcn/max_acc.pth",
#     "胃": f"{DEEPEMD_ROOT}/deepemd_trained_model/miniimagenet/fcn/max_acc.pth",
#     "十二指肠": f"{DEEPEMD_ROOT}/deepemd_trained_model/miniimagenet/fcn/max_acc.pth",
# }
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_USE_SFC = True


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DeepEMD test-test batch evaluation: 3 hospitals x 3 organs; "
            "runtime way = actual query-class count"
        )
    )

    parser.add_argument("-data_base", type=str, default=DEFAULT_DATA_BASE)
    parser.add_argument("-output_base", type=str, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("-hospitals", nargs="+", default=HOSPITALS)
    parser.add_argument("-organs", nargs="+", default=ORGANS)

    # Few-shot protocol.
    # way is intentionally not exposed as the normal control here: every dataset
    # uses all of its actual query classes.
    parser.add_argument("-shot", type=int, default=5)
    parser.add_argument("-query", type=int, default=1)
    parser.add_argument("-test_episode", type=int, default=100)
    parser.add_argument("-seed", type=int, default=1)

    # DeepEMD settings, aligned to the supplied eval_1.py.
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
    parser.add_argument("-solver", type=str, default="opencv", choices=["opencv", "qpth"])

    # SFC settings from the supplied file.
    parser.add_argument("-sfc_lr", type=float, default=100)
    parser.add_argument("-sfc_wd", type=float, default=0)
    parser.add_argument("-sfc_update_step", type=float, default=100)
    parser.add_argument("-sfc_bs", type=int, default=4)

    # Explicit SFC switch.
    sfc_group = parser.add_mutually_exclusive_group()
    sfc_group.add_argument(
        "--use_sfc",
        dest="use_sfc",
        action="store_true",
        help="Use DeepEMD get_sfc() for K-shot support adaptation (default).",
    )
    sfc_group.add_argument(
        "--no_sfc",
        dest="use_sfc",
        action="store_false",
        help="Disable SFC and use class-wise K-shot feature mean instead.",
    )
    parser.set_defaults(use_sfc=DEFAULT_USE_SFC)

    parser.add_argument("-gpu", default="0")
    parser.add_argument("-num_workers", type=int, default=8)

    # Compatibility fields required by the original DeepEMD utilities.
    # These are overwritten for every hospital-organ task.
    parser.add_argument("-way", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("-data_dir", type=str, default=DEFAULT_DATA_BASE, help=argparse.SUPPRESS)
    parser.add_argument("-model_dir", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("-save_dir", type=str, default=DEFAULT_OUTPUT_BASE, help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.feature_pyramid is not None:
        args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(",")]
    args.patch_list = [int(x) for x in args.patch_list.split(",")]
    return args


# =============================================================================
# Path / dataset helpers
# =============================================================================

def _looks_like_class_root(root: Path) -> bool:
    if not root.is_dir():
        return False
    for cdir in root.iterdir():
        if not cdir.is_dir() or cdir.name.startswith("."):
            continue
        for p in cdir.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                return True
    return False


def resolve_dataset_dir(data_base: str, hospital: str, organ: str, split: str) -> Path:
    """Resolve the actual class root used for episode inspection.

    Returns either:
        .../hospital/organ/test   (standard split layout)
    or:
        .../hospital/organ        (direct class-root fallback)

    NOTE: this returned class_root is NOT blindly passed to DeepEMD Dataset.
    Dataset(set, args) appends /{set} internally, so evaluate_one_dataset()
    passes the split parent as args.data_dir when class_root ends with /test.
    """
    base = Path(data_base) / hospital / organ
    if not base.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {base}")

    split_root = base / split
    if split_root.is_dir() and _looks_like_class_root(split_root):
        return split_root
    if _looks_like_class_root(base):
        return base

    raise RuntimeError(
        f"无法解析类别目录: {base}\n"
        f"既没有 {split_root}/类别/图像，也没有 {base}/类别/图像"
    )


def list_class_names(class_root: Path) -> List[str]:
    names = []
    for d in sorted(class_root.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if any(p.is_file() and p.suffix.lower() in IMAGE_EXTS for p in d.iterdir()):
            names.append(d.name)
    return names


def label_statistics(labels) -> Tuple[List[int], Dict[int, int]]:
    arr = np.asarray(labels, dtype=np.int64)
    unique, counts = np.unique(arr, return_counts=True)
    ids = [int(x) for x in unique.tolist()]
    count_map = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
    return ids, count_map


class TestTestSupportReplacementSampler:
    """Episode sampler for test-test evaluation with leakage-safe support replacement.

    For every class in every episode:
      1) query samples are drawn WITHOUT replacement;
      2) all chosen query indices are removed from the support pool;
      3) support samples are drawn WITHOUT replacement when enough remain;
      4) otherwise support samples are drawn WITH replacement from the remaining
         non-query images until exactly ``shot`` supports are obtained.

    The yielded batch keeps DeepEMD's expected shot-major / round-robin order:

        support shot 0 for all classes, ..., support shot K-1 for all classes,
        query 0 for all classes, ..., query Q-1 for all classes.

    Therefore the downstream split
        data[:way*shot], data[way*shot:]
    and ``reshape(shot, way, ...)`` semantics remain unchanged.

    To prevent query-support identity leakage, each class must still contain at
    least ``query + 1`` distinct images.  For the usual query=1 case this lowers
    the requirement from shot+1 images/class to only 2 images/class.
    """

    def __init__(
        self,
        labels,
        n_batch: int,
        n_cls: int,
        n_shot: int,
        n_query: int,
        seed: int,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.n_batch = int(n_batch)
        self.n_cls = int(n_cls)
        self.n_shot = int(n_shot)
        self.n_query = int(n_query)
        self.seed = int(seed)

        self.class_ids = sorted(int(x) for x in np.unique(self.labels).tolist())
        if len(self.class_ids) != self.n_cls:
            raise ValueError(
                f"n_cls={self.n_cls} but labels contain {len(self.class_ids)} classes: {self.class_ids}"
            )
        if self.n_shot < 1 or self.n_query < 1:
            raise ValueError("n_shot and n_query must both be >= 1")

        self.class_indices: Dict[int, np.ndarray] = {
            cid: np.flatnonzero(self.labels == cid).astype(np.int64)
            for cid in self.class_ids
        }
        too_small = {
            cid: int(idx.size)
            for cid, idx in self.class_indices.items()
            if int(idx.size) < self.n_query + 1
        }
        if too_small:
            raise RuntimeError(
                "Leakage-safe replacement sampling requires at least query+1 distinct "
                f"images per class (query={self.n_query}); insufficient={too_small}. "
                "A singleton class cannot provide a support image distinct from its query."
            )

    def __len__(self) -> int:
        return self.n_batch

    def __iter__(self):
        rng = np.random.RandomState(self.seed)

        for _ in range(self.n_batch):
            support_by_class: Dict[int, np.ndarray] = {}
            query_by_class: Dict[int, np.ndarray] = {}

            for cid in self.class_ids:
                idx = self.class_indices[cid]

                # Query is always sampled without replacement.
                query_idx = rng.choice(idx, size=self.n_query, replace=False).astype(np.int64)
                query_by_class[cid] = query_idx

                # Strictly exclude every query image from support.
                query_set = set(int(x) for x in query_idx.tolist())
                support_pool = np.asarray(
                    [int(x) for x in idx.tolist() if int(x) not in query_set],
                    dtype=np.int64,
                )
                if support_pool.size == 0:
                    raise RuntimeError(
                        f"class={cid}: no non-query image remains for support after selecting "
                        f"{self.n_query} query image(s)"
                    )

                # Preserve the old no-replacement behavior whenever possible.
                # Only undersized classes use replacement to reach K-shot.
                replace_support = bool(support_pool.size < self.n_shot)
                support_idx = rng.choice(
                    support_pool,
                    size=self.n_shot,
                    replace=replace_support,
                ).astype(np.int64)
                support_by_class[cid] = support_idx

            # DeepEMD expects shot-major support ordering.
            batch: List[int] = []
            for s in range(self.n_shot):
                for cid in self.class_ids:
                    batch.append(int(support_by_class[cid][s]))

            # Follow with query-major ordering, matching label = arange(way).repeat(query).
            for q in range(self.n_query):
                for cid in self.class_ids:
                    batch.append(int(query_by_class[cid][q]))

            yield batch


def sync_runtime_way(args: argparse.Namespace, model: nn.Module, way_eff: int) -> None:
    """Synchronize dynamic way into both argparse and DeepEMD runtime state."""
    args.way = int(way_eff)
    core = model.module if isinstance(model, nn.DataParallel) else model

    if hasattr(core, "args"):
        core.args.way = int(way_eff)
        core.args.shot = int(args.shot)
        core.args.query = int(args.query)
    if hasattr(core, "way"):
        core.way = int(way_eff)
    if hasattr(core, "shot"):
        core.shot = int(args.shot)


def classwise_shot_mean(data_shot: torch.Tensor, way: int, shot: int) -> torch.Tensor:
    """Convert shot-major [shot*way,C,H,W] support to [way,C,H,W] means.

    The episode sampler arranges support in round-robin / shot-major order.
    Therefore reshape(shot, way, ...) groups the same class along dim=0.
    This is the correct no-SFC fallback; data_shot.mean(0) would collapse all classes.
    """
    if data_shot.ndim != 4:
        raise RuntimeError(f"Expected support feature [shot*way,C,H,W], got {tuple(data_shot.shape)}")
    expected = int(way) * int(shot)
    if data_shot.shape[0] != expected:
        raise RuntimeError(
            f"Support count mismatch before class-wise mean: got={data_shot.shape[0]}, "
            f"expected=way*shot={way}*{shot}={expected}"
        )
    c, h, w = data_shot.shape[1:]
    return data_shot.reshape(shot, way, c, h, w).mean(dim=0).contiguous()


# =============================================================================
# Metric helpers
# =============================================================================

def safe_ci(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    return compute_confidence_interval(values)


def fisher_and_kl(intra: List[float], inter: List[float]) -> Tuple[float, float, float, float, float, float]:
    intra_np = np.asarray(intra, dtype=np.float64)
    inter_np = np.asarray(inter, dtype=np.float64)
    if intra_np.size == 0 or inter_np.size == 0:
        return 0.0, 0.0, float("nan"), float("nan"), float("nan"), float("nan")

    mu_intra, std_intra = float(intra_np.mean()), float(intra_np.std())
    mu_inter, std_inter = float(inter_np.mean()), float(inter_np.std())
    fisher = (mu_inter - mu_intra) ** 2 / (std_intra ** 2 + std_inter ** 2 + 1e-8)

    all_d = np.concatenate([intra_np, inter_np])
    if float(all_d.max()) <= float(all_d.min()):
        kl_div = 0.0
    else:
        bins = np.linspace(all_d.min(), all_d.max(), 60)
        p_intra, _ = np.histogram(intra_np, bins=bins, density=True)
        p_inter, _ = np.histogram(inter_np, bins=bins, density=True)
        p_intra = p_intra + 1e-8
        p_inter = p_inter + 1e-8
        p_intra /= p_intra.sum()
        p_inter /= p_inter.sum()
        kl_div = float(stats.entropy(p_inter, p_intra))

    return float(fisher), float(kl_div), mu_intra, std_intra, mu_inter, std_inter


def save_plots(
    *,
    save_dir: Path,
    confusion_matrix_accum: np.ndarray,
    intra_distances: List[float],
    inter_distances: List[float],
    latency_per_episode: List[float],
    test_acc_record: np.ndarray,
    fisher_ratio: float,
    accuracy_mean: float,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    way = confusion_matrix_accum.shape[0]

    # Confusion matrix.
    cm_norm = confusion_matrix_accum.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = cm_norm / (row_sums + 1e-8)

    fig, ax = plt.subplots(figsize=(max(6, way * 0.65), max(5, way * 0.60)))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalised Confusion Matrix")
    for r in range(way):
        for c in range(way):
            ax.text(
                c, r, f"{cm_norm[r, c]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[r, c] > 0.6 else "black",
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Distance distributions.
    intra = np.asarray(intra_distances, dtype=np.float64)
    inter = np.asarray(inter_distances, dtype=np.float64)
    if intra.size and inter.size:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(intra, bins=50, alpha=0.6, label="Intra-class", density=True)
        ax.hist(inter, bins=50, alpha=0.6, label="Inter-class", density=True)
        ax.axvline(intra.mean(), linestyle="--", linewidth=1.5, label=f"mu_intra={intra.mean():.3f}")
        ax.axvline(inter.mean(), linestyle="--", linewidth=1.5, label=f"mu_inter={inter.mean():.3f}")
        ax.set_xlabel("Distance (-logit / temperature)")
        ax.set_ylabel("Density")
        ax.set_title(f"Distance Distribution | Fisher Ratio = {fisher_ratio:.3f}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_dir / "distance_distribution.png", dpi=150)
        plt.close()

    # Latency / accuracy.
    latencies = np.asarray(latency_per_episode, dtype=np.float64)
    if latencies.size:
        window = min(50, len(test_acc_record))
        kernel = np.ones(window) / window
        rolled_acc = np.convolve(test_acc_record, kernel, mode="valid")
        rolled_lat = np.convolve(latencies * 1000, kernel, mode="valid")
        mean_lat = float(latencies.mean())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(latencies * 1000, alpha=0.6, linewidth=0.8)
        axes[0].axhline(mean_lat * 1000, linestyle="--", label=f"Mean = {mean_lat*1000:.1f} ms")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Latency (ms)")
        axes[0].set_title("Per-episode Matching Latency")
        axes[0].legend()

        axes[1].scatter(rolled_lat, rolled_acc, s=8, alpha=0.6)
        axes[1].set_xlabel("Avg Latency (ms, rolling window)")
        axes[1].set_ylabel("Avg Accuracy (%)")
        axes[1].set_title("Accuracy vs. Latency")
        axes[1].scatter([mean_lat * 1000], [accuracy_mean], s=120, zorder=5)

        plt.suptitle("Complexity-Performance Trade-off", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_dir / "latency_accuracy.png", dpi=150)
        plt.close()


# =============================================================================
# Single hospital-organ evaluation
# =============================================================================

def evaluate_one_dataset(
    *,
    base_args: argparse.Namespace,
    Dataset,
    model: nn.Module,
    num_gpu: int,
    hospital: str,
    organ: str,
    dataset_index: int,
) -> Dict:
    args = base_args

    # Deterministic but different stream for each hospital-organ dataset.
    dataset_seed = int(args.seed) + int(dataset_index) * 1009
    random.seed(dataset_seed)
    np.random.seed(dataset_seed)
    torch.manual_seed(dataset_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(dataset_seed)

    class_root = resolve_dataset_dir(args.data_base, hospital, organ, args.set)
    save_dir = Path(args.output_base) / hospital / organ / ("sfc" if args.use_sfc else "mean")
    save_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: DeepEMD Dataset(set, args) appends /{set} internally.
    # Therefore args.data_dir must be the PARENT of .../test, not .../test itself.
    # Keep class_root separately for class-folder inspection.
    if class_root.name == str(args.set):
        dataset_base = class_root.parent
    else:
        # Standard DeepEMD Dataset requires data_dir/{set}/class/image.
        # For safety, prefer the hospital-organ base when the resolved class root
        # itself is not named after the split.
        dataset_base = Path(args.data_base) / hospital / organ

    # Defensive guard against accidental .../test/test construction.
    while dataset_base.name == str(args.set):
        dataset_base = dataset_base.parent

    args.data_dir = str(dataset_base)
    args.save_dir = str(save_dir)

    expected_test_root = dataset_base / str(args.set)
    print(f"[PATH CHECK] Dataset data_dir = {dataset_base}")
    print(f"[PATH CHECK] Dataset internal split = {expected_test_root}")
    print(f"[PATH CHECK] class_root = {class_root}")

    if class_root.name == str(args.set) and expected_test_root != class_root:
        raise RuntimeError(
            f"Path invariant broken: Dataset split root={expected_test_root}, "
            f"but resolved class_root={class_root}"
        )

    test_set = Dataset(args.set, args)
    actual_label_ids, class_counts = label_statistics(test_set.label)
    way_eff = len(actual_label_ids)
    if way_eff < 2:
        raise RuntimeError(
            f"{hospital}-{organ}: actual query class count={way_eff}, need at least 2 classes"
        )

    # Dynamic-way rule requested by the user.
    # No 8-way/7-way cap: use every actual query class in this test-test dataset.
    sync_runtime_way(args, model, way_eff)

    folder_class_names = list_class_names(class_root)

    # Leakage-safe replacement protocol:
    # reserve query samples first, then sample support from the remaining images.
    # If fewer than K support candidates remain, support is sampled WITH replacement.
    # Thus each class only needs query+1 distinct images, not shot+query.
    min_distinct_per_class = int(args.query) + 1
    insufficient = {cid: n for cid, n in class_counts.items() if n < min_distinct_per_class}
    if insufficient:
        raise RuntimeError(
            f"{hospital}-{organ}: some classes have fewer than query+1={min_distinct_per_class} distinct images: "
            f"{insufficient}. At least one non-query image is required to avoid query/support leakage."
        )

    replacement_needed = {
        cid: n
        for cid, n in class_counts.items()
        if (int(n) - int(args.query)) < int(args.shot)
    }

    print("\n" + "=" * 110)
    print(f"[DATASET] {hospital}-{organ}")
    print(f"data_dir       = {class_root}")
    print(f"model          = {args.model_dir}")
    print(f"query classes  = {way_eff}  <-- runtime way")
    print(f"label ids      = {actual_label_ids}")
    print(f"folder classes = {len(folder_class_names)}: {folder_class_names}")
    print(f"protocol       = test-test | {way_eff}-way {args.shot}-shot {args.query}-query | episodes={args.test_episode}")
    print("support sample = query excluded; no-replacement if enough, WITH-replacement if undersized")
    if replacement_needed:
        print(f"replacement cls= {replacement_needed}  <-- class_count; support will contain repeats")
    else:
        print("replacement cls= none")
    print(f"support adapt  = {'SFC' if args.use_sfc else 'class-wise shot mean'}")
    if args.use_sfc:
        print(f"SFC            = lr={args.sfc_lr}, wd={args.sfc_wd}, steps={args.sfc_update_step}, bs={args.sfc_bs}")
    print(f"save_dir       = {save_dir}")
    print("=" * 110)

    # Leakage-safe test-test sampler: every episode contains all actual classes.
    # Query is distinct; support switches to replacement only when a class does
    # not have enough remaining non-query images for the requested K-shot.
    sampler = TestTestSupportReplacementSampler(
        test_set.label,
        n_batch=args.test_episode,
        n_cls=way_eff,
        n_shot=args.shot,
        n_query=args.query,
        seed=dataset_seed,
    )
    loader = DataLoader(
        test_set,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    tqdm_gen = tqdm.tqdm(loader, desc=f"{hospital}-{organ}")

    label = torch.arange(way_eff, device="cuda").repeat(args.query).long()

    confusion_matrix_accum = np.zeros((way_eff, way_eff), dtype=np.int64)
    intra_distances: List[float] = []
    inter_distances: List[float] = []
    latency_per_episode: List[float] = []
    test_acc_record = np.zeros((args.test_episode,), dtype=np.float64)
    ave_acc = Averager()

    core_model = model.module if isinstance(model, nn.DataParallel) else model

    for i, batch in enumerate(tqdm_gen, 1):
        data, _ = [x.cuda(non_blocking=True) for x in batch]
        k = way_eff * args.shot

        # Encoder stays exactly DeepEMD's own encoder.
        with torch.no_grad():
            core_model.mode = "encoder"
            data = model(data)
            data_shot, data_query = data[:k], data[k:]

        # SFC needs gradients internally, so do NOT wrap this branch in no_grad.
        if args.shot > 1:
            if args.use_sfc:
                data_shot = core_model.get_sfc(data_shot)
            else:
                with torch.no_grad():
                    data_shot = classwise_shot_mean(data_shot, way_eff, args.shot)

        # Meta / EMD matching.
        t0 = time.perf_counter()
        with torch.no_grad():
            core_model.mode = "meta"
            logits = model((
                data_shot.unsqueeze(0).repeat(num_gpu, 1, 1, 1, 1),
                data_query,
            ))
        torch.cuda.synchronize()
        latency_per_episode.append(time.perf_counter() - t0)

        if logits.shape != (way_eff * args.query, way_eff):
            raise RuntimeError(
                f"{hospital}-{organ}: logits shape={tuple(logits.shape)}, "
                f"expected={(way_eff * args.query, way_eff)}"
            )

        acc = count_acc(logits, label) * 100
        ave_acc.add(acc)
        test_acc_record[i - 1] = acc

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        true_labels = label.detach().cpu().numpy()
        for t, p in zip(true_labels, preds):
            confusion_matrix_accum[int(t), int(p)] += 1

        dist_matrix = (-logits / args.temperature).detach().cpu().numpy()
        for q_idx in range(dist_matrix.shape[0]):
            true_cls = int(true_labels[q_idx])
            for cls in range(way_eff):
                d = float(dist_matrix[q_idx, cls])
                if cls == true_cls:
                    intra_distances.append(d)
                else:
                    inter_distances.append(d)

        m_now, pm_now = safe_ci(test_acc_record[:i])
        tqdm_gen.set_description(
            f"{hospital}-{organ} | ep {i} acc={acc:.2f} avg={m_now:.4f}+/-{pm_now:.4f}"
        )

    m, pm = safe_ci(test_acc_record)
    fisher_ratio, kl_div, mu_intra, std_intra, mu_inter, std_inter = fisher_and_kl(
        intra_distances, inter_distances
    )

    cm_norm = confusion_matrix_accum.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = cm_norm / (row_sums + 1e-8)
    per_class_acc = np.diag(cm_norm) * 100.0

    off_diag = []
    for r in range(way_eff):
        for c in range(way_eff):
            if r != c:
                off_diag.append((int(confusion_matrix_accum[r, c]), r, c))
    off_diag.sort(reverse=True)

    latencies = np.asarray(latency_per_episode, dtype=np.float64)
    mean_lat = float(latencies.mean()) if latencies.size else 0.0
    std_lat = float(latencies.std()) if latencies.size else 0.0
    median_lat = float(np.median(latencies)) if latencies.size else 0.0

    print(f"\n[RESULT] {hospital}-{organ}")
    print(f"  way                  : {way_eff}")
    print(f"  support adaptation   : {'SFC' if args.use_sfc else 'class-wise mean'}")
    print(f"  Test Accuracy        : {m:.4f} +/- {pm:.4f} %")
    print(f"  Fisher Ratio         : {fisher_ratio:.4f}")
    print(f"  KL Divergence        : {kl_div:.4f}")
    print(f"  Mean matching latency: {mean_lat*1000:.2f} ms/episode")
    print("  Per-class accuracy:")
    for cls in range(way_eff):
        cls_name = folder_class_names[cls] if cls < len(folder_class_names) else f"label_{cls}"
        print(f"    Class {cls:<2d} {cls_name}: {per_class_acc[cls]:.2f}%")

    if off_diag:
        print("  Top-5 confused pairs:")
        for count, r, c in off_diag[:5]:
            denom = row_sums[r, 0]
            ratio = 100.0 * count / denom if denom > 0 else 0.0
            print(f"    Class {r} -> Class {c}: {count} ({ratio:.1f}%)")

    save_plots(
        save_dir=save_dir,
        confusion_matrix_accum=confusion_matrix_accum,
        intra_distances=intra_distances,
        inter_distances=inter_distances,
        latency_per_episode=latency_per_episode,
        test_acc_record=test_acc_record,
        fisher_ratio=fisher_ratio,
        accuracy_mean=m,
    )

    result = {
        "hospital": hospital,
        "organ": organ,
        "protocol": "test-test",
        "data_dir": str(class_root),
        "model_dir": args.model_dir,
        "way_rule": "actual_query_class_count",
        "way_effective": int(way_eff),
        "shot": int(args.shot),
        "query": int(args.query),
        "episodes": int(args.test_episode),
        "use_sfc": bool(args.use_sfc),
        "support_aggregation": "sfc" if args.use_sfc else "classwise_shot_mean",
        "support_sampling": "query_excluded_no_replace_if_possible_else_replace",
        "support_replacement_needed_classes": {str(k): int(v) for k, v in replacement_needed.items()},
        "minimum_distinct_images_per_class": int(args.query) + 1,
        "sfc": {
            "lr": float(args.sfc_lr),
            "wd": float(args.sfc_wd),
            "update_step": int(args.sfc_update_step),
            "bs": int(args.sfc_bs),
        } if args.use_sfc else None,
        "query_class_count": int(way_eff),
        "query_label_ids": actual_label_ids,
        "folder_class_names": folder_class_names,
        "class_counts": class_counts,
        "accuracy_percent": float(m),
        "ci95_percent": float(pm),
        "fisher_ratio": float(fisher_ratio),
        "kl_divergence_inter_intra": float(kl_div),
        "intra_mean": float(mu_intra),
        "intra_std": float(std_intra),
        "inter_mean": float(mu_inter),
        "inter_std": float(std_inter),
        "latency_mean_ms": mean_lat * 1000.0,
        "latency_std_ms": std_lat * 1000.0,
        "latency_median_ms": median_lat * 1000.0,
        "per_class_accuracy_percent": {
            str(i): float(per_class_acc[i]) for i in range(way_eff)
        },
        "confusion_matrix": confusion_matrix_accum.tolist(),
        "episode_accuracy_percent": test_acc_record.tolist(),
    }

    with open(save_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# =============================================================================
# Main batch loop
# =============================================================================

def main() -> None:
    args = parse_args()
    pprint(vars(args))

    if args.shot < 1 or args.query < 1 or args.test_episode < 1:
        raise ValueError("shot/query/test_episode must all be >= 1")

    for organ in args.organs:
        if organ not in ORGAN_MODEL_DIR:
            raise KeyError(f"Missing checkpoint config for organ={organ}")

    set_seed(args.seed)
    num_gpu = set_gpu(args)
    Dataset = set_up_datasets(args)
    Path(args.output_base).mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("[DeepEMD | test-test | hospital x organ | dynamic query-way]")
    print(f"data_base      = {args.data_base}")
    print(f"hospitals      = {args.hospitals}")
    print(f"organs         = {args.organs}")
    print(f"shot/query     = {args.shot}/{args.query}")
    print(f"episodes       = {args.test_episode}")
    print(f"support adapt  = {'SFC' if args.use_sfc else 'class-wise shot mean'}")
    print("way rule       = actual number of query classes in EACH hospital-organ dataset")
    print(f"output_base    = {args.output_base}")
    print("=" * 110)

    results: List[Dict] = []
    task_index = 0

    # Strict test-test dynamic-way evaluation.
    # For EVERY hospital-organ dataset:
    #   1) inspect the current test pool first;
    #   2) way = actual query/test class count;
    #   3) construct DeepEMD with that exact way;
    #   4) load checkpoint and evaluate.
    # There is no fixed organ-wise 8/8/7 way anywhere in test-test evaluation.
    for organ in args.organs:
        checkpoint = ORGAN_MODEL_DIR[organ]

        for hospital in args.hospitals:
            # Resolve the current test pool before model construction.
            class_root = resolve_dataset_dir(args.data_base, hospital, organ, args.set)
            if class_root.name == str(args.set):
                dataset_base = class_root.parent
            else:
                dataset_base = Path(args.data_base) / hospital / organ
            while dataset_base.name == str(args.set):
                dataset_base = dataset_base.parent

            # Probe the dataset only to obtain the actual query class count.
            probe_args = copy.deepcopy(args)
            probe_args.data_dir = str(dataset_base)
            probe_set = Dataset(probe_args.set, probe_args)
            actual_label_ids, _probe_counts = label_statistics(probe_set.label)
            actual_way = len(actual_label_ids)
            if actual_way < 2:
                raise RuntimeError(
                    f"{hospital}-{organ}: actual query class count={actual_way}, need at least 2 classes"
                )

            print("\n" + "#" * 110)
            print(f"[LOAD MODEL] {hospital}-{organ}")
            print(f"checkpoint       = {checkpoint}")
            print(f"dataset_base     = {dataset_base}")
            print(f"class_root       = {class_root}")
            print(f"actual query way = {actual_way}")
            print("#" * 110)

            model_args = copy.deepcopy(args)
            model_args.model_dir = checkpoint
            model_args.data_dir = str(dataset_base)
            model_args.way = int(actual_way)

            model = DeepEMD(model_args)
            model = load_model(model, model_args.model_dir)
            model = nn.DataParallel(model, list(range(num_gpu)))
            model = model.cuda()
            model.eval()

            result = evaluate_one_dataset(
                base_args=model_args,
                Dataset=Dataset,
                model=model,
                num_gpu=num_gpu,
                hospital=hospital,
                organ=organ,
                dataset_index=task_index,
            )

            # Strong invariant: evaluate_one_dataset() must see the same actual way
            # that was used to construct the model.
            if int(result['way_effective']) != int(actual_way):
                raise RuntimeError(
                    f"Dynamic-way invariant broken for {hospital}-{organ}: "
                    f"model way={actual_way}, evaluated way={result['way_effective']}"
                )

            results.append(result)
            task_index += 1

            del model, probe_set
            torch.cuda.empty_cache()

    # Summary outputs.
    mode_name = "sfc" if args.use_sfc else "mean"
    summary_csv = Path(args.output_base) / f"summary_{mode_name}.csv"
    summary_json = Path(args.output_base) / f"summary_{mode_name}.json"

    fields = [
        "hospital", "organ", "way_effective", "shot", "query", "episodes",
        "use_sfc", "support_aggregation", "accuracy_percent", "ci95_percent",
        "fisher_ratio", "kl_divergence_inter_intra", "latency_mean_ms", "model_dir",
    ]
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fields})

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol": "test-test",
                "way_rule": "actual_query_class_count_per_hospital_organ",
                "support_aggregation": "sfc" if args.use_sfc else "classwise_shot_mean",
                "hospitals": args.hospitals,
                "organs": args.organs,
                "shot": args.shot,
                "query": args.query,
                "episodes": args.test_episode,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 110)
    print("[DONE]")
    print(f"summary_csv  = {summary_csv}")
    print(f"summary_json = {summary_json}")
    print("=" * 110)


if __name__ == "__main__":
    main()