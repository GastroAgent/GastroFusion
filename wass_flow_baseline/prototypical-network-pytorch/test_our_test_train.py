#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ProtoNet test-train evaluation
==============================

Protocol:
    Support = DATA_ROOT/train
    Query   = DATA_ROOT/test

This script does NOT require command-line arguments.
Edit the CONFIG section below, then run:

    python test_tsy_test_train_fixed_config.py

Important:
- Class alignment is performed by class-folder NAME.
- WAY = 0 means using all matched eligible classes.
- Query comes only from test.
- Support comes only from train.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import mini_imagenet_tsy as mini_imagenet_module
from mini_imagenet_tsy import MiniImageNet
from convnet import Convnet
from utils import set_gpu, euclidean_metric


# =============================================================================
# 1. CONFIG — 只改这里即可
# =============================================================================

# 数据根目录：
# DATA_ROOT/
# ├── train/
# │   ├── class_1/
# │   ├── class_2/
# │   └── ...
# └── test/
#     ├── class_1/
#     ├── class_2/
#     └── ...
DATA_ROOT = Path(
    "/mnt/inaisfs/data/home/tansy_criait/jmf/"
    "data_tsy_827_doctor_test_split/十二指肠_1query"
)

# ProtoNet checkpoint
CHECKPOINT_PATH = Path(
    "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/prototypical-network-pytorch/save/tsy/proto-shierzhichang/max-acc.pth"
)

# 输出结果 JSON
OUTPUT_PATH = Path(
    "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/"
    "tsy/prototypical-network-pytorch_100eposide/result_retrain/"
    "shierzhichang_test_train_eval.json"
)

# GPU
GPU = "0"

# Few-shot evaluation
N_EPISODES = 100

# 0 = 使用 train/test 实际 matched 的全部 eligible classes
# 例如共有 8 类，则自动做 8-way。
# 若设为 5，则每个 episode 从 matched classes 中随机选 5 类。
WAY = 0

SHOT = 5
QUERY_PER_CLASS = 1

SEED = 42

# train 某类别少于 SHOT 张时：
# True  = support 允许 replacement 补足 SHOT
# False = 该类不 eligible / 严格要求至少 SHOT 张
SUPPORT_REPLACEMENT_IF_NEEDED = True


# =============================================================================
# 2. Helpers
# =============================================================================

def build_class_to_indices(dataset: MiniImageNet):
    """
    train/test 的数字 label 是各自按文件夹排序生成的，
    因此跨 split 必须按 class folder name 对齐，不能直接按数字 label 对齐。
    """
    out = defaultdict(list)

    for idx, lb in enumerate(dataset.label):
        class_name = dataset.wnids[int(lb)]
        out[class_name].append(idx)

    return dict(out)


def sample_indices(pool, n, rng: random.Random, replacement_if_needed: bool):
    """从一个类别 pool 中采 n 张。"""
    if len(pool) >= n:
        return rng.sample(pool, n)

    if len(pool) == 0:
        raise RuntimeError("Cannot sample from an empty class pool.")

    if not replacement_if_needed:
        raise RuntimeError(
            f"Class pool has only {len(pool)} images, "
            f"but {n} images are required."
        )

    return [rng.choice(pool) for _ in range(n)]


def load_image_batch(dataset: MiniImageNet, indices):
    images = []
    for idx in indices:
        image, _ = dataset[idx]
        images.append(image)

    return torch.stack(images, dim=0)


def load_checkpoint(model, ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif (
        isinstance(ckpt, dict)
        and "model" in ckpt
        and isinstance(ckpt["model"], dict)
    ):
        state = ckpt["model"]
    else:
        state = ckpt

    if isinstance(state, dict):
        state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

    model.load_state_dict(state, strict=True)


# =============================================================================
# 3. Main
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Basic checks
    # -------------------------------------------------------------------------
    data_root = DATA_ROOT.expanduser().resolve()
    checkpoint_path = CHECKPOINT_PATH.expanduser().resolve()
    output_path = OUTPUT_PATH.expanduser().resolve()

    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Support/train directory not found:\n{train_dir}")

    if not test_dir.is_dir():
        raise FileNotFoundError(f"Query/test directory not found:\n{test_dir}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found:\n{checkpoint_path}")

    print("=" * 100)
    print("[CONFIG]")
    print(f"DATA_ROOT       = {data_root}")
    print(f"Support/train   = {train_dir}")
    print(f"Query/test      = {test_dir}")
    print(f"Checkpoint      = {checkpoint_path}")
    print(f"Output          = {output_path}")
    print(f"GPU             = {GPU}")
    print(f"Episodes        = {N_EPISODES}")
    print(f"WAY             = {WAY}  (0 = all matched eligible classes)")
    print(f"SHOT            = {SHOT}")
    print(f"QUERY/class     = {QUERY_PER_CLASS}")
    print(f"Support replace = {SUPPORT_REPLACEMENT_IF_NEEDED}")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # MiniImageNet originally reads the module-global ROOT_PATH.
    # We overwrite it here so the real input path is controlled ONLY by
    # DATA_ROOT at the top of this file.
    # mini_imagenet.py itself does not need to be edited.
    # -------------------------------------------------------------------------
    mini_imagenet_module.ROOT_PATH = str(data_root)

    # GPU
    set_gpu(GPU)

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    rng = random.Random(SEED)

    # -------------------------------------------------------------------------
    # test-train protocol
    # -------------------------------------------------------------------------
    support_dataset = MiniImageNet("train")
    query_dataset = MiniImageNet("test")

    support_map = build_class_to_indices(support_dataset)
    query_map = build_class_to_indices(query_dataset)

    matched_classes = sorted(set(support_map) & set(query_map))

    eligible_classes = []

    for class_name in matched_classes:
        n_support = len(support_map[class_name])
        n_query = len(query_map[class_name])

        support_ok = (
            n_support >= SHOT
            or (
                SUPPORT_REPLACEMENT_IF_NEEDED
                and n_support > 0
            )
        )

        query_ok = n_query >= QUERY_PER_CLASS

        if support_ok and query_ok:
            eligible_classes.append(class_name)

    n_way = len(eligible_classes) if WAY <= 0 else WAY

    if n_way <= 0:
        raise RuntimeError("No eligible matched classes.")

    if len(eligible_classes) < n_way:
        raise RuntimeError(
            f"Requested {n_way}-way, but only "
            f"{len(eligible_classes)} matched classes are eligible."
        )

    print("\n" + "=" * 100)
    print("[TEST-TRAIN PROTOCOL]")
    print("Support = train")
    print("Query   = test")
    print(f"Train classes       = {len(support_map)}")
    print(f"Test classes        = {len(query_map)}")
    print(f"Matched classes     = {len(matched_classes)}")
    print(f"Eligible classes    = {len(eligible_classes)}")
    print(f"Effective N-way     = {n_way}")
    print(f"N-shot              = {SHOT}")
    print(f"N-query/class       = {QUERY_PER_CLASS}")
    print(f"Episodes            = {N_EPISODES}")
    print("=" * 100)

    print("\n[CLASS COUNTS]")
    for cls in matched_classes:
        n_s = len(support_map[cls])
        n_q = len(query_map[cls])
        mark = "OK" if cls in eligible_classes else "SKIP"

        print(
            f"[{mark:4s}] {cls:<42} "
            f"train/support={n_s:>4d}   "
            f"test/query={n_q:>4d}"
        )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = Convnet().to(device)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    episode_accs = []

    overall_correct = 0
    overall_total = 0

    per_class = {
        c: {
            "correct": 0,
            "total": 0,
        }
        for c in eligible_classes
    }

    with torch.no_grad():
        for episode_id in range(1, N_EPISODES + 1):

            # WAY=0 时每个 episode 使用全部 matched eligible classes。
            if n_way == len(eligible_classes):
                episode_classes = list(eligible_classes)
            else:
                episode_classes = rng.sample(
                    eligible_classes,
                    n_way,
                )

            support_indices = []
            query_indices = []

            query_gt_local = []
            query_gt_class = []

            # -------------------------------------------------------------
            # support 来自 train
            # query 来自 test
            # -------------------------------------------------------------
            for local_label, class_name in enumerate(episode_classes):

                support_idx = sample_indices(
                    support_map[class_name],
                    SHOT,
                    rng,
                    replacement_if_needed=SUPPORT_REPLACEMENT_IF_NEEDED,
                )

                query_idx = sample_indices(
                    query_map[class_name],
                    QUERY_PER_CLASS,
                    rng,
                    replacement_if_needed=False,
                )

                support_indices.extend(support_idx)
                query_indices.extend(query_idx)

                query_gt_local.extend(
                    [local_label] * QUERY_PER_CLASS
                )

                query_gt_class.extend(
                    [class_name] * QUERY_PER_CLASS
                )

            # -------------------------------------------------------------
            # Load images
            # -------------------------------------------------------------
            data_shot = load_image_batch(
                support_dataset,
                support_indices,
            ).to(
                device,
                non_blocking=True,
            )

            data_query = load_image_batch(
                query_dataset,
                query_indices,
            ).to(
                device,
                non_blocking=True,
            )

            # -------------------------------------------------------------
            # ProtoNet
            #
            # support_indices 是 class-major:
            # class0 的 SHOT 张 -> class1 的 SHOT 张 -> ...
            #
            # 所以 reshape:
            # [n_way, shot, feat_dim]
            # -------------------------------------------------------------
            support_feat = model(data_shot)

            proto = support_feat.reshape(
                n_way,
                SHOT,
                -1,
            ).mean(dim=1)

            query_feat = model(data_query)

            logits = euclidean_metric(
                query_feat,
                proto,
            )

            gt = torch.tensor(
                query_gt_local,
                dtype=torch.long,
                device=device,
            )

            pred = logits.argmax(dim=1)
            correct_mask = pred.eq(gt)

            ep_correct = int(correct_mask.sum().item())
            ep_total = int(gt.numel())
            ep_acc = ep_correct / ep_total

            episode_accs.append(ep_acc)

            overall_correct += ep_correct
            overall_total += ep_total

            for j, class_name in enumerate(query_gt_class):
                per_class[class_name]["total"] += 1

                if bool(correct_mask[j].item()):
                    per_class[class_name]["correct"] += 1

            running_acc = overall_correct / overall_total

            print(
                "episode {:4d}: "
                "running={:.2f}%   "
                "current={:.2f}%   "
                "({}/{})".format(
                    episode_id,
                    running_acc * 100.0,
                    ep_acc * 100.0,
                    ep_correct,
                    ep_total,
                )
            )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    episode_accs_np = np.asarray(
        episode_accs,
        dtype=np.float64,
    )

    episode_mean = float(
        episode_accs_np.mean()
    )

    episode_std = float(
        episode_accs_np.std(ddof=1)
        if len(episode_accs_np) > 1
        else 0.0
    )

    ci95 = float(
        1.96
        * episode_std
        / np.sqrt(len(episode_accs_np))
        if len(episode_accs_np) > 1
        else 0.0
    )

    overall_acc = (
        overall_correct / overall_total
        if overall_total > 0
        else 0.0
    )

    per_class_summary = {}

    for class_name, stat in per_class.items():
        acc = (
            stat["correct"] / stat["total"]
            if stat["total"] > 0
            else 0.0
        )

        per_class_summary[class_name] = {
            "correct": stat["correct"],
            "total": stat["total"],
            "accuracy": acc,
            "accuracy_percent": acc * 100.0,
        }

    # -------------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------------
    result = {
        "protocol": "test-train",
        "data_root": str(data_root),
        "support_root": str(train_dir),
        "query_root": str(test_dir),
        "support_split": "train",
        "query_split": "test",
        "checkpoint": str(checkpoint_path),

        "seed": SEED,
        "episodes": N_EPISODES,

        "n_way": n_way,
        "n_shot": SHOT,
        "n_query_per_class": QUERY_PER_CLASS,

        "matched_classes": matched_classes,
        "eligible_classes": eligible_classes,

        "support_replacement_if_needed": (
            SUPPORT_REPLACEMENT_IF_NEEDED
        ),

        "episode_accuracy_mean": episode_mean,
        "episode_accuracy_mean_percent": (
            episode_mean * 100.0
        ),
        "episode_accuracy_ci95_percent": (
            ci95 * 100.0
        ),

        "overall_query_accuracy": overall_acc,
        "overall_query_accuracy_percent": (
            overall_acc * 100.0
        ),

        "correct": overall_correct,
        "total_queries": overall_total,

        "per_class": per_class_summary,
    }

    # -------------------------------------------------------------------------
    # Print final
    # -------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("[FINAL]")
    print(
        f"Episode mean Acc = "
        f"{episode_mean * 100.0:.2f} "
        f"± {ci95 * 100.0:.2f}%"
    )
    print(
        f"Overall query Acc = "
        f"{overall_acc * 100.0:.2f}% "
        f"({overall_correct}/{overall_total})"
    )
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[SAVE] {output_path}")


if __name__ == "__main__":
    main()
