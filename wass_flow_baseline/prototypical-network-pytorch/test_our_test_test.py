#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ProtoNet test-test evaluation
=============================

Protocol:
    Support = DATA_ROOT/test
    Query   = DATA_ROOT/test

Important:
- Support and query are sampled from the SAME test split.
- Within each episode, support/query images are kept DISJOINT.
- WAY = 0 means using all eligible test classes.
- If a class has fewer than SHOT + QUERY_PER_CLASS images,
  support can use replacement when
  SUPPORT_REPLACEMENT_IF_NEEDED = True.
- Query images will never appear in support in the same episode.

Run:
    python test_tsy_test_test_fixed_config.py
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
# 1. CONFIG — 只改这里
# =============================================================================

# 数据根目录
#
# DATA_ROOT/
# └── test/
#     ├── class_1/
#     ├── class_2/
#     └── ...
#
DATA_ROOT = Path(
    "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/prototypical-network-pytorch/data_tsy_827_doctor_test_split_test_test/胃_1query"
)


# ProtoNet checkpoint
CHECKPOINT_PATH = Path(
    "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/"
    "prototypical-network-pytorch/save/tsy/"
    "proto-stomach/max-acc.pth"
)


# 输出结果
OUTPUT_PATH = Path(
    "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/"
    "tsy/prototypical-network-pytorch_100eposide/"
    "result_retrain_test_test/"
    "stomach_test_test_eval.json"
)


# GPU
GPU = "0"


# Few-shot evaluation
N_EPISODES = 100


# 0 = 使用 test 中所有 eligible classes
#
# 例如：
# test 有 7 个 eligible class
# WAY = 0
# -> 自动 7-way
#
WAY = 0


SHOT = 5

QUERY_PER_CLASS = 1

SEED = 42


# =============================================================================
# Rare-class support policy
# =============================================================================
#
# test-test 时：
#
# 如果一个类别：
#
#   image_num >= SHOT + QUERY
#
# 正常无放回抽：
#
#   support 5
#   query   1
#
#
# 如果：
#
#   QUERY < image_num < SHOT + QUERY
#
# 则：
#
#   先抽 query
#   support 从剩余图片中 replacement 补足 SHOT
#
# 这样可以保证：
#
#   support 和 query 不使用同一张图
#
SUPPORT_REPLACEMENT_IF_NEEDED = True


# =============================================================================
# 2. Helpers
# =============================================================================

def build_class_to_indices(dataset: MiniImageNet):
    """
    按 class-folder name 建立：

        class_name -> dataset indices

    即使数字 label 发生变化，也不依赖数字 label。
    """

    out = defaultdict(list)

    for idx, lb in enumerate(dataset.label):

        class_name = dataset.wnids[int(lb)]

        out[class_name].append(idx)

    return dict(out)


def sample_test_test_episode(
    pool,
    shot,
    query,
    rng: random.Random,
    replacement_if_needed=True,
):
    """
    从同一个 test class pool 中同时抽 support/query。

    核心要求：
        support 与 query 在同一个 episode 中不能重叠。

    Case 1:
        len(pool) >= shot + query

        直接无放回抽 shot+query 张。

    Case 2:
        query < len(pool) < shot+query

        先无放回抽 query；
        support 从剩余图片中抽。
        如果剩余数量 < shot，则 support replacement。

    要求至少存在：
        query + 1

    张不同图片，从而保证 query 与 support 不重叠。
    """

    pool = list(pool)

    n = len(pool)

    if n < query + 1:

        raise RuntimeError(
            f"Class only has {n} images, "
            f"but test-test needs at least "
            f"{query + 1} unique images "
            f"to keep support/query disjoint."
        )

    # -------------------------------------------------------------------------
    # 正常情况：足够 shot + query
    # -------------------------------------------------------------------------

    if n >= shot + query:

        picked = rng.sample(
            pool,
            shot + query,
        )

        support_indices = picked[:shot]

        query_indices = picked[
            shot:
            shot + query
        ]

        return (
            support_indices,
            query_indices,
            False,
        )

    # -------------------------------------------------------------------------
    # 图片不足 shot + query
    # -------------------------------------------------------------------------

    if not replacement_if_needed:

        raise RuntimeError(
            f"Class only has {n} images, "
            f"but {shot + query} are required."
        )

    # 先选 query
    query_indices = rng.sample(
        pool,
        query,
    )

    query_set = set(
        query_indices
    )

    # support 只能从 query 之外的图片中产生
    support_pool = [
        idx
        for idx in pool
        if idx not in query_set
    ]

    if len(support_pool) == 0:

        raise RuntimeError(
            "No image remains for support "
            "after selecting query."
        )

    # support pool 足够
    if len(support_pool) >= shot:

        support_indices = rng.sample(
            support_pool,
            shot,
        )

    # support pool 不够，replacement
    else:

        support_indices = [
            rng.choice(
                support_pool
            )
            for _ in range(shot)
        ]

    return (
        support_indices,
        query_indices,
        True,
    )


def load_image_batch(
    dataset: MiniImageNet,
    indices,
):

    images = []

    for idx in indices:

        image, _ = dataset[idx]

        images.append(image)

    return torch.stack(
        images,
        dim=0,
    )


def load_checkpoint(
    model,
    ckpt_path: Path,
):

    ckpt = torch.load(
        str(ckpt_path),
        map_location="cpu",
    )

    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
    ):

        state = ckpt[
            "state_dict"
        ]

    elif (
        isinstance(ckpt, dict)
        and "model" in ckpt
        and isinstance(
            ckpt["model"],
            dict,
        )
    ):

        state = ckpt[
            "model"
        ]

    else:

        state = ckpt

    if isinstance(
        state,
        dict,
    ):

        state = {
            (
                k[7:]
                if k.startswith(
                    "module."
                )
                else k
            ): v
            for k, v
            in state.items()
        }

    model.load_state_dict(
        state,
        strict=True,
    )


# =============================================================================
# 3. Main
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------

    data_root = (
        DATA_ROOT
        .expanduser()
        .resolve()
    )

    checkpoint_path = (
        CHECKPOINT_PATH
        .expanduser()
        .resolve()
    )

    output_path = (
        OUTPUT_PATH
        .expanduser()
        .resolve()
    )


    # =========================================================================
    # test-test:
    #
    # support = test
    # query   = test
    # =========================================================================

    test_dir = (
        data_root
        / "test"
    )


    if not test_dir.is_dir():

        raise FileNotFoundError(
            f"Test directory not found:\n"
            f"{test_dir}"
        )


    if not checkpoint_path.is_file():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )


    # -------------------------------------------------------------------------
    # Print CONFIG
    # -------------------------------------------------------------------------

    print(
        "=" * 100
    )

    print(
        "[CONFIG]"
    )

    print(
        f"DATA_ROOT       = "
        f"{data_root}"
    )

    print(
        f"Support/test    = "
        f"{test_dir}"
    )

    print(
        f"Query/test      = "
        f"{test_dir}"
    )

    print(
        f"Checkpoint      = "
        f"{checkpoint_path}"
    )

    print(
        f"Output          = "
        f"{output_path}"
    )

    print(
        f"GPU             = "
        f"{GPU}"
    )

    print(
        f"Episodes        = "
        f"{N_EPISODES}"
    )

    print(
        f"WAY             = "
        f"{WAY} "
        f"(0 = all eligible classes)"
    )

    print(
        f"SHOT            = "
        f"{SHOT}"
    )

    print(
        f"QUERY/class     = "
        f"{QUERY_PER_CLASS}"
    )

    print(
        f"Support replace = "
        f"{SUPPORT_REPLACEMENT_IF_NEEDED}"
    )

    print(
        "=" * 100
    )


    # -------------------------------------------------------------------------
    # MiniImageNet ROOT_PATH
    # -------------------------------------------------------------------------

    mini_imagenet_module.ROOT_PATH = (
        str(data_root)
    )


    # -------------------------------------------------------------------------
    # GPU
    # -------------------------------------------------------------------------

    set_gpu(GPU)


    # -------------------------------------------------------------------------
    # Seed
    # -------------------------------------------------------------------------

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
        )


    rng = random.Random(
        SEED
    )


    # =========================================================================
    # TEST-TEST protocol
    # =========================================================================

    # support / query 都来自同一个 test dataset
    test_dataset = MiniImageNet(
        "test"
    )


    test_map = build_class_to_indices(
        test_dataset
    )


    # -------------------------------------------------------------------------
    # Eligible classes
    # -------------------------------------------------------------------------

    eligible_classes = []

    for class_name in sorted(
        test_map.keys()
    ):

        n_test = len(
            test_map[
                class_name
            ]
        )

        # -------------------------------------------------------------
        # test-test 必须至少：
        #
        # QUERY + 1 unique image
        #
        # 因为 query 不能和 support 重合。
        # -------------------------------------------------------------

        if SUPPORT_REPLACEMENT_IF_NEEDED:

            eligible = (
                n_test
                >= QUERY_PER_CLASS + 1
            )

        else:

            eligible = (
                n_test
                >= SHOT
                + QUERY_PER_CLASS
            )

        if eligible:

            eligible_classes.append(
                class_name
            )


    n_way = (
        len(eligible_classes)
        if WAY <= 0
        else WAY
    )


    if n_way <= 0:

        raise RuntimeError(
            "No eligible test classes."
        )


    if (
        len(eligible_classes)
        < n_way
    ):

        raise RuntimeError(
            f"Requested {n_way}-way, "
            f"but only "
            f"{len(eligible_classes)} "
            f"test classes are eligible."
        )


    # -------------------------------------------------------------------------
    # Protocol info
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "[TEST-TEST PROTOCOL]"
    )

    print(
        "Support = test"
    )

    print(
        "Query   = test"
    )

    print(
        "Support/query overlap "
        "= FORBIDDEN"
    )

    print(
        f"Test classes        = "
        f"{len(test_map)}"
    )

    print(
        f"Eligible classes    = "
        f"{len(eligible_classes)}"
    )

    print(
        f"Effective N-way     = "
        f"{n_way}"
    )

    print(
        f"N-shot              = "
        f"{SHOT}"
    )

    print(
        f"N-query/class       = "
        f"{QUERY_PER_CLASS}"
    )

    print(
        f"Episodes            = "
        f"{N_EPISODES}"
    )

    print(
        "=" * 100
    )


    # -------------------------------------------------------------------------
    # Class count
    # -------------------------------------------------------------------------

    print(
        "\n[CLASS COUNTS]"
    )

    for cls in sorted(
        test_map.keys()
    ):

        n_test = len(
            test_map[cls]
        )

        mark = (
            "OK"
            if cls in eligible_classes
            else "SKIP"
        )

        print(
            f"[{mark:4s}] "
            f"{cls:<42} "
            f"test={n_test:>4d}"
        )


    # =========================================================================
    # Model
    # =========================================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    model = Convnet().to(
        device
    )


    load_checkpoint(
        model,
        checkpoint_path
    )


    model.eval()


    print(
        "\n[MODEL]"
    )

    print(
        f"Loaded checkpoint: "
        f"{checkpoint_path}"
    )


    # =========================================================================
    # Evaluation
    # =========================================================================

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


    replacement_episode_count = 0

    replacement_class_count = 0


    with torch.no_grad():

        for episode_id in range(
            1,
            N_EPISODES + 1
        ):

            # -------------------------------------------------------------
            # Episode classes
            # -------------------------------------------------------------

            if (
                n_way
                == len(
                    eligible_classes
                )
            ):

                episode_classes = list(
                    eligible_classes
                )

            else:

                episode_classes = rng.sample(
                    eligible_classes,
                    n_way,
                )


            support_indices = []

            query_indices = []

            query_gt_local = []

            query_gt_class = []


            episode_used_replacement = False


            # -------------------------------------------------------------
            # Support / Query BOTH from TEST
            # -------------------------------------------------------------

            for (
                local_label,
                class_name
            ) in enumerate(
                episode_classes
            ):

                (
                    support_idx,
                    query_idx,
                    used_replacement,
                ) = sample_test_test_episode(

                    test_map[
                        class_name
                    ],

                    SHOT,

                    QUERY_PER_CLASS,

                    rng,

                    replacement_if_needed=(
                        SUPPORT_REPLACEMENT_IF_NEEDED
                    ),
                )


                # ---------------------------------------------------------
                # Safety check:
                # support / query unique-index overlap forbidden
                # ---------------------------------------------------------

                overlap = (
                    set(
                        support_idx
                    )
                    &
                    set(
                        query_idx
                    )
                )

                if overlap:

                    raise RuntimeError(
                        f"Support/query overlap "
                        f"detected in class "
                        f"{class_name}: "
                        f"{overlap}"
                    )


                if used_replacement:

                    episode_used_replacement = True

                    replacement_class_count += 1


                support_indices.extend(
                    support_idx
                )

                query_indices.extend(
                    query_idx
                )


                query_gt_local.extend(
                    [
                        local_label
                    ]
                    * QUERY_PER_CLASS
                )


                query_gt_class.extend(
                    [
                        class_name
                    ]
                    * QUERY_PER_CLASS
                )


            if episode_used_replacement:

                replacement_episode_count += 1


            # -------------------------------------------------------------
            # Load images
            # -------------------------------------------------------------

            data_shot = load_image_batch(
                test_dataset,
                support_indices,
            ).to(
                device,
                non_blocking=True,
            )


            data_query = load_image_batch(
                test_dataset,
                query_indices,
            ).to(
                device,
                non_blocking=True,
            )


            # -------------------------------------------------------------
            # ProtoNet
            #
            # support:
            #
            # class0: SHOT
            # class1: SHOT
            # ...
            #
            # -> [n_way, shot, feat]
            # -> mean over shot
            # -------------------------------------------------------------

            support_feat = model(
                data_shot
            )


            proto = support_feat.reshape(
                n_way,
                SHOT,
                -1,
            ).mean(
                dim=1
            )


            query_feat = model(
                data_query
            )


            logits = euclidean_metric(
                query_feat,
                proto,
            )


            gt = torch.tensor(
                query_gt_local,
                dtype=torch.long,
                device=device,
            )


            pred = logits.argmax(
                dim=1
            )


            correct_mask = pred.eq(
                gt
            )


            ep_correct = int(
                correct_mask
                .sum()
                .item()
            )


            ep_total = int(
                gt.numel()
            )


            ep_acc = (
                ep_correct
                / ep_total
            )


            episode_accs.append(
                ep_acc
            )


            overall_correct += (
                ep_correct
            )

            overall_total += (
                ep_total
            )


            # -------------------------------------------------------------
            # Per-class
            # -------------------------------------------------------------

            for (
                j,
                class_name
            ) in enumerate(
                query_gt_class
            ):

                per_class[
                    class_name
                ][
                    "total"
                ] += 1


                if bool(
                    correct_mask[
                        j
                    ].item()
                ):

                    per_class[
                        class_name
                    ][
                        "correct"
                    ] += 1


            running_acc = (
                overall_correct
                / overall_total
            )


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


    # =========================================================================
    # Metrics
    # =========================================================================

    episode_accs_np = np.asarray(
        episode_accs,
        dtype=np.float64,
    )


    episode_mean = float(
        episode_accs_np.mean()
    )


    episode_std = float(

        episode_accs_np.std(
            ddof=1
        )

        if len(
            episode_accs_np
        ) > 1

        else 0.0
    )


    ci95 = float(

        1.96
        * episode_std
        / np.sqrt(
            len(
                episode_accs_np
            )
        )

        if len(
            episode_accs_np
        ) > 1

        else 0.0
    )


    overall_acc = (

        overall_correct
        / overall_total

        if overall_total > 0

        else 0.0
    )


    # -------------------------------------------------------------------------
    # Per-class summary
    # -------------------------------------------------------------------------

    per_class_summary = {}


    for (
        class_name,
        stat
    ) in per_class.items():

        acc = (

            stat[
                "correct"
            ]
            / stat[
                "total"
            ]

            if stat[
                "total"
            ] > 0

            else 0.0
        )


        per_class_summary[
            class_name
        ] = {

            "correct":
                stat[
                    "correct"
                ],

            "total":
                stat[
                    "total"
                ],

            "accuracy":
                acc,

            "accuracy_percent":
                acc * 100.0,
        }


    # =========================================================================
    # Result JSON
    # =========================================================================

    result = {

        "protocol":
            "test-test",

        "data_root":
            str(
                data_root
            ),

        "support_root":
            str(
                test_dir
            ),

        "query_root":
            str(
                test_dir
            ),

        "support_split":
            "test",

        "query_split":
            "test",

        "support_query_disjoint":
            True,

        "checkpoint":
            str(
                checkpoint_path
            ),

        "seed":
            SEED,

        "episodes":
            N_EPISODES,

        "n_way":
            n_way,

        "n_shot":
            SHOT,

        "n_query_per_class":
            QUERY_PER_CLASS,

        "eligible_classes":
            eligible_classes,

        "support_replacement_if_needed":
            SUPPORT_REPLACEMENT_IF_NEEDED,

        "replacement_episode_count":
            replacement_episode_count,

        "replacement_class_count":
            replacement_class_count,

        "episode_accuracy_mean":
            episode_mean,

        "episode_accuracy_mean_percent":
            episode_mean
            * 100.0,

        "episode_accuracy_std":
            episode_std,

        "episode_accuracy_std_percent":
            episode_std
            * 100.0,

        "episode_accuracy_ci95":
            ci95,

        "episode_accuracy_ci95_percent":
            ci95
            * 100.0,

        "overall_query_accuracy":
            overall_acc,

        "overall_query_accuracy_percent":
            overall_acc
            * 100.0,

        "correct":
            overall_correct,

        "total_queries":
            overall_total,

        "per_class":
            per_class_summary,
    }


    # =========================================================================
    # Final print
    # =========================================================================

    print()

    print(
        "=" * 100
    )

    print(
        "[FINAL]"
    )

    print(
        "Protocol = test-test"
    )

    print(
        "Support = test"
    )

    print(
        "Query   = test"
    )

    print(
        f"Episode mean Acc = "
        f"{episode_mean * 100.0:.2f} "
        f"± {ci95 * 100.0:.2f}%"
    )

    print(
        f"Episode std      = "
        f"{episode_std * 100.0:.2f}%"
    )

    print(
        f"Overall query Acc = "
        f"{overall_acc * 100.0:.2f}% "
        f"({overall_correct}/"
        f"{overall_total})"
    )

    print(
        f"Replacement episodes = "
        f"{replacement_episode_count}"
        f"/{N_EPISODES}"
    )

    print(
        "=" * 100
    )


    # =========================================================================
    # Save
    # =========================================================================

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


    print(
        f"[SAVE] {output_path}"
    )


if __name__ == "__main__":
    main()