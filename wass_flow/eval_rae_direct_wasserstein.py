#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct RAE Wasserstein evaluation (NO MetricEncoder / NO FSL model).

Purpose
-------
Use the frozen RAE latent itself as the empirical distribution for
query-support Wasserstein matching:

    image
      -> frozen RAE
      -> latent [B, 1024, 16, 16]
      -> spatial points [B, 256, 1024]
      -> L2-normalized points
      -> uniform-mass entropic Sinkhorn / W2 distance
      -> K-shot aggregation
      -> argmin(class distance)

There is NO:
    - MetricEncoder
    - FSL checkpoint
    - M0/M1/M2 checkpoint
    - meta checkpoint
    - learned Dynamic-SK mass

The only trainable-looking operation at evaluation time is NONE.
RAE is frozen and Wasserstein matching is purely non-parametric.

Supported settings
------------------
EVAL_SETTING = "test_train"
    query   = DATA_BASE/indomain_support_76/{hospital}/{part}
    support = DATA_BASE/{part}_1query/train

EVAL_SETTING = "test_test"
    query/support are from the same hospital/organ root.
    The sampled query image is explicitly excluded from support.

Outputs
-------
RESULT_DIR/
    summary.csv
    summary.json
    {part}/
        {hospital}_{part}_{EVAL_SETTING}_rae_wasserstein.json
"""

from __future__ import annotations

import csv
import json
import math
import os

# Choose one visible GPU before torch is imported.
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT_ROOT = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow"
DATA_BASE = os.path.join(PROJECT_ROOT, "data")

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "fsl_multi_vae/rae_flow_metric_v1",
    "results_rae_direct_wasserstein",
)

HOSPITALS = ["华西", "自贡", "南充", "苍溪"]
PARTS = ["十二指肠", "胃", "食管"]

# "test_train" or "test_test"
EVAL_SETTING = "test_test"

# Episode protocol.
# If a task has fewer valid classes than N_WAY, all valid classes are used.
N_WAY = 8
N_SHOT = 5
N_QUERY = 1
EPISODES_PER_TASK = 200
SEED = 42

SHUFFLE_CANDIDATE_SLOTS = True

# Frozen RAE.
IMAGE_SIZE = 256
RAE_CHUNK_SIZE = 8

# Wasserstein / Sinkhorn.
#
# RAE latent [B,1024,16,16] is treated as 16*16 empirical points in R^1024.
# OT_POOL_FACTOR=1 keeps all 256 spatial points.
# OT_POOL_FACTOR=2 -> 8*8=64 points, etc.
OT_POOL_FACTOR = 1

# Squared Euclidean cost between L2-normalized RAE points.
NORMALIZE_RAE_POINTS = True

# Entropic regularization and Sinkhorn iterations.
SINKHORN_REG = 0.10
SINKHORN_ITER = 50
SINKHORN_EPS = 1e-8

# When True, return sqrt(<pi, ||x-y||^2>), i.e. W2-style distance.
# Classification argmin is unchanged if this is toggled.
RETURN_W2_SQRT = True

# K-shot aggregation across independent support images.
SHOT_AGG = "softmin"   # "softmin", "min", "mean"
SHOT_TAU = 0.10

SAVE_EPISODE_DETAILS = True
LOG_EVERY = 25

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class EpisodePlan:
    episode_id: int
    candidate_labels: List[str]
    query_label: str
    query_path: str
    support_paths: Dict[str, List[str]]
    seed: int


@dataclass
class Accumulator:
    correct: int = 0
    total: int = 0
    pos_sum: float = 0.0
    neg_sum: float = 0.0
    margin_sum: float = 0.0
    pair_pos_sum: float = 0.0
    pair_neg_sum: float = 0.0


# =============================================================================
# Basic helpers
# =============================================================================

def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def fast_path_key(path: str) -> str:
    return os.path.normcase(
        os.path.abspath(
            os.path.normpath(
                os.path.expanduser(path)
            )
        )
    )


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =============================================================================
# Dataset scanning
# =============================================================================

def scan_class_folders(root: str) -> Dict[str, List[str]]:
    """Scan root/class_name/**/*image."""
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"Class root not found: {root}")

    result: Dict[str, List[str]] = {}

    for child in sorted(root_p.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue

        paths = [
            str(p)
            for p in child.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        paths.sort()

        seen = set()
        unique = []
        for p in paths:
            key = fast_path_key(p)
            if key not in seen:
                seen.add(key)
                unique.append(p)

        if unique:
            result[child.name] = unique

    if not result:
        raise RuntimeError(f"No class images found under: {root}")

    return result


def prepare_task_pools(
    hospital: str,
    part: str,
    *,
    eval_setting: str,
    n_shot: int,
) -> Tuple[
    Dict[str, List[str]],
    Dict[str, List[str]],
    List[str],
    str,
    str,
]:
    query_root = os.path.join(
        DATA_BASE,
        "indomain_support_76",
        hospital,
        part,
    )

    if eval_setting == "test_train":
        support_root = os.path.join(
            DATA_BASE,
            f"{part}_1query",
            "train",
        )
    elif eval_setting == "test_test":
        support_root = query_root
    else:
        raise ValueError(
            f"EVAL_SETTING must be test_train/test_test, got {eval_setting!r}"
        )

    query_pool = scan_class_folders(query_root)
    support_pool = (
        query_pool
        if eval_setting == "test_test"
        else scan_class_folders(support_root)
    )

    common_labels: List[str] = []
    all_common = sorted(set(query_pool) & set(support_pool))

    for label in all_common:
        if len(query_pool[label]) < 1:
            continue

        if eval_setting == "test_test":
            # Need one query + K distinct support images.
            if len(support_pool[label]) < n_shot + 1:
                continue
        else:
            if len(support_pool[label]) < n_shot:
                continue

        common_labels.append(label)

    if len(common_labels) < 2:
        counts = {
            c: (
                len(query_pool.get(c, [])),
                len(support_pool.get(c, [])),
            )
            for c in sorted(set(query_pool) | set(support_pool))
        }
        raise RuntimeError(
            f"{hospital}/{part}: fewer than 2 valid common classes. "
            f"counts(query,support)={counts}"
        )

    query_pool = {c: query_pool[c] for c in common_labels}
    support_pool = {c: support_pool[c] for c in common_labels}

    return (
        query_pool,
        support_pool,
        common_labels,
        query_root,
        support_root,
    )


# =============================================================================
# Episode generation
# =============================================================================

def generate_episode_plans(
    query_pool: Dict[str, List[str]],
    support_pool: Dict[str, List[str]],
    labels: Sequence[str],
    *,
    n_way: int,
    n_shot: int,
    n_episodes: int,
    seed: int,
) -> List[EpisodePlan]:
    rng = random.Random(seed)
    labels = list(labels)

    effective_way = min(int(n_way), len(labels))
    if effective_way < 2:
        raise ValueError(f"effective_way must be >=2, got {effective_way}")

    plans: List[EpisodePlan] = []

    for episode_id in range(int(n_episodes)):
        selected = rng.sample(labels, effective_way)
        query_label = rng.choice(selected)
        query_path = rng.choice(query_pool[query_label])
        query_key = fast_path_key(query_path)

        support_paths: Dict[str, List[str]] = {}

        for label in selected:
            candidates = [
                p
                for p in support_pool[label]
                if fast_path_key(p) != query_key
            ]

            if len(candidates) < int(n_shot):
                raise RuntimeError(
                    f"episode={episode_id}: support insufficient for "
                    f"class={label!r}; available={len(candidates)}, need={n_shot}"
                )

            support_paths[label] = rng.sample(
                candidates,
                int(n_shot),
            )

        if SHUFFLE_CANDIDATE_SLOTS:
            rng.shuffle(selected)

        plans.append(
            EpisodePlan(
                episode_id=episode_id,
                candidate_labels=list(selected),
                query_label=query_label,
                query_path=query_path,
                support_paths={
                    c: support_paths[c]
                    for c in selected
                },
                seed=seed,
            )
        )

    return plans


# =============================================================================
# Frozen RAE
# =============================================================================

def build_rae(project_root: str, device: torch.device) -> nn.Module:
    from rae.rae import RAE

    model = RAE(
        encoder_name="dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]",
        resolution=256,
        decoder_config_path=os.path.join(
            project_root,
            "rae/configs/decoder/ViTXL/config.json",
        ),
        pretrained_decoder_path=os.path.join(
            project_root,
            "flow_matcher_otcfm/rae_v2/decoder.pt",
        ),
        noise_tau=0.0,
        normalization_stat_path=os.path.join(
            project_root,
            "flow_matcher_otcfm/rae_v2/stats.pt",
        ),
    ).to(device).eval()

    model.requires_grad_(False)
    return model


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def load_images(
    paths: Sequence[str],
    transform,
) -> torch.Tensor:
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(
                transform(
                    im.convert("RGB")
                )
            )

    return torch.stack(images, dim=0)


@torch.inference_mode()
def encode_rae_chunked(
    rae: nn.Module,
    images: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size <= 0:
        chunk_size = int(images.shape[0])

    chunks = []
    for start in range(0, images.shape[0], chunk_size):
        chunks.append(
            rae.encode(
                images[start:start + chunk_size]
            )
        )

    latent = torch.cat(chunks, dim=0)

    if latent.dim() != 4:
        raise RuntimeError(
            f"RAE latent must be [B,C,H,W], got {tuple(latent.shape)}"
        )

    # This project normally produces [B,1024,16,16].
    if latent.shape[1] != 1024:
        raise RuntimeError(
            f"RAE latent channel must be 1024, got {tuple(latent.shape)}"
        )

    return latent.float()


# =============================================================================
# Direct RAE -> empirical points
# =============================================================================

def rae_latent_to_points(
    latent: torch.Tensor,
) -> torch.Tensor:
    """
    [B,C,H,W] -> [B,H*W,C].

    No MetricEncoder/FSL transformation is applied.
    """
    if latent.dim() != 4:
        raise ValueError(
            f"latent must be [B,C,H,W], got {tuple(latent.shape)}"
        )

    x = latent

    if OT_POOL_FACTOR > 1:
        if x.shape[-2] % OT_POOL_FACTOR != 0 or x.shape[-1] % OT_POOL_FACTOR != 0:
            raise ValueError(
                "RAE latent spatial shape must be divisible by OT_POOL_FACTOR. "
                f"shape={tuple(x.shape)}, factor={OT_POOL_FACTOR}"
            )

        x = F.avg_pool2d(
            x,
            kernel_size=OT_POOL_FACTOR,
            stride=OT_POOL_FACTOR,
        )

    points = (
        x.flatten(2)
        .transpose(1, 2)
        .contiguous()
    )

    if NORMALIZE_RAE_POINTS:
        points = F.normalize(
            points,
            p=2,
            dim=-1,
            eps=1e-8,
        )

    return points


# =============================================================================
# Uniform-mass Sinkhorn Wasserstein
# =============================================================================

def squared_euclidean_cost(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """
    x: [P,N,D]
    y: [P,M,D]
    -> cost [P,N,M]
    """
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1).unsqueeze(1)
    xy = torch.bmm(x, y.transpose(1, 2))

    cost = x2 + y2 - 2.0 * xy
    return cost.clamp_min_(0.0)


@torch.inference_mode()
def uniform_sinkhorn_distance(
    query_points: torch.Tensor,
    support_points: torch.Tensor,
    *,
    reg: float,
    n_iter: int,
) -> torch.Tensor:
    """
    Pairwise query-support entropic OT with UNIFORM empirical masses.

    query_points   : [Q,Nq,D]
    support_points : [S,Ns,D]

    Returns
    -------
    distance : [Q,S]

    No learned/dynamic node mass is used.
    """
    if reg <= 0:
        raise ValueError(f"SINKHORN_REG must be >0, got {reg}")
    if n_iter <= 0:
        raise ValueError(f"SINKHORN_ITER must be >0, got {n_iter}")

    q_count, nq, dim_q = query_points.shape
    s_count, ns, dim_s = support_points.shape

    if dim_q != dim_s:
        raise ValueError(
            f"point dims mismatch: query={dim_q}, support={dim_s}"
        )

    # Build all Q*S pairs.
    q_pair = (
        query_points[:, None, :, :]
        .expand(q_count, s_count, nq, dim_q)
        .reshape(q_count * s_count, nq, dim_q)
    )
    s_pair = (
        support_points[None, :, :, :]
        .expand(q_count, s_count, ns, dim_s)
        .reshape(q_count * s_count, ns, dim_s)
    )

    # Computation in fp32 for numerical stability.
    q_pair = q_pair.float()
    s_pair = s_pair.float()

    cost = squared_euclidean_cost(
        q_pair,
        s_pair,
    )  # [P,Nq,Ns]

    pair_count = cost.shape[0]

    log_a = torch.full(
        (pair_count, nq),
        fill_value=-math.log(float(nq)),
        device=cost.device,
        dtype=cost.dtype,
    )
    log_b = torch.full(
        (pair_count, ns),
        fill_value=-math.log(float(ns)),
        device=cost.device,
        dtype=cost.dtype,
    )

    log_k = -cost / float(reg)

    # Log-domain Sinkhorn scaling.
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(int(n_iter)):
        log_u = log_a - torch.logsumexp(
            log_k + log_v[:, None, :],
            dim=2,
        )
        log_v = log_b - torch.logsumexp(
            log_k + log_u[:, :, None],
            dim=1,
        )

    log_pi = (
        log_k
        + log_u[:, :, None]
        + log_v[:, None, :]
    )

    # Clamp only for exp numerical safety.
    pi = torch.exp(
        log_pi.clamp(min=-80.0, max=20.0)
    )

    transport_cost = (
        pi * cost
    ).sum(dim=(1, 2))

    transport_cost = transport_cost.clamp_min(
        SINKHORN_EPS
    )

    if RETURN_W2_SQRT:
        distance = transport_cost.sqrt()
    else:
        distance = transport_cost

    return distance.reshape(
        q_count,
        s_count,
    )


# =============================================================================
# K-shot classification
# =============================================================================

def aggregate_kshot(
    pair_distance: torch.Tensor,
) -> torch.Tensor:
    """
    pair_distance: [Q, way, shot]
    -> class_distance: [Q, way]
    """
    if SHOT_AGG == "min":
        return pair_distance.min(dim=-1).values

    if SHOT_AGG == "mean":
        return pair_distance.mean(dim=-1)

    if SHOT_AGG == "softmin":
        tau = float(SHOT_TAU)
        if tau <= 0:
            raise ValueError(
                "SHOT_TAU must be >0 for softmin"
            )

        return -tau * (
            torch.logsumexp(
                -pair_distance / tau,
                dim=-1,
            )
            - math.log(
                float(pair_distance.shape[-1])
            )
        )

    raise ValueError(
        f"Unsupported SHOT_AGG={SHOT_AGG!r}"
    )


@torch.inference_mode()
def rae_wasserstein_classify(
    support_z: torch.Tensor,
    query_z: torch.Tensor,
    *,
    n_way: int,
    n_shot: int,
) -> Dict[str, torch.Tensor]:
    expected_support = int(n_way) * int(n_shot)
    if support_z.shape[0] != expected_support:
        raise ValueError(
            f"support count={support_z.shape[0]} "
            f"!= n_way*n_shot={expected_support}"
        )

    support_points = rae_latent_to_points(
        support_z
    )
    query_points = rae_latent_to_points(
        query_z
    )

    flat_pair_distance = uniform_sinkhorn_distance(
        query_points,
        support_points,
        reg=SINKHORN_REG,
        n_iter=SINKHORN_ITER,
    )

    pair_distance = flat_pair_distance.reshape(
        query_z.shape[0],
        int(n_way),
        int(n_shot),
    )

    class_distance = aggregate_kshot(
        pair_distance
    )

    pred = class_distance.argmin(
        dim=-1
    )

    return {
        "pair_distance": pair_distance,
        "class_distance": class_distance,
        "pred": pred,
    }


# =============================================================================
# Metrics
# =============================================================================

def ci95_from_binary(
    correct: int,
    total: int,
) -> float:
    if total <= 0:
        return 0.0

    p = correct / total
    return 1.96 * math.sqrt(
        max(
            p * (1.0 - p),
            0.0,
        )
        / total
    )


def finalize_accumulator(
    acc: Accumulator,
) -> Dict[str, float]:
    n = max(acc.total, 1)
    accuracy = acc.correct / n
    ci95 = ci95_from_binary(
        acc.correct,
        acc.total,
    )

    return {
        "correct": int(acc.correct),
        "total": int(acc.total),
        "accuracy": float(accuracy),
        "accuracy_percent": float(
            100.0 * accuracy
        ),
        "ci95": float(ci95),
        "ci95_percent": float(
            100.0 * ci95
        ),
        "pos_class_distance": float(
            acc.pos_sum / n
        ),
        "nearest_neg_class_distance": float(
            acc.neg_sum / n
        ),
        "class_margin": float(
            acc.margin_sum / n
        ),
        "mean_positive_pair_distance": float(
            acc.pair_pos_sum / n
        ),
        "mean_nearest_negative_pair_distance": float(
            acc.pair_neg_sum / n
        ),
    }


# =============================================================================
# Evaluation
# =============================================================================

@torch.inference_mode()
def evaluate_task(
    *,
    hospital: str,
    part: str,
    rae: nn.Module,
    device: torch.device,
    transform,
) -> Dict[str, Any]:
    (
        query_pool,
        support_pool,
        labels,
        query_root,
        support_root,
    ) = prepare_task_pools(
        hospital,
        part,
        eval_setting=EVAL_SETTING,
        n_shot=N_SHOT,
    )

    effective_way = min(
        int(N_WAY),
        len(labels),
    )

    task_seed = (
        SEED
        + sum(ord(c) for c in hospital) * 1009
        + sum(ord(c) for c in part) * 9176
    )

    plans = generate_episode_plans(
        query_pool,
        support_pool,
        labels,
        n_way=effective_way,
        n_shot=N_SHOT,
        n_episodes=EPISODES_PER_TASK,
        seed=task_seed,
    )

    print("=" * 110)
    print(
        f"[Direct RAE Wasserstein] {hospital} / {part} | "
        f"{effective_way}-way {N_SHOT}-shot {N_QUERY}-query | "
        f"episodes={len(plans)}"
    )
    print(f"query_root   = {query_root}")
    print(f"support_root = {support_root}")
    print(f"valid classes = {len(labels)}")
    print("MetricEncoder/FSL model = DISABLED")
    print(
        "RAE latent -> spatial empirical points -> "
        f"{'L2 normalize -> ' if NORMALIZE_RAE_POINTS else ''}"
        "uniform-mass Sinkhorn Wasserstein"
    )
    print(
        f"Sinkhorn(reg={SINKHORN_REG}, iter={SINKHORN_ITER}, "
        f"pool_factor={OT_POOL_FACTOR}, sqrt_W2={RETURN_W2_SQRT})"
    )
    print(
        f"K-shot aggregation = {SHOT_AGG}, "
        f"tau={SHOT_TAU if SHOT_AGG == 'softmin' else 'N/A'}"
    )
    print("=" * 110)

    acc = Accumulator()
    episode_records: List[Dict[str, Any]] = []

    t0 = time.time()

    for ep_idx, ep in enumerate(plans):
        support_flat: List[str] = []
        for class_name in ep.candidate_labels:
            support_flat.extend(
                ep.support_paths[class_name]
            )

        all_paths = support_flat + [
            ep.query_path
        ]

        images = load_images(
            all_paths,
            transform,
        ).to(
            device,
            non_blocking=True,
        )

        latent = encode_rae_chunked(
            rae,
            images,
            RAE_CHUNK_SIZE,
        )

        support_z = latent[
            :len(support_flat)
        ]
        query_z = latent[
            len(support_flat):
        ]

        gt_index = ep.candidate_labels.index(
            ep.query_label
        )

        output = rae_wasserstein_classify(
            support_z,
            query_z,
            n_way=effective_way,
            n_shot=N_SHOT,
        )

        pair_d = output[
            "pair_distance"
        ][0]  # [way, shot]
        class_d = output[
            "class_distance"
        ][0]  # [way]

        pred_idx = int(
            output["pred"][0].item()
        )
        pred_label = ep.candidate_labels[
            pred_idx
        ]

        pos = float(
            class_d[gt_index].item()
        )

        neg_mask = torch.ones(
            effective_way,
            dtype=torch.bool,
            device=class_d.device,
        )
        neg_mask[gt_index] = False

        nearest_neg = float(
            class_d[neg_mask].min().item()
        )
        margin = nearest_neg - pos

        pair_pos = float(
            pair_d[gt_index].mean().item()
        )

        pair_neg_mask = torch.ones_like(
            pair_d,
            dtype=torch.bool,
        )
        pair_neg_mask[gt_index] = False

        pair_neg = float(
            pair_d.masked_select(
                pair_neg_mask
            ).min().item()
        )

        acc.total += 1
        acc.correct += int(
            pred_label == ep.query_label
        )
        acc.pos_sum += pos
        acc.neg_sum += nearest_neg
        acc.margin_sum += margin
        acc.pair_pos_sum += pair_pos
        acc.pair_neg_sum += pair_neg

        if SAVE_EPISODE_DETAILS:
            episode_records.append(
                {
                    "episode_id": int(
                        ep.episode_id
                    ),
                    "query_label": ep.query_label,
                    "pred_label": pred_label,
                    "correct": bool(
                        pred_label == ep.query_label
                    ),
                    "candidate_labels": list(
                        ep.candidate_labels
                    ),
                    "query_path": ep.query_path,
                    "support_paths_by_class": {
                        c: list(
                            ep.support_paths[c]
                        )
                        for c in ep.candidate_labels
                    },
                    "class_distances": {
                        name: float(
                            class_d[i].item()
                        )
                        for i, name in enumerate(
                            ep.candidate_labels
                        )
                    },
                    "pair_distances_by_class": {
                        name: [
                            float(v)
                            for v in pair_d[i]
                            .detach()
                            .cpu()
                            .tolist()
                        ]
                        for i, name in enumerate(
                            ep.candidate_labels
                        )
                    },
                    "pos_class_distance": pos,
                    "nearest_neg_class_distance": nearest_neg,
                    "margin": margin,
                }
            )

        if (
            LOG_EVERY > 0
            and (ep_idx + 1) % LOG_EVERY == 0
        ):
            running_acc = (
                100.0
                * acc.correct
                / max(acc.total, 1)
            )
            print(
                f"  [{ep_idx+1:04d}/{len(plans):04d}] "
                f"acc={running_acc:.2f}%"
            )

        del (
            images,
            latent,
            support_z,
            query_z,
            output,
            pair_d,
            class_d,
        )

    elapsed = time.time() - t0
    metrics = finalize_accumulator(
        acc
    )

    result: Dict[str, Any] = {
        "hospital": hospital,
        "part": part,
        "eval_setting": EVAL_SETTING,
        "query_root": query_root,
        "support_root": support_root,
        "valid_classes": labels,
        "protocol": {
            "n_way": effective_way,
            "n_shot": N_SHOT,
            "n_query_total": N_QUERY,
            "episodes": EPISODES_PER_TASK,
            "seed": task_seed,
            "query_excluded_from_support": True,
        },
        "representation": {
            "encoder": "frozen RAE only",
            "metric_encoder": False,
            "fsl_model": False,
            "latent_shape_expected": "[B,1024,16,16]",
            "point_representation": (
                "[B,H*W,C] from raw RAE latent spatial positions"
            ),
        },
        "wasserstein": {
            "type": (
                "uniform-mass entropic Sinkhorn W2"
                if RETURN_W2_SQRT
                else "uniform-mass entropic Sinkhorn transport cost"
            ),
            "metric_model_used": False,
            "dynamic_sk_mass_used": False,
            "point_l2_normalization": NORMALIZE_RAE_POINTS,
            "ground_cost": (
                "squared Euclidean on "
                + (
                    "L2-normalized RAE spatial points"
                    if NORMALIZE_RAE_POINTS
                    else "raw RAE spatial points"
                )
            ),
            "sinkhorn_reg": SINKHORN_REG,
            "sinkhorn_iter": SINKHORN_ITER,
            "ot_pool_factor": OT_POOL_FACTOR,
            "return_w2_sqrt": RETURN_W2_SQRT,
            "kshot_aggregation": SHOT_AGG,
            "shot_tau": (
                SHOT_TAU
                if SHOT_AGG == "softmin"
                else None
            ),
        },
        "metrics": metrics,
        "elapsed_sec": elapsed,
    }

    if SAVE_EPISODE_DETAILS:
        result["episodes"] = episode_records

    return result


# =============================================================================
# Output
# =============================================================================

def save_summary_csv(
    path: str,
    rows: List[Dict[str, Any]],
) -> None:
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    fields = [
        "hospital",
        "part",
        "eval_setting",
        "n_way",
        "n_shot",
        "episodes",
        "correct",
        "accuracy_percent",
        "ci95_percent",
        "pos_class_distance",
        "nearest_neg_class_distance",
        "class_margin",
        "mean_positive_pair_distance",
        "mean_nearest_negative_pair_distance",
    ]

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fields
                }
            )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if EVAL_SETTING not in {
        "test_train",
        "test_test",
    }:
        raise ValueError(
            f"Bad EVAL_SETTING={EVAL_SETTING!r}"
        )

    if N_QUERY != 1:
        raise ValueError(
            "This evaluator currently follows the canonical "
            "1-query episode protocol. Keep N_QUERY=1."
        )

    if (
        N_WAY < 2
        or N_SHOT < 1
        or EPISODES_PER_TASK < 1
    ):
        raise ValueError(
            "Bad N_WAY/N_SHOT/EPISODES_PER_TASK"
        )

    if OT_POOL_FACTOR < 1:
        raise ValueError(
            "OT_POOL_FACTOR must be >=1"
        )

    setup_seed(SEED)
    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 110)
    print("[Direct RAE Wasserstein Evaluation]")
    print(f"device       = {device}")
    print(f"hospitals    = {HOSPITALS}")
    print(f"parts        = {PARTS}")
    print(
        f"protocol     = up to {N_WAY}-way "
        f"{N_SHOT}-shot 1-query, "
        f"{EPISODES_PER_TASK} episodes/task"
    )
    print(f"setting      = {EVAL_SETTING}")
    print("MetricEncoder= DISABLED")
    print("FSL model    = DISABLED")
    print(f"result_dir   = {RESULT_DIR}")
    print("=" * 110)

    rae = build_rae(
        PROJECT_ROOT,
        device,
    )
    transform = build_transform(
        IMAGE_SIZE
    )

    summary_rows: List[Dict[str, Any]] = []
    summary_nested: Dict[str, Any] = {}

    for part in PARTS:
        for hospital in HOSPITALS:
            result = evaluate_task(
                hospital=hospital,
                part=part,
                rae=rae,
                device=device,
                transform=transform,
            )

            out_json = os.path.join(
                RESULT_DIR,
                part,
                (
                    f"{hospital}_{part}_{EVAL_SETTING}"
                    "_rae_wasserstein.json"
                ),
            )
            save_json(
                out_json,
                result,
            )

            metrics = result["metrics"]
            effective_way = result[
                "protocol"
            ]["n_way"]

            summary_nested.setdefault(
                part,
                {},
            )[hospital] = metrics

            print(
                f"\n[RESULT] {hospital} / {part} | "
                f"{effective_way}-way"
            )
            print(
                f"  acc={metrics['accuracy_percent']:.2f}% "
                f"±{metrics['ci95_percent']:.2f}% "
                f"margin={metrics['class_margin']:.6f}"
            )

            summary_rows.append(
                {
                    "hospital": hospital,
                    "part": part,
                    "eval_setting": EVAL_SETTING,
                    "n_way": effective_way,
                    "n_shot": N_SHOT,
                    "episodes": EPISODES_PER_TASK,
                    "correct": metrics["correct"],
                    "accuracy_percent": metrics[
                        "accuracy_percent"
                    ],
                    "ci95_percent": metrics[
                        "ci95_percent"
                    ],
                    "pos_class_distance": metrics[
                        "pos_class_distance"
                    ],
                    "nearest_neg_class_distance": metrics[
                        "nearest_neg_class_distance"
                    ],
                    "class_margin": metrics[
                        "class_margin"
                    ],
                    "mean_positive_pair_distance": metrics[
                        "mean_positive_pair_distance"
                    ],
                    "mean_nearest_negative_pair_distance": metrics[
                        "mean_nearest_negative_pair_distance"
                    ],
                }
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()

    save_summary_csv(
        os.path.join(
            RESULT_DIR,
            "summary.csv",
        ),
        summary_rows,
    )

    save_json(
        os.path.join(
            RESULT_DIR,
            "summary.json",
        ),
        {
            "config": {
                "hospitals": HOSPITALS,
                "parts": PARTS,
                "eval_setting": EVAL_SETTING,
                "n_way_max": N_WAY,
                "n_shot": N_SHOT,
                "n_query": N_QUERY,
                "episodes_per_task": EPISODES_PER_TASK,
                "seed": SEED,
                "metric_encoder_used": False,
                "fsl_model_used": False,
                "wasserstein": {
                    "uniform_mass": True,
                    "dynamic_sk": False,
                    "point_l2_normalization": NORMALIZE_RAE_POINTS,
                    "sinkhorn_reg": SINKHORN_REG,
                    "sinkhorn_iter": SINKHORN_ITER,
                    "ot_pool_factor": OT_POOL_FACTOR,
                    "return_w2_sqrt": RETURN_W2_SQRT,
                    "shot_agg": SHOT_AGG,
                    "shot_tau": (
                        SHOT_TAU
                        if SHOT_AGG == "softmin"
                        else None
                    ),
                },
            },
            "results": summary_nested,
        },
    )

    print("\n" + "=" * 110)
    print("[DONE]")
    print(
        "summary CSV  = "
        + os.path.join(
            RESULT_DIR,
            "summary.csv",
        )
    )
    print(
        "summary JSON = "
        + os.path.join(
            RESULT_DIR,
            "summary.json",
        )
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
