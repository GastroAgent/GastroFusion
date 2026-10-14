#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAE + MetricEncoder counterpart of eval_1.py (test-test episodic evaluation).

Preserved task semantics:
    support and query come from the same class-folder pool
    N-way K-shot episodic sampling
    query is excluded from support within each episode
    shot-major support order
    per-episode SFC + DeepEMD EMD matching

Only the feature extraction backend changes:
    image -> frozen RAE -> MetricEncoder -> dense local feature map.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm
from PIL import Image
from scipy import stats
from torchvision import transforms


# =============================================================================
# USER CONFIG
# =============================================================================

PROJECT_ROOT = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow"
DEEPEMD_ROOT = "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/DeepEMD"

# Protocol aligned to eval_1.py, extended to batch hospital x organ evaluation:
#   data pool = DATA_BASE/{hospital}/{organ}/[test]/class/image
#   query and support are sampled from the SAME current hospital-organ pool.
#   The selected query image is strictly excluded from support in every episode.
DATA_BASE = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/data/data_tsy_827_doctor_test_split/indomain_support_76_test"

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "fsl_multi_vae/rae_flow_metric_v1",
    "results_rae_metric_sfc_test_test_batch",
    "食管"
)

# Batch setting: 3 external hospitals x 3 organs.
HOSPITALS = ["华西", "自贡", "苍溪", "南充"]
PARTS = [
    "食管", 
    # "胃", 
    # "十二指肠"
]

# Use the same organ-wise maximum way convention as the other batch evaluators.
# Actual way is automatically reduced when a hospital-organ pool has fewer classes.
ORGAN_CONFIG = {
    "食管": {"way": 8},
    "胃": {"way": 8},
    "十二指肠": {"way": 7},
}

# MetricEncoder checkpoints. Multiple stages are allowed; all stages consume the
# same episode and the same frozen RAE latent tensor.
STAGE_CHECKPOINTS = {
    "M2_best": os.path.join(
        PROJECT_ROOT, "weights_metric_v1", "m2_食管-test-test", "metric_direct_episode.pt"
    ),
}

N_SHOT = 5
N_QUERY = 1  # IMPORTANT: query images PER GT CLASS, matching eval_huaxi_support.py
EPISODES_PER_TASK = 100
SEED = 1
IMAGE_SIZE = 256
RAE_CHUNK_SIZE = 16  # faster default; automatically halves on CUDA OOM

# Spatial pooling applied AFTER MetricEncoder and BEFORE SFC/DeepEMD.
# 1 = disabled (exactly preserves the current feature map).
# 2 = 16x16 -> 8x8 (256 -> 64 local tokens); 4 = 16x16 -> 4x4.
# This mainly reduces repeated SFC/EMD cost, not RAE/Metric encoding cost.
LOCAL_POOL = 1

# -------------------------------------------------------------------------
# Speed / profiling options. These do NOT change SFC mathematics.
# -------------------------------------------------------------------------
# Only evaluate shared checkpoints plus the checkpoint specialized for the
# current organ. Example: 胃 will not waste time evaluating M2_食管.
FILTER_ORGAN_SPECIFIC_STAGES = True

# Before the episode loop, encode every image actually referenced by the
# generated episode plans once: image -> RAE -> each MetricEncoder -> dense map.
# Dense maps are cached on CPU in float32 and reused by every episode.
PRECOMPUTE_DENSE_FEATURES = True
PRECOMPUTE_IMAGE_BATCH = 32
IMAGE_IO_WORKERS = 8

# Keep cached dense maps on GPU to avoid CPU->GPU copies in every episode.
# Set False only if GPU memory becomes tight. This does NOT change the math.
CACHE_FEATURES_ON_GPU = True

# Progress visibility. DeepEMD-like episode information is printed every episode.
PRECOMPUTE_LOG_EVERY = 1

# Detailed per-episode JSON is large and slow to serialize. Accuracy/statistics
# are unaffected when this is False. Turn it back on only when debugging.
SAVE_EPISODE_DETAILS = False

# Print progress + elapsed/ETA. First episode is also reported so that a rough
# total runtime becomes visible quickly.
LOG_EVERY = 1
PROFILE_TIMING = True

# DeepEMD/SFC settings corresponding to eval_1.py.
DEEPEMD_TEMPERATURE = 12.5
DEEPEMD_METRIC = "cosine"
DEEPEMD_NORM = "center"
DEEPEMD_MODE = "fcn"
DEEPEMD_SOLVER = "opencv"
DEEPEMD_NUM_PATCH = 9
DEEPEMD_PATCH_LIST = [2, 3]
DEEPEMD_PATCH_RATIO = 2.0
DEEPEMD_FEATURE_PYRAMID = None
# SFC COST NOTE: calls/episode ~= update_step * ceil(way*shot / bs).
# If speed matters more than SFC adaptation, set USE_SFC=False; the fallback is
# class-wise shot mean (one prototype per class), never a global mean.
USE_SFC = True
SFC_LR = 0.1
SFC_WD = 0.0
SFC_UPDATE_STEP = 10
SFC_BS = 4

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# Imports
# =============================================================================

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if DEEPEMD_ROOT not in sys.path:
    sys.path.insert(0, DEEPEMD_ROOT)
sys.path.insert(0, "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae")
from rae_flow_metric_v1.models.metric_encoder import MetricEncoder
from Models.models.Network import DeepEMD


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class EpisodePlan:
    episode_id: int
    episode_classes: List[str]
    gt_classes: List[str]
    query_paths_by_class: Dict[str, List[str]]
    support_paths_by_class: Dict[str, List[str]]


@dataclass
class StageAccumulator:
    query_correct: int = 0
    query_total: int = 0
    episode_accuracies: List[float] = field(default_factory=list)
    intra_distances: List[float] = field(default_factory=list)
    inter_distances: List[float] = field(default_factory=list)
    latency_ms: List[float] = field(default_factory=list)
    feature_fetch_ms: List[float] = field(default_factory=list)
    sfc_ms: List[float] = field(default_factory=list)
    matching_ms: List[float] = field(default_factory=list)
    confusion: Dict[Tuple[str, str], int] = field(default_factory=dict)


# =============================================================================
# Basic helpers
# =============================================================================

def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def as_args_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def safe_norm(path: str) -> str:
    """Fast path identity key with no realpath/stat calls on network storage."""
    return os.path.normcase(
        os.path.abspath(os.path.normpath(os.path.expanduser(str(path))))
    )


def sample_n(rng: random.Random, candidates: Sequence[str], n: int) -> List[str]:
    candidates = list(candidates)
    if not candidates:
        return []
    if len(candidates) >= n:
        return rng.sample(candidates, n)
    return [rng.choice(candidates) for _ in range(n)]


def looks_like_class_root(root: Path) -> bool:
    if not root.is_dir():
        return False
    for cdir in root.iterdir():
        if not cdir.is_dir() or cdir.name.startswith("."):
            continue
        for p in cdir.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                return True
    return False


def resolve_split_root(base: str | Path, preferred_split: str) -> Path:
    base = Path(base)
    if not base.is_dir():
        raise FileNotFoundError(f"基础目录不存在: {base}")
    if base.name == preferred_split and looks_like_class_root(base):
        return base
    split_root = base / preferred_split
    if split_root.is_dir() and looks_like_class_root(split_root):
        return split_root
    if looks_like_class_root(base):
        return base
    raise RuntimeError(
        f"无法解析类别目录: base={base}, preferred_split={preferred_split}; "
        f"既没有 {split_root}/类别/图像，也没有 {base}/类别/图像"
    )


def resolve_dataset_root(hospital: str, organ: str) -> Path:
    """Resolve DATA_BASE/{hospital}/{organ}/[test]/class/image."""
    return resolve_split_root(Path(DATA_BASE) / hospital / organ, "test")


def resolve_query_root(hospital: str, organ: str) -> Path:
    # eval_1 protocol: query comes from the same hospital-organ test pool.
    return resolve_dataset_root(hospital, organ)


def resolve_support_root(hospital: str, organ: str) -> Path:
    # eval_1 protocol: support comes from exactly the same pool as query.
    return resolve_dataset_root(hospital, organ)


def list_class_images(root: str | Path) -> Dict[str, List[str]]:
    """Match eval_huaxi_support.py: direct class folders and direct image files."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root}")
    out: Dict[str, List[str]] = {}
    for cdir in sorted(root.iterdir(), key=lambda p: p.name):
        if not cdir.is_dir() or cdir.name.startswith("."):
            continue
        imgs = sorted(
            str(p)
            for p in cdir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if imgs:
            out[cdir.name] = imgs
    return out


# =============================================================================
# Episode construction -- intentionally mirrors eval_huaxi_support.py
# =============================================================================

def build_episode_plan(
    *,
    rng: random.Random,
    episode_id: int,
    query_files: Dict[str, List[str]],
    support_files: Dict[str, List[str]],
    query_classes: Sequence[str],
    candidate_pool: Sequence[str],
    way_eff: int,
    n_shot: int,
    n_query_per_gt_class: int,
) -> EpisodePlan:
    query_classes = list(query_classes)
    candidate_pool = list(candidate_pool)

    # First guarantee GT/query classes in the episode, then add support-only
    # distractors if the requested way is wider than the query-class set.
    if len(query_classes) > way_eff:
        gt_in_episode = rng.sample(query_classes, way_eff)
        distractors: List[str] = []
    else:
        gt_in_episode = list(query_classes)
        n_distract = way_eff - len(gt_in_episode)
        gt_set = set(gt_in_episode)
        distractor_pool = [c for c in candidate_pool if c not in gt_set]
        distractors = rng.sample(distractor_pool, n_distract) if n_distract > 0 else []

    episode_classes = gt_in_episode + distractors
    rng.shuffle(episode_classes)

    # Query is sampled for EVERY GT class in this episode, not one random class.
    query_paths_by_class: Dict[str, List[str]] = {}
    for c in gt_in_episode:
        query_paths_by_class[c] = sample_n(rng, query_files[c], n_query_per_gt_class)

    # Strictly exclude any selected query path from support whenever the roots
    # overlap. If fewer than shot remain, sample WITH replacement from the
    # remaining support, exactly like eval_huaxi_support.py.
    support_paths_by_class: Dict[str, List[str]] = {}
    for c in episode_classes:
        support_candidates = list(support_files[c])
        if c in query_paths_by_class:
            query_norm = {safe_norm(p) for p in query_paths_by_class[c]}
            filtered = [p for p in support_candidates if safe_norm(p) not in query_norm]
            if len(filtered) < len(support_candidates):
                if not filtered:
                    raise RuntimeError(
                        f"episode={episode_id} 排除 query 后 class={c} 没有可用 support"
                    )
                support_candidates = filtered
        sampled = sample_n(rng, support_candidates, n_shot)
        if len(sampled) != n_shot:
            raise RuntimeError(
                f"episode={episode_id} support 抽样失败: class={c}, need={n_shot}"
            )
        support_paths_by_class[c] = sampled

    return EpisodePlan(
        episode_id=episode_id,
        episode_classes=list(episode_classes),
        gt_classes=list(gt_in_episode),
        query_paths_by_class=query_paths_by_class,
        support_paths_by_class=support_paths_by_class,
    )


def generate_episode_plans(
    *,
    query_files: Dict[str, List[str]],
    support_files: Dict[str, List[str]],
    query_classes: Sequence[str],
    candidate_pool: Sequence[str],
    way_eff: int,
    n_episodes: int,
    seed: int,
) -> List[EpisodePlan]:
    rng = random.Random(seed)
    return [
        build_episode_plan(
            rng=rng,
            episode_id=i,
            query_files=query_files,
            support_files=support_files,
            query_classes=query_classes,
            candidate_pool=candidate_pool,
            way_eff=way_eff,
            n_shot=N_SHOT,
            n_query_per_gt_class=N_QUERY,
        )
        for i in range(n_episodes)
    ]


# =============================================================================
# Frozen RAE
# =============================================================================

def build_rae(project_root: str, device: torch.device):
    from rae.rae import RAE

    model = RAE(
        encoder_name="dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]",
        resolution=IMAGE_SIZE,
        decoder_config_path=os.path.join(project_root, "rae/configs/decoder/ViTXL/config.json"),
        pretrained_decoder_path=os.path.join(project_root, "flow_matcher_otcfm/rae_v2/decoder.pt"),
        noise_tau=0.0,
        normalization_stat_path=os.path.join(project_root, "flow_matcher_otcfm/rae_v2/stats.pt"),
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_transform():
    # RAE/M2 validation preprocessing, intentionally not the DeepEMD encoder
    # transform because the image encoder is the only part that differs.
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


def _load_one_image(path: str, transform) -> torch.Tensor:
    with Image.open(path) as im:
        return transform(im.convert("RGB"))


def load_images(paths: Sequence[str], transform) -> torch.Tensor:
    paths = list(paths)
    if not paths:
        raise RuntimeError("load_images received an empty path list")
    workers = min(max(1, int(IMAGE_IO_WORKERS)), len(paths))
    if workers <= 1:
        images = [_load_one_image(p, transform) for p in paths]
    else:
        # PIL decode / NFS reads are independent per image. Parallel loading keeps
        # the GPU from waiting on dozens of serial file opens.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            images = list(pool.map(lambda p: _load_one_image(p, transform), paths))
    return torch.stack(images, dim=0)


@torch.no_grad()
def encode_rae_chunked(rae: nn.Module, images: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """RAE encode with an adaptive chunk size.

    We start from RAE_CHUNK_SIZE for speed. If CUDA OOM occurs, the current
    chunk is retried with half the size until it succeeds. Numerical behavior
    of RAE itself is unchanged.
    """
    chunks = []
    start = 0
    current_chunk = max(1, int(chunk_size))
    while start < images.shape[0]:
        take = min(current_chunk, int(images.shape[0] - start))
        try:
            chunks.append(rae.encode(images[start:start + take]))
            start += take
        except torch.cuda.OutOfMemoryError:
            if current_chunk <= 1:
                raise
            current_chunk = max(1, current_chunk // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[RAE OOM fallback] reduce chunk -> {current_chunk} and retry",
                flush=True,
            )
    latent = torch.cat(chunks, dim=0)
    if latent.ndim != 4 or latent.shape[1:] != (1024, 16, 16):
        raise RuntimeError(f"RAE latent expected [B,1024,16,16], got {tuple(latent.shape)}")
    return latent


# =============================================================================
# MetricEncoder checkpoint loading
# =============================================================================

def strip_prefix_if_all(state: Dict[str, torch.Tensor], prefix: str):
    if state and all(str(k).startswith(prefix) for k in state):
        return {str(k)[len(prefix):]: v for k, v in state.items()}
    return state


def extract_metric_state(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise TypeError("checkpoint must be a dict")
    for key in ("metric_encoder", "metric"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return strip_prefix_if_all(value, "module.")
    for key in ("model", "model_state_dict", "state_dict", "params"):
        value = ckpt.get(key)
        if not isinstance(value, dict):
            continue
        state = strip_prefix_if_all(value, "module.")
        metric_prefixed = {
            str(k)[len("metric."):]: v for k, v in state.items() if str(k).startswith("metric.")
        }
        if metric_prefixed:
            return metric_prefixed
        metric_encoder_prefixed = {
            str(k)[len("metric_encoder."):]: v
            for k, v in state.items()
            if str(k).startswith("metric_encoder.")
        }
        if metric_encoder_prefixed:
            return metric_encoder_prefixed
        return state
    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return strip_prefix_if_all(ckpt, "module.")
    raise KeyError("Cannot locate MetricEncoder state in checkpoint")


def resolve_metric_arch(ckpt: Dict[str, Any]) -> Tuple[int, int, float]:
    args = as_args_dict(ckpt.get("args", {}))
    cfg = ckpt.get("metric_config", {})
    cfg = cfg if isinstance(cfg, dict) else {}
    hidden_dim = int(args.get("hidden_dim", args.get("metric_hidden_dim", cfg.get("model_dim", 256))))
    metric_dim = int(args.get("metric_dim", cfg.get("output_dim", 256)))
    dropout = float(args.get("dropout", args.get("metric_dropout", 0.0)))
    return hidden_dim, metric_dim, dropout


def resolve_stage_path(template: str, part: str) -> str:
    return template.format(part=part)


def load_stage_metrics_for_part(
    part: str,
    device: torch.device,
) -> Tuple[Dict[str, nn.Module], Dict[str, Dict[str, Any]]]:
    models: Dict[str, nn.Module] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    for stage, template in STAGE_CHECKPOINTS.items():
        if not template:
            continue

        if FILTER_ORGAN_SPECIFIC_STAGES:
            mentioned_organs = [organ_name for organ_name in PARTS if organ_name in stage]
            if mentioned_organs and part not in mentioned_organs:
                print(
                    f"[Metric Skip] {stage:<18} is organ-specific for {mentioned_organs}; "
                    f"current organ={part}"
                )
                continue

        path = resolve_stage_path(template, part)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{stage} checkpoint not found for part={part}: {path}")
        ckpt = torch_load(path)
        hidden_dim, metric_dim, dropout = resolve_metric_arch(ckpt)
        metric = MetricEncoder(
            in_dim=1024,
            hidden_dim=hidden_dim,
            out_dim=metric_dim,
            dropout=dropout,
        )
        incompatible = metric.load_state_dict(extract_metric_state(ckpt), strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"{stage} MetricEncoder mismatch:\n"
                f"  checkpoint={path}\n"
                f"  missing={list(incompatible.missing_keys)[:20]}\n"
                f"  unexpected={list(incompatible.unexpected_keys)[:20]}"
            )
        metric = metric.to(device).eval().requires_grad_(False)
        models[stage] = metric
        metadata[stage] = {
            "path": path,
            "checkpoint_stage": ckpt.get("stage"),
            "checkpoint_epoch": ckpt.get("epoch"),
            "hidden_dim": hidden_dim,
            "metric_dim": metric_dim,
            "dropout": dropout,
        }
        print(
            f"[Metric Load] {stage:<18} hidden={hidden_dim} metric={metric_dim} "
            f"ckpt_stage={ckpt.get('stage', 'unknown')} path={path}"
        )
    if not models:
        raise RuntimeError("No stage checkpoints configured")
    return models, metadata


@torch.no_grad()
def metric_dense_map(metric: nn.Module, latent: torch.Tensor) -> torch.Tensor:
    """MetricEncoder.raw [B,N,D] -> pooled DeepEMD local map [B,D,H',W'].

    LOCAL_POOL is a spatial compression factor applied only after MetricEncoder:
      1: no pooling, keep the original local map exactly
      2: e.g. 16x16 -> 8x8
      4: e.g. 16x16 -> 4x4

    Pooling here directly reduces the number of local patches seen by SFC/EMD.
    """
    feat = metric(latent)
    raw = feat.raw
    if raw.ndim != 3:
        raise RuntimeError(f"MetricEncoder.raw expected [B,N,D], got {tuple(raw.shape)}")
    b, n, d = raw.shape
    side = int(round(math.sqrt(int(n))))
    if side * side != int(n):
        raise RuntimeError(f"Metric token count must be square, got N={n}")

    dense = raw.transpose(1, 2).contiguous().reshape(b, d, side, side)

    pool = int(LOCAL_POOL)
    if pool < 1:
        raise ValueError(f"LOCAL_POOL must be >= 1, got {LOCAL_POOL}")
    if pool > 1:
        if pool > dense.shape[-2] or pool > dense.shape[-1]:
            raise ValueError(
                f"LOCAL_POOL={pool} is larger than local map size "
                f"{tuple(dense.shape[-2:])}"
            )
        dense = torch.nn.functional.avg_pool2d(
            dense, kernel_size=pool, stride=pool, ceil_mode=False
        )

    return dense.contiguous()




def _cuda_sync(device: torch.device) -> None:
    if PROFILE_TIMING and device.type == "cuda":
        torch.cuda.synchronize(device)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes:d}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h{minutes:02d}m"


class DenseFeatureCache:
    """Cached frozen RAE -> MetricEncoder dense maps.

    The cache is intentionally organ-scoped and survives across the three
    hospitals. Test-Train therefore computes the common Huaxi support features
    only once for an organ. No SFC result is cached: SFC remains episode-specific.
    """

    def __init__(
        self,
        *,
        rae: nn.Module,
        stage_models: Dict[str, nn.Module],
        device: torch.device,
        transform,
    ) -> None:
        self.rae = rae
        self.stage_models = stage_models
        self.device = device
        self.transform = transform
        self.maps: Dict[str, Dict[str, torch.Tensor]] = {
            stage: {} for stage in stage_models
        }
        self.path_by_key: Dict[str, str] = {}
        self.cache_bytes = 0
        self.total_image_io_s = 0.0
        self.total_rae_s = 0.0
        self.total_metric_s: Dict[str, float] = {stage: 0.0 for stage in stage_models}
        self.cache_device = (
            device if (CACHE_FEATURES_ON_GPU and device.type == "cuda") else torch.device("cpu")
        )
        print(f"[Feature Cache Device] {self.cache_device}", flush=True)

    def _missing_paths(self, paths: Sequence[str]) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for path in paths:
            key = safe_norm(path)
            self.path_by_key.setdefault(key, str(path))
            if key in seen:
                continue
            seen.add(key)
            if any(key not in self.maps[stage] for stage in self.stage_models):
                ordered.append(str(path))
        return ordered

    @property
    def n_cached_images(self) -> int:
        if not self.maps:
            return 0
        first_stage = next(iter(self.maps))
        return len(self.maps[first_stage])

    def ensure(self, paths: Sequence[str], *, label: str = "") -> Dict[str, Any]:
        unique_requested = len({safe_norm(p) for p in paths})
        missing = self._missing_paths(paths)
        cache_hits = max(0, unique_requested - len(missing))
        prefix = f"[{label}] " if label else ""

        print(
            f"[Feature Cache] {prefix}requested_unique={unique_requested}, "
            f"hit={cache_hits}, miss={len(missing)}, cached_before={self.n_cached_images}"
        )
        if not missing or not PRECOMPUTE_DENSE_FEATURES:
            return {
                "requested_unique": unique_requested,
                "cache_hits": cache_hits,
                "cache_misses": len(missing),
                "precompute_seconds": 0.0,
            }

        t_all = time.perf_counter()
        batch_size = max(1, int(PRECOMPUTE_IMAGE_BATCH))
        n_batches = math.ceil(len(missing) / batch_size)
        local_io = 0.0
        local_rae = 0.0
        local_metric = {stage: 0.0 for stage in self.stage_models}

        for batch_idx, start in enumerate(range(0, len(missing), batch_size), start=1):
            batch_paths = missing[start:start + batch_size]

            t0 = time.perf_counter()
            images_cpu = load_images(batch_paths, self.transform)
            images = images_cpu.to(self.device, non_blocking=True)
            _cuda_sync(self.device)
            io_s = time.perf_counter() - t0
            local_io += io_s
            self.total_image_io_s += io_s

            t0 = time.perf_counter()
            with torch.no_grad():
                latent = encode_rae_chunked(self.rae, images, RAE_CHUNK_SIZE)
            _cuda_sync(self.device)
            rae_s = time.perf_counter() - t0
            local_rae += rae_s
            self.total_rae_s += rae_s

            for stage, metric in self.stage_models.items():
                t0 = time.perf_counter()
                with torch.no_grad():
                    dense = metric_dense_map(metric, latent)
                _cuda_sync(self.device)
                metric_s = time.perf_counter() - t0
                local_metric[stage] += metric_s
                self.total_metric_s[stage] += metric_s

                dense_store = dense.detach().to(self.cache_device, dtype=torch.float32).contiguous()
                for i, path in enumerate(batch_paths):
                    key = safe_norm(path)
                    if key not in self.maps[stage]:
                        tensor = dense_store[i].clone()
                        self.maps[stage][key] = tensor
                        self.cache_bytes += tensor.numel() * tensor.element_size()
                del dense, dense_store

            del images_cpu, images, latent

            if (
                batch_idx == 1
                or batch_idx == n_batches
                or (PRECOMPUTE_LOG_EVERY > 0 and batch_idx % PRECOMPUTE_LOG_EVERY == 0)
            ):
                elapsed = time.perf_counter() - t_all
                avg_batch = elapsed / batch_idx
                eta = avg_batch * (n_batches - batch_idx)
                processed = min(start + len(batch_paths), len(missing))
                img_per_s = processed / max(elapsed, 1e-9)
                metric_batch_s = sum(local_metric.values())
                print(
                    f"  [PRECOMPUTE {batch_idx:03d}/{n_batches:03d}] "
                    f"images={processed:04d}/{len(missing):04d} ({img_per_s:.2f} img/s) | "
                    f"last_io={io_s:.2f}s last_rae={rae_s:.2f}s | "
                    f"elapsed={format_duration(elapsed)} ETA={format_duration(eta)}",
                    flush=True,
                )

        elapsed_all = time.perf_counter() - t_all
        print(
            f"[Feature Cache Done] {prefix}new={len(missing)} images | "
            f"time={format_duration(elapsed_all)} | I/O={format_duration(local_io)} | "
            f"RAE={format_duration(local_rae)} | "
            f"Metric(sum)={format_duration(sum(local_metric.values()))} | "
            f"cache≈{self.cache_bytes / (1024**3):.2f} GiB on {self.cache_device}"
        )
        return {
            "requested_unique": unique_requested,
            "cache_hits": cache_hits,
            "cache_misses": len(missing),
            "precompute_seconds": elapsed_all,
            "image_io_seconds": local_io,
            "rae_seconds": local_rae,
            "metric_seconds_by_stage": local_metric,
            "cache_gib_after": self.cache_bytes / (1024**3),
        }

    def batch(self, stage: str, paths: Sequence[str]) -> torch.Tensor:
        stage_cache = self.maps[stage]
        tensors = []
        missing = []
        for path in paths:
            key = safe_norm(path)
            value = stage_cache.get(key)
            if value is None:
                missing.append(str(path))
            else:
                tensors.append(value)
        if missing:
            raise KeyError(
                f"Feature cache miss for stage={stage}: {missing[:5]} "
                f"(total {len(missing)})"
            )
        batch = torch.stack(tensors, dim=0)
        if batch.device != self.device:
            batch = batch.to(self.device, non_blocking=True)
        return batch


# =============================================================================
# DeepEMD SFC/meta head -- same logic as eval_huaxi_support.py
# =============================================================================

class FlexibleSFCDeepEMD(DeepEMD):
    """DeepEMD SFC with channel count inferred from MetricEncoder feature map."""

    def get_sfc(self, support: torch.Tensor) -> torch.Tensor:
        support = support.squeeze(0)
        if support.ndim != 4:
            raise RuntimeError(f"SFC expects [way*shot,C,H,W], got {tuple(support.shape)}")
        expected = int(self.args.way) * int(self.args.shot)
        if support.shape[0] != expected:
            raise RuntimeError(
                f"SFC support count mismatch: got={support.shape[0]}, expected={expected}"
            )
        c, h, w = support.shape[1:]
        # IMPORTANT: support is shot-major, so view(shot, way, ...) is correct.
        sfc = (
            support.reshape(self.args.shot, self.args.way, c, h, w)
            .mean(dim=0)
            .clone()
            .detach()
        )
        sfc = nn.Parameter(sfc, requires_grad=True)
        optimizer = torch.optim.SGD(
            [sfc],
            lr=float(self.args.sfc_lr),
            momentum=0.9,
            dampening=0.9,
            weight_decay=float(self.args.sfc_wd),
        )
        label_shot = torch.arange(self.args.way, device=support.device).repeat(self.args.shot).long()

        with torch.enable_grad():
            for _ in range(int(self.args.sfc_update_step)):
                random_ids = torch.randperm(expected, device=support.device)
                for start in range(0, expected, int(self.args.sfc_bs)):
                    selected_ids = random_ids[start:min(start + int(self.args.sfc_bs), expected)]
                    batch_support = support[selected_ids]
                    batch_labels = label_shot[selected_ids]
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.emd_forward_1shot(sfc, batch_support.detach())
                    loss = torch.nn.functional.cross_entropy(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
        return sfc.detach()


def build_deepemd_args(way: int) -> argparse.Namespace:
    # Contains the fields used by DeepEMD/SFC while deliberately avoiding the
    # original image encoder as final features come from RAE + MetricEncoder.
    return argparse.Namespace(
        way=int(way),
        shot=int(N_SHOT),
        query=int(N_QUERY),
        dataset="cifar_fs",
        temperature=float(DEEPEMD_TEMPERATURE),
        metric=DEEPEMD_METRIC,
        norm=DEEPEMD_NORM,
        deepemd=DEEPEMD_MODE,
        feature_pyramid=DEEPEMD_FEATURE_PYRAMID,
        num_patch=int(DEEPEMD_NUM_PATCH),
        patch_list=list(DEEPEMD_PATCH_LIST),
        patch_ratio=float(DEEPEMD_PATCH_RATIO),
        solver=DEEPEMD_SOLVER,
        form="L2",
        l2_strength=1e-6,
        sfc_lr=float(SFC_LR),
        sfc_wd=float(SFC_WD),
        sfc_update_step=int(SFC_UPDATE_STEP),
        sfc_bs=int(SFC_BS),
        gpu="0",
        set="test",
        data_dir="",
        model_dir="",
        save_dir=RESULT_DIR,
        support_dir="",
        support_set="train",
        num_class=64,
        backbone="resnet12",
    )


def build_meta_head(way: int, device: torch.device) -> FlexibleSFCDeepEMD:
    args = build_deepemd_args(way)
    try:
        model = FlexibleSFCDeepEMD(args, model_type="latent")
    except TypeError:
        model = FlexibleSFCDeepEMD(args)
    model = model.to(device)
    model.eval()
    # The network's own encoder is not used. Freeze persistent model parameters;
    # get_sfc() creates a temporary per-episode trainable nn.Parameter itself.
    for p in model.parameters():
        p.requires_grad_(False)
    model.args.way = int(way)
    model.args.shot = int(N_SHOT)
    return model


# =============================================================================
# Metrics
# =============================================================================

def confidence_interval(values: Sequence[float]) -> Tuple[float, float]:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    mean = float(values.mean())
    if values.size <= 1:
        return mean, 0.0
    pm = float(1.96 * values.std(ddof=1) / math.sqrt(values.size))
    return mean, pm


def fisher_and_kl(intra: Sequence[float], inter: Sequence[float]) -> Tuple[float, float]:
    intra = np.asarray(list(intra), dtype=np.float64)
    inter = np.asarray(list(inter), dtype=np.float64)
    if intra.size == 0 or inter.size == 0:
        return 0.0, 0.0
    mu_i, std_i = float(intra.mean()), float(intra.std())
    mu_e, std_e = float(inter.mean()), float(inter.std())
    fisher = float((mu_e - mu_i) ** 2 / (std_i ** 2 + std_e ** 2 + 1e-8))
    all_d = np.concatenate([intra, inter])
    if float(all_d.max()) <= float(all_d.min()):
        return fisher, 0.0
    bins = np.linspace(all_d.min(), all_d.max(), 60)
    p_i, _ = np.histogram(intra, bins=bins, density=True)
    p_e, _ = np.histogram(inter, bins=bins, density=True)
    p_i = p_i + 1e-8
    p_e = p_e + 1e-8
    p_i /= p_i.sum()
    p_e /= p_e.sum()
    return fisher, float(stats.entropy(p_e, p_i))


def finalize_stage(acc: StageAccumulator, candidate_pool: Sequence[str]) -> Dict[str, Any]:
    ep_mean, ep_pm = confidence_interval(acc.episode_accuracies)
    query_acc = 100.0 * acc.query_correct / max(acc.query_total, 1)
    fisher, kl = fisher_and_kl(acc.intra_distances, acc.inter_distances)
    latency = np.asarray(acc.latency_ms, dtype=np.float64)
    feature_fetch = np.asarray(acc.feature_fetch_ms, dtype=np.float64)
    sfc_latency = np.asarray(acc.sfc_ms, dtype=np.float64)
    matching_latency = np.asarray(acc.matching_ms, dtype=np.float64)

    per_class_total = {c: 0 for c in candidate_pool}
    per_class_correct = {c: 0 for c in candidate_pool}
    for (gt, pred), count in acc.confusion.items():
        per_class_total[gt] = per_class_total.get(gt, 0) + int(count)
        if gt == pred:
            per_class_correct[gt] = per_class_correct.get(gt, 0) + int(count)
    per_class_acc = {
        c: 100.0 * per_class_correct[c] / per_class_total[c]
        for c in candidate_pool
        if per_class_total[c] > 0
    }

    return {
        "accuracy_percent": float(ep_mean),
        "ci95_percent": float(ep_pm),
        "query_micro_accuracy_percent": float(query_acc),
        "query_correct": int(acc.query_correct),
        "query_total": int(acc.query_total),
        "fisher_ratio": fisher,
        "kl_divergence_inter_intra": kl,
        "latency_mean_ms": float(latency.mean()) if latency.size else 0.0,
        "latency_std_ms": float(latency.std()) if latency.size else 0.0,
        "feature_fetch_mean_ms": float(feature_fetch.mean()) if feature_fetch.size else 0.0,
        "sfc_mean_ms": float(sfc_latency.mean()) if sfc_latency.size else 0.0,
        "matching_mean_ms": float(matching_latency.mean()) if matching_latency.size else 0.0,
        "per_class_accuracy_percent": per_class_acc,
    }


# =============================================================================
# Single hospital-organ evaluation
# =============================================================================

def evaluate_one_dataset(
    *,
    hospital: str,
    organ: str,
    rae: nn.Module,
    stage_models: Dict[str, nn.Module],
    stage_metadata: Dict[str, Dict[str, Any]],
    meta_head: FlexibleSFCDeepEMD,
    feature_cache: DenseFeatureCache,
    device: torch.device,
    transform,
    dataset_seed: int,
) -> Dict[str, Any]:
    dataset_start = time.perf_counter()
    query_root = resolve_query_root(hospital, organ)
    support_root = resolve_support_root(hospital, organ)
    query_files = list_class_images(query_root)
    support_files = list_class_images(support_root)

    candidate_pool = [
        c for c in sorted(query_files.keys())
        if c in support_files and query_files[c] and support_files[c]
    ]
    query_classes = list(candidate_pool)
    if not candidate_pool:
        raise RuntimeError(f"test pool has no usable classes: {query_root}")

    way_requested = int(ORGAN_CONFIG[organ]["way"])
    way_eff = min(way_requested, len(candidate_pool))
    if way_eff < 2:
        raise RuntimeError(f"有效 way 太小: requested={way_requested}, support={len(candidate_pool)}")

    meta_head.args.way = int(way_eff)
    meta_head.args.shot = int(N_SHOT)

    plans = generate_episode_plans(
        query_files=query_files,
        support_files=support_files,
        query_classes=query_classes,
        candidate_pool=candidate_pool,
        way_eff=way_eff,
        n_episodes=EPISODES_PER_TASK,
        seed=dataset_seed,
    )

    print("\n" + "=" * 110)
    print(f"[DATASET] {hospital}-{organ}")
    print(f"query_root       = {query_root}")
    print(f"support_root     = {support_root}")
    print(f"protocol         = {way_eff}-way {N_SHOT}-shot, query/GT-class={N_QUERY}, episodes={EPISODES_PER_TASK}")
    print(f"stages           = {list(stage_models.keys())}")
    print(
        f"SFC              = enabled={USE_SFC and N_SHOT > 1}, lr={SFC_LR}, "
        f"steps={SFC_UPDATE_STEP}, bs={SFC_BS} (UNCHANGED)"
    )
    print("speed            = GPU feature cache + adaptive RAE chunks + per-episode profiling")
    print("=" * 110)
    print("[CLASS MAP]")
    for ci, cname in enumerate(candidate_pool):
        print(f"  Class {ci:02d} -> {cname}")
    sfc_calls = 0
    if USE_SFC and N_SHOT > 1:
        sfc_calls = int(SFC_UPDATE_STEP) * math.ceil((way_eff * N_SHOT) / max(int(SFC_BS), 1))
    print(
        f"[COST HINT] each episode: SFC_EMD_calls≈{sfc_calls} + final_matching; "
        f"USE_SFC={USE_SFC}",
        flush=True,
    )

    # Precompute only images that the deterministic episode plans will actually use.
    planned_paths: List[str] = []
    for ep in plans:
        for values in ep.support_paths_by_class.values():
            planned_paths.extend(values)
        for values in ep.query_paths_by_class.values():
            planned_paths.extend(values)
    cache_stats = feature_cache.ensure(planned_paths, label=f"{hospital}-{organ}")

    accumulators = {stage: StageAccumulator() for stage in stage_models}
    episode_records: List[Dict[str, Any]] = []
    loop_start = time.perf_counter()

    for ep_idx, ep in enumerate(plans):
        support_flat: List[str] = []
        for s in range(N_SHOT):
            for c in ep.episode_classes:
                support_flat.append(ep.support_paths_by_class[c][s])

        query_flat: List[str] = []
        query_targets: List[int] = []
        query_labels: List[str] = []
        for ci, c in enumerate(ep.episode_classes):
            if c not in ep.query_paths_by_class:
                continue
            for p in ep.query_paths_by_class[c]:
                query_flat.append(p)
                query_targets.append(ci)
                query_labels.append(c)

        if not query_flat:
            raise RuntimeError(f"episode={ep_idx} contains no query images")

        all_paths = support_flat + query_flat
        support_count = len(support_flat)
        target = torch.tensor(query_targets, dtype=torch.long, device=device)

        ep_record: Dict[str, Any] = {
            "episode_id": int(ep.episode_id),
            "episode_classes": list(ep.episode_classes),
            "gt_classes": list(ep.gt_classes),
            "query_paths_by_class": {k: list(v) for k, v in ep.query_paths_by_class.items()},
            "support_paths_by_class": {k: list(v) for k, v in ep.support_paths_by_class.items()},
            "query_local_targets": list(query_targets),
            "stage_results": {},
        }

        episode_stage_status: List[str] = []
        for stage, _metric in stage_models.items():
            t_stage = time.perf_counter()

            t0 = time.perf_counter()
            dense = feature_cache.batch(stage, all_paths)
            _cuda_sync(device)
            fetch_ms = (time.perf_counter() - t0) * 1000.0
            support_feat = dense[:support_count]
            query_feat = dense[support_count:]

            if ep_idx == 0:
                patch_count = int(support_feat.shape[-2] * support_feat.shape[-1])
                print(
                    f"[FEATURE SHAPE] stage={stage} support={tuple(support_feat.shape)} "
                    f"query={tuple(query_feat.shape)} local_patches={patch_count}",
                    flush=True,
                )

            if support_feat.shape[0] != way_eff * N_SHOT:
                raise RuntimeError(
                    f"{stage}: support feature count={support_feat.shape[0]} "
                    f"!= way*shot={way_eff*N_SHOT}"
                )

            sfc_seed = int(dataset_seed) + int(ep_idx) * 100003
            torch.manual_seed(sfc_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sfc_seed)

            t0 = time.perf_counter()
            if N_SHOT > 1 and USE_SFC:
                support_proto = meta_head.get_sfc(support_feat)
            elif N_SHOT > 1:
                # No-SFC baseline: preserve one prototype per class.
                # support_feat is shot-major [shot*way,C,H,W].
                c, h, w = support_feat.shape[1:]
                support_proto = (
                    support_feat.reshape(N_SHOT, way_eff, c, h, w)
                    .mean(dim=0)
                    .contiguous()
                )
            else:
                support_proto = support_feat
            _cuda_sync(device)
            sfc_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = meta_head.emd_forward_1shot(support_proto, query_feat)
            _cuda_sync(device)
            matching_ms = (time.perf_counter() - t0) * 1000.0

            if logits.shape != (len(query_flat), way_eff):
                raise RuntimeError(
                    f"{stage}: logits shape={tuple(logits.shape)} expected={(len(query_flat), way_eff)}"
                )

            pred = logits.argmax(dim=1)
            n_correct = int(pred.eq(target).sum().item())
            n_query = int(target.numel())
            ep_acc = 100.0 * n_correct / max(n_query, 1)

            acc = accumulators[stage]
            acc.query_correct += n_correct
            acc.query_total += n_query
            acc.episode_accuracies.append(ep_acc)
            acc.feature_fetch_ms.append(fetch_ms)
            acc.sfc_ms.append(sfc_ms)
            acc.matching_ms.append(matching_ms)
            stage_total_ms = (time.perf_counter() - t_stage) * 1000.0
            acc.latency_ms.append(stage_total_ms)
            running_mean, running_ci = confidence_interval(acc.episode_accuracies)
            episode_stage_status.append(
                f"{stage}: this={ep_acc:.2f}% avg={running_mean:.2f}+/-{running_ci:.2f}% "
                f"[fetch={fetch_ms:.1f}ms SFC={sfc_ms/1000:.2f}s "
                f"match={matching_ms/1000:.2f}s total={stage_total_ms/1000:.2f}s]"
            )

            dist = (-logits / DEEPEMD_TEMPERATURE).detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            for qi in range(dist.shape[0]):
                true_local = int(target_np[qi])
                gt_name = ep.episode_classes[true_local]
                pred_name = ep.episode_classes[int(pred_np[qi])]
                acc.confusion[(gt_name, pred_name)] = acc.confusion.get((gt_name, pred_name), 0) + 1
                for cls_idx in range(dist.shape[1]):
                    value = float(dist[qi, cls_idx])
                    if cls_idx == true_local:
                        acc.intra_distances.append(value)
                    else:
                        acc.inter_distances.append(value)

            if SAVE_EPISODE_DETAILS:
                pred_list = [int(x) for x in pred.detach().cpu().tolist()]
                ep_record["stage_results"][stage] = {
                    "accuracy_percent": ep_acc,
                    "correct": n_correct,
                    "total_queries": n_query,
                    "predicted_local": pred_list,
                    "predicted_labels": [ep.episode_classes[x] for x in pred_list],
                    "query_labels": list(query_labels),
                    "logits": logits.detach().cpu().tolist(),
                    "timing_ms": {
                        "feature_fetch": fetch_ms,
                        "sfc": sfc_ms,
                        "matching": matching_ms,
                    },
                }

            del dense, support_feat, query_feat, support_proto, logits

        if SAVE_EPISODE_DETAILS:
            episode_records.append(ep_record)

        should_log = LOG_EVERY > 0 and ((ep_idx + 1) % LOG_EVERY == 0)
        if should_log:
            done = ep_idx + 1
            elapsed = time.perf_counter() - loop_start
            avg_ep = elapsed / done
            eta = avg_ep * (len(plans) - done)
            print(
                f"[EPISODE {done:03d}/{len(plans):03d}] {hospital}-{organ} | "
                + " | ".join(episode_stage_status)
                + f" | elapsed={format_duration(elapsed)} ETA={format_duration(eta)}",
                flush=True,
            )

    metrics = {stage: finalize_stage(acc, candidate_pool) for stage, acc in accumulators.items()}
    dataset_seconds = time.perf_counter() - dataset_start

    result: Dict[str, Any] = {
        "hospital": hospital,
        "organ": organ,
        "protocol": "test-test; support/query sampled from same pool with within-episode exclusion",
        "query_root": str(query_root),
        "support_root": str(support_root),
        "way_requested": way_requested,
        "way_effective": way_eff,
        "shot": N_SHOT,
        "query_per_gt_class": N_QUERY,
        "episodes": EPISODES_PER_TASK,
        "seed": dataset_seed,
        "query_class_count": len(query_files),
        "support_class_count": len(support_files),
        "shared_gt_class_count": len(query_classes),
        "query_classes": sorted(query_files.keys()),
        "support_classes": candidate_pool,
        "shared_gt_classes": query_classes,
        "episode_construction": {
            "gt_first": True,
            "support_only_distractors_fill_remaining_way": False,
            "query_for_every_gt_class": True,
            "query_excluded_from_support_if_paths_overlap": True,
            "support_order": "shot-major",
        },
        "encoder": "frozen RAE -> stage-specific MetricEncoder.raw -> spatial pool -> dense local map",
        "local_pool": int(LOCAL_POOL),
        "speed_optimization": {
            "filter_organ_specific_stages": FILTER_ORGAN_SPECIFIC_STAGES,
            "precompute_dense_features": PRECOMPUTE_DENSE_FEATURES,
            "precompute_image_batch": PRECOMPUTE_IMAGE_BATCH,
            "image_io_workers": IMAGE_IO_WORKERS,
            "cache_features_on_gpu": CACHE_FEATURES_ON_GPU,
            "local_pool": int(LOCAL_POOL),
            "rae_chunk_size_initial": RAE_CHUNK_SIZE,
            "cache_stats": cache_stats,
            "dataset_seconds": dataset_seconds,
        },
        "sfc": {
            "enabled": USE_SFC and N_SHOT > 1,
            "implementation": "DeepEMD get_sfc semantics",
            "lr": SFC_LR,
            "weight_decay": SFC_WD,
            "update_step": SFC_UPDATE_STEP,
            "batch_size": SFC_BS,
        },
        "matching": {
            "head": "DeepEMD emd_forward_1shot",
            "metric": DEEPEMD_METRIC,
            "norm": DEEPEMD_NORM,
            "solver": DEEPEMD_SOLVER,
            "temperature": DEEPEMD_TEMPERATURE,
        },
        "stage_checkpoints": stage_metadata,
        "metrics": metrics,
    }
    if SAVE_EPISODE_DETAILS:
        result["episode_details"] = episode_records
    return result


# =============================================================================
# Output
# =============================================================================

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_summary_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "stage", "hospital", "organ", "way_effective", "shot", "query_per_gt_class",
        "episodes", "accuracy_percent", "ci95_percent", "query_micro_accuracy_percent",
        "query_correct", "query_total", "fisher_ratio", "kl_divergence_inter_intra",
        "latency_mean_ms", "feature_fetch_mean_ms", "sfc_mean_ms", "matching_mean_ms", "checkpoint",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if N_SHOT < 1 or N_QUERY < 1 or EPISODES_PER_TASK < 1:
        raise ValueError("N_SHOT/N_QUERY/EPISODES_PER_TASK must be >= 1")
    for organ in PARTS:
        if organ not in ORGAN_CONFIG:
            raise KeyError(f"Missing ORGAN_CONFIG for {organ}")

    setup_seed(SEED)
    os.makedirs(RESULT_DIR, exist_ok=True)
    run_start = time.perf_counter()
    total_tasks = len(PARTS) * len(HOSPITALS)
    completed_tasks = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 110)
    print("[RAE + MetricEncoder | eval_1 test-test | hospital x organ batch | SFC]")
    print(f"device       = {device}")
    print(f"data_base    = {DATA_BASE}")
    print("same_pool    = DATA_BASE/{hospital}/{organ}/[test]")
    print(f"hospitals    = {HOSPITALS}")
    print(f"organs       = {PARTS}")
    print(f"shot         = {N_SHOT}")
    print(f"query/class  = {N_QUERY}")
    print(f"episodes     = {EPISODES_PER_TASK}")
    print(f"result_dir   = {RESULT_DIR}")
    print(f"stage filter = {FILTER_ORGAN_SPECIFIC_STAGES}")
    print(f"precompute   = {PRECOMPUTE_DENSE_FEATURES}, batch={PRECOMPUTE_IMAGE_BATCH}, io_workers={IMAGE_IO_WORKERS}, RAE_chunk={RAE_CHUNK_SIZE}")
    print(f"cache device = {'GPU' if CACHE_FEATURES_ON_GPU else 'CPU'}")
    print(f"local pool   = {LOCAL_POOL} (1=no pooling; applied Metric -> SFC/EMD)")
    print(f"episode log  = every {LOG_EVERY} episode(s)")
    print(f"support adapt = {'SFC' if USE_SFC else 'class-wise shot mean (no SFC)'}")
    print("=" * 110)

    rae = build_rae(PROJECT_ROOT, device)
    transform = build_transform()

    summary_rows: List[Dict[str, Any]] = []
    summary_nested: Dict[str, Any] = {}

    for organ_idx, organ in enumerate(PARTS):
        way = int(ORGAN_CONFIG[organ]["way"])
        stage_models, stage_meta = load_stage_metrics_for_part(organ, device)
        meta_head = build_meta_head(way, device)
        feature_cache = DenseFeatureCache(
            rae=rae,
            stage_models=stage_models,
            device=device,
            transform=transform,
        )

        print(
            f"[WORKLOAD] organ={organ} stages={len(stage_models)} episodes/dataset={EPISODES_PER_TASK} "
            f"SFC_steps={SFC_UPDATE_STEP}; hospitals={len(HOSPITALS)}"
        )

        for hospital_idx, hospital in enumerate(HOSPITALS):
            task_start = time.perf_counter()
            dataset_seed = int(SEED) + organ_idx * 10000 + hospital_idx * 1009
            result = evaluate_one_dataset(
                hospital=hospital,
                organ=organ,
                rae=rae,
                stage_models=stage_models,
                stage_metadata=stage_meta,
                meta_head=meta_head,
                feature_cache=feature_cache,
                device=device,
                transform=transform,
                dataset_seed=dataset_seed,
            )

            out_dir = os.path.join(RESULT_DIR, hospital, organ)
            save_json(os.path.join(out_dir, "result.json"), result)
            summary_nested.setdefault(organ, {})[hospital] = result["metrics"]

            print(f"\n[RESULT] {hospital}-{organ}")
            for stage, m in result["metrics"].items():
                print(
                    f"  {stage:<18} Acc={m['accuracy_percent']:.2f} +/- {m['ci95_percent']:.2f}% "
                    f"| micro={m['query_micro_accuracy_percent']:.2f}% "
                    f"| time/stage-ep={m['latency_mean_ms']/1000:.2f}s "
                    f"(fetch={m['feature_fetch_mean_ms']:.1f}ms, "
                    f"SFC={m['sfc_mean_ms']/1000:.2f}s, match={m['matching_mean_ms']/1000:.2f}s)"
                )
                summary_rows.append({
                    "stage": stage,
                    "hospital": hospital,
                    "organ": organ,
                    "way_effective": result["way_effective"],
                    "shot": N_SHOT,
                    "query_per_gt_class": N_QUERY,
                    "episodes": EPISODES_PER_TASK,
                    "accuracy_percent": m["accuracy_percent"],
                    "ci95_percent": m["ci95_percent"],
                    "query_micro_accuracy_percent": m["query_micro_accuracy_percent"],
                    "query_correct": m["query_correct"],
                    "query_total": m["query_total"],
                    "fisher_ratio": m["fisher_ratio"],
                    "kl_divergence_inter_intra": m["kl_divergence_inter_intra"],
                    "latency_mean_ms": m["latency_mean_ms"],
                    "feature_fetch_mean_ms": m["feature_fetch_mean_ms"],
                    "sfc_mean_ms": m["sfc_mean_ms"],
                    "matching_mean_ms": m["matching_mean_ms"],
                    "checkpoint": stage_meta[stage]["path"],
                })

            completed_tasks += 1
            task_seconds = time.perf_counter() - task_start
            run_elapsed = time.perf_counter() - run_start
            mean_task = run_elapsed / completed_tasks
            eta_all = mean_task * (total_tasks - completed_tasks)
            print(
                f"[TIME] dataset={hospital}-{organ} took {format_duration(task_seconds)} | "
                f"whole-run elapsed={format_duration(run_elapsed)} | "
                f"rough ETA remaining={format_duration(eta_all)} "
                f"({completed_tasks}/{total_tasks} datasets done)"
            )

        del feature_cache, meta_head, stage_models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_summary_csv(os.path.join(RESULT_DIR, "summary.csv"), summary_rows)
    total_seconds = time.perf_counter() - run_start
    save_json(
        os.path.join(RESULT_DIR, "summary.json"),
        {
            "protocol": "batch hospital x organ test-test; same pool support/query with strict overlap exclusion",
            "reference": "eval_1.py test-test episodic protocol extended to hospital x organ + RAE/Metric backend",
            "data_base": DATA_BASE,
            "only_encoder_difference": "frozen RAE + stage-specific MetricEncoder",
            "optimization": {
                "filter_organ_specific_stages": FILTER_ORGAN_SPECIFIC_STAGES,
                "precompute_dense_features": PRECOMPUTE_DENSE_FEATURES,
                "precompute_image_batch": PRECOMPUTE_IMAGE_BATCH,
                "image_io_workers": IMAGE_IO_WORKERS,
            "cache_features_on_gpu": CACHE_FEATURES_ON_GPU,
            "local_pool": int(LOCAL_POOL),
            "rae_chunk_size_initial": RAE_CHUNK_SIZE,
                "sfc_math_changed": False,
                "save_episode_details": SAVE_EPISODE_DETAILS,
            },
            "hospitals": HOSPITALS,
            "organs": PARTS,
            "organ_config": ORGAN_CONFIG,
            "shot": N_SHOT,
            "query_per_gt_class": N_QUERY,
            "episodes_per_dataset": EPISODES_PER_TASK,
            "sfc": {
                "lr": SFC_LR,
                "wd": SFC_WD,
                "update_step": SFC_UPDATE_STEP,
                "bs": SFC_BS,
            },
            "stage_checkpoints": STAGE_CHECKPOINTS,
            "total_runtime_seconds": total_seconds,
            "results": summary_nested,
        },
    )

    print("\n" + "=" * 110)
    print("[DONE]")
    print(f"total runtime = {format_duration(total_seconds)}")
    print(f"summary CSV   = {os.path.join(RESULT_DIR, 'summary.csv')}")
    print(f"summary JSON  = {os.path.join(RESULT_DIR, 'summary.json')}")
    print("=" * 110)


if __name__ == "__main__":
    main()
