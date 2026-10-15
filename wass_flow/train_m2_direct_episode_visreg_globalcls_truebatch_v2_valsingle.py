#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2: Direct N-way K-shot episodic calibration for RAE-Flow-Metric V1.

Canonical M2 path
-----------------
query/support image
  -> frozen RAE: (B,1024,16,16)
  -> shared MetricEncoder
  -> raw metric tokens for dynamic SK masses
  -> L2-normalized metric tokens for squared-Euclidean ground cost
  -> entropic Sinkhorn for every query-support image pair
  -> K-shot soft-min per class
  -> negative class energy / learnable classification temperature
  -> episodic cross entropy + VISReg representation regularization
  -> optional per-image global classification auxiliary loss
  -> true multi-episode mini-batch training (independent episodes, one forward/backward)

The default episode protocol matches the project's recent 1-query + N-distractor
setting: one query class is sampled from the N candidate classes, each class has
K support images, and all ``n_query`` query images come from that one query class.
The query image(s) are never reused as support in the same episode.

M2 deliberately contains NO Flow trajectory and NO SFC.  Its output checkpoint
is the direct metric calibration checkpoint consumed by F1.

Data input supports three compatible modes:
1) class-folder root,
2) class-pool JSON glob,
3) episode manifest with query_path/query_label/candidate_labels/support_dirs.
"""

from __future__ import annotations

# import debugpy
# try:
#     print("Waiting for debugger attach")
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9504))
#     # debugpy.listen(("172.16.0.108", 9504)) 
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import argparse
import glob
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms

# Make ``import rae_flow_metric_v1`` work when this file is executed directly.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from models.metric_encoder import MetricEncoder, pool_square_tokens
from models.sk_sinkhorn import pairwise_sk_sinkhorn_distance

try:
    from data.class_pool import load_records
except ImportError:
    # Backward-compatible direct execution from inside rae_flow_metric_v1/.
    from data.class_pool import load_records

try:
    from losss.visreg import VISReg
except ImportError:
    # Backward-compatible direct execution from inside rae_flow_metric_v1/.
    from losss.visreg import VISReg


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class EpisodePlan:
    classes: list[str]
    support_paths: list[list[str]]  # [way][shot]
    query_paths: list[str]         # [n_query], all from one GT class
    query_label: int               # local slot after class shuffle
    metadata: Optional[dict] = None


@dataclass
class EpisodeManifestRecord:
    """One fixed candidate-set episode template.

    The query/candidate set comes from the manifest. K-shot support images are
    sampled online from each candidate's support directory.
    """
    query_paths: list[str]
    query_label: str
    candidate_labels: list[str]
    support_pools: list[list[str]]
    support_dirs: list[str]
    metadata: dict


@dataclass
class EpisodeStats:
    loss_sum: float = 0.0
    ce_sum: float = 0.0
    vis_sum: float = 0.0
    global_cls_loss_sum: float = 0.0  # sample-weighted sum
    global_cls_correct: float = 0.0
    global_cls_count: int = 0
    acc_sum: float = 0.0
    acc_sq_sum: float = 0.0
    pos_sum: float = 0.0
    neg_sum: float = 0.0
    margin_sum: float = 0.0
    count: int = 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "M2: Direct episodic calibration with Dynamic-SK weighted Sinkhorn"
    )

    train = p.add_mutually_exclusive_group(required=True)
    train.add_argument(
        "--train_root", type=str,
        help="Class-folder root: root/class_name/images",
    )
    train.add_argument(
        "--train_json_glob", type=str,
        help="Glob of class-pool JSON files (one or more files per class)",
    )
    train.add_argument(
        "--train_episode_manifest", type=str,
        help=(
            "Episode manifest JSON/JSONL (or directory/glob) with "
            "query_path/query_label/candidate_labels/support_dirs"
        ),
    )

    val = p.add_mutually_exclusive_group(required=False)
    val.add_argument("--val_root", type=str, default=None)
    val.add_argument("--val_json_glob", type=str, default=None)
    val.add_argument(
        "--val_episode_manifest", type=str, default=None,
        help="Validation episode manifest in the same format as train manifest",
    )
    p.add_argument(
        "--val_fraction", type=float, default=0.10,
        help=(
            "Validation split when no explicit validation source is supplied. "
            "Class-pool mode splits per class; manifest mode splits records."
        ),
    )

    ckpt = p.add_mutually_exclusive_group(required=True)
    ckpt.add_argument(
        "--init_checkpoint", type=str,
        help="M1 metric_medical_adapt.pt; only MetricEncoder weights are inherited",
    )
    ckpt.add_argument(
        "--resume_checkpoint", type=str,
        help="Resume a complete M2 checkpoint including temperatures/optimizer",
    )

    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    # Episode protocol: total-query protocol, not n_query per class.
    p.add_argument("--n_way", type=int, default=16)
    p.add_argument("--n_shot", type=int, default=5)
    p.add_argument("--n_query", type=int, default=1)
    p.add_argument("--episodes_per_epoch", type=int, default=500)
    p.add_argument("--val_episodes", type=int, default=200)
    p.add_argument(
        "--episode_batch_size", type=int, default=4,
        help=(
            "TRUE episode mini-batch size per GPU. Several independent N-way episodes "
            "are encoded and classified in one forward/backward; n_way is NOT enlarged. "
            "With DDP the effective episode batch is world_size * episode_batch_size."
        ),
    )

    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--rae_chunk_size", type=int, default=8)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)

    # Metric / OT.
    p.add_argument("--hidden_dim", type=int, default=256, help="Legacy compatibility arg")
    p.add_argument("--metric_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument(
        "--ot_pool_factor", type=int, default=1,
        help="1=full 16x16=256 nodes; 2=8x8=64 nodes (debug/OOM fallback)",
    )
    p.add_argument("--sinkhorn_reg", type=float, default=0.1)
    p.add_argument("--sinkhorn_iter", type=int, default=30)
    p.add_argument("--sk_weight_eps", type=float, default=1e-3)

    # Trainable aggregation/calibration temperatures.
    p.add_argument("--shot_tau_init", type=float, default=0.10)
    p.add_argument("--cls_tau_init", type=float, default=0.10)
    p.add_argument("--tau_min", type=float, default=1e-3)
    p.add_argument("--tau_max", type=float, default=5.0)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--metric_lr", type=float, default=1e-5)
    p.add_argument("--temperature_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument(
        "--freeze_metric_epochs", type=int, default=0,
        help="Optional warm-up: only temperatures train for the first N epochs",
    )

    # VISReg: same defaults and implementation style as train_m1_medical_metric.py.
    # M2 objective:
    #   L_total = L_episode_CE + visreg_weight * L_VISReg
    p.add_argument("--visreg_weight", type=float, default=1e-3)
    p.add_argument("--visreg_num_proj", type=int, default=1024)
    p.add_argument("--visreg_projection_chunk", type=int, default=128)

    # Optional per-image global classification auxiliary task.
    # Every support image and every query image in the current episode contributes
    # one global-class CE term.  The global class IDs are stable across episodes.
    p.add_argument(
        "--global_cls_weight", type=float, default=0.0,
        help="Auxiliary per-image global classification loss weight; 0 disables it",
    )
    p.add_argument(
        "--global_cls_lr", type=float, default=1e-3,
        help="Learning rate of the auxiliary global classification head",
    )
    p.add_argument(
        "--global_cls_label_smoothing", type=float, default=0.0,
        help="Label smoothing for per-image global classification CE",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log_every", type=int, default=25)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------------


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    return device, rank, world_size, rank == 0


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    s = int(seed) + int(rank) * 100_003
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def all_reduce_stats(stats: EpisodeStats, device: torch.device) -> EpisodeStats:
    values = torch.tensor(
        [
            stats.loss_sum,
            stats.ce_sum,
            stats.vis_sum,
            stats.global_cls_loss_sum,
            stats.global_cls_correct,
            float(stats.global_cls_count),
            stats.acc_sum,
            stats.acc_sq_sum,
            stats.pos_sum,
            stats.neg_sum,
            stats.margin_sum,
            float(stats.count),
        ],
        device=device,
        dtype=torch.float64,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return EpisodeStats(
        loss_sum=float(values[0].item()),
        ce_sum=float(values[1].item()),
        vis_sum=float(values[2].item()),
        global_cls_loss_sum=float(values[3].item()),
        global_cls_correct=float(values[4].item()),
        global_cls_count=int(round(values[5].item())),
        acc_sum=float(values[6].item()),
        acc_sq_sum=float(values[7].item()),
        pos_sum=float(values[8].item()),
        neg_sum=float(values[9].item()),
        margin_sum=float(values[10].item()),
        count=int(round(values[11].item())),
    )


# -----------------------------------------------------------------------------
# Medical class-pool loading
# -----------------------------------------------------------------------------


def _scan_image_dir(path: str) -> list[str]:
    root = Path(path)
    if not root.exists():
        return []
    return sorted(
        str(p.resolve())
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def load_class_folders(root: str) -> dict[str, list[str]]:
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"train/val root 不存在或不是目录: {root}")

    out: dict[str, list[str]] = {}
    for child in sorted(root_p.iterdir()):
        if not child.is_dir():
            continue
        images = _scan_image_dir(str(child))
        if images:
            out[child.name] = images
    if not out:
        raise RuntimeError(f"未在 {root} 下找到 class-folder 图像")
    return out


def _resolve_json_image_path(value: str, base_dir: Path) -> str:
    """Resolve one image path stored inside a class-pool JSON.

    The Huaxi/ImageNet medical pools used by this project are list[dict] records
    such as {"x": "/abs/path.jpg", "label": "class name", ...}.  We do NOT
    stat every image here: the existing SingleGroupDataset follows the same
    convention because per-file os.path.exists() checks are very expensive on
    the shared filesystem.  Image.open() remains the final runtime validation.
    """
    p = Path(str(value).strip())
    if not p.is_absolute():
        p = base_dir / p
    return str(p)


def _entry_label(entry: dict) -> Optional[str]:
    """Return the class label carried by one JSON sample record."""
    for key in ("label", "class_name", "class", "name"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)

    # Compatibility fallback for pools that only store a numeric global ID.
    value = entry.get("label_id")
    if isinstance(value, (str, int)) and str(value).strip():
        return str(value).strip()
    return None


def _entry_image_path(entry: dict, base_dir: Path) -> Optional[str]:
    """Return an image path from the record formats used in this project."""
    for key in (
        "x",                 # canonical data_pool_huaxi / SingleGroupDataset key
        "path",
        "image",
        "image_path",
        "img",
        "img_path",
        "file",
        "filepath",
        "x0_path",
        "x1_path",
    ):
        value = entry.get(key)
        if isinstance(value, (str, os.PathLike)) and str(value).strip():
            return _resolve_json_image_path(os.fspath(value), base_dir)
    return None


def _record_list_from_class_pool_json(obj) -> Optional[list[dict]]:
    """Recognize sample-record JSON containers without treating metadata as data."""
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if isinstance(obj, dict):
        for key in ("data", "items", "samples", "records"):
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return None


def _collect_legacy_paths_from_json_obj(obj, base_dir: Path) -> list[str]:
    """Compatibility path extractor for old one-class JSON pool layouts."""
    paths: list[str] = []
    path_keys = {
        "x", "path", "image", "image_path", "img", "img_path", "file",
        "filepath", "x0_path", "x1_path",
    }
    list_keys = {
        "images", "paths", "image_paths", "files", "items", "samples", "data",
    }

    def add_path(v):
        if isinstance(v, (str, os.PathLike)) and str(v).strip():
            paths.append(_resolve_json_image_path(os.fspath(v), base_dir))

    def walk(x):
        if isinstance(x, (str, os.PathLike)):
            add_path(x)
        elif isinstance(x, list):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if lk in path_keys:
                    add_path(v)
                elif lk in list_keys:
                    walk(v)
                elif isinstance(v, (list, dict)):
                    walk(v)

    walk(obj)
    return list(dict.fromkeys(paths))


def _record_path_for_episode(record) -> str:
    """Extract the image path from data.class_pool.ImageRecord.

    The canonical project loader returns ImageRecord objects.  Keep this adapter
    intentionally tolerant to field-name changes while treating ``record.label``
    as the authoritative class name.
    """
    for attr in ("path", "image_path", "x", "img_path", "file", "filepath"):
        value = getattr(record, attr, None)
        if isinstance(value, (str, os.PathLike)) and str(value).strip():
            return os.fspath(value)

    # Dataclass / simple object compatibility.
    payload = getattr(record, "__dict__", None)
    if isinstance(payload, dict):
        for key in ("path", "image_path", "x", "img_path", "file", "filepath"):
            value = payload.get(key)
            if isinstance(value, (str, os.PathLike)) and str(value).strip():
                return os.fspath(value)

    raise AttributeError(
        "Cannot locate image path on ImageRecord returned by data.class_pool.load_records. "
        f"type={type(record).__name__}, fields={sorted(vars(record).keys()) if hasattr(record, '__dict__') else '<unknown>'}"
    )


def load_class_json_glob(pattern: str) -> dict[str, list[str]]:
    """Load JSON class pools through the project's canonical data.class_pool loader.

    This deliberately reuses the exact loader used by M0-A/M0-B/M1 instead of
    maintaining an M2-specific JSON parser.  ``load_records(None, pattern)``
    parses the project JSON format and returns records carrying authoritative
    ``label`` plus image path information; this function only groups them into
    the dict[label] -> list[path] representation needed by episodic M2.
    """
    records = load_records(None, pattern)
    if not records:
        raise RuntimeError(f"data.class_pool.load_records returned no records: {pattern}")

    merged: dict[str, list[str]] = {}
    bad_label = 0
    bad_path = 0

    for record in records:
        label = getattr(record, "label", None)
        if not isinstance(label, str) or not label.strip():
            bad_label += 1
            continue

        try:
            path = _record_path_for_episode(record)
        except (AttributeError, TypeError, ValueError):
            bad_path += 1
            continue

        merged.setdefault(label.strip(), []).append(str(path))

    # Stable de-duplication without stat/realpath on every image.  This follows
    # the shared-filesystem-friendly behavior of the project's data loaders.
    merged = {
        label: list(dict.fromkeys(paths))
        for label, paths in merged.items()
        if paths
    }

    if not merged:
        sample_fields = []
        if records and hasattr(records[0], "__dict__"):
            sample_fields = sorted(vars(records[0]).keys())
        raise RuntimeError(
            "Canonical data.class_pool loader returned records, but M2 could not "
            "group any usable label/path pairs.\n"
            f"  pattern={pattern}\n"
            f"  records={len(records)}\n"
            f"  bad_label={bad_label}, bad_path={bad_path}\n"
            f"  first_record_fields={sample_fields}"
        )

    print(
        "[M2 class pool via data.class_pool.load_records] "
        f"records={len(records)} classes={len(merged)} "
        f"images={sum(len(v) for v in merged.values())} "
        f"bad_label={bad_label} bad_path={bad_path}"
    )
    return merged


def load_class_pool(root: Optional[str], json_glob: Optional[str]) -> dict[str, list[str]]:
    if bool(root) == bool(json_glob):
        raise ValueError("必须且只能指定 root 或 json_glob 之一")
    return load_class_folders(root) if root else load_class_json_glob(json_glob)



def _resolve_manifest_path(value: str, base_dir: Path) -> str:
    p = Path(value)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    else:
        p = p.resolve()
    return str(p)


def _expand_manifest_files(source: str) -> list[Path]:
    """Accept one JSON/JSONL file, a directory, or a glob pattern."""
    p = Path(source)
    if p.is_file():
        return [p.resolve()]
    if p.is_dir():
        files = sorted(
            x.resolve()
            for x in p.rglob("*")
            if x.is_file() and x.suffix.lower() in {".json", ".jsonl"}
        )
    else:
        files = [
            Path(x).resolve()
            for x in sorted(glob.glob(source))
            if Path(x).is_file()
        ]
    if not files:
        raise FileNotFoundError(
            f"episode manifest 未匹配 JSON/JSONL 文件: {source}"
        )
    return files


def _records_from_manifest_file(path: Path) -> list[dict]:
    """Load list/single-record/wrapped-list JSON, plus JSONL."""
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(
                        f"{path}:{line_no} JSONL 每行必须是 dict"
                    )
                records.append(obj)
        return records

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        # Single record.
        if "query_path" in obj or "query_paths" in obj:
            return [obj]

        # Common wrappers.
        for key in ("episodes", "records", "items", "data"):
            value = obj.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(
        f"无法识别 episode manifest 结构: {path}. "
        "需要 list[record]、单个 record，或 episodes/records/items/data 列表。"
    )


def load_episode_manifest(
    source: str,
    *,
    n_way: int,
    n_shot: int,
    n_query: int,
) -> list[EpisodeManifestRecord]:
    """Parse fixed-query/fixed-candidate episode templates.

    Required record fields:
        query_path (or query_paths)
        query_label
        candidate_labels
        support_dirs

    support_dirs are scanned once at startup. The query image is removed from
    *all* support pools, so it can never leak into support sampling.  A support
    pool only needs one usable image; when its size is < n_shot, K-shot support
    is sampled with replacement.  Records with any empty support pool are skipped.
    """
    output: list[EpisodeManifestRecord] = []
    # Many manifest records reuse the same candidate support directories.
    # Scan each directory only once per process.
    support_dir_cache: dict[str, list[str]] = {}

    for manifest_path in _expand_manifest_files(source):
        base_dir = manifest_path.parent
        raw_records = _records_from_manifest_file(manifest_path)

        for rec_idx, rec in enumerate(raw_records):
            if not isinstance(rec, dict):
                raise ValueError(
                    f"{manifest_path} record[{rec_idx}] 必须是 dict"
                )

            query_label = rec.get("query_label")
            if not isinstance(query_label, str) or not query_label.strip():
                raise ValueError(
                    f"{manifest_path} record[{rec_idx}] 缺少有效 query_label"
                )
            query_label = query_label.strip()

            candidate_labels = rec.get("candidate_labels")
            support_dirs = rec.get("support_dirs")
            if not isinstance(candidate_labels, list) or not all(
                isinstance(x, str) and x.strip() for x in candidate_labels
            ):
                raise ValueError(
                    f"{manifest_path} record[{rec_idx}] candidate_labels 必须是字符串列表"
                )
            candidate_labels = [x.strip() for x in candidate_labels]

            if not isinstance(support_dirs, list) or not all(
                isinstance(x, str) and x.strip() for x in support_dirs
            ):
                raise ValueError(
                    f"{manifest_path} record[{rec_idx}] support_dirs 必须是字符串列表"
                )

            if len(candidate_labels) != len(support_dirs):
                raise ValueError(
                    f"{manifest_path} record[{rec_idx}] candidate_labels/support_dirs "
                    f"长度不一致: {len(candidate_labels)} vs {len(support_dirs)}"
                )
            # Manifest mode is variable-way: the actual episode width is defined
            # by this record's candidate_labels rather than the CLI --n_way.
            # --n_way remains the fixed width only for class-pool sampling.
            record_n_way = len(candidate_labels)
            if record_n_way < 2:
                raise ValueError(
                    f"{manifest_path} record[{rec_idx}] 候选类数="
                    f"{record_n_way}，至少需要 2 个候选类"
                )
            if len(set(candidate_labels)) != len(candidate_labels):
                raise ValueError(
                    f"{manifest_path} record[{rec_idx}] candidate_labels 含重复类别"
                )
            if query_label not in candidate_labels:
                raise ValueError(
                    f"{manifest_path} record[{rec_idx}] query_label={query_label!r} "
                    "不在 candidate_labels 中"
                )

            raw_query_paths = rec.get("query_paths")
            if raw_query_paths is None:
                one = rec.get("query_path")
                raw_query_paths = [one] if isinstance(one, str) else None
            if not isinstance(raw_query_paths, list) or not all(
                isinstance(x, str) and x.strip() for x in raw_query_paths
            ):
                raise ValueError(
                    f"{manifest_path} record[{rec_idx}] 需要 query_path "
                    "或 query_paths"
                )

            query_paths = [
                _resolve_manifest_path(x.strip(), base_dir)
                for x in raw_query_paths
            ]
            query_paths = list(dict.fromkeys(query_paths))
            if len(query_paths) < int(n_query):
                raise ValueError(
                    f"{manifest_path} record[{rec_idx}] query 数="
                    f"{len(query_paths)} < n_query={n_query}"
                )
            missing_query = [p for p in query_paths if not Path(p).is_file()]
            if missing_query:
                raise FileNotFoundError(
                    f"{manifest_path} record[{rec_idx}] query_path 不存在: "
                    f"{missing_query[:3]}"
                )

            resolved_support_dirs = [
                _resolve_manifest_path(x.strip(), base_dir)
                for x in support_dirs
            ]

            # Exclude every possible query in this record from every support pool.
            #
            # IMPORTANT: a support pool no longer needs to contain >= n_shot
            # distinct images.  As long as at least one image remains after query
            # exclusion, the record is valid; sample_manifest_episode() will sample
            # with replacement only when that pool has fewer than n_shot images.
            # This avoids physically duplicating file paths at manifest-load time.
            query_real = {os.path.realpath(p) for p in query_paths}
            support_pools: list[list[str]] = []
            empty_support = None
            for label, support_dir in zip(candidate_labels, resolved_support_dirs):
                if not Path(support_dir).is_dir():
                    raise FileNotFoundError(
                        f"{manifest_path} record[{rec_idx}] support_dir 不存在: "
                        f"label={label!r}, dir={support_dir}"
                    )
                if support_dir not in support_dir_cache:
                    support_dir_cache[support_dir] = _scan_image_dir(support_dir)
                images = [
                    p for p in support_dir_cache[support_dir]
                    if os.path.realpath(p) not in query_real
                ]

                # If one candidate class has no usable support image at all, this
                # manifest record cannot form a valid N-way episode.  Skip this
                # record instead of crashing or attempting division-by-zero
                # replication.
                if len(images) == 0:
                    empty_support = (label, support_dir)
                    break

                # Keep the real pool unchanged even when len(images) < n_shot.
                # Repetition is handled online at episode sampling time.
                support_pools.append(images)

            if empty_support is not None:
                empty_label, empty_dir = empty_support
                print(
                    f"[SKIP] {manifest_path} record[{rec_idx}] 无可用 support: "
                    f"label={empty_label!r}, after_query_exclusion=0, "
                    f"dir={empty_dir}"
                )
                continue

            metadata = {
                k: v for k, v in rec.items()
                if k not in {
                    "query_path", "query_paths", "query_label",
                    "candidate_labels", "support_dirs",
                }
            }
            metadata.update(
                {
                    "manifest_file": str(manifest_path),
                    "manifest_record_index": int(rec_idx),
                    "record_n_way": int(record_n_way),
                }
            )

            output.append(
                EpisodeManifestRecord(
                    query_paths=query_paths,
                    query_label=query_label,
                    candidate_labels=candidate_labels,
                    support_pools=support_pools,
                    support_dirs=resolved_support_dirs,
                    metadata=metadata,
                )
            )

    if not output:
        raise RuntimeError(f"episode manifest 没有有效记录: {source}")
    return output


def split_manifest_records(
    records: list[EpisodeManifestRecord],
    val_fraction: float,
    *,
    seed: int,
) -> tuple[list[EpisodeManifestRecord], list[EpisodeManifestRecord]]:
    """Deterministically split episode templates when val manifest is omitted."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction 必须位于 (0,1)")
    if len(records) < 2:
        raise RuntimeError(
            "manifest 只有 1 条记录，无法自动 train/val 划分；"
            "请显式提供 --val_episode_manifest"
        )

    idx = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * float(val_fraction))))
    n_val = min(n_val, len(idx) - 1)
    val_idx = set(idx[:n_val])
    train_records = [r for i, r in enumerate(records) if i not in val_idx]
    val_records = [r for i, r in enumerate(records) if i in val_idx]
    return train_records, val_records


def stratified_split(
    pool: dict[str, list[str]],
    val_fraction: float,
    *,
    seed: int,
    min_train: int,
    min_val: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction 必须位于 (0,1)")

    rng = random.Random(seed)
    train, val = {}, {}
    for cls, paths in sorted(pool.items()):
        items = list(paths)
        rng.shuffle(items)
        if len(items) < min_train + min_val:
            continue
        n_val = max(min_val, int(round(len(items) * val_fraction)))
        n_val = min(n_val, len(items) - min_train)
        val[cls] = sorted(items[:n_val])
        train[cls] = sorted(items[n_val:])

    if not train or not val:
        raise RuntimeError(
            "自动 train/val 划分后无有效类别；请提供显式 val_root/val_json_glob "
            "或降低 n_way/n_shot。"
        )
    return train, val


def filter_episode_classes(
    pool: dict[str, list[str]],
    *,
    n_shot: int,
    n_query: int,
) -> dict[str, list[str]]:
    # Every class can become the query class, so require K+Q samples for all.
    need = int(n_shot) + int(n_query)
    valid = {c: sorted(v) for c, v in pool.items() if len(v) >= need}
    if len(valid) < 2:
        counts = sorted((c, len(v)) for c, v in pool.items())
        raise RuntimeError(
            f"有效类别不足。每类至少需要 n_shot+n_query={need} 张。"
            f" 当前有效={len(valid)}; 示例计数={counts[:10]}"
        )
    return valid


# -----------------------------------------------------------------------------
# Global classification label space
# -----------------------------------------------------------------------------


def build_global_label_map(
    pool: Optional[dict[str, list[str]]],
    manifest_records: Optional[list[EpisodeManifestRecord]],
) -> dict[str, int]:
    """Build one stable class-name -> contiguous head-ID map for all train episodes.

    This mapping is global across episodes and is intentionally different from the
    shuffled episode-local slot IDs used by the N-way episodic CE.
    """
    labels: set[str] = set()
    if pool is not None:
        labels.update(str(x) for x in pool.keys())
    if manifest_records is not None:
        for record in manifest_records:
            labels.update(str(x) for x in record.candidate_labels)
            labels.add(str(record.query_label))
    if not labels:
        raise RuntimeError("cannot build global classification label map from empty training data")
    return {label: idx for idx, label in enumerate(sorted(labels))}


def episode_global_targets(
    ep: EpisodePlan,
    *,
    n_shot: int,
    n_query: int,
    label_to_id: dict[str, int],
    device: torch.device,
    require_all_known: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-image global targets aligned with [support..., query...].

    Support order is class-major, exactly matching load_episode_images().  Every
    query belongs to the episode GT class by the existing protocol.  Unknown labels
    are masked only for validation compatibility; training requires every sample to
    have a global target whenever the auxiliary task is enabled.
    """
    names = [
        cls
        for cls in ep.classes
        for _ in range(int(n_shot))
    ]
    query_name = ep.classes[int(ep.query_label)]
    names.extend([query_name] * int(n_query))

    ids = [label_to_id.get(name, -1) for name in names]
    target = torch.tensor(ids, dtype=torch.long, device=device)
    known = target >= 0
    if require_all_known and not bool(known.all().item()):
        missing = sorted({name for name, idx in zip(names, ids) if idx < 0})
        raise RuntimeError(
            "global classification target missing from training label map: "
            f"{missing[:20]}"
        )
    return target, known


# -----------------------------------------------------------------------------
# Episode planning / image loading
# -----------------------------------------------------------------------------


def sample_episode(
    pool: dict[str, list[str]],
    *,
    n_way: int,
    n_shot: int,
    n_query: int,
    rng: random.Random,
) -> EpisodePlan:
    classes = sorted(pool.keys())
    if len(classes) < n_way:
        raise RuntimeError(f"可用类别 {len(classes)} < n_way={n_way}")

    selected = rng.sample(classes, n_way)
    rng.shuffle(selected)
    query_label = rng.randrange(n_way)

    support_paths: list[list[str]] = []
    query_paths: list[str] = []

    for local_id, cls in enumerate(selected):
        items = pool[cls]
        if local_id == query_label:
            chosen = rng.sample(items, n_shot + n_query)
            support_paths.append(chosen[:n_shot])
            query_paths = chosen[n_shot:]
        else:
            support_paths.append(rng.sample(items, n_shot))

    return EpisodePlan(
        classes=selected,
        support_paths=support_paths,
        query_paths=query_paths,
        query_label=query_label,
    )



def sample_manifest_episode(
    record: EpisodeManifestRecord,
    *,
    n_shot: int,
    n_query: int,
    rng: random.Random,
) -> EpisodePlan:
    """Instantiate one manifest template into a concrete K-shot episode.

    Candidate labels and support directories are shuffled *together* every time,
    preventing any fixed candidate position (e.g. GT always at slot 0) from
    leaking into training. Candidate membership itself never changes.
    """
    if len(record.query_paths) < int(n_query):
        raise RuntimeError(
            f"manifest record query 数={len(record.query_paths)} < n_query={n_query}"
        )

    pairs = list(
        zip(
            record.candidate_labels,
            record.support_pools,
            record.support_dirs,
        )
    )
    rng.shuffle(pairs)

    classes = [x[0] for x in pairs]
    query_label = classes.index(record.query_label)
    support_paths: list[list[str]] = []
    for label, pool, support_dir in pairs:
        if not pool:
            # load_episode_manifest() filters these records, so reaching this branch
            # means the record was mutated/corrupted after loading.
            raise RuntimeError(
                f"manifest support pool unexpectedly empty: "
                f"label={label!r}, dir={support_dir}"
            )

        if len(pool) >= int(n_shot):
            # Preserve the original no-replacement behavior whenever possible.
            chosen = rng.sample(pool, int(n_shot))
        else:
            # Low-resource class: K-shot is still allowed.  Draw with replacement
            # from the real support pool instead of duplicating the pool itself.
            chosen = rng.choices(pool, k=int(n_shot))
        support_paths.append(chosen)

    query_paths = rng.sample(record.query_paths, int(n_query))

    meta = dict(record.metadata)
    meta.update(
        {
            "source_mode": "episode_manifest",
            "query_label_name": record.query_label,
            "candidate_labels_after_shuffle": list(classes),
            "support_dirs_after_shuffle": [x[2] for x in pairs],
            "episode_n_way": int(len(classes)),
        }
    )

    return EpisodePlan(
        classes=classes,
        support_paths=support_paths,
        query_paths=query_paths,
        query_label=query_label,
        metadata=meta,
    )


def build_transform(image_size: int, augment: bool):
    ops = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.append(transforms.ToTensor())  # RAE V1 uses [0,1] input
    return transforms.Compose(ops)


def load_episode_images(
    ep: EpisodePlan,
    transform,
) -> tuple[torch.Tensor, torch.Tensor]:
    support = []
    for class_paths in ep.support_paths:
        for path in class_paths:
            with Image.open(path) as im:
                support.append(transform(im.convert("RGB")))

    query = []
    for path in ep.query_paths:
        with Image.open(path) as im:
            query.append(transform(im.convert("RGB")))

    return torch.stack(support, dim=0), torch.stack(query, dim=0)


# -----------------------------------------------------------------------------
# Frozen RAE runtime
# -----------------------------------------------------------------------------


def build_rae(project_root: str, device: torch.device):
    project_root = str(Path(project_root).resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from rae.rae import RAE

    model = RAE(
        encoder_name="dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]",
        resolution=256,
        decoder_config_path=os.path.join(
            project_root, "rae/configs/decoder/ViTXL/config.json"
        ),
        pretrained_decoder_path=os.path.join(
            project_root, "flow_matcher_otcfm/rae_v2/decoder.pt"
        ),
        noise_tau=0.0,
        normalization_stat_path=os.path.join(
            project_root, "flow_matcher_otcfm/rae_v2/stats.pt"
        ),
    ).to(device).eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def encode_rae_chunked(
    rae,
    images: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size <= 0:
        chunk_size = images.shape[0]
    chunks = []
    for start in range(0, images.shape[0], chunk_size):
        chunks.append(rae.encode(images[start:start + chunk_size]))
    latent = torch.cat(chunks, dim=0)
    if latent.ndim != 4 or latent.shape[1:] != (1024, 16, 16):
        raise RuntimeError(
            f"RAE 输出应为 (B,1024,16,16)，实际={tuple(latent.shape)}"
        )
    return latent


# -----------------------------------------------------------------------------
# M2 model
# -----------------------------------------------------------------------------


class DirectEpisodeMetricModel(nn.Module):
    """Metric Encoder + batched Dynamic-SK Sinkhorn + K-shot soft-min classifier.

    Forward accepts either one episode:
        support_latent: (S,C,H,W), query_latent: (Q,C,H,W)
    or a TRUE episode mini-batch with identical episode widths:
        support_latent: (E,S,C,H,W), query_latent: (E,Q,C,H,W)

    The leading E dimension is preserved by the pairwise OT implementation, so
    query/support pairs are Cartesian *within each episode only*.  Cross-episode
    matching is never constructed.
    """

    def __init__(
        self,
        metric: MetricEncoder,
        *,
        n_way: int,
        n_shot: int,
        sinkhorn_reg: float,
        sinkhorn_iter: int,
        sk_weight_eps: float,
        shot_tau_init: float,
        cls_tau_init: float,
        tau_min: float,
        tau_max: float,
        ot_pool_factor: int = 1,
        num_global_classes: int = 0,
    ) -> None:
        super().__init__()
        if n_way < 2 or n_shot < 1:
            raise ValueError("n_way>=2, n_shot>=1 are required")
        if shot_tau_init <= 0 or cls_tau_init <= 0:
            raise ValueError("temperature initial values must be >0")

        self.metric = metric
        self.n_way = int(n_way)
        self.n_shot = int(n_shot)
        self.sinkhorn_reg = float(sinkhorn_reg)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.sk_weight_eps = float(sk_weight_eps)
        self.ot_pool_factor = int(ot_pool_factor)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.num_global_classes = int(num_global_classes)

        self.log_shot_tau = nn.Parameter(torch.tensor(math.log(shot_tau_init)))
        self.log_cls_tau = nn.Parameter(torch.tensor(math.log(cls_tau_init)))

        self.global_cls_head: Optional[nn.Module]
        if self.num_global_classes > 0:
            self.global_cls_head = nn.Linear(int(metric.out_dim), self.num_global_classes)
        else:
            self.global_cls_head = None

    def _temperature(self, log_tau: torch.Tensor) -> torch.Tensor:
        return log_tau.exp().clamp(self.tau_min, self.tau_max)

    @property
    def shot_tau(self) -> torch.Tensor:
        return self._temperature(self.log_shot_tau)

    @property
    def cls_tau(self) -> torch.Tensor:
        return self._temperature(self.log_cls_tau)

    def _pool_features(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ot_pool_factor > 1:
            raw = pool_square_tokens(raw, self.ot_pool_factor)
        normalized = F.normalize(raw, p=2, dim=-1, eps=1e-8)
        return raw, normalized

    def _cloud_to_global_feature(self, cloud: torch.Tensor) -> torch.Tensor:
        """Convert each image MetricEncoder cloud to one D-dimensional vector."""
        if cloud.ndim < 2:
            raise RuntimeError(f"MetricEncoder cloud must be at least 2D, got {tuple(cloud.shape)}")
        if cloud.ndim == 2:
            feat = cloud
        else:
            feat = cloud.reshape(cloud.shape[0], -1, cloud.shape[-1]).mean(dim=1)
        if feat.shape[-1] != int(self.metric.out_dim):
            raise RuntimeError(
                f"global feature dim={feat.shape[-1]} != MetricEncoder out_dim={self.metric.out_dim}"
            )
        return feat

    def forward(
        self,
        support_latent: torch.Tensor,
        query_latent: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        single_episode = support_latent.ndim == 4
        if single_episode:
            if query_latent.ndim != 4:
                raise ValueError("single support/query latent dimensions do not match")
            support_latent = support_latent.unsqueeze(0)
            query_latent = query_latent.unsqueeze(0)
        elif support_latent.ndim != 5 or query_latent.ndim != 5:
            raise ValueError(
                "support/query latent must be 4D single-episode or 5D episode-batch, "
                f"got support={tuple(support_latent.shape)}, query={tuple(query_latent.shape)}"
            )

        e = int(support_latent.shape[0])
        if int(query_latent.shape[0]) != e:
            raise ValueError("support/query episode batch size mismatch")
        support_count = int(support_latent.shape[1])
        query_count = int(query_latent.shape[1])
        if support_count <= 0 or support_count % self.n_shot != 0:
            raise ValueError(
                f"support count={support_count} cannot form complete n_shot={self.n_shot} groups"
            )
        episode_n_way = support_count // self.n_shot
        if episode_n_way < 2:
            raise ValueError(f"current episode_n_way={episode_n_way}; at least 2 ways are required")

        # TRUE metric batch: every image from every episode is encoded in ONE
        # MetricEncoder call.  Ordering is supports first, then queries.
        support_flat = support_latent.reshape(e * support_count, *support_latent.shape[2:])
        query_flat = query_latent.reshape(e * query_count, *query_latent.shape[2:])
        all_latent = torch.cat([support_flat, query_flat], dim=0)
        all_feat = self.metric(all_latent)

        s_raw_flat = all_feat.raw[: e * support_count]
        q_raw_flat = all_feat.raw[e * support_count :]
        token_shape = s_raw_flat.shape[1:]
        s_raw = s_raw_flat.reshape(e, support_count, *token_shape)
        q_raw = q_raw_flat.reshape(e, query_count, *token_shape)
        s_raw, s_norm = self._pool_features(s_raw)
        q_raw, q_norm = self._pool_features(q_raw)

        # sk_sinkhorn pairwise APIs support arbitrary broadcastable leading dims.
        # Therefore E is a real batch dimension and OT shape is (E,Q,S), not
        # (E*Q,E*S): no cross-episode query/support pairs are ever computed.
        ot = pairwise_sk_sinkhorn_distance(
            q_norm,
            s_norm,
            query_raw=q_raw,
            support_raw=s_raw,
            reg=self.sinkhorn_reg,
            n_iter=self.sinkhorn_iter,
            weight_eps=self.sk_weight_eps,
            normalize_cost_points=False,
            return_transport=False,
            return_cost=False,
        )
        pair_distance = ot.distance.reshape(
            e, query_count, episode_n_way, self.n_shot
        )

        shot_tau = self.shot_tau
        class_energy = -shot_tau * (
            torch.logsumexp(-pair_distance / shot_tau, dim=-1)
            - math.log(float(self.n_shot))
        )
        cls_tau = self.cls_tau
        logits = -class_energy / cls_tau  # (E,Q,N)

        # MetricEncoder.cloud is the existing per-image cloud/global embedding used
        # by VISReg.  Reshape it back to the true episode batch.
        cloud_feat = self._cloud_to_global_feature(all_feat.cloud)
        s_cloud = cloud_feat[: e * support_count].reshape(e, support_count, -1)
        q_cloud = cloud_feat[e * support_count :].reshape(e, query_count, -1)

        global_logits = None
        if self.global_cls_head is not None:
            global_feat = torch.cat([s_cloud, q_cloud], dim=1)  # (E,S+Q,D)
            global_logits = self.global_cls_head(global_feat)

        out = {
            "logits": logits,
            "class_energy": class_energy,
            "pair_distance": pair_distance,
            "episode_n_way": torch.tensor(episode_n_way, device=logits.device),
            "support_cloud": s_cloud,
            "query_cloud": q_cloud,
            "global_logits": global_logits,
            "shot_tau": shot_tau,
            "cls_tau": cls_tau,
        }
        if single_episode:
            for key in ("logits", "class_energy", "pair_distance", "support_cloud", "query_cloud"):
                out[key] = out[key].squeeze(0)
            if out["global_logits"] is not None:
                out["global_logits"] = out["global_logits"].squeeze(0)
        return out


# -----------------------------------------------------------------------------
# Checkpoint I/O
# -----------------------------------------------------------------------------


def _torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _strip_prefix(state: dict, prefix: str) -> dict:
    if state and all(str(k).startswith(prefix) for k in state):
        return {str(k)[len(prefix):]: v for k, v in state.items()}
    return state


def extract_metric_state(ckpt) -> dict:
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint must be a dict")

    for key in ("metric_encoder", "metric"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            return _strip_prefix(value, "module.")

    # Common full-model layouts.
    for key in ("model", "model_state_dict", "state_dict", "params"):
        value = ckpt.get(key)
        if isinstance(value, dict):
            state = _strip_prefix(value, "module.")
            metric_prefixed = {
                k[len("metric."):]: v for k, v in state.items()
                if str(k).startswith("metric.")
            }
            if metric_prefixed:
                return metric_prefixed
            return state

    # Last resort: checkpoint itself may be a state_dict.
    if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        return _strip_prefix(ckpt, "module.")

    raise KeyError("Cannot locate MetricEncoder weights in checkpoint")


def load_m1_metric(metric: MetricEncoder, path: str) -> None:
    ckpt = _torch_load(path, "cpu")
    state = extract_metric_state(ckpt)
    missing, unexpected = metric.load_state_dict(state, strict=False)

    # A real architecture mismatch should not silently pass.  Allow only no-op
    # compatibility leftovers; current V1 M1->M2 should normally be exact.
    if missing or unexpected:
        raise RuntimeError(
            "M1 -> M2 MetricEncoder checkpoint mismatch.\n"
            f"missing={missing[:20]}\nunexpected={unexpected[:20]}"
        )


def save_m2_checkpoint(
    path: str,
    *,
    raw_model: DirectEpisodeMetricModel,
    optimizer,
    scheduler,
    epoch: int,
    args,
    val_metrics: dict,
    global_label_to_id: Optional[dict[str, int]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "stage": "M2",
        "epoch": int(epoch),
        "metric_encoder": raw_model.metric.state_dict(),
        "log_shot_tau": raw_model.log_shot_tau.detach().cpu(),
        "log_cls_tau": raw_model.log_cls_tau.detach().cpu(),
        "global_cls_head": (
            raw_model.global_cls_head.state_dict()
            if raw_model.global_cls_head is not None
            else None
        ),
        "global_label_to_id": dict(global_label_to_id or {}),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
        "val": val_metrics,
        "metric_config": {
            "input_dim": raw_model.metric.in_dim,
            "output_dim": raw_model.metric.out_dim,
            "model_dim": raw_model.metric.model_dim,
            "depth": raw_model.metric.depth,
            "heads": raw_model.metric.num_heads,
        },
        "ot_config": {
            "ground_cost": "l2-normalized squared-euclidean",
            "node_mass": "pair-conditioned dynamic SK from raw metric tokens",
            "sinkhorn_reg": raw_model.sinkhorn_reg,
            "sinkhorn_iter": raw_model.sinkhorn_iter,
            "sk_weight_eps": raw_model.sk_weight_eps,
            "ot_pool_factor": raw_model.ot_pool_factor,
        },
        "global_cls_config": {
            "enabled": raw_model.global_cls_head is not None,
            "num_classes": int(raw_model.num_global_classes),
            "weight": float(getattr(args, "global_cls_weight", 0.0)),
            "lr": float(getattr(args, "global_cls_lr", 0.0)),
            "label_smoothing": float(getattr(args, "global_cls_label_smoothing", 0.0)),
            "target_scope": "every support image + every query image",
            "feature": "MetricEncoder cloud mean/global vector",
        },
        "episode_config": {
            # Keep the nominal CLI n_way for backward compatibility, but record
            # that manifest episodes may use their own candidate-set widths.
            "n_way": raw_model.n_way,
            "n_way_mode": (
                "dynamic_from_manifest_candidate_labels"
                if getattr(args, "train_episode_manifest", None)
                else "fixed_from_cli"
            ),
            "n_shot": raw_model.n_shot,
            "n_query_total": args.n_query,
            "episode_batch_size": int(getattr(args, "episode_batch_size", 1)),
            "episode_batch_semantics": "true vectorized independent episodes; no gradient accumulation",
            "query_protocol": "single query class + N-way distractors",
            "shot_aggregation": "differentiable soft-min",
            "data_mode": (
                "episode_manifest"
                if getattr(args, "train_episode_manifest", None)
                else "class_pool"
            ),
            "manifest_semantics": (
                "fixed query/candidates + online K-shot support sampling; "
                "candidate/support pairs shuffled together; query excluded from support; "
                "low-resource support pools sampled with replacement; empty-pool records skipped"
                if getattr(args, "train_episode_manifest", None)
                else None
            ),
        },
    }
    torch.save(state, path)


def resume_m2(
    raw_model: DirectEpisodeMetricModel,
    optimizer,
    scheduler,
    path: str,
    *,
    global_label_to_id: Optional[dict[str, int]] = None,
) -> int:
    ckpt = _torch_load(path, "cpu")
    raw_model.metric.load_state_dict(extract_metric_state(ckpt), strict=True)

    if "log_shot_tau" in ckpt:
        raw_model.log_shot_tau.data.copy_(ckpt["log_shot_tau"].to(raw_model.log_shot_tau))
    if "log_cls_tau" in ckpt:
        raw_model.log_cls_tau.data.copy_(ckpt["log_cls_tau"].to(raw_model.log_cls_tau))

    saved_map = ckpt.get("global_label_to_id") or {}
    if raw_model.global_cls_head is not None:
        if saved_map and global_label_to_id is not None and dict(saved_map) != dict(global_label_to_id):
            raise RuntimeError(
                "resume checkpoint global label map differs from current training data; "
                "refusing to silently change auxiliary class semantics"
            )
        head_state = ckpt.get("global_cls_head")
        if isinstance(head_state, dict):
            raw_model.global_cls_head.load_state_dict(head_state, strict=True)
        elif float(getattr(raw_model, "num_global_classes", 0)) > 0:
            print("[WARN] resume checkpoint has no global_cls_head; auxiliary head starts fresh")

    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, RuntimeError) as exc:
            print(f"[WARN] optimizer state not restored because parameter groups changed: {exc}")
    if "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except (ValueError, RuntimeError) as exc:
            print(f"[WARN] scheduler state not restored: {exc}")
    return int(ckpt.get("epoch", 0))


# -----------------------------------------------------------------------------
# Training / validation
# -----------------------------------------------------------------------------


def batch_episode_metrics(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-episode accuracy / positive energy / negative energy / margin."""
    logits = output["logits"]
    energy = output["class_energy"]
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
        energy = energy.unsqueeze(0)
        target = target.unsqueeze(0) if target.ndim == 1 else target
    if logits.ndim != 3 or target.ndim != 2:
        raise ValueError(
            f"expected logits=(E,Q,N), target=(E,Q), got {tuple(logits.shape)}, {tuple(target.shape)}"
        )

    pred = logits.argmax(dim=-1)
    episode_acc = pred.eq(target).float().mean(dim=1)  # (E,)

    e, q, n = energy.shape
    pos = energy.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mask = F.one_hot(target, num_classes=n).bool()
    neg = energy.masked_fill(mask, float("inf")).min(dim=-1).values
    episode_pos = pos.mean(dim=1)
    episode_neg = neg.mean(dim=1)
    episode_margin = (neg - pos).mean(dim=1)
    return episode_acc, episode_pos, episode_neg, episode_margin


def finalize_stats(stats: EpisodeStats, device: torch.device) -> dict[str, float]:
    total = all_reduce_stats(stats, device)
    n = max(total.count, 1)
    mean_acc = total.acc_sum / n
    var = max(total.acc_sq_sum / n - mean_acc * mean_acc, 0.0)
    global_n = max(total.global_cls_count, 1)
    # Episode-level normal-approximation CI. This matches the project's usual
    # mean +/- 95% CI reporting and works for n_query=1 binary episode accuracy.
    ci95 = 1.96 * math.sqrt(var / n)
    return {
        "loss": total.loss_sum / n,
        "ce_loss": total.ce_sum / n,
        "vis_loss": total.vis_sum / n,
        "global_cls_loss": (
            total.global_cls_loss_sum / global_n
            if total.global_cls_count > 0 else 0.0
        ),
        "global_cls_accuracy": (
            total.global_cls_correct / global_n
            if total.global_cls_count > 0 else 0.0
        ),
        "global_cls_samples": float(total.global_cls_count),
        "accuracy": mean_acc,
        "ci95": ci95,
        "pos_energy": total.pos_sum / n,
        "neg_energy": total.neg_sum / n,
        "margin": total.margin_sum / n,
        "episodes": float(total.count),
    }


def set_metric_trainable(raw_model: DirectEpisodeMetricModel, trainable: bool) -> None:
    for p in raw_model.metric.parameters():
        p.requires_grad_(trainable)


def run_epoch(
    *,
    model,
    raw_model: DirectEpisodeMetricModel,
    rae,
    visreg: VISReg,
    pool: Optional[dict[str, list[str]]],
    manifest_records: Optional[list[EpisodeManifestRecord]],
    transform,
    device: torch.device,
    rng: random.Random,
    n_episodes: int,
    args,
    global_label_to_id: Optional[dict[str, int]] = None,
    optimizer=None,
    scaler=None,
    training: bool,
    rank: int,
) -> dict[str, float]:
    if training:
        model.train()
    else:
        model.eval()

    stats = EpisodeStats()
    amp_enabled = bool(args.amp and device.type == "cuda")
    # True episode mini-batching is a TRAINING optimization only.
    # Validation intentionally remains one episode per forward so the evaluation
    # protocol/semantics stay identical to the original M2 evaluator.
    episode_batch_size = (
        max(1, int(getattr(args, "episode_batch_size", 1))) if training else 1
    )

    use_manifest = manifest_records is not None
    if use_manifest == (pool is not None):
        raise ValueError("run_epoch 必须且只能提供 pool 或 manifest_records 之一")

    use_global_cls = raw_model.global_cls_head is not None
    if use_global_cls and not global_label_to_id:
        raise RuntimeError("global classification is enabled but global_label_to_id is empty")

    manifest_order: list[int] = []
    if use_manifest:
        if not manifest_records:
            raise RuntimeError("manifest_records 为空")
        manifest_order = list(range(len(manifest_records)))
        rng.shuffle(manifest_order)

    episode_cursor = 0
    batch_index = 0
    while episode_cursor < n_episodes:
        requested_bs = min(episode_batch_size, n_episodes - episode_cursor)
        episodes: list[EpisodePlan] = []
        for local_i in range(requested_bs):
            step = episode_cursor + local_i
            if use_manifest:
                cycle_pos = step % len(manifest_order)
                if step > 0 and cycle_pos == 0:
                    rng.shuffle(manifest_order)
                record = manifest_records[manifest_order[cycle_pos]]
                ep = sample_manifest_episode(
                    record,
                    n_shot=args.n_shot,
                    n_query=args.n_query,
                    rng=rng,
                )
            else:
                ep = sample_episode(
                    pool,
                    n_way=args.n_way,
                    n_shot=args.n_shot,
                    n_query=args.n_query,
                    rng=rng,
                )
            episodes.append(ep)

        # TRUE vectorized episode batches require the same tensor width.  Normal
        # class-pool training always satisfies this.  Dynamic-way manifest batches
        # are split into same-way subgroups without ever padding fake candidates.
        groups: dict[int, list[EpisodePlan]] = {}
        for ep in episodes:
            groups.setdefault(len(ep.classes), []).append(ep)

        for episode_n_way, group in groups.items():
            if episode_n_way < 2:
                raise RuntimeError(f"episode has only {episode_n_way} candidate classes")
            bs = len(group)
            support_count = episode_n_way * int(args.n_shot)

            support_imgs = []
            query_imgs = []
            targets = []
            global_targets = []
            global_known_masks = []
            for ep in group:
                support_img, query_img = load_episode_images(ep, transform)
                if support_img.shape[0] != support_count:
                    raise RuntimeError(
                        f"support images={support_img.shape[0]} != way*shot={support_count}"
                    )
                if query_img.shape[0] != int(args.n_query):
                    raise RuntimeError(
                        f"query images={query_img.shape[0]} != n_query={args.n_query}"
                    )
                support_imgs.append(support_img)
                query_imgs.append(query_img)
                targets.append(
                    torch.full((args.n_query,), int(ep.query_label), dtype=torch.long)
                )
                if use_global_cls:
                    gt, known = episode_global_targets(
                        ep,
                        n_shot=args.n_shot,
                        n_query=args.n_query,
                        label_to_id=global_label_to_id,
                        device=torch.device("cpu"),
                        require_all_known=training,
                    )
                    global_targets.append(gt)
                    global_known_masks.append(known)

            # TRUE RAE image batch: all supports from all episodes, then all queries.
            support_img_all = torch.cat(support_imgs, dim=0).to(device, non_blocking=True)
            query_img_all = torch.cat(query_imgs, dim=0).to(device, non_blocking=True)
            all_img = torch.cat([support_img_all, query_img_all], dim=0)
            with torch.no_grad():
                latent = encode_rae_chunked(rae, all_img, args.rae_chunk_size)

            ns_total = bs * support_count
            support_z = latent[:ns_total].reshape(
                bs, support_count, *latent.shape[1:]
            )
            query_z = latent[ns_total:].reshape(
                bs, int(args.n_query), *latent.shape[1:]
            )
            target = torch.stack(targets, dim=0).to(device, non_blocking=True)

            global_target = None
            global_known = None
            if use_global_cls:
                global_target = torch.stack(global_targets, dim=0).to(device, non_blocking=True)
                global_known = torch.stack(global_known_masks, dim=0).to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            grad_context = torch.enable_grad() if training else torch.no_grad()
            with grad_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    # ONE model forward for E independent episodes.  Inside the model:
                    #   1) ONE MetricEncoder call for all E*(S+Q) images;
                    #   2) ONE batched Dynamic-SK/Sinkhorn call with leading E dim.
                    output = model(support_z, query_z)

                    logits = output["logits"]
                    ce_loss = F.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        target.reshape(-1),
                    )

                    if float(args.visreg_weight) > 0.0:
                        # MetricEncoder.cloud is already one D-vector per image.
                        # Aggregate all independent cloud embeddings in the true
                        # episode meta-batch, matching VISReg's (1,B,D) semantics.
                        episode_cloud = torch.cat(
                            [output["support_cloud"], output["query_cloud"]], dim=1
                        )  # (E,S+Q,D)
                        vis_input = episode_cloud.reshape(
                            1, -1, episode_cloud.shape[-1]
                        ).float()
                        vis_loss = visreg(vis_input)
                    else:
                        vis_loss = ce_loss.new_zeros(())

                    if use_global_cls and bool(global_known.any().item()):
                        global_logits = output["global_logits"]
                        if global_logits is None:
                            raise RuntimeError("global_cls_head enabled but forward returned no logits")
                        if global_logits.shape[:2] != global_target.shape:
                            raise RuntimeError(
                                f"global logits leading shape={global_logits.shape[:2]} != "
                                f"targets={tuple(global_target.shape)}"
                            )
                        global_cls_loss = F.cross_entropy(
                            global_logits[global_known].float(),
                            global_target[global_known],
                            label_smoothing=float(args.global_cls_label_smoothing),
                        )
                    else:
                        global_cls_loss = ce_loss.new_zeros(())

                    loss = (
                        ce_loss
                        + float(args.visreg_weight) * vis_loss
                        + float(args.global_cls_weight) * global_cls_loss
                    )

                if training:
                    # This is NOT gradient accumulation: one batch -> one backward -> one step.
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.grad_clip,
                    )
                    scaler.step(optimizer)
                    scaler.update()

            with torch.no_grad():
                ep_acc, ep_pos, ep_neg, ep_margin = batch_episode_metrics(output, target)
                stats.loss_sum += float(loss.detach().item()) * bs
                stats.ce_sum += float(ce_loss.detach().item()) * bs
                stats.vis_sum += float(vis_loss.detach().item()) * bs
                stats.acc_sum += float(ep_acc.sum().item())
                stats.acc_sq_sum += float((ep_acc * ep_acc).sum().item())
                stats.pos_sum += float(ep_pos.sum().item())
                stats.neg_sum += float(ep_neg.sum().item())
                stats.margin_sum += float(ep_margin.sum().item())
                stats.count += bs

                global_cls_v = 0.0
                global_acc_v = 0.0
                global_count_v = 0
                if use_global_cls and bool(global_known.any().item()):
                    global_count_v = int(global_known.sum().item())
                    global_cls_v = float(global_cls_loss.detach().item())
                    pred_global = output["global_logits"][global_known].argmax(dim=-1)
                    correct_global = int(
                        (pred_global == global_target[global_known]).sum().item()
                    )
                    global_acc_v = correct_global / max(global_count_v, 1)
                    stats.global_cls_loss_sum += global_cls_v * global_count_v
                    stats.global_cls_correct += float(correct_global)
                    stats.global_cls_count += global_count_v

            batch_index += 1
            if training and rank == 0 and args.log_every > 0 and batch_index % args.log_every == 0:
                msg = (
                    f"  [batch {batch_index:05d}] episodes={stats.count}/{n_episodes} "
                    f"true_bs={bs} way={episode_n_way} "
                    f"loss={float(loss.detach()):.4f} epi_ce={float(ce_loss.detach()):.4f} "
                    f"vis={float(vis_loss.detach()):.4f} "
                )
                if use_global_cls:
                    msg += (
                        f"global_ce={global_cls_v:.4f} global_acc={100*global_acc_v:.1f}% "
                        f"global_n={global_count_v} "
                    )
                msg += (
                    f"epi_acc={float(ep_acc.mean()):.3f} "
                    f"margin={float(ep_margin.mean()):.4f} "
                    f"tau_shot={float(raw_model.shot_tau.detach()):.4f} "
                    f"tau_cls={float(raw_model.cls_tau.detach()):.4f}"
                )
                print(msg)

            del all_img, latent, support_z, query_z, output

        episode_cursor += requested_bs

    return finalize_stats(stats, device)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.n_way < 2:
        raise ValueError("--n_way must be >=2")
    if args.n_shot < 1 or args.n_query < 1:
        raise ValueError("--n_shot/--n_query must be >=1")
    if args.ot_pool_factor < 1:
        raise ValueError("--ot_pool_factor must be >=1")
    if args.freeze_metric_epochs < 0:
        raise ValueError("--freeze_metric_epochs must be >=0")
    if args.episode_batch_size < 1:
        raise ValueError("--episode_batch_size must be >=1")
    if args.global_cls_weight < 0:
        raise ValueError("--global_cls_weight must be >=0")
    if args.global_cls_lr <= 0:
        raise ValueError("--global_cls_lr must be >0")
    if not (0.0 <= args.global_cls_label_smoothing < 1.0):
        raise ValueError("--global_cls_label_smoothing must be in [0,1)")
    if args.visreg_weight < 0:
        raise ValueError("--visreg_weight must be >=0")
    if args.visreg_num_proj <= 0:
        raise ValueError("--visreg_num_proj must be >0")
    if args.visreg_projection_chunk <= 0:
        raise ValueError("--visreg_projection_chunk must be >0")

    device, rank, world_size, is_main = setup_distributed()
    seed_everything(args.seed, rank)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------- data -------------------------
    train_pool: Optional[dict[str, list[str]]] = None
    val_pool: Optional[dict[str, list[str]]] = None
    train_manifest_records: Optional[list[EpisodeManifestRecord]] = None
    val_manifest_records: Optional[list[EpisodeManifestRecord]] = None

    if args.train_episode_manifest:
        # New preferred mode: each record fixes query + candidate set + support dirs.
        if args.val_root or args.val_json_glob:
            raise ValueError(
                "train 使用 --train_episode_manifest 时，验证请使用 "
                "--val_episode_manifest，或不提供验证源并用 --val_fraction 自动划分 manifest"
            )

        all_train_manifest = load_episode_manifest(
            args.train_episode_manifest,
            n_way=args.n_way,
            n_shot=args.n_shot,
            n_query=args.n_query,
        )
        if args.val_episode_manifest:
            train_manifest_records = all_train_manifest
            val_manifest_records = load_episode_manifest(
                args.val_episode_manifest,
                n_way=args.n_way,
                n_shot=args.n_shot,
                n_query=args.n_query,
            )
        else:
            train_manifest_records, val_manifest_records = split_manifest_records(
                all_train_manifest,
                args.val_fraction,
                seed=args.seed,
            )
        data_mode = "episode_manifest"

    else:
        # Train uses class-pool mode (--train_root or --train_json_glob).
        # Validation is intentionally independent and may use:
        #   1) --val_episode_manifest  (preferred fixed validation protocol),
        #   2) --val_root / --val_json_glob, or
        #   3) an automatic split from the training class pool.
        train_pool_raw = load_class_pool(args.train_root, args.train_json_glob)

        if args.val_episode_manifest:
            # Mixed mode requested by this experiment:
            #   train = random N-way episodes sampled from a JSON class pool
            #   val   = fixed query/candidate episode manifest with online K-shot support
            train_pool = train_pool_raw
            val_manifest_records = load_episode_manifest(
                args.val_episode_manifest,
                n_way=args.n_way,
                n_shot=args.n_shot,
                n_query=args.n_query,
            )
            data_mode = "class_pool_train+manifest_val"

        elif args.val_root or args.val_json_glob:
            train_pool = train_pool_raw
            val_pool = load_class_pool(args.val_root, args.val_json_glob)
            data_mode = "class_pool"

        else:
            # Need K+Q samples on both sides because any class can become GT query.
            min_side = args.n_shot + args.n_query
            train_pool, val_pool = stratified_split(
                train_pool_raw,
                args.val_fraction,
                seed=args.seed,
                min_train=min_side,
                min_val=min_side,
            )
            data_mode = "class_pool"

        # Training pool always needs enough examples to form a complete random episode.
        train_pool = filter_episode_classes(
            train_pool, n_shot=args.n_shot, n_query=args.n_query
        )
        if len(train_pool) < args.n_way:
            raise RuntimeError(
                f"train valid classes={len(train_pool)} < n_way={args.n_way}"
            )

        # Only class-pool validation needs class-count filtering.  Manifest validation
        # defines its own candidate width per record and is validated when loaded.
        if val_pool is not None:
            val_pool = filter_episode_classes(
                val_pool, n_shot=args.n_shot, n_query=args.n_query
            )
            if len(val_pool) < args.n_way:
                raise RuntimeError(
                    f"val valid classes={len(val_pool)} < n_way={args.n_way}"
                )

    # Stable global label space is built ONCE from the training source.
    # It is independent of the shuffled local episode slots.
    global_label_to_id: dict[str, int] = {}
    if float(args.global_cls_weight) > 0.0:
        global_label_to_id = build_global_label_map(train_pool, train_manifest_records)

    if is_main:
        print("=" * 96)
        print("[M2] Direct episodic SK-Wasserstein calibration")
        print(f"data mode = {data_mode}")
        if data_mode == "episode_manifest":
            print(
                f"train manifest records={len(train_manifest_records)}, "
                f"val manifest records={len(val_manifest_records)}"
            )
            print(
                "manifest protocol = fixed query + fixed candidate set + "
                "online K-shot support sampling (replacement allowed for low-resource pools)"
            )
            print(
                "candidate label/support-dir pairs are shuffled together; "
                "query paths are excluded from all support pools"
            )
            train_way_values = sorted({len(r.candidate_labels) for r in train_manifest_records})
            val_way_values = sorted({len(r.candidate_labels) for r in val_manifest_records})
            print(
                f"manifest candidate ways = train{train_way_values}, val{val_way_values}; "
                "actual n-way is taken from each record"
            )
            print(
                f"episode=dynamic-way {args.n_shot}-shot "
                f"{args.n_query}-query(total, single GT class)"
            )
        elif data_mode == "class_pool_train+manifest_val":
            val_way_values = sorted({len(r.candidate_labels) for r in val_manifest_records})
            print(f"train classes={len(train_pool)}")
            print(f"val manifest records={len(val_manifest_records)}")
            print(
                f"train episode={args.n_way}-way {args.n_shot}-shot "
                f"{args.n_query}-query(total, single GT class), sampled online from class pool"
            )
            print(
                f"val manifest candidate ways={val_way_values}; "
                "actual validation n-way is taken from each manifest record"
            )
            print(
                "val manifest protocol = fixed query + fixed candidate set + "
                "online K-shot support sampling"
            )
        else:
            print(f"train classes={len(train_pool)}, val classes={len(val_pool)}")
            print(
                f"episode={args.n_way}-way {args.n_shot}-shot "
                f"{args.n_query}-query(total, single GT class)"
            )
        print(
            "OT = raw-token Dynamic SK + L2-normalized squared-Euclidean "
            f"+ Sinkhorn(reg={args.sinkhorn_reg}, iter={args.sinkhorn_iter})"
        )
        print(f"OT nodes = {256 // (args.ot_pool_factor ** 2)}")
        print("K-shot aggregation = differentiable soft-min")
        print(
            "loss = episodic CE + "
            f"{args.visreg_weight:g} * VISReg + "
            f"{args.global_cls_weight:g} * global per-image CE"
        )
        print(
            f"TRUE episode batch = {args.episode_batch_size} episode(s)/rank/forward; "
            f"effective DDP episode batch = {args.episode_batch_size * world_size} episode(s)/update"
        )
        print("VISReg input = all support/query cloud embeddings across the TRUE episode mini-batch")
        if global_label_to_id:
            print(
                f"global CLS = enabled, classes={len(global_label_to_id)}, "
                f"head_lr={args.global_cls_lr:g}, targets=EVERY support+query image"
            )
        else:
            print("global CLS = disabled")
        print("RAE = frozen; Flow = disabled")
        print("=" * 96)

    train_transform = build_transform(args.image_size, augment=args.augment)
    val_transform = build_transform(args.image_size, augment=False)

    # ------------------------- models -------------------------
    rae = build_rae(args.project_root, device)
    metric = MetricEncoder(
        in_dim=1024,
        hidden_dim=args.hidden_dim,
        out_dim=args.metric_dim,
        dropout=args.dropout,
    )

    raw_model = DirectEpisodeMetricModel(
        metric,
        n_way=args.n_way,
        n_shot=args.n_shot,
        sinkhorn_reg=args.sinkhorn_reg,
        sinkhorn_iter=args.sinkhorn_iter,
        sk_weight_eps=args.sk_weight_eps,
        shot_tau_init=args.shot_tau_init,
        cls_tau_init=args.cls_tau_init,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        ot_pool_factor=args.ot_pool_factor,
        num_global_classes=len(global_label_to_id),
    ).to(device)

    visreg = VISReg(
        num_projections=args.visreg_num_proj,
        projection_chunk_size=args.visreg_projection_chunk,
    ).to(device)

    # Metric moves gently; temperatures calibrate faster; the fresh auxiliary
    # classification head gets its own LR.
    param_groups = [
        {
            "params": list(raw_model.metric.parameters()),
            "lr": args.metric_lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [raw_model.log_shot_tau, raw_model.log_cls_tau],
            "lr": args.temperature_lr,
            "weight_decay": 0.0,
        },
    ]
    if raw_model.global_cls_head is not None:
        param_groups.append(
            {
                "params": list(raw_model.global_cls_head.parameters()),
                "lr": args.global_cls_lr,
                "weight_decay": args.weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=0.0
    )

    start_epoch = 0
    if args.init_checkpoint:
        load_m1_metric(raw_model.metric, args.init_checkpoint)
        if is_main:
            print(f"[M2] initialized MetricEncoder from M1: {args.init_checkpoint}")

    # DDP wraps only the trainable M2 model, not the frozen RAE.
    model = raw_model
    if world_size > 1:
        model = DDP(
            raw_model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    if args.resume_checkpoint:
        start_epoch = resume_m2(
            raw_model, optimizer, scheduler, args.resume_checkpoint,
            global_label_to_id=global_label_to_id,
        )
        if is_main:
            print(f"[M2] resumed from epoch={start_epoch}: {args.resume_checkpoint}")

    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(args.amp and device.type == "cuda")
    )

    best_acc = -1.0
    best_margin = -float("inf")

    # Fixed validation seed every epoch -> comparable val episodes.
    val_seed = args.seed + 9_000_001 + rank * 100_003

    for epoch in range(start_epoch, args.epochs):
        metric_trainable = epoch >= args.freeze_metric_epochs
        set_metric_trainable(raw_model, metric_trainable)

        # Different deterministic stream per epoch and rank.
        train_rng = random.Random(
            args.seed + epoch * 1_000_003 + rank * 100_003
        )
        val_rng = random.Random(val_seed)

        if is_main:
            print(
                f"\n[M2][Epoch {epoch+1}/{args.epochs}] "
                f"metric_trainable={metric_trainable} "
                f"metric_lr={optimizer.param_groups[0]['lr']:.3e} "
                f"temp_lr={optimizer.param_groups[1]['lr']:.3e}"
                + (
                    f" global_cls_lr={optimizer.param_groups[2]['lr']:.3e}"
                    if raw_model.global_cls_head is not None else ""
                )
            )

        train_metrics = run_epoch(
            model=model,
            raw_model=raw_model,
            rae=rae,
            visreg=visreg,
            pool=train_pool,
            manifest_records=train_manifest_records,
            transform=train_transform,
            device=device,
            rng=train_rng,
            n_episodes=args.episodes_per_epoch,
            args=args,
            global_label_to_id=global_label_to_id,
            optimizer=optimizer,
            scaler=scaler,
            training=True,
            rank=rank,
        )

        val_metrics = run_epoch(
            model=model,
            raw_model=raw_model,
            rae=rae,
            visreg=visreg,
            pool=val_pool,
            manifest_records=val_manifest_records,
            transform=val_transform,
            device=device,
            rng=val_rng,
            n_episodes=args.val_episodes,
            args=args,
            global_label_to_id=global_label_to_id,
            training=False,
            rank=rank,
        )
        scheduler.step()

        if is_main:
            print(
                f"[M2][{epoch+1:03d}/{args.epochs}] "
                f"train loss={train_metrics['loss']:.4f} "
                f"(epi_ce={train_metrics['ce_loss']:.4f}, vis={train_metrics['vis_loss']:.4f}, "
                f"global_ce={train_metrics['global_cls_loss']:.4f}, "
                f"global_acc={100*train_metrics['global_cls_accuracy']:.2f}%) "
                f"epi_acc={100*train_metrics['accuracy']:.2f}% "
                f"margin={train_metrics['margin']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} "
                f"(epi_ce={val_metrics['ce_loss']:.4f}, vis={val_metrics['vis_loss']:.4f}, "
                f"global_ce={val_metrics['global_cls_loss']:.4f}, "
                f"global_acc={100*val_metrics['global_cls_accuracy']:.2f}%) "
                f"epi_acc={100*val_metrics['accuracy']:.2f}% "
                f"±{100*val_metrics['ci95']:.2f}% "
                f"margin={val_metrics['margin']:.4f} | "
                f"tau_shot={float(raw_model.shot_tau.detach()):.4f} "
                f"tau_cls={float(raw_model.cls_tau.detach()):.4f}"
            )

            save_m2_checkpoint(
                os.path.join(args.output_dir, "metric_direct_episode_last.pt"),
                raw_model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                args=args,
                val_metrics=val_metrics,
                global_label_to_id=global_label_to_id,
            )

            acc = val_metrics["accuracy"]
            margin = val_metrics["margin"]
            better = acc > best_acc + 1e-12 or (
                abs(acc - best_acc) <= 1e-12 and margin > best_margin
            )
            if better:
                best_acc = acc
                best_margin = margin
                save_m2_checkpoint(
                    os.path.join(args.output_dir, "metric_direct_episode.pt"),
                    raw_model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    args=args,
                    val_metrics=val_metrics,
                    global_label_to_id=global_label_to_id,
                )
                print(
                    f"[M2] new best: val_acc={100*best_acc:.2f}% "
                    f"margin={best_margin:.4f}"
                )

    cleanup_distributed()


if __name__ == "__main__":
    main()