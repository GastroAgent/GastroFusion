#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline RAE + Metric + fixed-8-step RAE-Flow FSL service.

Goal
----
Drop-in conceptual replacement for the old VAE-based ``offline_align_fsl_service.py``
while preserving its external/offline protocol:

    pre-existing flat MLLM/FSL query-support JSON
        -> pair-level FSL evidence only (NO MLLM rerun, NO episode resampling)
        -> group by query and support class
        -> support-level aggregation
        -> top1/top2/gap/ranking
        -> fusion-compatible JSON

This implementation upgrades only the internal FSL geometry:

    image -> frozen RAE -> MetricEncoder -> Sinkhorn OT

and evaluates TWO OT mass variants under the SAME ground cost:

1) uniform:
       uniform node masses on both token sets.
2) dynamic:
       DeepEMD-style pair-conditioned SK masses from raw metric tokens.

For Flow, a frozen Vision-Hint DiT generates a FIXED 8-step trajectory.  There is
NO similarity network, NO early-stop, NO stop_beta/stop_method.  Dynamic-SK masses
are computed ONCE from the direct t=0 query-support pair and reused over the entire
trajectory, matching the canonical fixed-SK-across-trajectory rule of this project.

For every query-support pair, both uniform and dynamic variants save:

    direct
    tolast      : D(z_t, z_8), t=0..7
    neighbor    : D(z_t, z_{t+1}), t=0..7
    tosupport   : D(z_t, z_support), t=0..8

plus mean/sum/std/min/max/slope/AUC/monotonic summaries and all per-step values.

Outputs
-------
output_dir/
    result.json
        Original flat pair records enriched with all RAE/Metric/Flow evidence.
    query_results.json
        Per-query class distances/rankings for every score x aggregation.
    summary.json
        Overall/per-class metrics, rescue/degradation, step-wise metrics, oracles.
    score_summary.csv
        Flat overview of every score x aggregation.
    fsl_full_results_for_fusion_uniform.json
    fsl_full_results_for_fusion_dynamic.json
        Old offline FSL/MLLM fusion field names retained.

Primary fusion policy (kept intentionally conservative/compatible)
------------------------------------------------------------------
    trajectory score : tolast_mean
    direct score     : direct
    support aggregate: min

The additional Flow distances and mean/softmin support aggregations are analysis
outputs; they do NOT silently change the legacy fusion-ready primary fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter, OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# =============================================================================
# Defaults
#
# All path-like CLI arguments have editable defaults here.  Nothing in this
# script is argparse-required, so the common workflow is simply:
#   1) edit the DEFAULT_* paths below once;
#   2) run: python offline_align_rae_metric_flow_service.py
#
# CLI arguments are still kept as optional overrides for temporary experiments.
# =============================================================================

SCRIPT_VERSION = "2026-08-19-allsupport-v1"

DEFAULT_PROJECT_ROOT = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow"

# Existing flat query-support JSON produced by the prior MLLM/FSL pipeline.
DEFAULT_DATA_PATH = (
    "/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/"
    "data_tsy_deepemd/mllm_fsl_subset_agent_lot_way/苍溪/十二指肠/"
    "final_doctor_exam_苍溪_十二指肠_huaxi_support_flat.json"
)

# RAE MetricEncoder checkpoints: organ-specific test-test M2 models.
# When --metric_checkpoint is omitted, the script infers the organ from --data_path
# and automatically selects the corresponding checkpoint below.
DEFAULT_METRIC_CHECKPOINTS = {
    "十二指肠": os.path.join(
        DEFAULT_PROJECT_ROOT,
        "weights_metric_v1",
        "new_m2_十二指肠-test-train",
        "metric_direct_episode.pt",
    ),
    "食管": os.path.join(
        DEFAULT_PROJECT_ROOT,
        "weights_metric_v1",
        "new_m2_食管-test-train",
        "metric_direct_episode.pt",
    ),
    "胃": os.path.join(
        DEFAULT_PROJECT_ROOT,
        "weights_metric_v1",
        "new_m2_胃-test-train",
        "metric_direct_episode.pt",
    ),
}


def infer_organ_from_data_path(data_path: str) -> str:
    """Infer organ from the input JSON path.

    Match longer organ names first so that 十二指肠 is resolved before any
    shorter substring.  The current project uses exactly these three organs.
    """
    text = str(data_path)
    for organ in ("十二指肠", "食管", "胃"):
        if organ in text:
            return organ
    raise ValueError(
        "Cannot infer organ from --data_path. Expected one of "
        "['十二指肠', '食管', '胃'] in the path, but got:\n"
        f"  {data_path}\n"
        "Pass --metric_checkpoint explicitly if this input uses a non-standard path."
    )


def resolve_default_metric_checkpoint(data_path: str) -> tuple[str, str]:
    """Return (organ, checkpoint) for the current input data path."""
    organ = infer_organ_from_data_path(data_path)
    checkpoint = DEFAULT_METRIC_CHECKPOINTS[organ]
    return organ, checkpoint

# Frozen RAE Vision-Hint DiT Flow checkpoint.
DEFAULT_FLOW_CHECKPOINT = os.path.join(
    DEFAULT_PROJECT_ROOT,
    "weights_dit_flow_v1",
    "vision_hint_inclass",
    "otcfm_dit_L_step_265000.pt",
)

# Offline result directory.  The script creates it automatically.
DEFAULT_OUTPUT_DIR = (
    "/mnt/inaisfs/data/home/tansy_criait/qsz/qcjw/our_method/批量测试/result/"
    "offline_align_rae_metric_flow_service/mllm_fsl_subset_agent_lot_way"
)

# -----------------------------------------------------------------------------
# SUPPORT PROTOCOL
# -----------------------------------------------------------------------------
# This version intentionally matches the old VAE offline protocol:
# use ALL valid query-support pair rows already present in the input flat JSON.
# There is NO N-shot cap, NO support resampling, and NO episode regeneration.
#
# ``MAX_N_SHOT`` is retained only for backward CLI compatibility with old shell
# scripts that still pass ``--max_n_shot``.  It is NOT used for selection in
# this all-support version.
MAX_N_SHOT = 0

DEFAULT_NUM_STEPS = 8
DEFAULT_SOLVER = "heun"
DEFAULT_IMAGE_SIZE = 256
DEFAULT_RAE_CHUNK_SIZE = 8
DEFAULT_METRIC_CHUNK_SIZE = 16
DEFAULT_FLOW_PAIR_CHUNK = 2
DEFAULT_OT_POOL_FACTOR = 1
DEFAULT_SINKHORN_REG = 0.1
DEFAULT_SINKHORN_ITER = 30
DEFAULT_SK_WEIGHT_EPS = 1e-3
DEFAULT_SOFTMIN_TAU = 0.10
DEFAULT_SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# Generic helpers
# =============================================================================


def _torch_load(path: str, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        # Some older project checkpoints contain metadata that requires the
        # historical pickle path. This fallback is only for trusted local files.
        return torch.load(path, map_location=map_location, weights_only=False)


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _first_path(value: Any) -> Optional[str]:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _resolve_image_path(value: Any, base_dir: Path) -> Optional[str]:
    path = _first_path(value)
    if path is None:
        return None
    p = Path(path).expanduser()
    if not p.is_absolute():
        candidate = (base_dir / p).resolve()
        if candidate.exists():
            p = candidate
    return str(p)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        x = float(value)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return x if np.isfinite(x) else None
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _json_safe(value.item())
        return [_json_safe(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64))) if values else 0.0


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


def _ci95_binary(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    p = correct / total
    return 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / total)


def _linear_slope(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    y = np.asarray(values, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64)
    x = x - x.mean()
    denom = float(np.square(x).sum())
    if denom <= 0:
        return 0.0
    return float((x * (y - y.mean())).sum() / denom)


def _normalized_auc(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    y = np.asarray(values, dtype=np.float64)
    return float(np.trapz(y, dx=1.0) / (len(y) - 1))


def _decreasing_rate(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 1.0
    y = np.asarray(values, dtype=np.float64)
    return float(np.mean(np.diff(y) <= 0.0))


def trajectory_stats(values: Sequence[float], *, step_offset: int = 0) -> Dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {
            "all": [], "mean": None, "sum": None, "std": None,
            "min": None, "max": None, "first": None, "last": None,
            "delta_last_minus_first": None, "slope": None,
            "auc": None, "decreasing_rate": None, "argmin_step": None,
        }
    arr = np.asarray(vals, dtype=np.float64)
    argmin = int(np.argmin(arr)) + int(step_offset)
    return {
        "all": vals,
        "mean": float(arr.mean()),
        "sum": float(arr.sum()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "first": float(arr[0]),
        "last": float(arr[-1]),
        "delta_last_minus_first": float(arr[-1] - arr[0]),
        "slope": _linear_slope(vals),
        "auc": _normalized_auc(vals),
        "decreasing_rate": _decreasing_rate(vals),
        "argmin_step": argmin,
    }


def _softmin(values: Sequence[float], tau: float) -> float:
    if not values:
        return float("nan")
    if tau <= 0:
        raise ValueError("softmin tau must be >0")
    x = np.asarray(values, dtype=np.float64)
    z = -x / float(tau)
    m = float(z.max())
    logmeanexp = m + math.log(float(np.exp(z - m).mean()))
    return float(-tau * logmeanexp)


def aggregate_support(values: Sequence[float], method: str, tau: float) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        return float("nan")
    if method == "min":
        return float(min(vals))
    if method == "mean":
        return float(np.mean(vals))
    if method == "softmin":
        return _softmin(vals, tau)
    raise ValueError(f"unknown support aggregation: {method}")


# =============================================================================
# Input flat JSON
# =============================================================================


def load_flat_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict):
        records = None
        for key in ("results", "records", "data", "items"):
            if isinstance(obj.get(key), list):
                records = obj[key]
                break
        if records is None:
            raise ValueError(
                "Flat JSON dict must contain one list field among results/records/data/items"
            )
    else:
        raise TypeError(f"Unsupported JSON root type: {type(obj).__name__}")
    return [dict(r) for r in records if isinstance(r, dict)]


@dataclass
class PairSpec:
    index: int
    query_path: str
    support_path: str
    matched_class: str
    ground_truth: Optional[str]


def _first_present(rec: Mapping[str, Any], keys: Sequence[str]) -> Tuple[Any, Optional[str]]:
    """Return the first non-empty value and the key that supplied it.

    Important for legacy offline JSON: the raw flat file stores image paths in
    ``x0`` / ``x1``.  ``MedicalJsonDataset`` only creates ``x0_path`` /
    ``x1_path`` later inside ``__getitem__``.  Because this RAE offline service
    reads the raw JSON directly, x0/x1 MUST be supported here.
    """
    for key in keys:
        value = rec.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, tuple)) and len(value) == 0:
            continue
        return value, key
    return None, None


def parse_pair_specs(
    records: Sequence[Mapping[str, Any]],
    data_path: str,
) -> Tuple[List[PairSpec], List[int]]:
    """Parse the SAME raw pair protocol consumed by old MedicalJsonDataset.

    Canonical legacy raw fields:
        x0 / label_A = query image / ground-truth class
        x1 / label_B = support image / support class

    ``x0_path`` / ``x1_path`` are accepted too because old result.json files
    contain those aliases after MedicalJsonDataset processing.
    """
    base_dir = Path(data_path).expanduser().resolve().parent
    pairs: List[PairSpec] = []
    skipped: List[int] = []

    query_keys = (
        "x0",              # canonical RAW flat JSON used by MedicalJsonDataset
        "x0_path",         # processed/offline result alias
        "target_image",
        "query_path",
        "query_image",
        "image_path",
    )
    support_keys = (
        "x1",              # canonical RAW flat JSON used by MedicalJsonDataset
        "x1_path",         # processed/offline result alias
        "support_image",
        "support_path",
    )
    class_keys = (
        "label_B",
        "matched_class",
        "support_label",
        "class_name",
    )
    gt_keys = (
        "label_A",
        "ground_truth",
        "ground_truth_name",
        "query_label",
        "label",
    )

    source_counter: Counter = Counter()
    skip_reason_counter: Counter = Counter()

    for idx, rec in enumerate(records):
        q_value, q_key = _first_present(rec, query_keys)
        s_value, s_key = _first_present(rec, support_keys)
        cls, cls_key = _first_present(rec, class_keys)
        gt, gt_key = _first_present(rec, gt_keys)

        q = _resolve_image_path(q_value, base_dir)
        s = _resolve_image_path(s_value, base_dir)

        missing = []
        if q is None:
            missing.append("query")
        if s is None:
            missing.append("support")
        if cls is None:
            missing.append("support_class")
        if missing:
            skipped.append(idx)
            skip_reason_counter["+".join(missing)] += 1
            continue

        source_counter[f"query:{q_key}"] += 1
        source_counter[f"support:{s_key}"] += 1
        source_counter[f"class:{cls_key}"] += 1
        if gt_key is not None:
            source_counter[f"gt:{gt_key}"] += 1

        pairs.append(
            PairSpec(
                index=idx,
                query_path=str(q),
                support_path=str(s),
                matched_class=str(cls),
                ground_truth=None if gt is None else str(gt),
            )
        )

    if records:
        first_keys = sorted(str(k) for k in records[0].keys())
        print(f"[Input parser] first_record_keys={first_keys}")
    if source_counter:
        print(f"[Input parser] field_sources={dict(source_counter)}")
    if skip_reason_counter:
        print(f"[Input parser] skipped_reasons={dict(skip_reason_counter)}")

    return pairs, skipped


def select_support_pairs_per_query_class(
    pairs: Sequence[PairSpec],
    max_n_shot: int = 0,
) -> Tuple[List[PairSpec], Dict[str, Any]]:
    """Keep ALL valid support rows from the existing flat JSON.

    This all-support version is intentionally aligned with the old VAE offline
    evaluation protocol: every valid query-support row in the input JSON is kept.
    There is no N-shot cap, no support resampling, and no episode regeneration.

    ``max_n_shot`` is accepted only for backward compatibility with existing
    shell scripts.  Its value is ignored.  Expensive RAE/Flow computation is
    still de-duplicated later by the exact ``(query_path, support_path)`` pair,
    but all original rows remain in ``result.json`` and in class aggregation.
    Therefore the primary support-min classifier is exactly:

        D(q, c) = min_{s appearing for class c in the input JSON} D(q, s)
    """
    requested_max_n_shot = int(max_n_shot)

    selected: List[PairSpec] = list(pairs)
    raw_count: Counter = Counter()
    unique_supports_by_group: Dict[Tuple[str, str], set] = {}

    for pair in pairs:
        group_key = (pair.query_path, pair.matched_class)
        raw_count[group_key] += 1
        unique_supports_by_group.setdefault(group_key, set()).add(
            os.path.realpath(pair.support_path)
        )

    selected_count_distribution = Counter(raw_count.values())
    unique_count_distribution = Counter(
        len(v) for v in unique_supports_by_group.values()
    )
    duplicate_support_rows_present = sum(
        int(raw_count[k]) - len(unique_supports_by_group.get(k, set()))
        for k in raw_count
    )

    diagnostics = {
        "support_mode": "all_valid_pair_rows_from_json",
        "shot_cap_enabled": False,
        "requested_max_n_shot_ignored": requested_max_n_shot,
        "effective_max_n_shot": None,
        "num_query_class_groups": len(raw_count),
        "num_input_valid_pair_rows": len(pairs),
        "num_selected_pair_rows": len(selected),
        "duplicate_support_rows_present": int(duplicate_support_rows_present),
        "duplicate_support_rows_removed": 0,
        "unique_support_rows_dropped_by_shot_cap": 0,
        "raw_rows_per_query_class_distribution": {
            str(k): int(v) for k, v in sorted(selected_count_distribution.items())
        },
        "unique_supports_per_query_class_distribution": {
            str(k): int(v) for k, v in sorted(unique_count_distribution.items())
        },
        # Backward-compatible field name.  In all-support mode this is the
        # distribution of retained JSON rows per query/class, not a capped shot count.
        "selected_supports_per_query_class_distribution": {
            str(k): int(v) for k, v in sorted(selected_count_distribution.items())
        },
    }
    return selected, diagnostics


# =============================================================================
# RAE + Metric loading
# =============================================================================


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((int(image_size), int(image_size))),
        transforms.ToTensor(),  # [0,1], same as M2 validation / RAE Flow target input
    ])


def load_images(paths: Sequence[str], transform) -> torch.Tensor:
    images = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"unsupported image extension: {path}")
        with Image.open(path) as im:
            images.append(transform(im.convert("RGB")))
    return torch.stack(images, dim=0)


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
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_rae_chunked(rae: nn.Module, images: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_size = int(chunk_size) if int(chunk_size) > 0 else int(images.shape[0])
    parts = []
    for start in range(0, images.shape[0], chunk_size):
        parts.append(rae.encode(images[start : start + chunk_size]))
    latent = torch.cat(parts, dim=0)
    if latent.ndim != 4 or tuple(latent.shape[1:]) != (1024, 16, 16):
        raise RuntimeError(
            f"RAE latent expected (B,1024,16,16), got {tuple(latent.shape)}"
        )
    return latent.float()


def strip_prefix_if_all(state: Dict[str, torch.Tensor], prefix: str):
    if state and all(str(k).startswith(prefix) for k in state):
        return {str(k)[len(prefix):]: v for k, v in state.items()}
    return state


def extract_metric_state(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise TypeError("metric checkpoint must be a dict")
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
            str(k)[len("metric."):]: v
            for k, v in state.items()
            if str(k).startswith("metric.")
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
    raise KeyError("Cannot locate MetricEncoder weights in checkpoint")


def resolve_metric_arch(ckpt: Dict[str, Any]) -> Tuple[int, int, float]:
    saved_args = _as_dict(ckpt.get("args", {}))
    cfg = ckpt.get("metric_config", {})
    cfg = cfg if isinstance(cfg, dict) else {}
    hidden_dim = int(saved_args.get("hidden_dim", cfg.get("model_dim", 256)))
    metric_dim = int(saved_args.get("metric_dim", cfg.get("output_dim", 256)))
    dropout = float(saved_args.get("dropout", 0.0))
    return hidden_dim, metric_dim, dropout


def load_metric(project_root: str, checkpoint: str, device: torch.device):
    parent = str(Path(project_root).resolve() / "fsl_multi_vae")
    if parent not in sys.path:
        sys.path.insert(0, parent)
    project_root = str(Path(project_root).resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from rae_flow_metric_v1.models.metric_encoder import MetricEncoder

    ckpt = _torch_load(checkpoint, "cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"metric checkpoint must be dict: {checkpoint}")
    hidden_dim, metric_dim, dropout = resolve_metric_arch(ckpt)
    metric = MetricEncoder(
        in_dim=1024,
        hidden_dim=hidden_dim,
        out_dim=metric_dim,
        dropout=dropout,
    )
    state = extract_metric_state(ckpt)
    incompatible = metric.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "MetricEncoder checkpoint mismatch:\n"
            f"  checkpoint={checkpoint}\n"
            f"  missing={missing[:30]}\n"
            f"  unexpected={unexpected[:30]}"
        )
    metric = metric.to(device).eval().requires_grad_(False)
    metadata = {
        "path": checkpoint,
        "stage": ckpt.get("stage"),
        "epoch": ckpt.get("epoch"),
        "hidden_dim": hidden_dim,
        "metric_dim": metric_dim,
        "dropout": dropout,
        "ot_config": ckpt.get("ot_config"),
        "episode_config": ckpt.get("episode_config"),
    }
    return metric, metadata


# =============================================================================
# Small CPU LRU caches.  Cache entries are model outputs only; no gradients.
# =============================================================================


class TensorLRU:
    def __init__(self, max_items: int):
        self.max_items = max(int(max_items), 0)
        self.data: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key not in self.data:
            return None
        value = self.data.pop(key)
        self.data[key] = value
        return value

    def put(self, key: str, value: torch.Tensor) -> None:
        if self.max_items == 0:
            return
        key = str(key)
        if key in self.data:
            self.data.pop(key)
        self.data[key] = value.detach().cpu().float().contiguous()
        while len(self.data) > self.max_items:
            self.data.popitem(last=False)

    def __len__(self) -> int:
        return len(self.data)


class FeatureRuntime:
    def __init__(
        self,
        *,
        rae: nn.Module,
        metric: nn.Module,
        transform,
        device: torch.device,
        rae_chunk_size: int,
        metric_chunk_size: int,
        latent_cache_items: int,
        metric_cache_items: int,
    ):
        self.rae = rae
        self.metric = metric
        self.transform = transform
        self.device = device
        self.rae_chunk_size = int(rae_chunk_size)
        self.metric_chunk_size = int(metric_chunk_size)
        self.latent_cache = TensorLRU(latent_cache_items)
        self.metric_cache = TensorLRU(metric_cache_items)

    @torch.inference_mode()
    def latents(self, paths: Sequence[str], *, out_device: Optional[torch.device] = None) -> torch.Tensor:
        out_device = self.device if out_device is None else out_device
        results: List[Optional[torch.Tensor]] = [None] * len(paths)
        missing_positions: Dict[str, List[int]] = OrderedDict()
        for i, p in enumerate(paths):
            key = str(p)
            cached = self.latent_cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                missing_positions.setdefault(key, []).append(i)

        missing_paths = list(missing_positions)
        chunk = max(self.rae_chunk_size, 1)
        for start in range(0, len(missing_paths), chunk):
            sub_paths = missing_paths[start : start + chunk]
            images = load_images(sub_paths, self.transform).to(self.device, non_blocking=True)
            latent = encode_rae_chunked(self.rae, images, self.rae_chunk_size).cpu()
            for j, p in enumerate(sub_paths):
                z = latent[j].contiguous()
                self.latent_cache.put(p, z)
                for pos in missing_positions[p]:
                    results[pos] = z
            del images, latent

        if any(x is None for x in results):
            raise RuntimeError("internal latent-cache error")
        return torch.stack([x for x in results if x is not None], dim=0).to(
            out_device, non_blocking=True
        )

    @torch.inference_mode()
    def metric_raw(self, paths: Sequence[str]) -> torch.Tensor:
        results: List[Optional[torch.Tensor]] = [None] * len(paths)
        missing_positions: Dict[str, List[int]] = OrderedDict()
        for i, p in enumerate(paths):
            key = str(p)
            cached = self.metric_cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                missing_positions.setdefault(key, []).append(i)

        missing_paths = list(missing_positions)
        chunk = max(self.metric_chunk_size, 1)
        for start in range(0, len(missing_paths), chunk):
            sub_paths = missing_paths[start : start + chunk]
            z = self.latents(sub_paths, out_device=self.device)
            feat = self.metric(z)
            raw = feat.raw.detach().cpu().float()
            for j, p in enumerate(sub_paths):
                r = raw[j].contiguous()
                self.metric_cache.put(p, r)
                for pos in missing_positions[p]:
                    results[pos] = r
            del z, feat, raw

        if any(x is None for x in results):
            raise RuntimeError("internal metric-cache error")
        return torch.stack([x for x in results if x is not None], dim=0).to(
            self.device, non_blocking=True
        )


# =============================================================================
# Fixed 8-step RAE Flow runtime
# =============================================================================


class FrozenFlowRuntime:
    def __init__(self, args, primary_device: torch.device):
        parent = str(Path(args.project_root).resolve() / "fsl_multi_vae")
        if parent not in sys.path:
            sys.path.insert(0, parent)
        try:
            from rae_flow_metric_v1.flow_adapter import FrozenVisionHintDiTFlow
        except ImportError:
            from flow_adapter import FrozenVisionHintDiTFlow

        requested = str(args.flow_device).strip()
        self.device = primary_device if requested in {"", "same", "primary"} else torch.device(requested)
        if self.device.type != "cuda":
            raise RuntimeError("RAE DiT Flow generation requires CUDA")
        self.num_steps = DEFAULT_NUM_STEPS  # intentionally fixed; no similarity/early stop
        self.stochastic_hint = bool(args.flow_stochastic_hint)
        self.flow = FrozenVisionHintDiTFlow(
            args.project_root,
            args.flow_checkpoint,
            self.device,
            dit_variant=args.flow_dit_variant,
            endovit_path=args.flow_endovit_path,
            num_steps=self.num_steps,
            solver=DEFAULT_SOLVER,
            hint_signal_ratio=args.flow_hint_signal_ratio,
            hint_dropout=args.flow_hint_dropout,
            amp=args.amp,
            allow_condition_mismatch=args.flow_allow_condition_mismatch,
        ).to(self.device).eval()
        for p in self.flow.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def generate(self, source_z: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        traj = self.flow.generate(
            source_z.to(self.device, non_blocking=True),
            target_images.to(self.device, non_blocking=True),
            stochastic_hint=self.stochastic_hint,
        )
        if traj.ndim != 5:
            raise RuntimeError(
                f"Flow trajectory must be (B,T+1,C,H,W), got {tuple(traj.shape)}"
            )
        if traj.shape[1] != self.num_steps + 1:
            raise RuntimeError(
                f"Flow trajectory length={traj.shape[1]} != {self.num_steps + 1}"
            )
        if tuple(traj.shape[2:]) != (1024, 16, 16):
            raise RuntimeError(
                f"Flow state expected (1024,16,16), got {tuple(traj.shape[2:])}"
            )
        return traj.detach().float()


# =============================================================================
# OT geometry: same ground cost, only uniform-vs-dynamic masses differ.
# =============================================================================


class OTGeometry:
    def __init__(self, args):
        parent = str(Path(args.project_root).resolve() / "fsl_multi_vae")
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from rae_flow_metric_v1.models.metric_encoder import pool_square_tokens
        from rae_flow_metric_v1.models.sk_sinkhorn import (
            aligned_sinkhorn_from_weights,
            compute_aligned_dynamic_sk_weights,
        )

        self.pool_square_tokens = pool_square_tokens
        self.aligned_sinkhorn_from_weights = aligned_sinkhorn_from_weights
        self.compute_aligned_dynamic_sk_weights = compute_aligned_dynamic_sk_weights
        self.pool_factor = int(args.ot_pool_factor)
        self.reg = float(args.sinkhorn_reg)
        self.n_iter = int(args.sinkhorn_iter)
        self.weight_eps = float(args.sk_weight_eps)

    def pool_and_normalize(self, raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_factor > 1:
            raw = self.pool_square_tokens(raw, self.pool_factor)
        raw = raw.float()
        norm = F.normalize(raw, p=2, dim=-1, eps=1e-8)
        return raw, norm

    def uniform_weights(self, raw: torch.Tensor) -> torch.Tensor:
        n = int(raw.shape[-2])
        return torch.full(
            raw.shape[:-2] + (n,),
            1.0 / float(n),
            dtype=torch.float32,
            device=raw.device,
        )

    @torch.inference_mode()
    def fixed_weights(
        self,
        q_raw: torch.Tensor,
        s_raw: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        uq = self.uniform_weights(q_raw)
        us = self.uniform_weights(s_raw)
        dyn = self.compute_aligned_dynamic_sk_weights(
            q_raw, s_raw, weight_eps=self.weight_eps
        )
        return {
            "uniform": (uq, us),
            "dynamic": (dyn.query.float(), dyn.support.float()),
        }

    @torch.inference_mode()
    def distance(
        self,
        left_norm: torch.Tensor,
        right_norm: torch.Tensor,
        w_left: torch.Tensor,
        w_right: torch.Tensor,
    ) -> torch.Tensor:
        out = self.aligned_sinkhorn_from_weights(
            left_norm,
            right_norm,
            w_left,
            w_right,
            reg=self.reg,
            n_iter=self.n_iter,
            normalize_cost_points=False,
            return_transport=False,
            return_cost=False,
        )
        return out.distance.float()


# =============================================================================
# Pair evidence
# =============================================================================


def _flow_block_from_values(
    *,
    tolast: Sequence[float],
    neighbor: Sequence[float],
    tosupport: Sequence[float],
    direct: float,
) -> Dict[str, Any]:
    tolast_stats = trajectory_stats(tolast, step_offset=0)
    neighbor_stats = trajectory_stats(neighbor, step_offset=0)
    tosupport_stats = trajectory_stats(tosupport, step_offset=0)
    post = list(tosupport[1:])
    post_stats = trajectory_stats(post, step_offset=1)
    endpoint = float(tosupport[-1])
    best = float(min(tosupport))
    return {
        "tolast": tolast_stats,
        "neighbor": neighbor_stats,
        "tosupport": {
            **tosupport_stats,
            "start": float(tosupport[0]),
            "post_mean": _mean(post),
            "post_std": _std(post),
            "endpoint": endpoint,
            "endpoint_gain_vs_direct": float(direct - endpoint),
            "best_gain_vs_direct": float(direct - best),
            "best_step": int(np.argmin(np.asarray(tosupport, dtype=np.float64))),
        },
        "tosupport_post": post_stats,
    }


@torch.inference_mode()
def compute_pair_chunk_evidence(
    *,
    q_paths: Sequence[str],
    s_paths: Sequence[str],
    feature_runtime: FeatureRuntime,
    flow_runtime: FrozenFlowRuntime,
    metric: nn.Module,
    geometry: OTGeometry,
    transform,
    metric_device: torch.device,
) -> List[Dict[str, Any]]:
    if len(q_paths) != len(s_paths):
        raise ValueError("q_paths/s_paths size mismatch")
    B = len(q_paths)
    if B == 0:
        return []

    # Direct image states / metric tokens.
    q_z_metric = feature_runtime.latents(q_paths, out_device=metric_device)
    s_z_metric = feature_runtime.latents(s_paths, out_device=metric_device)
    q_raw_cached = feature_runtime.metric_raw(q_paths)
    s_raw_cached = feature_runtime.metric_raw(s_paths)
    q_raw, q_norm = geometry.pool_and_normalize(q_raw_cached)
    s_raw, s_norm = geometry.pool_and_normalize(s_raw_cached)

    weights = geometry.fixed_weights(q_raw, s_raw)
    direct_distance: Dict[str, torch.Tensor] = {}
    for mode, (wq, ws) in weights.items():
        direct_distance[mode] = geometry.distance(q_norm, s_norm, wq, ws)

    # Flow source is query RAE latent; condition is CURRENT support image.
    q_z_flow = q_z_metric.to(flow_runtime.device, non_blocking=True)
    target_images = load_images(s_paths, transform).to(flow_runtime.device, non_blocking=True)
    traj = flow_runtime.generate(q_z_flow, target_images)  # (B,9,1024,16,16)

    # Sanity: adapter should return source as trajectory state 0.
    source_delta = (
        traj[:, 0].float() - q_z_flow.float()
    ).abs().flatten(1).max(dim=1).values

    # Metric tokens for all 9 states.
    flat_states = traj.reshape(B * traj.shape[1], *traj.shape[2:])
    state_raw_parts = []
    metric_chunk = max(int(feature_runtime.metric_chunk_size), 1)
    for start in range(0, flat_states.shape[0], metric_chunk):
        z = flat_states[start : start + metric_chunk].to(metric_device, non_blocking=True)
        feat = metric(z)
        state_raw_parts.append(feat.raw.detach().float())
    state_raw = torch.cat(state_raw_parts, dim=0).reshape(
        B, traj.shape[1], state_raw_parts[0].shape[-2], state_raw_parts[0].shape[-1]
    )
    state_raw, state_norm = geometry.pool_and_normalize(state_raw)

    evidence_by_mode: Dict[str, List[Dict[str, Any]]] = {"uniform": [], "dynamic": []}
    for mode, (wq0, ws0) in weights.items():
        # Same t=0 query-support masses are fixed across ALL trajectory distances.
        wq = wq0.unsqueeze(1)  # (B,1,N)
        ws = ws0.unsqueeze(1)  # (B,1,N)

        # z_t -> actual support, t=0..8 (9 distances)
        d_support = geometry.distance(
            state_norm,
            s_norm.unsqueeze(1),
            wq,
            ws,
        )
        # z_t -> z_8, t=0..7 (8 distances)
        d_last = geometry.distance(
            state_norm[:, :-1],
            state_norm[:, -1:],
            wq,
            ws,
        )
        # z_t -> z_{t+1}, t=0..7 (8 distances)
        d_neighbor = geometry.distance(
            state_norm[:, :-1],
            state_norm[:, 1:],
            wq,
            ws,
        )

        for i in range(B):
            direct_i = float(direct_distance[mode][i].item())
            tosupport_i = [float(x) for x in d_support[i].detach().cpu().tolist()]
            tolast_i = [float(x) for x in d_last[i].detach().cpu().tolist()]
            neighbor_i = [float(x) for x in d_neighbor[i].detach().cpu().tolist()]
            block = _flow_block_from_values(
                tolast=tolast_i,
                neighbor=neighbor_i,
                tosupport=tosupport_i,
                direct=direct_i,
            )
            block["direct"] = direct_i
            block["sanity_direct_minus_tosupport_step0"] = float(
                direct_i - tosupport_i[0]
            )
            evidence_by_mode[mode].append(block)

    out = []
    for i in range(B):
        out.append({
            "uniform": evidence_by_mode["uniform"][i],
            "dynamic": evidence_by_mode["dynamic"][i],
            "flow": {
                "num_steps": DEFAULT_NUM_STEPS,
                "solver": DEFAULT_SOLVER,
                "trajectory_length": DEFAULT_NUM_STEPS + 1,
                "source_state_max_abs_diff": float(source_delta[i].item()),
                "dynamic_mass_rule": "direct_t0_query_support_pair_fixed_across_trajectory",
            },
        })

    del q_z_metric, s_z_metric, q_z_flow, target_images, traj, flat_states
    return out


# =============================================================================
# Score extraction / query aggregation
# =============================================================================


def score_names() -> List[str]:
    names = [
        "direct",
        "tolast_mean",
        "neighbor_mean",
        "tosupport_all_mean",
        "tosupport_post_mean",
        "tosupport_endpoint",
    ]
    names += [f"tolast_step{i}" for i in range(DEFAULT_NUM_STEPS)]
    names += [f"neighbor_step{i}" for i in range(DEFAULT_NUM_STEPS)]
    names += [f"tosupport_step{i}" for i in range(DEFAULT_NUM_STEPS + 1)]
    return names


def extract_pair_score(evidence: Mapping[str, Any], mode: str, score: str) -> float:
    block = evidence[mode]
    if score == "direct":
        return float(block["direct"])
    if score == "tolast_mean":
        return float(block["tolast"]["mean"])
    if score == "neighbor_mean":
        return float(block["neighbor"]["mean"])
    if score == "tosupport_all_mean":
        return float(block["tosupport"]["mean"])
    if score == "tosupport_post_mean":
        return float(block["tosupport"]["post_mean"])
    if score == "tosupport_endpoint":
        return float(block["tosupport"]["endpoint"])
    if score.startswith("tolast_step"):
        i = int(score[len("tolast_step"):])
        return float(block["tolast"]["all"][i])
    if score.startswith("neighbor_step"):
        i = int(score[len("neighbor_step"):])
        return float(block["neighbor"]["all"][i])
    if score.startswith("tosupport_step"):
        i = int(score[len("tosupport_step"):])
        return float(block["tosupport"]["all"][i])
    raise KeyError(score)


@dataclass
class QueryGroup:
    query_path: str
    ground_truth: Optional[str]
    rows: List[Dict[str, Any]]


def group_rows_by_query(rows: Sequence[Dict[str, Any]]) -> List[QueryGroup]:
    grouped: "OrderedDict[str, QueryGroup]" = OrderedDict()
    for row in rows:
        q = str(row["target_image"])
        if q not in grouped:
            grouped[q] = QueryGroup(q, row.get("ground_truth"), [])
        g = grouped[q]
        if g.ground_truth is None and row.get("ground_truth") is not None:
            g.ground_truth = str(row["ground_truth"])
        g.rows.append(row)
    return list(grouped.values())


def make_ranking(class_distances: Mapping[str, float]) -> List[Dict[str, Any]]:
    ranking = [
        {"class": str(cls), "distance": float(dist)}
        for cls, dist in class_distances.items()
        if dist is not None and np.isfinite(float(dist))
    ]
    ranking.sort(key=lambda x: x["distance"])
    return ranking


def query_score_view(
    group: QueryGroup,
    *,
    mode: str,
    score: str,
    aggregation: str,
    softmin_tau: float,
) -> Dict[str, Any]:
    class_shot_scores: "OrderedDict[str, List[float]]" = OrderedDict()
    for row in group.rows:
        cls = str(row["matched_class"])
        value = extract_pair_score(row["rae_metric"], mode, score)
        class_shot_scores.setdefault(cls, []).append(float(value))

    class_distances: "OrderedDict[str, float]" = OrderedDict()
    for cls, values in class_shot_scores.items():
        class_distances[cls] = aggregate_support(values, aggregation, softmin_tau)

    ranking = make_ranking(class_distances)
    pred = ranking[0]["class"] if ranking else "未知"
    top2 = ranking[1]["class"] if len(ranking) > 1 else None
    gap = (
        float(ranking[1]["distance"] - ranking[0]["distance"])
        if len(ranking) > 1 else 999.0
    )
    gt = group.ground_truth
    evaluable = gt is not None and gt in class_distances and bool(ranking)
    correct = bool(evaluable and pred == gt)
    pos_distance = None
    nearest_neg = None
    margin = None
    if evaluable:
        pos_distance = float(class_distances[str(gt)])
        negatives = [
            float(v) for c, v in class_distances.items()
            if c != gt and np.isfinite(float(v))
        ]
        if negatives:
            nearest_neg = float(min(negatives))
            margin = float(nearest_neg - pos_distance)

    return {
        "mode": mode,
        "score": score,
        "aggregation": aggregation,
        "query_path": group.query_path,
        "ground_truth": gt,
        "candidate_labels": list(class_distances.keys()),
        "n_way": len(class_distances),
        "class_shot_scores": class_shot_scores,
        "class_distances": class_distances,
        "ranking": ranking,
        "pred": pred,
        "top1": pred,
        "top2": top2,
        "gap": gap,
        "evaluable": evaluable,
        "correct": correct,
        "positive_class_distance": pos_distance,
        "nearest_negative_distance": nearest_neg,
        "margin": margin,
    }


def summarize_views(views: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    evaluable = [v for v in views if bool(v.get("evaluable"))]
    correct = sum(int(bool(v.get("correct"))) for v in evaluable)
    total = len(evaluable)
    gaps = [float(v["gap"]) for v in evaluable if _safe_float(v.get("gap")) is not None]
    margins = [float(v["margin"]) for v in evaluable if _safe_float(v.get("margin")) is not None]
    pos = [
        float(v["positive_class_distance"])
        for v in evaluable
        if _safe_float(v.get("positive_class_distance")) is not None
    ]
    neg = [
        float(v["nearest_negative_distance"])
        for v in evaluable
        if _safe_float(v.get("nearest_negative_distance")) is not None
    ]
    ways = [int(v.get("n_way", 0)) for v in views]

    per_class: Dict[str, Dict[str, Any]] = OrderedDict()
    for v in evaluable:
        gt = str(v["ground_truth"])
        d = per_class.setdefault(gt, {
            "total": 0, "correct": 0, "pred_counts": Counter(),
            "margins": [], "gaps": [], "positive_distances": [],
            "nearest_negative_distances": [],
        })
        d["total"] += 1
        d["correct"] += int(bool(v["correct"]))
        d["pred_counts"][str(v["pred"])] += 1
        if v.get("margin") is not None:
            d["margins"].append(float(v["margin"]))
        if v.get("gap") is not None:
            d["gaps"].append(float(v["gap"]))
        if v.get("positive_class_distance") is not None:
            d["positive_distances"].append(float(v["positive_class_distance"]))
        if v.get("nearest_negative_distance") is not None:
            d["nearest_negative_distances"].append(float(v["nearest_negative_distance"]))

    per_class_out: Dict[str, Dict[str, Any]] = OrderedDict()
    for cls, d in per_class.items():
        n = int(d["total"])
        c = int(d["correct"])
        confusion = [
            {"pred": pred, "count": int(count)}
            for pred, count in d["pred_counts"].most_common()
            if pred != cls
        ]
        per_class_out[cls] = {
            "total": n,
            "correct": c,
            "accuracy": c / max(n, 1),
            "accuracy_percent": 100.0 * c / max(n, 1),
            "mean_margin": _mean(d["margins"]),
            "mean_gap": _mean(d["gaps"]),
            "mean_positive_distance": _mean(d["positive_distances"]),
            "mean_nearest_negative_distance": _mean(d["nearest_negative_distances"]),
            "top_confusions": confusion[:10],
        }

    way_counter = Counter(ways)
    return {
        "n_queries": len(views),
        "n_evaluable_queries": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "accuracy_percent": 100.0 * correct / max(total, 1),
        "ci95": _ci95_binary(correct, total),
        "ci95_percent": 100.0 * _ci95_binary(correct, total),
        "gap_mean": _mean(gaps),
        "gap_std": _std(gaps),
        "gap_median": _median(gaps),
        "margin_mean": _mean(margins),
        "margin_std": _std(margins),
        "margin_median": _median(margins),
        "positive_margin_rate": float(np.mean(np.asarray(margins) > 0)) if margins else 0.0,
        "positive_class_distance_mean": _mean(pos),
        "nearest_negative_distance_mean": _mean(neg),
        "candidate_way_min": min(ways) if ways else 0,
        "candidate_way_max": max(ways) if ways else 0,
        "candidate_way_mean": _mean(ways),
        "candidate_way_distribution": {str(k): int(v) for k, v in sorted(way_counter.items())},
        "per_class": per_class_out,
    }


def compare_predictions(
    base_views: Sequence[Mapping[str, Any]],
    alt_views: Sequence[Mapping[str, Any]],
    *,
    base_name: str,
    alt_name: str,
) -> Dict[str, Any]:
    base_map = {str(v["query_path"]): v for v in base_views if v.get("evaluable")}
    alt_map = {str(v["query_path"]): v for v in alt_views if v.get("evaluable")}
    keys = [k for k in base_map if k in alt_map]
    both_correct = base_only = alt_only = both_wrong = 0
    agree = 0
    for k in keys:
        b = base_map[k]
        a = alt_map[k]
        bc = bool(b["correct"])
        ac = bool(a["correct"])
        agree += int(str(b["pred"]) == str(a["pred"]))
        if bc and ac:
            both_correct += 1
        elif bc and not ac:
            base_only += 1
        elif (not bc) and ac:
            alt_only += 1
        else:
            both_wrong += 1
    n = len(keys)
    rescue = alt_only
    degradation = base_only
    return {
        "base": base_name,
        "alternative": alt_name,
        "n_queries": n,
        "both_correct": both_correct,
        "base_only_correct": base_only,
        "alternative_only_correct": alt_only,
        "both_wrong": both_wrong,
        "rescue": rescue,
        "degradation": degradation,
        "net_rescue": rescue - degradation,
        "rescue_rate_percent": 100.0 * rescue / max(n, 1),
        "degradation_rate_percent": 100.0 * degradation / max(n, 1),
        "override_precision_rescue_vs_degradation": rescue / max(rescue + degradation, 1),
        "agreement_rate": agree / max(n, 1),
        "disagreement_rate": 1.0 - agree / max(n, 1),
        "base_accuracy": (both_correct + base_only) / max(n, 1),
        "alternative_accuracy": (both_correct + alt_only) / max(n, 1),
        "gain_vs_base_pp": 100.0 * (alt_only - base_only) / max(n, 1),
    }


def oracle_stats(named_views: Sequence[Tuple[str, Sequence[Mapping[str, Any]]]]) -> Dict[str, Any]:
    if not named_views:
        return {}
    maps = [
        {str(v["query_path"]): v for v in views if v.get("evaluable")}
        for _, views in named_views
    ]
    common = set(maps[0])
    for m in maps[1:]:
        common &= set(m)
    keys = sorted(common)
    oracle_correct = 0
    all_wrong = 0
    best_single = 0.0
    expert_acc = {}
    for (name, _), m in zip(named_views, maps):
        acc = sum(int(bool(m[k]["correct"])) for k in keys) / max(len(keys), 1)
        expert_acc[name] = acc
        best_single = max(best_single, acc)
    for k in keys:
        any_correct = any(bool(m[k]["correct"]) for m in maps)
        oracle_correct += int(any_correct)
        all_wrong += int(not any_correct)
    acc = oracle_correct / max(len(keys), 1)
    return {
        "experts": [name for name, _ in named_views],
        "n_queries": len(keys),
        "oracle_correct": oracle_correct,
        "oracle_accuracy": acc,
        "oracle_accuracy_percent": 100.0 * acc,
        "all_experts_wrong": all_wrong,
        "best_single_accuracy": best_single,
        "best_single_accuracy_percent": 100.0 * best_single,
        "oracle_gain_vs_best_single_pp": 100.0 * (acc - best_single),
        "expert_accuracy": expert_acc,
    }


# =============================================================================
# Legacy fusion-compatible view
# =============================================================================


def pack_fsl_view(
    class_traj: Mapping[str, float],
    class_direct: Mapping[str, float],
    *,
    trajectory_score_key: str,
) -> Dict[str, Any]:
    ranking = make_ranking(class_traj)
    ranking_direct = make_ranking(class_direct)
    top1 = ranking[0]["class"] if ranking else "未知"
    top2 = ranking[1]["class"] if len(ranking) > 1 else None
    distance = ranking[0]["distance"] if ranking else None
    gap = ranking[1]["distance"] - ranking[0]["distance"] if len(ranking) > 1 else 999.0
    top1_direct = ranking_direct[0]["class"] if ranking_direct else "未知"
    top2_direct = ranking_direct[1]["class"] if len(ranking_direct) > 1 else None
    distance_direct = ranking_direct[0]["distance"] if ranking_direct else None
    gap_direct = (
        ranking_direct[1]["distance"] - ranking_direct[0]["distance"]
        if len(ranking_direct) > 1 else 999.0
    )
    return {
        "trajectory_score_key": trajectory_score_key,
        "top1": top1,
        "top2": top2,
        "gap": float(gap),
        "distance": distance,
        "ranking": ranking,
        "class_min": dict(class_traj),
        "class_min_distances": dict(class_traj),
        "class_distances": dict(class_traj),
        "top1_direct": top1_direct,
        "top2_direct": top2_direct,
        "gap_direct": float(gap_direct),
        "distance_direct": distance_direct,
        "ranking_direct": ranking_direct,
        "class_min_direct": dict(class_direct),
        "class_min_distances_direct": dict(class_direct),
        "class_distances_direct": dict(class_direct),
        "valid": bool(ranking),
    }


def build_fusion_ready(
    groups: Sequence[QueryGroup],
    *,
    mode: str,
    softmin_tau: float,
) -> Dict[str, Any]:
    output_records = []
    for g in groups:
        traj_view = query_score_view(
            g,
            mode=mode,
            score="tolast_mean",
            aggregation="min",
            softmin_tau=softmin_tau,
        )
        direct_view = query_score_view(
            g,
            mode=mode,
            score="direct",
            aggregation="min",
            softmin_tau=softmin_tau,
        )
        pair_results = []
        for row in g.rows:
            pair_results.append({
                "matched_class": row["matched_class"],
                "support_image": row["support_image"],
                "compare_distance": extract_pair_score(row["rae_metric"], mode, "tolast_mean"),
                "final_distance": extract_pair_score(row["rae_metric"], mode, "tolast_mean"),
                "compare_distance_direct": extract_pair_score(row["rae_metric"], mode, "direct"),
                "final_distance_direct": extract_pair_score(row["rae_metric"], mode, "direct"),
                "direct_distance": extract_pair_score(row["rae_metric"], mode, "direct"),
            })

        score_key = f"tolast_rae_metric_{mode}_mean"
        fsl_full_result = pack_fsl_view(
            traj_view["class_distances"],
            direct_view["class_distances"],
            trajectory_score_key=score_key,
        )
        output_records.append({
            "image_id": os.path.basename(g.query_path),
            "image_path": g.query_path,
            "target_image": g.query_path,
            "ground_truth": g.ground_truth,
            "ok": True,
            "class_distances": traj_view["class_distances"],
            "class_min_distances": traj_view["class_distances"],
            "class_distances_direct": direct_view["class_distances"],
            "class_min_distances_direct": direct_view["class_distances"],
            "results": pair_results,
            "fsl_trajectory_score_key": score_key,
            "fsl_full_result": fsl_full_result,
        })

    return {
        "results": output_records,
        "summary": {
            "n_queries": len(output_records),
            "contains_traj": True,
            "contains_direct": True,
            "sk_mode": mode,
            "trajectory_score_key": f"tolast_rae_metric_{mode}_mean",
            "support_aggregation": "min",
            "num_steps": DEFAULT_NUM_STEPS,
            "solver": DEFAULT_SOLVER,
            "format": "offline_rae_metric_flow_fsl_service_and_fusion_compatible_v1",
        },
    }


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline RAE+Metric+8-step RAE-Flow FSL service (uniform + dynamic SK)"
    )

    # All paths have editable DEFAULT_* values above; CLI is override-only.
    p.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Existing flat query-support JSON produced by the prior MLLM/FSL pipeline",
    )
    p.add_argument(
        "--metric_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional explicit RAE MetricEncoder checkpoint. If omitted, infer the organ "
            "from --data_path and use DEFAULT_METRIC_CHECKPOINTS[organ]."
        ),
    )
    p.add_argument(
        "--flow_checkpoint",
        type=str,
        default=DEFAULT_FLOW_CHECKPOINT,
        help="RAE Vision-Hint DiT Flow checkpoint",
    )
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--project_root", type=str, default=DEFAULT_PROJECT_ROOT)

    # Backward-compatible argument only.  This all-support script ignores the cap
    # and always keeps every valid query-support row from the input JSON.
    p.add_argument(
        "--max_n_shot",
        type=int,
        default=MAX_N_SHOT,
        help=(
            "Deprecated/ignored in this all-support version. All valid support rows "
            "already present in the flat JSON are used for each query x candidate class."
        ),
    )

    # Devices/runtime.
    p.add_argument("--device", type=str, default="cuda:0",
                   help="RAE + Metric device")
    p.add_argument("--flow_device", type=str, default="same",
                   help="Flow device; 'same' uses --device")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--log_every", type=int, default=25)

    # RAE / caches.
    p.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--rae_chunk_size", type=int, default=DEFAULT_RAE_CHUNK_SIZE)
    p.add_argument("--metric_chunk_size", type=int, default=DEFAULT_METRIC_CHUNK_SIZE)
    p.add_argument("--latent_cache_items", type=int, default=256,
                   help="CPU LRU cache entries for RAE latents; 0 disables")
    p.add_argument("--metric_cache_items", type=int, default=1024,
                   help="CPU LRU cache entries for metric raw tokens; 0 disables")

    # OT geometry -- project defaults.
    p.add_argument("--ot_pool_factor", type=int, default=DEFAULT_OT_POOL_FACTOR,
                   help="1 = full 16x16=256 nodes; 2 = pooled 8x8=64 nodes")
    p.add_argument("--sinkhorn_reg", type=float, default=DEFAULT_SINKHORN_REG)
    p.add_argument("--sinkhorn_iter", type=int, default=DEFAULT_SINKHORN_ITER)
    p.add_argument("--sk_weight_eps", type=float, default=DEFAULT_SK_WEIGHT_EPS)
    p.add_argument("--softmin_tau", type=float, default=DEFAULT_SOFTMIN_TAU,
                   help="Analysis-only K-shot softmin; primary fusion remains support-min")

    # Flow: num_steps and solver intentionally NOT exposed as variable semantics.
    # They are fixed to 8 / Heun to remove the old self.similarity early-stop path.
    p.add_argument("--flow_pair_chunk", type=int, default=DEFAULT_FLOW_PAIR_CHUNK)
    p.add_argument("--flow_dit_variant", type=str, default="L")
    p.add_argument("--flow_endovit_path", type=str, default=None)
    p.add_argument("--flow_hint_signal_ratio", type=float, default=None,
                   help="None => restore/resolve from Flow checkpoint condition_config")
    p.add_argument("--flow_hint_dropout", type=float, default=0.0)
    p.add_argument("--flow_stochastic_hint", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--flow_allow_condition_mismatch", action="store_true")

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    print(f"[SCRIPT_VERSION] {SCRIPT_VERSION}")
    args = parse_args()

    # Resolve organ-specific MetricEncoder checkpoint unless the caller explicitly
    # provides --metric_checkpoint.  This makes one script safe for batch-running
    # 十二指肠 / 食管 / 胃 inputs without manually editing the checkpoint each time.
    if args.metric_checkpoint is None or not str(args.metric_checkpoint).strip():
        inferred_organ, resolved_metric_checkpoint = resolve_default_metric_checkpoint(
            args.data_path
        )
        args.metric_checkpoint = resolved_metric_checkpoint
        metric_checkpoint_source = "auto-by-organ"
    else:
        inferred_organ = infer_organ_from_data_path(args.data_path)
        metric_checkpoint_source = "explicit-cli"

    args.metric_checkpoint = str(Path(args.metric_checkpoint).expanduser())
    if not Path(args.metric_checkpoint).is_file():
        raise FileNotFoundError(
            f"Metric checkpoint not found for organ={inferred_organ}: "
            f"{args.metric_checkpoint}"
        )

    if DEFAULT_NUM_STEPS != 8:
        raise AssertionError("This offline service is intentionally fixed at 8 Flow steps")
    if DEFAULT_SOLVER != "heun":
        raise AssertionError("This offline service is intentionally fixed to Heun")
    if int(args.image_size) != DEFAULT_IMAGE_SIZE:
        raise ValueError(
            f"This RAE/Flow pipeline is fixed at image_size={DEFAULT_IMAGE_SIZE}; got {args.image_size}"
        )
    if args.ot_pool_factor < 1:
        raise ValueError("ot_pool_factor must be >=1")
    if args.sinkhorn_reg <= 0 or args.sinkhorn_iter <= 0:
        raise ValueError("Sinkhorn reg/iter must be positive")
    if args.softmin_tau <= 0:
        raise ValueError("softmin_tau must be >0")
    # NOTE: --max_n_shot is intentionally ignored in this all-support version.

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    metric_device = torch.device(args.device)
    if metric_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import project modules before model construction.
    project_root = str(Path(args.project_root).expanduser().resolve())
    args.project_root = project_root
    for pth in (project_root, str(Path(project_root) / "fsl_multi_vae")):
        if pth not in sys.path:
            sys.path.insert(0, pth)

    print("=" * 110)
    print("[Offline RAE FSL Service] existing flat MLLM/FSL pairs -> RAE Metric + fixed 8-step Flow")
    print(f"data_path          = {args.data_path}")
    print(f"inferred_organ     = {inferred_organ}")
    print(f"metric_ckpt_source = {metric_checkpoint_source}")
    print(f"metric_checkpoint  = {args.metric_checkpoint}")
    print(f"flow_checkpoint    = {args.flow_checkpoint}")
    print(f"output_dir         = {args.output_dir}")
    print("support_protocol    = ALL valid query-support rows from input JSON (no shot cap)")
    print(f"max_n_shot arg     = {args.max_n_shot} (ignored for backward compatibility)")
    print(f"device             = {metric_device}")
    print(f"flow_device        = {args.flow_device}")
    print("Flow               = FIXED 8-step Heun; similarity/early-stop DISABLED")
    print(
        "OT                 = same L2-normalized squared-Euclidean ground cost; "
        "Uniform masses + Dynamic-SK masses"
    )
    print(
        "Dynamic Flow SK    = masses computed ONCE from direct t0 query-support pair "
        "and fixed across trajectory"
    )
    print("Primary fusion      = tolast_mean + support-min; Direct retained separately")
    print("Analysis aggregation= min + mean + softmin")
    print("=" * 110)

    raw_records = load_flat_json(args.data_path)
    pair_specs, skipped_indices = parse_pair_specs(raw_records, args.data_path)
    if not pair_specs:
        sample_keys = sorted(str(k) for k in raw_records[0].keys()) if raw_records else []
        sample = raw_records[0] if raw_records else {}
        raise RuntimeError(
            "No valid query-support pairs found in input JSON.\n"
            f"data_path={args.data_path}\n"
            f"num_records={len(raw_records)}\n"
            f"first_record_keys={sample_keys}\n"
            "Expected legacy raw fields x0/x1/label_B (preferred), or "
            "x0_path/x1_path/label_B and compatible aliases.\n"
            f"first_record_preview={_json_safe(sample)}"
        )

    # The flat JSON has one query-support pair per row.  Match the old VAE
    # protocol exactly at the support-selection level: keep ALL valid rows.
    # No shot cap, no random sampling, and no episode regeneration are applied.
    all_valid_pair_specs = pair_specs
    pair_specs, shot_diagnostics = select_support_pairs_per_query_class(
        all_valid_pair_specs, args.max_n_shot
    )
    if not pair_specs:
        raise RuntimeError("No query-support pairs remain after support selection")

    print(
        "[All-support] "
        f"query_class_groups={shot_diagnostics['num_query_class_groups']} "
        f"valid_rows={len(all_valid_pair_specs)} selected_rows={len(pair_specs)} "
        f"duplicate_rows_present={shot_diagnostics['duplicate_support_rows_present']} "
        "dropped_by_cap=0"
    )
    if int(args.max_n_shot) != 0:
        print(
            f"[All-support] NOTE: --max_n_shot={args.max_n_shot} was supplied but is ignored."
        )
    print(
        "[All-support] JSON rows/query/class distribution="
        f"{shot_diagnostics['selected_supports_per_query_class_distribution']}"
    )
    print(
        "[All-support] unique support paths/query/class distribution="
        f"{shot_diagnostics['unique_supports_per_query_class_distribution']}"
    )

    # Deduplicate expensive q/s computation among the SELECTED pairs.
    unique_keys: "OrderedDict[Tuple[str, str], None]" = OrderedDict()
    for p in pair_specs:
        unique_keys[(p.query_path, p.support_path)] = None
    unique_pairs = list(unique_keys.keys())
    print(
        f"[Input] raw_records={len(raw_records)} valid_rows={len(all_valid_pair_specs)} "
        f"selected_rows={len(pair_specs)} skipped_parse={len(skipped_indices)} "
        f"unique_selected_query_support_pairs={len(unique_pairs)}"
    )

    transform = build_transform(args.image_size)
    rae = build_rae(project_root, metric_device)
    metric, metric_metadata = load_metric(
        project_root, args.metric_checkpoint, metric_device
    )
    flow_runtime = FrozenFlowRuntime(args, metric_device)
    geometry = OTGeometry(args)
    feature_runtime = FeatureRuntime(
        rae=rae,
        metric=metric,
        transform=transform,
        device=metric_device,
        rae_chunk_size=args.rae_chunk_size,
        metric_chunk_size=args.metric_chunk_size,
        latent_cache_items=args.latent_cache_items,
        metric_cache_items=args.metric_cache_items,
    )

    pair_evidence: Dict[Tuple[str, str], Dict[str, Any]] = {}
    chunk_size = max(int(args.flow_pair_chunk), 1)
    pbar = tqdm(range(0, len(unique_pairs), chunk_size), desc="RAE Flow unique q-support pairs")
    for chunk_idx, start in enumerate(pbar):
        chunk_pairs = unique_pairs[start : start + chunk_size]
        q_paths = [q for q, _ in chunk_pairs]
        s_paths = [s for _, s in chunk_pairs]
        evidence_list = compute_pair_chunk_evidence(
            q_paths=q_paths,
            s_paths=s_paths,
            feature_runtime=feature_runtime,
            flow_runtime=flow_runtime,
            metric=metric,
            geometry=geometry,
            transform=transform,
            metric_device=metric_device,
        )
        for key, evidence in zip(chunk_pairs, evidence_list):
            pair_evidence[key] = evidence

        done = min(start + len(chunk_pairs), len(unique_pairs))
        pbar.set_postfix({
            "cached_latent": len(feature_runtime.latent_cache),
            "cached_metric": len(feature_runtime.metric_cache),
        })
        if args.log_every > 0 and done % args.log_every == 0:
            if torch.cuda.is_available() and metric_device.type == "cuda":
                alloc = torch.cuda.memory_allocated(metric_device) / (1024 ** 3)
                reserv = torch.cuda.memory_reserved(metric_device) / (1024 ** 3)
                print(f"[Progress] {done}/{len(unique_pairs)} metric_gpu={alloc:.2f}G reserved={reserv:.2f}G")

    # Enrich the SELECTED flat rows in original JSON order.
    enriched_rows: List[Dict[str, Any]] = []
    spec_by_index = {p.index: p for p in pair_specs}
    for idx, original in enumerate(raw_records):
        if idx not in spec_by_index:
            continue
        spec = spec_by_index[idx]
        evidence = pair_evidence[(spec.query_path, spec.support_path)]
        row = deepcopy(original)
        # Preserve both RAW legacy fields (x0/x1) and processed aliases
        # (x0_path/x1_path), so result.json remains service-compatible.
        row["x0"] = spec.query_path
        row["x1"] = spec.support_path
        row["x0_path"] = spec.query_path
        row["x1_path"] = spec.support_path
        row["target_image"] = spec.query_path
        row["support_image"] = spec.support_path
        row["matched_class"] = spec.matched_class
        row["ground_truth"] = spec.ground_truth
        row["rae_metric"] = evidence

        # Explicit flat aliases for downstream analysis.
        row["direct_uniform"] = extract_pair_score(evidence, "uniform", "direct")
        row["direct_dynamic"] = extract_pair_score(evidence, "dynamic", "direct")
        row["tolast_uniform_mean"] = extract_pair_score(evidence, "uniform", "tolast_mean")
        row["tolast_dynamic_mean"] = extract_pair_score(evidence, "dynamic", "tolast_mean")
        row["neighbor_uniform_mean"] = extract_pair_score(evidence, "uniform", "neighbor_mean")
        row["neighbor_dynamic_mean"] = extract_pair_score(evidence, "dynamic", "neighbor_mean")
        row["tosupport_uniform_mean"] = extract_pair_score(evidence, "uniform", "tosupport_all_mean")
        row["tosupport_dynamic_mean"] = extract_pair_score(evidence, "dynamic", "tosupport_all_mean")

        # Old result.json aliases use Dynamic-SK as the canonical upgraded geometry.
        # Uniform has its own explicit fields and its own fusion-ready JSON.
        row["trajectory_score_key"] = "tolast_rae_metric_dynamic_mean"
        row["compare_distance"] = row["tolast_dynamic_mean"]
        row["final_distance"] = row["tolast_dynamic_mean"]
        row["compare_distance_direct"] = row["direct_dynamic"]
        row["final_distance_direct"] = row["direct_dynamic"]
        row["compare_distance_uniform"] = row["tolast_uniform_mean"]
        row["compare_distance_direct_uniform"] = row["direct_uniform"]
        enriched_rows.append(row)

    result_path = output_dir / "result.json"
    _save_json(result_path, enriched_rows)
    print(f"[Saved] {result_path}")

    groups = group_rows_by_query(enriched_rows)
    modes = ["uniform", "dynamic"]
    aggregations = ["min", "mean", "softmin"]
    scores = score_names()

    # Per-query full matrix of score/aggregation views.
    query_output: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    views_lookup: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    score_summaries: Dict[str, Any] = OrderedDict()
    csv_rows: List[Dict[str, Any]] = []

    for group in groups:
        query_output[group.query_path] = {
            "image_id": os.path.basename(group.query_path),
            "query_path": group.query_path,
            "ground_truth": group.ground_truth,
            "scores": {},
        }

    print(f"[Stats] building query/class summaries for {len(groups)} unique queries ...")
    for mode in modes:
        for agg in aggregations:
            for score in scores:
                views = [
                    query_score_view(
                        g,
                        mode=mode,
                        score=score,
                        aggregation=agg,
                        softmin_tau=args.softmin_tau,
                    )
                    for g in groups
                ]
                key = (mode, agg, score)
                views_lookup[key] = views
                summary = summarize_views(views)
                summary_key = f"{mode}/{agg}/{score}"
                score_summaries[summary_key] = summary
                csv_rows.append({
                    "mode": mode,
                    "aggregation": agg,
                    "score": score,
                    "n_queries": summary["n_queries"],
                    "n_evaluable_queries": summary["n_evaluable_queries"],
                    "correct": summary["correct"],
                    "accuracy_percent": summary["accuracy_percent"],
                    "ci95_percent": summary["ci95_percent"],
                    "gap_mean": summary["gap_mean"],
                    "gap_std": summary["gap_std"],
                    "margin_mean": summary["margin_mean"],
                    "margin_std": summary["margin_std"],
                    "positive_margin_rate": summary["positive_margin_rate"],
                    "candidate_way_min": summary["candidate_way_min"],
                    "candidate_way_max": summary["candidate_way_max"],
                    "candidate_way_mean": summary["candidate_way_mean"],
                })
                for v in views:
                    qobj = query_output[str(v["query_path"])]
                    qobj["scores"].setdefault(mode, {}).setdefault(agg, {})[score] = {
                        "candidate_labels": v["candidate_labels"],
                        "class_shot_scores": v["class_shot_scores"],
                        "class_distances": v["class_distances"],
                        "ranking": v["ranking"],
                        "pred": v["pred"],
                        "top2": v["top2"],
                        "gap": v["gap"],
                        "correct": v["correct"],
                        "evaluable": v["evaluable"],
                        "positive_class_distance": v["positive_class_distance"],
                        "nearest_negative_distance": v["nearest_negative_distance"],
                        "margin": v["margin"],
                    }

    # Comparisons: primary legacy aggregation=min.
    comparisons: Dict[str, Any] = OrderedDict()
    for mode in modes:
        base = views_lookup[(mode, "min", "direct")]
        for score in scores:
            if score == "direct":
                continue
            alt = views_lookup[(mode, "min", score)]
            name = f"{mode}:direct_vs_{score}"
            comparisons[name] = compare_predictions(
                base,
                alt,
                base_name=f"{mode}/min/direct",
                alt_name=f"{mode}/min/{score}",
            )

    # Uniform -> Dynamic changes under identical score/min aggregation.
    for score in scores:
        comparisons[f"uniform_vs_dynamic:{score}"] = compare_predictions(
            views_lookup[("uniform", "min", score)],
            views_lookup[("dynamic", "min", score)],
            base_name=f"uniform/min/{score}",
            alt_name=f"dynamic/min/{score}",
        )

    # Primary expert oracles.
    primary_views = {
        "direct_uniform": views_lookup[("uniform", "min", "direct")],
        "flow_uniform": views_lookup[("uniform", "min", "tolast_mean")],
        "direct_dynamic": views_lookup[("dynamic", "min", "direct")],
        "flow_dynamic": views_lookup[("dynamic", "min", "tolast_mean")],
    }
    oracles = {
        "uniform_direct_flow": oracle_stats([
            ("direct_uniform", primary_views["direct_uniform"]),
            ("flow_uniform", primary_views["flow_uniform"]),
        ]),
        "dynamic_direct_flow": oracle_stats([
            ("direct_dynamic", primary_views["direct_dynamic"]),
            ("flow_dynamic", primary_views["flow_dynamic"]),
        ]),
        "direct_uniform_dynamic": oracle_stats([
            ("direct_uniform", primary_views["direct_uniform"]),
            ("direct_dynamic", primary_views["direct_dynamic"]),
        ]),
        "flow_uniform_dynamic": oracle_stats([
            ("flow_uniform", primary_views["flow_uniform"]),
            ("flow_dynamic", primary_views["flow_dynamic"]),
        ]),
        "four_expert": oracle_stats(list(primary_views.items())),
    }

    # Pair-level numerical sanity diagnostics.
    uniform_direct_delta = []
    dynamic_direct_delta = []
    source_state_delta = []
    for ev in pair_evidence.values():
        uniform_direct_delta.append(abs(float(ev["uniform"]["sanity_direct_minus_tosupport_step0"])))
        dynamic_direct_delta.append(abs(float(ev["dynamic"]["sanity_direct_minus_tosupport_step0"])))
        source_state_delta.append(abs(float(ev["flow"]["source_state_max_abs_diff"])))
    sanity = {
        "uniform_direct_vs_tosupport_step0_max_abs_diff": max(uniform_direct_delta, default=0.0),
        "dynamic_direct_vs_tosupport_step0_max_abs_diff": max(dynamic_direct_delta, default=0.0),
        "flow_state0_vs_input_query_latent_max_abs_diff": max(source_state_delta, default=0.0),
    }

    query_results_path = output_dir / "query_results.json"
    _save_json(query_results_path, {
        "queries": list(query_output.values()),
        "protocol": {
            "candidate_way_mode": "from_existing_flat_json_per_query",
            "max_n_shot": None,
            "requested_max_n_shot_ignored": args.max_n_shot,
            "support_selection": "all_valid_pair_rows_from_existing_flat_json_no_cap",
            "support_aggregations": aggregations,
            "softmin_tau": args.softmin_tau,
            "primary_support_aggregation": "min",
            "primary_flow_score": "tolast_mean",
        },
    })

    summary = {
        "config": vars(args),
        "protocol": {
            "input_mode": "existing_flat_mllm_fsl_query_support_pairs",
            "mllm_rerun": False,
            "episode_resampling": False,
            "candidate_way_mode": "actual_unique_support_labels_per_query",
            "num_raw_records": len(raw_records),
            "num_valid_pair_rows_before_support_selection": len(all_valid_pair_specs),
            # Backward-compatible alias retained for old parsers. No cap is applied.
            "num_valid_pair_rows_before_shot_cap": len(all_valid_pair_specs),
            "num_selected_pair_rows": len(pair_specs),
            "num_skipped_pair_rows": len(skipped_indices),
            "num_unique_query_support_pairs": len(unique_pairs),
            "max_n_shot": None,
            "requested_max_n_shot_ignored": args.max_n_shot,
            "support_selection": "all_valid_pair_rows_from_existing_flat_json_no_cap",
            "shot_selection_diagnostics": shot_diagnostics,
            "num_unique_queries": len(groups),
            "flow_num_steps": DEFAULT_NUM_STEPS,
            "flow_solver": DEFAULT_SOLVER,
            "similarity_model_used": False,
            "early_stop_used": False,
            "primary_flow_score": "tolast_mean",
            "primary_support_aggregation": "min",
            "analysis_support_aggregations": aggregations,
        },
        "geometry": {
            "rae_latent": "(1024,16,16)",
            "metric_checkpoint": metric_metadata,
            "ground_cost": "squared-euclidean on L2-normalized metric tokens",
            "uniform_mass": "1/N on both token sets",
            "dynamic_mass": "DeepEMD-style pair-conditioned SK from raw metric tokens",
            "dynamic_flow_mass_rule": "compute once from direct t0 query-support pair; reuse across all 8 steps and all trajectory distance families",
            "sinkhorn_reg": args.sinkhorn_reg,
            "sinkhorn_iter": args.sinkhorn_iter,
            "sk_weight_eps": args.sk_weight_eps,
            "ot_pool_factor": args.ot_pool_factor,
        },
        "score_summaries": score_summaries,
        "comparisons_min_support_agg": comparisons,
        "oracles_primary": oracles,
        "sanity": sanity,
    }
    summary_path = output_dir / "summary.json"
    _save_json(summary_path, summary)

    csv_path = output_dir / "score_summary.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    fusion_uniform = build_fusion_ready(groups, mode="uniform", softmin_tau=args.softmin_tau)
    fusion_dynamic = build_fusion_ready(groups, mode="dynamic", softmin_tau=args.softmin_tau)
    fusion_uniform_path = output_dir / "fsl_full_results_for_fusion_uniform.json"
    fusion_dynamic_path = output_dir / "fsl_full_results_for_fusion_dynamic.json"
    _save_json(fusion_uniform_path, fusion_uniform)
    _save_json(fusion_dynamic_path, fusion_dynamic)

    # Console headline: primary min-support results + four-expert oracle.
    def acc(mode: str, score: str) -> float:
        return float(score_summaries[f"{mode}/min/{score}"]["accuracy_percent"])

    print("\n" + "=" * 110)
    print("[DONE] Offline RAE FSL service finished")
    print(f"Uniform Direct       : {acc('uniform', 'direct'):.4f}%")
    print(f"Uniform Flow-ToLast  : {acc('uniform', 'tolast_mean'):.4f}%")
    print(f"Dynamic Direct       : {acc('dynamic', 'direct'):.4f}%")
    print(f"Dynamic Flow-ToLast  : {acc('dynamic', 'tolast_mean'):.4f}%")
    print(f"4-expert Oracle      : {oracles['four_expert'].get('oracle_accuracy_percent', 0.0):.4f}%")
    print(f"Sanity direct~step0  : uniform={sanity['uniform_direct_vs_tosupport_step0_max_abs_diff']:.3e}, "
          f"dynamic={sanity['dynamic_direct_vs_tosupport_step0_max_abs_diff']:.3e}")
    print("Outputs:")
    for pth in (
        result_path,
        query_results_path,
        summary_path,
        csv_path,
        fusion_uniform_path,
        fusion_dynamic_path,
    ):
        print(f"  {pth}")
    print("=" * 110)


if __name__ == "__main__":
    main()