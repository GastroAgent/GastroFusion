#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark exact Wasserstein/EMD vs Dynamic-SK Sinkhorn on Direct Q->Support classification.

Only two requested outputs are benchmarked:
  1) Exact-Wass: exact unregularized OT solved by OpenCV EMD.
  2) Dynamic-SK: current entropy-regularized Dynamic-SK Sinkhorn.

For a fair solver comparison, BOTH use exactly the same:
  - Q2L/M2 MetricEncoder tokens,
  - pooled token sets,
  - pair-conditioned Dynamic-SK node masses,
  - L2-normalized tokens,
  - squared-Euclidean ground cost,
  - K-shot aggregation.

The exact solver follows the cv2.EMD strategy in the supplied wassertain.py,
while Dynamic-SK uses rae_flow_metric_v1.models.sk_sinkhorn.
Flow/Q2Last/Q2StepMean/fusion/oracle/rescue and other diagnostics are removed.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow"
DATA_BASE = os.path.join(PROJECT_ROOT, "fsl_multi_vae/data/data_tsy_827_doctor_test_split")
EVAL_SETTING = "test_test"
RESULT_DIR = os.path.join(PROJECT_ROOT, "fsl_multi_vae/rae_flow_metric_v1/results_benchmark_wass_vs_dynamic_sk_testtest")
HOSPITALS = ["华西", "自贡", "南充", "苍溪"]
PARTS = ["十二指肠", "胃", "食管"]
METRIC_CHECKPOINTS = {
    "十二指肠": os.path.join(PROJECT_ROOT, "weights_metric_v1", "十二指肠-test-test", "metric_q2l_m0b_rae_manifest.pt"),
    "胃": os.path.join(PROJECT_ROOT, "weights_metric_v1", "胃-test-test", "metric_q2l_m0b_rae_manifest.pt"),
    "食管": os.path.join(PROJECT_ROOT, "weights_metric_v1", "食管-test-test", "metric_q2l_m0b_rae_manifest.pt"),
}
def N_WAY_FOR_PART(part: str) -> int:
    return 8

def EFFECTIVE_N_WAY(requested_way: int, candidate_labels: Sequence[str]) -> int:
    return min(requested_way, len(candidate_labels))

N_SHOT = 5
N_QUERY = 1
EPISODES_PER_TASK = int(os.environ.get("EPISODES_PER_TASK", "100"))
SEED = 42
IMAGE_SIZE = 256
RAE_CHUNK_SIZE = 8
SHOT_AGG = "softmin"
SHOT_TAU = 0.10
LOG_EVERY = 10

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rae_flow_metric_v1.models.metric_encoder import MetricEncoder, pool_square_tokens
from rae_flow_metric_v1.models.sk_sinkhorn import (
    aligned_sk_sinkhorn_distance,
    compute_aligned_dynamic_sk_weights,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class EpisodePlan:
    episode_id: int
    candidate_labels: List[str]
    query_label: str
    query_path: str
    support_paths: Dict[str, List[str]]


@dataclass
class MetricRuntime:
    model: nn.Module
    checkpoint_path: str
    ot_pool_factor: int
    sinkhorn_reg: float
    sinkhorn_iter: int
    sk_weight_eps: float


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


def extract_metric_state(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = None
    for key in ("metric_encoder", "metric", "metric_state_dict", "fsl_model"):
        if isinstance(ckpt.get(key), dict):
            state = ckpt[key]
            break
    if state is None and isinstance(ckpt.get("model"), dict):
        state = ckpt["model"]
    if state is None:
        raise KeyError("checkpoint has no MetricEncoder state")
    cleaned = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("module.", "metric.", "metric_encoder."):
            while name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def load_metric_for_part(part: str, device: torch.device) -> MetricRuntime:
    path = METRIC_CHECKPOINTS[part]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{part} checkpoint missing: {path}")
    ckpt = torch_load(path)
    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint must be dict: {path}")
    a = as_args_dict(ckpt.get("args", {}))
    hidden = int(a.get("hidden_dim", 256))
    metric_dim = int(a.get("metric_dim", 256))
    dropout = float(a.get("dropout", 0.0))
    model = MetricEncoder(1024, hidden, metric_dim, dropout=dropout)
    model.load_state_dict(extract_metric_state(ckpt), strict=True)
    model = model.to(device).eval().requires_grad_(False)
    rt = MetricRuntime(
        model=model,
        checkpoint_path=path,
        ot_pool_factor=max(int(a.get("ot_pool_factor", 1)), 1),
        sinkhorn_reg=float(a.get("sinkhorn_reg", 0.1)),
        sinkhorn_iter=int(a.get("sinkhorn_iter", 30)),
        sk_weight_eps=float(a.get("sk_weight_eps", 1e-3)),
    )
    print(
        f"[Metric] {part}: pool={rt.ot_pool_factor} reg={rt.sinkhorn_reg} "
        f"iter={rt.sinkhorn_iter} eps={rt.sk_weight_eps} checkpoint={path}"
    )
    return rt


def build_rae(project_root: str, device: torch.device):
    from rae.rae import RAE
    model = RAE(
        encoder_name="dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]",
        resolution=256,
        decoder_config_path=os.path.join(project_root, "rae/configs/decoder/ViTXL/config.json"),
        pretrained_decoder_path=os.path.join(project_root, "flow_matcher_otcfm/rae_v2/decoder.pt"),
        noise_tau=0.0,
        normalization_stat_path=os.path.join(project_root, "flow_matcher_otcfm/rae_v2/stats.pt"),
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def scan_class_folders(root: str) -> Dict[str, List[str]]:
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"Class root not found: {root}")
    result: Dict[str, List[str]] = {}
    for child in sorted(root_p.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        paths = sorted(str(p) for p in child.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if paths:
            result[child.name] = paths
    if not result:
        raise RuntimeError(f"No class images found under: {root}")
    return result


def prepare_task_pools(hospital: str, part: str):
    query_root = os.path.join(DATA_BASE, "indomain_support_76", hospital, part)
    support_root = query_root
    query_pool = scan_class_folders(query_root)
    support_pool = query_pool
    candidate_labels = [c for c in sorted(query_pool) if len(query_pool[c]) >= N_SHOT + 1]
    query_labels = list(candidate_labels)
    query_pool = {c: query_pool[c] for c in candidate_labels}
    support_pool = {c: support_pool[c] for c in candidate_labels}
    if len(candidate_labels) < 2:
        raise RuntimeError(f"{hospital}/{part}: fewer than 2 valid classes")
    return query_pool, support_pool, candidate_labels, query_labels, query_root, support_root


def generate_episode_plans(
    query_pool: Dict[str, List[str]],
    support_pool: Dict[str, List[str]],
    candidate_labels: Sequence[str],
    query_labels: Sequence[str],
    *, n_way: int, n_shot: int, n_episodes: int, seed: int,
) -> List[EpisodePlan]:
    rng = random.Random(seed)
    candidate_labels = list(candidate_labels)
    query_labels = list(query_labels)
    n_way = min(int(n_way), len(candidate_labels))
    plans: List[EpisodePlan] = []
    for episode_id in range(n_episodes):
        query_label = rng.choice(query_labels)
        distractors = [c for c in candidate_labels if c != query_label]
        selected = [query_label] + rng.sample(distractors, n_way - 1)
        query_path = rng.choice(query_pool[query_label])
        query_key = os.path.normcase(os.path.abspath(query_path))
        support_paths: Dict[str, List[str]] = {}
        for label in selected:
            candidates = [p for p in support_pool[label] if os.path.normcase(os.path.abspath(p)) != query_key]
            if len(candidates) < n_shot:
                raise RuntimeError(f"episode={episode_id} class={label}: support={len(candidates)} < shot={n_shot}")
            support_paths[label] = rng.sample(candidates, n_shot)
        rng.shuffle(selected)
        plans.append(EpisodePlan(
            episode_id=episode_id,
            candidate_labels=list(selected),
            query_label=query_label,
            query_path=query_path,
            support_paths={c: support_paths[c] for c in selected},
        ))
    return plans


def load_images(paths: Sequence[str], transform) -> torch.Tensor:
    out = []
    for path in paths:
        with Image.open(path) as im:
            out.append(transform(im.convert("RGB")))
    return torch.stack(out, dim=0)


@torch.inference_mode()
def encode_rae_chunked(rae: nn.Module, images: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_size = int(images.shape[0]) if chunk_size <= 0 else int(chunk_size)
    return torch.cat([
        rae.encode(images[i:i + chunk_size])
        for i in range(0, images.shape[0], chunk_size)
    ], dim=0).float()


def aggregate_kshot(pair_distance: torch.Tensor) -> torch.Tensor:
    if SHOT_AGG == "mean":
        return pair_distance.mean(dim=-1)
    if SHOT_AGG == "min":
        return pair_distance.min(dim=-1).values
    if SHOT_AGG == "softmin":
        tau = float(SHOT_TAU)
        return -tau * (
            torch.logsumexp(-pair_distance / tau, dim=-1)
            - math.log(float(pair_distance.shape[-1]))
        )
    raise ValueError(SHOT_AGG)


@torch.inference_mode()
def extract_aligned_tokens(
    metric: nn.Module,
    query_z: torch.Tensor,
    support_z: torch.Tensor,
    pool_factor: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One MetricEncoder pass; return aligned raw and normalized token pairs."""
    all_z = torch.cat([query_z, support_z], dim=0)
    feat = metric(all_z)
    q_raw_one = pool_square_tokens(feat.raw[:1], pool_factor)
    s_raw = pool_square_tokens(feat.raw[1:], pool_factor)
    q_raw = q_raw_one.expand(s_raw.shape[0], -1, -1).contiguous()
    q_norm = F.normalize(q_raw.float(), p=2, dim=-1)
    s_norm = F.normalize(s_raw.float(), p=2, dim=-1)
    return q_raw, s_raw, q_norm, s_norm


@torch.inference_mode()
def exact_wasserstein_distance(
    q_norm: torch.Tensor,
    s_norm: torch.Tensor,
    q_raw: torch.Tensor,
    s_raw: torch.Tensor,
    *, weight_eps: float,
) -> torch.Tensor:
    """Exact unregularized OT via cv2.EMD, aligned pair form [P,M,D] -> [P].

    This mirrors the exact OpenCV EMD route in the supplied wassertain.py, but
    deliberately uses the SAME squared-Euclidean ground cost as Dynamic-SK so
    the benchmark isolates the OT solver approximation rather than cost scaling.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Exact-Wass benchmark requires OpenCV (cv2).") from exc

    weights = compute_aligned_dynamic_sk_weights(q_raw, s_raw, weight_eps=weight_eps)
    cost = (
        (q_norm.unsqueeze(-2) - s_norm.unsqueeze(-3)).pow(2).sum(dim=-1)
    ).clamp_min(0.0)

    p, mq, ms = cost.shape
    dim = q_norm.shape[-1]
    flat_cost = cost.reshape(p, mq, ms)
    flat_wq = weights.query.reshape(p, mq)
    flat_ws = weights.support.reshape(p, ms)
    flat_q = q_norm.reshape(p, mq, dim)
    flat_s = s_norm.reshape(p, ms, dim)

    distances: List[float] = []
    for idx in range(p):
        c_np = flat_cost[idx].detach().cpu().numpy().astype(np.float32, copy=False)
        wq_np = flat_wq[idx].detach().cpu().numpy().astype(np.float32, copy=False)
        ws_np = flat_ws[idx].detach().cpu().numpy().astype(np.float32, copy=False)
        q_np = flat_q[idx].detach().cpu().numpy().astype(np.float32, copy=False)
        s_np = flat_s[idx].detach().cpu().numpy().astype(np.float32, copy=False)
        sig_q = np.concatenate([wq_np[:, None], q_np], axis=1)
        sig_s = np.concatenate([ws_np[:, None], s_np], axis=1)
        _, _, flow = cv2.EMD(sig_q, sig_s, cv2.DIST_USER, c_np)
        distances.append(float((flow * c_np).sum()))
    return torch.tensor(distances, device=q_norm.device, dtype=torch.float32)


@torch.inference_mode()
def dynamic_sk_distance(
    q_norm: torch.Tensor,
    s_norm: torch.Tensor,
    q_raw: torch.Tensor,
    s_raw: torch.Tensor,
    *, reg: float, n_iter: int, weight_eps: float,
) -> torch.Tensor:
    return aligned_sk_sinkhorn_distance(
        q_norm,
        s_norm,
        query_raw=q_raw,
        support_raw=s_raw,
        reg=reg,
        n_iter=n_iter,
        weight_eps=weight_eps,
        normalize_cost_points=False,  # already L2-normalized above
        return_transport=False,
        return_cost=False,
    ).distance.float()


def timed_cuda_call(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


@torch.inference_mode()
def evaluate_task(
    hospital: str,
    part: str,
    *,
    rae: nn.Module,
    metric_rt: MetricRuntime,
    device: torch.device,
    transform,
) -> Dict[str, Any]:
    query_pool, support_pool, candidate_labels, query_labels, query_root, support_root = prepare_task_pools(hospital, part)
    requested_way = N_WAY_FOR_PART(part)
    episode_n_way = EFFECTIVE_N_WAY(requested_way, candidate_labels)
    task_seed = SEED + sum(ord(c) for c in hospital) * 1009 + sum(ord(c) for c in part) * 9176
    plans = generate_episode_plans(
        query_pool, support_pool, candidate_labels, query_labels,
        n_way=episode_n_way, n_shot=N_SHOT, n_episodes=EPISODES_PER_TASK, seed=task_seed,
    )

    print("=" * 118)
    print(f"[Exact-Wass vs Dynamic-SK] {hospital}/{part} | {episode_n_way}-way {N_SHOT}-shot 1-query | episodes={len(plans)}")
    print(f"setting={EVAL_SETTING} | query={query_root} | support={support_root}")
    print(
        f"pool={metric_rt.ot_pool_factor} | Dynamic-SK reg={metric_rt.sinkhorn_reg} "
        f"iter={metric_rt.sinkhorn_iter} eps={metric_rt.sk_weight_eps}"
    )
    print("Timing scope: DISTANCE ONLY; RAE + MetricEncoder feature extraction excluded.")
    print("=" * 118)

    exact_correct = 0
    sk_correct = 0
    exact_seconds = 0.0
    sk_seconds = 0.0
    total_pairs = 0
    warmed = False

    for ep_idx, ep in enumerate(plans):
        support_flat: List[str] = []
        for c in ep.candidate_labels:
            support_flat.extend(ep.support_paths[c])

        support_images = load_images(support_flat, transform).to(device, non_blocking=True)
        query_image = load_images([ep.query_path], transform).to(device, non_blocking=True)
        all_images = torch.cat([support_images, query_image], dim=0)
        all_z = encode_rae_chunked(rae, all_images, RAE_CHUNK_SIZE)
        support_z = all_z[:-1]
        query_z = all_z[-1:]

        q_raw, s_raw, q_norm, s_norm = extract_aligned_tokens(
            metric_rt.model, query_z, support_z, metric_rt.ot_pool_factor
        )

        # One discarded warm-up removes first-call CUDA/OpenCV initialization from timing.
        if not warmed:
            _ = dynamic_sk_distance(
                q_norm, s_norm, q_raw, s_raw,
                reg=metric_rt.sinkhorn_reg,
                n_iter=metric_rt.sinkhorn_iter,
                weight_eps=metric_rt.sk_weight_eps,
            )
            _ = exact_wasserstein_distance(
                q_norm, s_norm, q_raw, s_raw,
                weight_eps=metric_rt.sk_weight_eps,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            warmed = True

        # Exact-Wass is intentionally the first/default benchmark branch.
        exact_flat, dt_exact = timed_cuda_call(lambda: exact_wasserstein_distance(
            q_norm, s_norm, q_raw, s_raw, weight_eps=metric_rt.sk_weight_eps
        ))
        sk_flat, dt_sk = timed_cuda_call(lambda: dynamic_sk_distance(
            q_norm, s_norm, q_raw, s_raw,
            reg=metric_rt.sinkhorn_reg,
            n_iter=metric_rt.sinkhorn_iter,
            weight_eps=metric_rt.sk_weight_eps,
        ))

        way = len(ep.candidate_labels)
        exact_pair = exact_flat.reshape(way, N_SHOT)
        sk_pair = sk_flat.reshape(way, N_SHOT)
        exact_class = aggregate_kshot(exact_pair)
        sk_class = aggregate_kshot(sk_pair)
        exact_pred = ep.candidate_labels[int(exact_class.argmin().item())]
        sk_pred = ep.candidate_labels[int(sk_class.argmin().item())]

        exact_correct += int(exact_pred == ep.query_label)
        sk_correct += int(sk_pred == ep.query_label)
        exact_seconds += dt_exact
        sk_seconds += dt_sk
        total_pairs += len(support_flat)

        if LOG_EVERY > 0 and (ep_idx + 1) % LOG_EVERY == 0:
            n = ep_idx + 1
            print(
                f"  [{n:04d}/{len(plans):04d}] "
                f"Exact={100.0*exact_correct/n:.2f}% ({exact_seconds:.3f}s) | "
                f"Dynamic-SK={100.0*sk_correct/n:.2f}% ({sk_seconds:.3f}s)"
            )

        del support_images, query_image, all_images, all_z, support_z, query_z
        del q_raw, s_raw, q_norm, s_norm, exact_flat, sk_flat, exact_pair, sk_pair

    n = len(plans)
    result = {
        "hospital": hospital,
        "part": part,
        "eval_setting": EVAL_SETTING,
        "protocol": {
            "n_way": episode_n_way,
            "n_shot": N_SHOT,
            "n_query": N_QUERY,
            "episodes": n,
            "total_query_support_pairs": total_pairs,
            "kshot_aggregation": SHOT_AGG,
            "shot_tau": SHOT_TAU,
            "timing_scope": "distance_only_excludes_rae_and_metric_encoder",
        },
        "checkpoint": metric_rt.checkpoint_path,
        "distance_config": {
            "exact_wass": "OpenCV EMD exact unregularized OT",
            "dynamic_sk": "entropy-regularized Dynamic-SK Sinkhorn",
            "shared_node_mass": "pair-conditioned Dynamic-SK",
            "shared_ground_cost": "squared Euclidean on L2-normalized metric tokens",
            "ot_pool_factor": metric_rt.ot_pool_factor,
            "sinkhorn_reg": metric_rt.sinkhorn_reg,
            "sinkhorn_iter": metric_rt.sinkhorn_iter,
            "sk_weight_eps": metric_rt.sk_weight_eps,
        },
        "exact_wass_accuracy_percent": 100.0 * exact_correct / n,
        "dynamic_sk_accuracy_percent": 100.0 * sk_correct / n,
        "exact_wass_total_seconds": exact_seconds,
        "dynamic_sk_total_seconds": sk_seconds,
        "exact_wass_ms_per_episode": 1000.0 * exact_seconds / n,
        "dynamic_sk_ms_per_episode": 1000.0 * sk_seconds / n,
        "exact_wass_ms_per_pair": 1000.0 * exact_seconds / max(total_pairs, 1),
        "dynamic_sk_ms_per_pair": 1000.0 * sk_seconds / max(total_pairs, 1),
        "dynamic_sk_speedup_x": exact_seconds / max(sk_seconds, 1e-12),
    }
    return result


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "hospital", "part", "eval_setting", "n_way", "n_shot", "episodes", "total_pairs",
        "exact_wass_accuracy_percent", "dynamic_sk_accuracy_percent",
        "exact_wass_total_seconds", "dynamic_sk_total_seconds",
        "exact_wass_ms_per_episode", "dynamic_sk_ms_per_episode",
        "exact_wass_ms_per_pair", "dynamic_sk_ms_per_pair", "dynamic_sk_speedup_x",
        "checkpoint",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main() -> None:
    if N_QUERY != 1 or N_SHOT < 1 or EPISODES_PER_TASK < 1:
        raise ValueError("requires 1-query, shot>=1, episodes>=1")
    setup_seed(SEED)
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("benchmark expects CUDA for the current Dynamic-SK implementation")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 118)
    print("[BENCHMARK] Exact Wasserstein/EMD vs Dynamic-SK Sinkhorn")
    print(f"setting={EVAL_SETTING} | episodes/task={EPISODES_PER_TASK}")
    print(f"result_dir={RESULT_DIR}")
    print("Only Accuracy + Distance Time are retained.")
    print("=" * 118)

    rae = build_rae(PROJECT_ROOT, device)
    transform = build_transform(IMAGE_SIZE)
    metric_cache: Dict[str, MetricRuntime] = {}
    rows: List[Dict[str, Any]] = []

    for part in PARTS:
        metric_rt = metric_cache.setdefault(part, load_metric_for_part(part, device))
        for hospital in HOSPITALS:
            result = evaluate_task(
                hospital, part, rae=rae, metric_rt=metric_rt,
                device=device, transform=transform,
            )
            out_json = os.path.join(RESULT_DIR, part, f"{hospital}_{part}_{EVAL_SETTING}_wass_vs_dynamic_sk.json")
            save_json(out_json, result)
            p = result["protocol"]
            rows.append({
                "hospital": hospital,
                "part": part,
                "eval_setting": EVAL_SETTING,
                "n_way": p["n_way"],
                "n_shot": p["n_shot"],
                "episodes": p["episodes"],
                "total_pairs": p["total_query_support_pairs"],
                **{k: result[k] for k in (
                    "exact_wass_accuracy_percent", "dynamic_sk_accuracy_percent",
                    "exact_wass_total_seconds", "dynamic_sk_total_seconds",
                    "exact_wass_ms_per_episode", "dynamic_sk_ms_per_episode",
                    "exact_wass_ms_per_pair", "dynamic_sk_ms_per_pair", "dynamic_sk_speedup_x",
                )},
                "checkpoint": result["checkpoint"],
            })
            print(
                f"[RESULT] {hospital}/{part}: "
                f"Exact Acc={result['exact_wass_accuracy_percent']:.2f}% "
                f"Time={result['exact_wass_total_seconds']:.3f}s | "
                f"Dynamic-SK Acc={result['dynamic_sk_accuracy_percent']:.2f}% "
                f"Time={result['dynamic_sk_total_seconds']:.3f}s | "
                f"Speedup={result['dynamic_sk_speedup_x']:.2f}x"
            )

    summary_csv = os.path.join(RESULT_DIR, "summary_wass_vs_dynamic_sk.csv")
    summary_json = os.path.join(RESULT_DIR, "summary_wass_vs_dynamic_sk.json")
    save_csv(summary_csv, rows)
    save_json(summary_json, {
        "setting": EVAL_SETTING,
        "default_reference": "exact_wass",
        "timing_scope": "distance_only_excludes_rae_and_metric_encoder",
        "rows": rows,
    })
    print("=" * 118)
    print(f"[DONE] CSV : {summary_csv}")
    print(f"[DONE] JSON: {summary_json}")
    print("=" * 118)


if __name__ == "__main__":
    main()
