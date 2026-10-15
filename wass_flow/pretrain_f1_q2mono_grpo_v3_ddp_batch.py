#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q2Mono-GRPO v1 pretraining on class-JSON roots.

Design summary
--------------
Frozen modules:
  * RAE
  * support-conditioned Flow / DiT

Trainable modules:
  * Compressor (the existing MetricEncoder architecture; structure is NOT changed)
  * Q2MonoRetentionPolicy (support-conditioned LSTM + retention head)

The Q2Mono candidate rule itself is fixed and matches the evaluator semantics:
for each Query--Support trajectory pair, retain only strict running-record
Query-to-Step distances as candidate states.  The learned policy may keep or
reject those candidate states; it cannot create candidates outside Q2Mono.

Objectives:
  1) Compressor Direct classification CE
  2) Compressor VISReg on cloud-level pre-L2 embeddings
  3) joint differentiable soft-Q2Mono classification CE
  4) group-relative PPO/GRPO loss on sampled hard retention paths
  5) Direct-safe hinge for the policy
  6) Bernoulli KL to an EMA old policy

Crucial gradient separation:
  * Direct CE + VISReg train Compressor.
  * soft-Q2Mono CE jointly trains Compressor + policy.
  * PPO/GRPO, safe and KL use DETACHED Compressor-derived states/distances,
    so RL cannot improve reward by merely distorting the metric scale.

Data:
  --train_jsons_root and optional --val_jsons_root each point to a directory
  containing many class JSON files.  The loader accepts common JSON layouts:
  a list of image paths; a list of dict records containing path/image_path/file;
  or a dict with images/items/data/samples/paths/files.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from tqdm import tqdm

# Keep this script beside the existing rae_flow_metric_v1 training scripts.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

import train_f1_support_conditioned_lstm_v5_medical_manifest as base
from models.metric_encoder import MetricEncoder, pool_square_tokens
from models.sk_sinkhorn import aligned_sk_sinkhorn_distance

try:
    from visreg import VISReg, cloud_features_to_visreg_input
except ImportError:
    # Allows using the exact uploaded/reference implementation renamed as visreg.py.
    from visreg import VISReg, cloud_features_to_visreg_input


STAGE = "F1-Q2Mono-GRPO-v1-DDP-classjson-pretrain"

FEATURE_NAMES = [
    # Pair-local geometry.
    "q2step",
    "direct_qs",
    "q2step_over_direct",
    "delta_prev",
    "record_gain",
    "prefix_mean",
    "prefix_std",
    "prefix_slope",
    "record_count_frac",
    "time_frac",
    # Class-level discriminability at this trajectory step.
    "class_top1_prob",
    "class_prob_margin",
    "class_entropy",
    "class_delta_margin_vs_direct",
    "class_delta_entropy_vs_direct",
    "class_agree_direct",
    # K-shot consistency for the pair's class.
    "class_shot_mean",
    "class_shot_std",
    "class_shot_cv",
]


# -----------------------------------------------------------------------------
# CLI / basic helpers
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Q2Mono-GRPO v3 DDP batched class-JSON pretraining")

    p.add_argument("--train_jsons_root", type=str, required=True)
    p.add_argument("--val_jsons_root", type=str, default="")
    p.add_argument("--val_fraction", type=float, default=0.10,
                   help="Used only when --val_jsons_root is empty; per-class deterministic split.")
    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--m2_checkpoint", type=str, required=True,
                   help="Initial Compressor checkpoint + OT/temperature configuration.")
    p.add_argument("--dit_checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume_checkpoint", type=str, default=None)

    # Episode sampling.
    p.add_argument("--n_way", type=int, default=8)
    p.add_argument("--n_shot", type=int, default=5)
    p.add_argument("--n_query", type=int, default=1)
    p.add_argument("--episodes_per_epoch", type=int, default=500)
    p.add_argument("--episode_batch_size", type=int, default=4,
                   help="True per-rank episode batch size; episodes are merged for RAE/Flow/Compressor/policy forwards.")
    p.add_argument("--val_episodes", type=int, default=200)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--rae_chunk_size", type=int, default=32)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)

    # Devices.
    p.add_argument("--compressor_device", type=str, default="cuda:0")
    p.add_argument("--flow_device", type=str, default="auto")
    p.add_argument("--rae_device", type=str, default="auto")

    # Flow.
    p.add_argument("--dit_variant", type=str, default="L", choices=["S", "B", "L", "L12"])
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--num_steps", type=int, default=8)
    p.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    p.add_argument("--pair_chunk_size", type=int, default=16)
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hint_signal_ratio", type=float, default=None)
    p.add_argument("--stochastic_hint", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pure_noise_hint", action="store_true")
    p.add_argument("--allow_condition_mismatch", action="store_true")

    # Existing Compressor / OT properties.  Architecture is intentionally unchanged.
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--metric_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--ot_pool_factor", type=int, default=None)
    p.add_argument("--sinkhorn_reg", type=float, default=None)
    p.add_argument("--sinkhorn_iter", type=int, default=None)
    p.add_argument("--sk_weight_eps", type=float, default=None)
    p.add_argument("--shot_tau_override", type=float, default=0.10)

    # Q2Mono policy.
    p.add_argument("--policy_hidden_dim", type=int, default=256)
    p.add_argument("--policy_relation_dim", type=int, default=256)
    p.add_argument("--policy_condition_dim", type=int, default=128)
    p.add_argument("--policy_dropout", type=float, default=0.05)
    p.add_argument("--retain_init_prob", type=float, default=0.80)
    p.add_argument("--feature_ratio_clip", type=float, default=10.0)
    p.add_argument("--feature_eps", type=float, default=1e-6)

    # VISReg: exact supplied objective, applied to Compressor cloud embeddings.
    p.add_argument("--lambda_visreg", type=float, default=0.05)
    p.add_argument("--visreg_num_projections", type=int, default=1024)
    p.add_argument("--visreg_projection_chunk_size", type=int, default=128)
    p.add_argument("--visreg_center_weight", type=float, default=1.0)
    p.add_argument("--visreg_scale_weight", type=float, default=1.0)
    p.add_argument("--visreg_shape_weight", type=float, default=1.0)

    # Objective weights.
    p.add_argument("--lambda_direct_ce", type=float, default=1.0)
    p.add_argument("--lambda_joint_ce", type=float, default=1.0)
    p.add_argument("--lambda_grpo", type=float, default=0.25)
    p.add_argument("--lambda_safe", type=float, default=0.50)
    p.add_argument("--lambda_kl", type=float, default=0.02)

    # Group-relative PPO.
    p.add_argument("--grpo_group_size", type=int, default=8)
    p.add_argument("--grpo_warmup_epochs", type=int, default=1)
    p.add_argument("--grpo_clip_eps", type=float, default=0.20)
    p.add_argument("--grpo_adv_eps", type=float, default=1e-6)
    p.add_argument("--grpo_adv_clip", type=float, default=5.0)
    p.add_argument("--reward_margin_weight", type=float, default=0.25)
    p.add_argument("--reward_rescue_bonus", type=float, default=0.50)
    p.add_argument("--reward_degrade_penalty", type=float, default=1.00)
    p.add_argument("--safe_margin", type=float, default=0.0)
    p.add_argument("--ema_decay", type=float, default=0.995)

    # Optimization.
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--compressor_lr", type=float, default=1e-5)
    p.add_argument("--policy_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--show_progress", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ddp_backend", type=str, default="nccl")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def init_distributed(args) -> Tuple[int, int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the Flow runtime")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.ddp_backend, init_method="env://")
    else:
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def unwrap_model(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, DDP) else module


def resolve_devices(args, local_rank: int) -> Tuple[torch.device, torch.device, torch.device]:
    # Standard one-process-per-GPU DDP: each rank owns a complete frozen+trainable
    # pipeline on its LOCAL_RANK GPU.  Cross-rank communication is only for
    # trainable gradients / metrics / VISReg population statistics.
    dev = torch.device(f"cuda:{local_rank}")
    return dev, dev, dev


def ddp_global_visreg_input(local_vis: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """Gather cloud populations across ranks while preserving gradients locally.

    local_vis is [1,B,D]. Remote tensors are detached. During backward each rank
    contributes the derivative for its own B samples; DDP averages those partial
    gradients, so the caller multiplies VISReg by world_size to recover the global
    population objective's gradient scale.
    """
    if world_size <= 1:
        return local_vis
    gathered = [torch.empty_like(local_vis) for _ in range(world_size)]
    dist.all_gather(gathered, local_vis.detach().contiguous())
    gathered[rank] = local_vis
    return torch.cat(gathered, dim=1).contiguous()


# -----------------------------------------------------------------------------
# Flexible class-JSON pool
# -----------------------------------------------------------------------------

_PATH_KEYS = ("image_path", "path", "file", "filepath", "image", "img")
_CONTAINER_KEYS = ("images", "items", "data", "samples", "paths", "files", "records")


def _extract_paths(obj: Any) -> List[str]:
    out: List[str] = []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        for x in obj:
            out.extend(_extract_paths(x))
        return out
    if isinstance(obj, dict):
        for key in _PATH_KEYS:
            v = obj.get(key)
            if isinstance(v, str):
                out.append(v)
                return out
        for key in _CONTAINER_KEYS:
            if key in obj:
                out.extend(_extract_paths(obj[key]))
                return out
        # Last-resort: recurse into values. Useful for {id: path} / nested records.
        for v in obj.values():
            out.extend(_extract_paths(v))
    return out




def resolve_pretrain_from_m2(args, m2_ckpt: dict) -> None:
    """Inherit only Compressor/OT settings required for geometry compatibility.

    Pretraining episode width/shot are task-level choices and are intentionally NOT
    constrained by the n_way/n_shot saved in the source M2 checkpoint.
    """
    ep = m2_ckpt.get("episode_config") or {}
    ot = m2_ckpt.get("ot_config") or {}

    if args.ot_pool_factor is None:
        args.ot_pool_factor = int(ot.get("ot_pool_factor", 1))
    if args.sinkhorn_reg is None:
        args.sinkhorn_reg = float(ot.get("sinkhorn_reg", 0.1))
    if args.sinkhorn_iter is None:
        args.sinkhorn_iter = int(ot.get("sinkhorn_iter", 30))
    if args.sk_weight_eps is None:
        args.sk_weight_eps = float(ot.get("sk_weight_eps", 1e-3))

    if is_main_process():
        saved_way = ep.get("n_way")
        saved_shot = ep.get("n_shot")
        if saved_way is not None and int(saved_way) != int(args.n_way):
            print(
                f"[INFO] source M2 n_way={saved_way}; current pretrain n_way={args.n_way}. "
                "Allowed: episode width is not a Compressor architecture constraint.",
                flush=True,
            )
        if saved_shot is not None and int(saved_shot) != int(args.n_shot):
            print(
                f"[INFO] source M2 n_shot={saved_shot}; current pretrain n_shot={args.n_shot}. "
                "Allowed: K-shot aggregation is configured by the current task.",
                flush=True,
            )

def load_class_json_root(root: str, *, show_progress: bool = False, desc: str = "scan JSONs") -> Dict[str, List[str]]:
    root_p = Path(root).expanduser().resolve()
    if not root_p.is_dir():
        raise FileNotFoundError(f"JSONs root is not a directory: {root}")
    files = sorted(list(root_p.glob("*.json")) + list(root_p.glob("*.jsonl")))
    if not files:
        raise RuntimeError(f"No .json/.jsonl files found under {root_p}")

    pool: Dict[str, List[str]] = {}
    iterator = tqdm(files, desc=desc, unit="file", dynamic_ncols=True, leave=False) if show_progress else files
    for fp in iterator:
        if fp.suffix.lower() == ".jsonl":
            records = []
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            obj: Any = records
        else:
            with fp.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        paths = _extract_paths(obj)
        resolved: List[str] = []
        seen = set()
        for p in paths:
            pp = Path(os.path.expandvars(os.path.expanduser(p)))
            if not pp.is_absolute():
                pp = (fp.parent / pp).resolve()
            s = str(pp)
            if s not in seen:
                resolved.append(s)
                seen.add(s)
        if resolved:
            pool[fp.stem] = resolved
    if not pool:
        raise RuntimeError(f"Class JSON files were found, but no image paths could be parsed under {root_p}")
    return pool


def split_pool(pool: Dict[str, List[str]], val_fraction: float, seed: int) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be in (0,0.5)")
    train: Dict[str, List[str]] = {}
    val: Dict[str, List[str]] = {}
    for ci, (cls, paths) in enumerate(sorted(pool.items())):
        xs = list(paths)
        rng = random.Random(seed + 1009 * ci)
        rng.shuffle(xs)
        nv = max(1, int(round(len(xs) * val_fraction)))
        if len(xs) - nv < 2:
            continue
        val[cls] = xs[:nv]
        train[cls] = xs[nv:]
    return train, val


@dataclass
class Episode:
    classes: List[str]
    support_paths: List[str]
    query_paths: List[str]
    query_labels: torch.Tensor


def sample_episode(pool: Dict[str, List[str]], n_way: int, n_shot: int, n_query: int, rng: random.Random) -> Episode:
    eligible = [c for c, xs in pool.items() if len(xs) >= n_shot + n_query]
    if len(eligible) < n_way:
        raise RuntimeError(f"Need {n_way} eligible classes, only {len(eligible)} have >= {n_shot+n_query} images")
    classes = rng.sample(eligible, n_way)
    support_paths: List[str] = []
    query_paths: List[str] = []
    query_labels: List[int] = []
    # Support is K-shot for every candidate class.  n_query is TOTAL queries per
    # episode (matching the project's 8/7-way 5-shot 1-query protocol), not
    # n_query queries for every class.  Query classes are sampled independently
    # from the selected candidate set.
    chosen_by_class: Dict[str, List[str]] = {}
    for c in classes:
        chosen_by_class[c] = rng.sample(pool[c], n_shot + n_query)
        support_paths.extend(chosen_by_class[c][:n_shot])

    for _ in range(n_query):
        ci = rng.randrange(n_way)
        c = classes[ci]
        # Supports and queries are disjoint because they come from the tail of
        # this class's pre-sampled list.  For n_query>1, consume distinct tails.
        used_same_class = sum(1 for y in query_labels if y == ci)
        tail = chosen_by_class[c][n_shot:]
        if used_same_class >= len(tail):
            # Rare only when repeated query-class draws exhaust the tail.
            remaining = [x for x in pool[c] if x not in support_paths and x not in query_paths]
            if not remaining:
                raise RuntimeError(f"class {c!r} has no disjoint query left")
            qp = rng.choice(remaining)
        else:
            qp = tail[used_same_class]
        query_paths.append(qp)
        query_labels.append(ci)
    return Episode(classes, support_paths, query_paths, torch.tensor(query_labels, dtype=torch.long))


def load_images(paths: Sequence[str], transform) -> torch.Tensor:
    imgs = []
    for p in paths:
        with Image.open(p) as im:
            imgs.append(transform(im.convert("RGB")))
    return torch.stack(imgs, dim=0)


# -----------------------------------------------------------------------------
# Policy model
# -----------------------------------------------------------------------------

class Q2MonoRetentionPolicy(nn.Module):
    """Pair-conditioned LSTM over trajectory states.

    Compressor remains a single-input module.  Query, Support and each Flow state
    are independently compressed first.  Their interaction is formed only here.
    """

    def __init__(self, repr_dim: int, stat_dim: int, hidden_dim: int, relation_dim: int,
                 condition_dim: int, dropout: float, retain_init_prob: float) -> None:
        super().__init__()
        self.repr_dim = int(repr_dim)
        self.stat_dim = int(stat_dim)
        self.hidden_dim = int(hidden_dim)

        # Query--Support / Query--Step relation representation.
        relation_in = 8 * self.repr_dim
        self.relation = nn.Sequential(
            nn.LayerNorm(relation_in),
            nn.Linear(relation_in, relation_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(relation_dim, hidden_dim),
            nn.GELU(),
        )
        # Explicit trajectory/class statistics condition all LSTM gates separately.
        self.stat_encoder = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, condition_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, condition_dim),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, condition_dim), nn.GELU(), nn.Linear(condition_dim, condition_dim)
        )
        self.cond_norm = nn.LayerNorm(condition_dim)

        # Initial recurrent state uses Query and Support roles explicitly.
        self.query_init = nn.Sequential(nn.LayerNorm(repr_dim), nn.Linear(repr_dim, hidden_dim), nn.Tanh())
        self.support_init = nn.Sequential(nn.LayerNorm(repr_dim), nn.Linear(repr_dim, hidden_dim), nn.Tanh())
        self.cell = base.ConditionalLSTMCell(hidden_dim, hidden_dim, condition_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.retain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        p = min(max(float(retain_init_prob), 1e-4), 1 - 1e-4)
        with torch.no_grad():
            last = self.retain_head[-1]
            assert isinstance(last, nn.Linear)
            nn.init.normal_(last.weight, 0.0, 1e-3)
            nn.init.constant_(last.bias, math.log(p / (1.0 - p)))

    def forward(self, query_repr: torch.Tensor, support_repr: torch.Tensor,
                step_repr: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        """Return retention logits [P,T].

        query_repr:   [P,D]
        support_repr: [P,D]
        step_repr:    [P,T,D]
        stats:        [P,T,F]
        """
        p, t, d = step_repr.shape
        if query_repr.shape != (p, d) or support_repr.shape != (p, d):
            raise ValueError("query/support representation shape mismatch")
        if stats.shape[:2] != (p, t) or stats.shape[-1] != self.stat_dim:
            raise ValueError("policy statistics shape mismatch")

        q = query_repr[:, None, :].expand(-1, t, -1)
        s = support_repr[:, None, :].expand(-1, t, -1)
        rel = torch.cat([
            q, s, step_repr,
            (q - s).abs(), q * s,
            (step_repr - s).abs(), step_repr * s,
            (step_repr - q).abs(),
        ], dim=-1)
        x = self.relation(rel.float())

        tau = torch.arange(1, t + 1, device=stats.device, dtype=torch.float32).view(1, t, 1) / float(max(t, 1))
        cond = self.cond_norm(self.stat_encoder(stats.float()) + self.time_encoder(tau).expand(p, -1, -1))

        h = self.query_init(query_repr.float())
        c = self.support_init(support_repr.float())
        outs = []
        for j in range(t):
            h, c = self.cell(x[:, j], h, c, cond[:, j])
            outs.append(self.retain_head(self.out_norm(h)).squeeze(-1))
        return torch.stack(outs, dim=1)


# -----------------------------------------------------------------------------
# Geometry and Q2Mono
# -----------------------------------------------------------------------------

def _compress_raw(compressor: nn.Module, z: torch.Tensor, pool_factor: int) -> torch.Tensor:
    feat = compressor(z)
    raw = feat.raw
    if pool_factor and pool_factor > 1:
        raw = pool_square_tokens(raw, pool_factor)
    return raw.float()


def _aligned_distance(q_raw: torch.Tensor, s_raw: torch.Tensor, args) -> torch.Tensor:
    qn = F.normalize(q_raw.float(), p=2, dim=-1, eps=1e-8)
    sn = F.normalize(s_raw.float(), p=2, dim=-1, eps=1e-8)
    return aligned_sk_sinkhorn_distance(
        qn, sn,
        query_raw=q_raw,
        support_raw=s_raw,
        reg=float(args.sinkhorn_reg),
        n_iter=int(args.sinkhorn_iter),
        weight_eps=float(args.sk_weight_eps),
        return_transport=False,
    ).distance.float()


def shot_softmin(pair_scores: torch.Tensor, shot_tau: torch.Tensor) -> torch.Tensor:
    # [..., N, K] -> [..., N], preserving the reference normalization by log(K).
    k = pair_scores.shape[-1]
    tau = shot_tau.float().clamp_min(1e-6)
    return -tau * (torch.logsumexp(-pair_scores.float() / tau, dim=-1) - math.log(float(k)))


def class_logits_from_pair_scores(pair_scores: torch.Tensor, shot_tau: torch.Tensor, cls_tau: torch.Tensor) -> torch.Tensor:
    class_d = shot_softmin(pair_scores, shot_tau)
    return -class_d / cls_tau.float().clamp_min(1e-6)


def strict_record_mask(step_dist: torch.Tensor) -> torch.Tensor:
    """Exact Q2Mono strict running-record candidates, [P,T] bool."""
    # cummax up to previous step; first trajectory state is always a record.
    prev = torch.cat([
        torch.full_like(step_dist[:, :1], -float("inf")),
        torch.cummax(step_dist[:, :-1].detach(), dim=1).values,
    ], dim=1)
    return step_dist.detach() > prev


def soft_q2mono_pair_score(direct: torch.Tensor, steps: torch.Tensor, retain_prob: torch.Tensor,
                           candidate_mask: torch.Tensor) -> torch.Tensor:
    w = retain_prob * candidate_mask.float()
    return (direct + (w * steps).sum(dim=-1)) / (1.0 + w.sum(dim=-1))


def hard_q2mono_pair_score(direct: torch.Tensor, steps: torch.Tensor, actions: torch.Tensor,
                           candidate_mask: torch.Tensor) -> torch.Tensor:
    # actions [...,P,T]
    w = actions.float() * candidate_mask.float()
    return (direct.unsqueeze(0) + (w * steps.unsqueeze(0)).sum(dim=-1)) / (1.0 + w.sum(dim=-1))


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits.float(), dim=-1)
    n = logits.shape[-1]
    if n <= 1:
        return torch.zeros(logits.shape[:-1], device=logits.device)
    return -(p * p.clamp_min(1e-9).log()).sum(dim=-1) / math.log(float(n))


def _margin(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits.float(), dim=-1)
    top2 = probs.topk(min(2, probs.shape[-1]), dim=-1).values
    return top2[..., 0] - (top2[..., 1] if probs.shape[-1] >= 2 else 0.0)


def build_policy_statistics(step_dist: torch.Tensor, direct_pair: torch.Tensor,
                            n_way: int, n_shot: int, shot_tau: torch.Tensor,
                            cls_tau: torch.Tensor, eps: float, ratio_clip: float) -> torch.Tensor:
    """Observable GT-free pair-level statistics [P,T,F]."""
    p, t = step_dist.shape
    if p % (n_way * n_shot) != 0:
        raise ValueError("pair count must be Q*N*K")
    qn = p // (n_way * n_shot)
    ds = step_dist.reshape(qn, n_way, n_shot, t)
    dd = direct_pair.reshape(qn, n_way, n_shot)

    # Class distributions at every step and Direct.
    direct_logits = class_logits_from_pair_scores(dd, shot_tau, cls_tau)  # Q,N
    direct_margin = _margin(direct_logits)
    direct_ent = normalized_entropy(direct_logits)
    direct_pred = direct_logits.argmax(dim=-1)

    step_logits = []
    for j in range(t):
        step_logits.append(class_logits_from_pair_scores(ds[..., j], shot_tau, cls_tau))
    step_logits = torch.stack(step_logits, dim=1)  # Q,T,N
    probs = F.softmax(step_logits.float(), dim=-1)
    top2 = probs.topk(min(2, n_way), dim=-1).values
    top1 = top2[..., 0]
    margin = top2[..., 0] - (top2[..., 1] if n_way >= 2 else 0.0)
    ent = normalized_entropy(step_logits)
    pred = step_logits.argmax(dim=-1)

    # Pair-local prefix geometry.
    prefix_mean = []
    prefix_std = []
    for j in range(t):
        x = step_dist[:, :j+1]
        prefix_mean.append(x.mean(dim=1))
        prefix_std.append(x.std(dim=1, unbiased=False))
    prefix_mean = torch.stack(prefix_mean, 1)
    prefix_std = torch.stack(prefix_std, 1)
    prev = torch.cat([step_dist[:, :1], step_dist[:, :-1]], 1)
    delta = step_dist - prev
    prev_max = torch.cat([
        torch.full_like(step_dist[:, :1], -float("inf")),
        torch.cummax(step_dist[:, :-1].detach(), dim=1).values,
    ], 1)
    record_gain = torch.where(torch.isfinite(prev_max), step_dist - prev_max, step_dist)
    rec = strict_record_mask(step_dist).float()
    rec_count_frac = torch.cumsum(rec, dim=1) / torch.arange(1, t+1, device=step_dist.device).float().view(1, -1)
    slope = (step_dist - step_dist[:, :1]) / torch.arange(1, t+1, device=step_dist.device).float().view(1, -1)
    time_frac = torch.arange(1, t+1, device=step_dist.device).float().view(1, -1).expand(p, -1) / float(t)

    # Expand class-level and K-shot statistics to each pair.
    pair_class_index = torch.arange(n_way, device=step_dist.device).repeat_interleave(n_shot)
    pair_class_index = pair_class_index.repeat(qn)
    pair_query_index = torch.arange(qn, device=step_dist.device).repeat_interleave(n_way*n_shot)

    cls_top1 = top1[pair_query_index]
    cls_margin = margin[pair_query_index]
    cls_ent = ent[pair_query_index]
    dm = direct_margin[pair_query_index].unsqueeze(1)
    de = direct_ent[pair_query_index].unsqueeze(1)
    agree = (pred[pair_query_index] == direct_pred[pair_query_index].unsqueeze(1)).float()

    shot_mean = ds.mean(dim=2)  # Q,N,T
    shot_std = ds.std(dim=2, unbiased=False)
    shot_cv = shot_std / shot_mean.abs().clamp_min(eps)
    sm = shot_mean[pair_query_index, pair_class_index]
    ss = shot_std[pair_query_index, pair_class_index]
    scv = shot_cv[pair_query_index, pair_class_index]

    direct_expand = direct_pair.unsqueeze(1).expand(-1, t)
    ratio = (step_dist / direct_expand.abs().clamp_min(eps)).clamp(-ratio_clip, ratio_clip)

    feats = torch.stack([
        step_dist,
        direct_expand,
        ratio,
        delta,
        record_gain,
        prefix_mean,
        prefix_std,
        slope,
        rec_count_frac,
        time_frac,
        cls_top1,
        cls_margin,
        cls_ent,
        cls_margin - dm,
        cls_ent - de,
        agree,
        sm,
        ss,
        scv,
    ], dim=-1)
    if feats.shape[-1] != len(FEATURE_NAMES):
        raise AssertionError((feats.shape, len(FEATURE_NAMES)))
    return torch.nan_to_num(feats.float(), nan=0.0, posinf=ratio_clip, neginf=-ratio_clip)


@torch.no_grad()
def generate_post_states(flow, query_z: torch.Tensor, support_images_cpu: torch.Tensor, args,
                         n_way: int, n_shot: int) -> torch.Tensor:
    """Return [Q,N,K,T,C,H,W] support-conditioned Flow states."""
    qn = query_z.shape[0]
    ns = support_images_cpu.shape[0]
    target_vision = flow.encode_target_vision(support_images_cpu)
    hint = flow.build_hint(
        support_images_cpu,
        stochastic=bool(args.stochastic_hint),
        pure_noise=bool(args.pure_noise_hint),
        seed=None,
    )
    outs = []
    for qi in range(qn):
        qrep = query_z[qi:qi+1].expand(ns, -1, -1, -1).contiguous().to(flow.device)
        chunks = []
        for st in range(0, ns, max(1, int(args.pair_chunk_size))):
            ed = min(st + max(1, int(args.pair_chunk_size)), ns)
            chunks.append(flow.generate_post_states(
                qrep[st:ed],
                target_vision=target_vision[st:ed],
                hint=hint[st:ed],
                num_steps=int(args.num_steps),
                solver=args.solver,
            ).detach().cpu())
        outs.append(torch.cat(chunks, dim=0))
    z = torch.stack(outs, dim=0)  # Q,S,T,C,H,W
    return z.reshape(qn, n_way, n_shot, args.num_steps, *z.shape[-3:])


def compute_episode_geometry(compressor: nn.Module, support_z: torch.Tensor, query_z: torch.Tensor,
                             post_states_cpu: torch.Tensor, args, comp_device: torch.device):
    """Differentiable Compressor geometry with one Compressor forward per episode.

    A single forward is important for DDP correctness because support, query and
    trajectory states all share the same trainable Compressor parameters.
    """
    qn = query_z.shape[0]
    n_way, n_shot, t = post_states_cpu.shape[1:4]
    ns = n_way * n_shot

    support_dev = support_z.to(comp_device, non_blocking=True)
    query_dev = query_z.to(comp_device, non_blocking=True)
    post_flat = post_states_cpu.reshape(qn*ns*t, *post_states_cpu.shape[-3:]).to(comp_device, non_blocking=True)

    n_support = support_dev.shape[0]
    n_query = query_dev.shape[0]
    all_latent = torch.cat([support_dev, query_dev, post_flat], dim=0)
    all_raw = _compress_raw(compressor, all_latent, args.ot_pool_factor)

    support_raw_img = all_raw[:n_support]
    query_raw_img = all_raw[n_support:n_support+n_query]
    post_raw_flat = all_raw[n_support+n_query:]

    # Direct Q--Support pairs.
    qraw_direct = query_raw_img[:, None].expand(qn, ns, -1, -1).reshape(qn*ns, *query_raw_img.shape[1:])
    sraw_direct = support_raw_img[None].expand(qn, ns, -1, -1).reshape(qn*ns, *support_raw_img.shape[1:])
    direct = _aligned_distance(qraw_direct, sraw_direct, args)

    nodes, dim = post_raw_flat.shape[-2:]
    post_raw = post_raw_flat.reshape(qn*ns, t, nodes, dim)
    q2step = torch.stack([_aligned_distance(qraw_direct, post_raw[:, j], args) for j in range(t)], dim=1)

    # Mean pre-L2 Compressor features used by the policy, not raw RAE features.
    q_repr_img = query_raw_img.mean(dim=-2)
    s_repr_img = support_raw_img.mean(dim=-2)
    step_repr = post_raw.mean(dim=-2)
    q_repr_pair = q_repr_img[:, None].expand(qn, ns, -1).reshape(qn*ns, dim)
    s_repr_pair = s_repr_img[None].expand(qn, ns, -1).reshape(qn*ns, dim)

    return {
        "direct_pair": direct,
        "q2step": q2step,
        "query_repr_pair": q_repr_pair,
        "support_repr_pair": s_repr_pair,
        "step_repr": step_repr,
        "query_cloud": q_repr_img,
        "support_image_repr": s_repr_img,
    }


# -----------------------------------------------------------------------------
# PPO / reward
# -----------------------------------------------------------------------------

def bernoulli_logprob(logits: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lp1 = F.logsigmoid(logits.float())
    lp0 = F.logsigmoid(-logits.float())
    lp = actions * lp1 + (1.0-actions) * lp0
    return (lp * mask.float()).sum(dim=(-1, -2))


def bernoulli_kl(old_logits: torch.Tensor, new_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    po = torch.sigmoid(old_logits.float()).clamp(1e-6, 1-1e-6)
    pn = torch.sigmoid(new_logits.float()).clamp(1e-6, 1-1e-6)
    kl = po * (po.log()-pn.log()) + (1-po) * ((1-po).log()-(1-pn).log())
    return (kl * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def true_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # logits [...,Q,N], target [Q]
    probs = F.softmax(logits.float(), dim=-1)
    q = probs.shape[-2]
    y = target.view(*([1] * (probs.ndim-2)), q, 1).expand(*probs.shape[:-1], 1)
    pt = probs.gather(-1, y).squeeze(-1)
    other = probs.clone()
    other.scatter_(-1, y, -1.0)
    return pt - other.max(dim=-1).values


def compute_group_reward(group_logits: torch.Tensor, direct_logits: torch.Tensor,
                         target: torch.Tensor, args) -> torch.Tensor:
    """Return reward [G,Q].  No gradients are needed through rewards."""
    g, q, n = group_logits.shape
    target_g = target.unsqueeze(0).expand(g, -1)
    ce_g = F.cross_entropy(group_logits.reshape(g*q, n), target_g.reshape(-1), reduction="none").reshape(g, q)
    ce_d = F.cross_entropy(direct_logits.float(), target, reduction="none").unsqueeze(0)
    reward = ce_d - ce_g

    md = true_margin(direct_logits.unsqueeze(0), target).expand(g, -1)
    mg = true_margin(group_logits, target)
    reward = reward + float(args.reward_margin_weight) * (mg - md)

    pred_d = direct_logits.argmax(dim=-1).unsqueeze(0)
    pred_g = group_logits.argmax(dim=-1)
    yy = target.unsqueeze(0)
    rescue = (pred_d != yy) & (pred_g == yy)
    degrade = (pred_d == yy) & (pred_g != yy)
    reward = reward + float(args.reward_rescue_bonus) * rescue.float()
    reward = reward - float(args.reward_degrade_penalty) * degrade.float()
    return reward


def group_relative_ppo(current_logits: torch.Tensor, old_logits: torch.Tensor,
                       candidate_mask: torch.Tensor, direct_pair_det: torch.Tensor,
                       steps_det: torch.Tensor, target: torch.Tensor,
                       n_way: int, n_shot: int, shot_tau: torch.Tensor,
                       cls_tau: torch.Tensor, args) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Sample G hard Q2Mono-retention paths from EMA old policy and optimize clipped PPO."""
    pair_count, t = current_logits.shape
    qn = pair_count // (n_way*n_shot)
    g = int(args.grpo_group_size)

    with torch.no_grad():
        old_prob = torch.sigmoid(old_logits.float())
        probs = old_prob.unsqueeze(0).expand(g, -1, -1)
        actions = torch.bernoulli(probs)
        actions = actions * candidate_mask.float().unsqueeze(0)

        pair_scores = hard_q2mono_pair_score(direct_pair_det, steps_det, actions, candidate_mask)
        pair_scores = pair_scores.reshape(g, qn, n_way, n_shot)
        group_logits = class_logits_from_pair_scores(pair_scores, shot_tau, cls_tau)
        direct_logits = class_logits_from_pair_scores(
            direct_pair_det.reshape(qn, n_way, n_shot), shot_tau, cls_tau
        )
        reward = compute_group_reward(group_logits, direct_logits, target, args)
        mean = reward.mean(dim=0, keepdim=True)
        std = reward.std(dim=0, unbiased=False, keepdim=True)
        adv = (reward - mean) / std.clamp_min(float(args.grpo_adv_eps))
        if args.grpo_adv_clip > 0:
            adv = adv.clamp(-float(args.grpo_adv_clip), float(args.grpo_adv_clip))

    # Each query owns N*K pair trajectories. Sum path log-probs over all candidate records.
    mask_q = candidate_mask.reshape(qn, n_way*n_shot, t)
    cur_q = current_logits.reshape(qn, n_way*n_shot, t)
    old_q = old_logits.reshape(qn, n_way*n_shot, t)
    act_q = actions.reshape(g, qn, n_way*n_shot, t)

    cur_lp = []
    old_lp = []
    for gi in range(g):
        cur_lp.append(bernoulli_logprob(cur_q, act_q[gi], mask_q))
        old_lp.append(bernoulli_logprob(old_q, act_q[gi], mask_q))
    cur_lp = torch.stack(cur_lp, 0)  # G,Q
    old_lp = torch.stack(old_lp, 0)
    ratio = torch.exp((cur_lp - old_lp).clamp(-20, 20))
    clipped = ratio.clamp(1.0-float(args.grpo_clip_eps), 1.0+float(args.grpo_clip_eps))
    surrogate = torch.minimum(ratio * adv, clipped * adv)
    loss = -surrogate.mean()
    info = {
        "reward": float(reward.mean().item()),
        "reward_std": float(std.mean().item()),
        "adv_abs": float(adv.abs().mean().item()),
        "ratio": float(ratio.mean().item()),
        "clip_frac": float((ratio.ne(clipped)).float().mean().item()),
    }
    return loss, info


@torch.no_grad()
def ema_update(old: nn.Module, new: nn.Module, decay: float) -> None:
    new = unwrap_model(new)
    for po, pn in zip(old.parameters(), new.parameters()):
        po.mul_(decay).add_(pn.detach(), alpha=1.0-decay)
    for bo, bn in zip(old.buffers(), new.buffers()):
        bo.copy_(bn)


# -----------------------------------------------------------------------------
# One episode / epoch
# -----------------------------------------------------------------------------


@torch.no_grad()
def generate_post_states_batch(flow, query_z: torch.Tensor, support_images_cpu: torch.Tensor,
                               args, n_way: int, n_shot: int) -> torch.Tensor:
    """True batched Flow generation.

    query_z: [B,Q,C,H,W]
    support_images_cpu: [B,N*K,3,H,W]
    returns: [B,Q,N,K,T,C,H,W]
    """
    bsz, qn = query_z.shape[:2]
    ns = support_images_cpu.shape[1]
    support_flat = support_images_cpu.reshape(bsz * ns, *support_images_cpu.shape[2:])
    target = flow.encode_target_vision(support_flat)
    hint = flow.build_hint(
        support_flat,
        stochastic=bool(args.stochastic_hint),
        pure_noise=bool(args.pure_noise_hint),
        seed=None,
    )
    # [B,NS,...] -> [B,Q,NS,...] -> flattened pair stream.
    target = target.reshape(bsz, ns, *target.shape[1:])[:, None].expand(-1, qn, -1, *([-1] * (target.ndim - 1)))
    hint = hint.reshape(bsz, ns, *hint.shape[1:])[:, None].expand(-1, qn, -1, *([-1] * (hint.ndim - 1)))
    target = target.reshape(bsz * qn * ns, *target.shape[3:]).contiguous()
    hint = hint.reshape(bsz * qn * ns, *hint.shape[3:]).contiguous()
    qrep = query_z[:, :, None].expand(-1, -1, ns, -1, -1, -1).reshape(
        bsz * qn * ns, *query_z.shape[2:]
    ).contiguous().to(flow.device)

    chunks = []
    pcs = max(1, int(args.pair_chunk_size))
    for st in range(0, qrep.shape[0], pcs):
        ed = min(st + pcs, qrep.shape[0])
        chunks.append(flow.generate_post_states(
            qrep[st:ed],
            target_vision=target[st:ed],
            hint=hint[st:ed],
            num_steps=int(args.num_steps),
            solver=args.solver,
        ).detach().cpu())
    z = torch.cat(chunks, dim=0)
    return z.reshape(bsz, qn, n_way, n_shot, args.num_steps, *z.shape[-3:])


def compute_batch_geometry(compressor: nn.Module, support_z: torch.Tensor, query_z: torch.Tensor,
                           post_states_cpu: torch.Tensor, args, comp_device: torch.device):
    """Batched differentiable Compressor geometry with one Compressor forward.

    support_z: [B,NS,C,H,W], query_z: [B,Q,C,H,W],
    post_states_cpu: [B,Q,N,K,T,C,H,W].
    """
    bsz, ns = support_z.shape[:2]
    qn = query_z.shape[1]
    n_way, n_shot, t = post_states_cpu.shape[2:5]

    support_dev = support_z.reshape(bsz * ns, *support_z.shape[2:]).to(comp_device, non_blocking=True)
    query_dev = query_z.reshape(bsz * qn, *query_z.shape[2:]).to(comp_device, non_blocking=True)
    post_flat = post_states_cpu.reshape(bsz * qn * ns * t, *post_states_cpu.shape[-3:]).to(comp_device, non_blocking=True)

    n_support = support_dev.shape[0]
    n_query = query_dev.shape[0]
    all_latent = torch.cat([support_dev, query_dev, post_flat], dim=0)
    all_raw = _compress_raw(compressor, all_latent, args.ot_pool_factor)

    support_raw = all_raw[:n_support].reshape(bsz, ns, *all_raw.shape[1:])
    query_raw = all_raw[n_support:n_support+n_query].reshape(bsz, qn, *all_raw.shape[1:])
    post_raw = all_raw[n_support+n_query:].reshape(bsz, qn, ns, t, *all_raw.shape[1:])

    qraw_direct = query_raw[:, :, None].expand(-1, -1, ns, -1, -1).reshape(bsz * qn * ns, *query_raw.shape[2:])
    sraw_direct = support_raw[:, None].expand(-1, qn, -1, -1, -1).reshape(bsz * qn * ns, *support_raw.shape[2:])
    direct = _aligned_distance(qraw_direct, sraw_direct, args)

    post_pair = post_raw.reshape(bsz * qn * ns, t, *post_raw.shape[-2:])
    q2step = torch.stack([_aligned_distance(qraw_direct, post_pair[:, j], args) for j in range(t)], dim=1)

    q_repr_img = query_raw.mean(dim=-2)       # [B,Q,D]
    s_repr_img = support_raw.mean(dim=-2)     # [B,NS,D]
    step_repr = post_pair.mean(dim=-2)        # [B*Q*NS,T,D]
    dim = q_repr_img.shape[-1]
    q_repr_pair = q_repr_img[:, :, None].expand(-1, -1, ns, -1).reshape(bsz * qn * ns, dim)
    s_repr_pair = s_repr_img[:, None].expand(-1, qn, -1, -1).reshape(bsz * qn * ns, dim)

    return {
        "direct_pair": direct,
        "q2step": q2step,
        "query_repr_pair": q_repr_pair,
        "support_repr_pair": s_repr_pair,
        "step_repr": step_repr,
        "query_cloud": q_repr_img,
        "support_image_repr": s_repr_img,
    }


def run_episode_batch(episodes: Sequence[Episode], *, compressor, policy, old_policy, visreg, rae, flow,
                      transform, args, devices, shot_tau, cls_tau, training: bool, epoch: int,
                      progress=None):
    """Run a true per-rank batch of same-width episodes in one model step."""
    if not episodes:
        raise ValueError("empty episode batch")
    comp_device, flow_device, rae_device = devices
    bsz = len(episodes)
    n_way = len(episodes[0].classes)
    n_shot = int(args.n_shot)
    qn = len(episodes[0].query_paths)
    ns = n_way * n_shot
    if any(len(ep.classes) != n_way or len(ep.query_paths) != qn for ep in episodes):
        raise ValueError("batched episodes must have identical N-way and query count")

    if progress is not None:
        progress.set_postfix_str("stage=load-images")
    support_imgs = torch.stack([load_images(ep.support_paths, transform) for ep in episodes], dim=0)
    query_imgs = torch.stack([load_images(ep.query_paths, transform) for ep in episodes], dim=0)
    targets = torch.cat([ep.query_labels for ep in episodes], dim=0).to(comp_device)

    if progress is not None:
        progress.set_postfix_str("stage=RAE")
    # One RAE stream for the whole episode batch.
    all_img = torch.cat([
        support_imgs.reshape(bsz * ns, *support_imgs.shape[2:]),
        query_imgs.reshape(bsz * qn, *query_imgs.shape[2:]),
    ], dim=0).to(rae_device, non_blocking=True)
    with torch.no_grad():
        latent = base.encode_rae_chunked(rae, all_img, int(args.rae_chunk_size))
    support_z = latent[:bsz*ns].reshape(bsz, ns, *latent.shape[1:]).detach()
    query_z = latent[bsz*ns:].reshape(bsz, qn, *latent.shape[1:]).detach()

    if progress is not None:
        progress.set_postfix_str("stage=Flow")
    # One merged Query--Support pair stream for Flow.
    post = generate_post_states_batch(flow, query_z, support_imgs, args, n_way, n_shot)
    if progress is not None:
        progress.set_postfix_str("stage=Compressor+geometry")
    geo = compute_batch_geometry(compressor, support_z, query_z, post, args, comp_device)
    direct_pair = geo["direct_pair"]
    steps = geo["q2step"]
    pair_count, t = steps.shape
    total_q = bsz * qn

    direct_logits = class_logits_from_pair_scores(
        direct_pair.reshape(total_q, n_way, n_shot), shot_tau, cls_tau
    )
    direct_ce = F.cross_entropy(direct_logits, targets)

    candidate_mask = strict_record_mask(steps)
    stats = build_policy_statistics(
        steps, direct_pair, n_way, n_shot, shot_tau, cls_tau,
        eps=float(args.feature_eps), ratio_clip=float(args.feature_ratio_clip),
    )

    q_joint = geo["query_repr_pair"]
    s_joint = geo["support_repr_pair"]
    z_joint = geo["step_repr"]
    pair_n = q_joint.shape[0]
    if progress is not None:
        progress.set_postfix_str("stage=LSTM-policy")
    policy_logits_all = policy(
        torch.cat([q_joint, q_joint.detach()], dim=0),
        torch.cat([s_joint, s_joint.detach()], dim=0),
        torch.cat([z_joint, z_joint.detach()], dim=0),
        torch.cat([stats, stats.detach()], dim=0),
    )
    retain_logits = policy_logits_all[:pair_n]
    pol_det = policy_logits_all[pair_n:]

    retain_prob = torch.sigmoid(retain_logits) * candidate_mask.float()
    soft_pair = soft_q2mono_pair_score(direct_pair, steps, retain_prob, candidate_mask)
    soft_logits = class_logits_from_pair_scores(
        soft_pair.reshape(total_q, n_way, n_shot), shot_tau, cls_tau
    )
    joint_ce = F.cross_entropy(soft_logits, targets)

    # VISReg population keeps the natural episode axis: [B,N,D] + [B,Q,D].
    support_cloud = geo["support_image_repr"].reshape(bsz, n_way, n_shot, -1).mean(dim=2)
    query_cloud = geo["query_cloud"]
    vis_in_local = cloud_features_to_visreg_input(support_cloud, query_cloud)
    rank = int(getattr(args, "rank", 0))
    world_size = int(getattr(args, "world_size", 1))
    vis_in = ddp_global_visreg_input(vis_in_local.float(), world_size, rank)
    vis_loss_raw = visreg(vis_in)
    vis_loss = vis_loss_raw * float(world_size) if training and world_size > 1 else vis_loss_raw

    zero = retain_logits.sum() * 0.0
    grpo_loss = zero
    safe_loss = zero
    kl_loss = zero
    grpo_info = {"reward": 0.0, "reward_std": 0.0, "adv_abs": 0.0, "ratio": 1.0, "clip_frac": 0.0}

    with torch.no_grad():
        old_det = old_policy(
            geo["query_repr_pair"].detach(), geo["support_repr_pair"].detach(),
            geo["step_repr"].detach(), stats.detach()
        )
    if training and epoch >= int(args.grpo_warmup_epochs) and float(args.lambda_grpo) > 0:
        grpo_loss, grpo_info = group_relative_ppo(
            pol_det, old_det, candidate_mask, direct_pair.detach(), steps.detach(), targets,
            n_way, n_shot, shot_tau.detach(), cls_tau.detach(), args
        )

    det_prob = torch.sigmoid(pol_det) * candidate_mask.float()
    safe_pair = soft_q2mono_pair_score(direct_pair.detach(), steps.detach(), det_prob, candidate_mask)
    safe_logits = class_logits_from_pair_scores(
        safe_pair.reshape(total_q, n_way, n_shot), shot_tau.detach(), cls_tau.detach()
    )
    safe_ce_q = F.cross_entropy(safe_logits, targets, reduction="none")
    direct_ce_q_det = F.cross_entropy(direct_logits.detach(), targets, reduction="none")
    direct_correct = direct_logits.detach().argmax(-1).eq(targets)
    if direct_correct.any():
        safe_loss = F.relu(
            safe_ce_q[direct_correct] - direct_ce_q_det[direct_correct] - float(args.safe_margin)
        ).mean()

    kl_loss = bernoulli_kl(old_det, pol_det, candidate_mask)
    total = (
        float(args.lambda_direct_ce) * direct_ce
        + float(args.lambda_visreg) * vis_loss
        + float(args.lambda_joint_ce) * joint_ce
        + float(args.lambda_grpo) * grpo_loss
        + float(args.lambda_safe) * safe_loss
        + float(args.lambda_kl) * kl_loss
    )

    with torch.no_grad():
        pred_direct = direct_logits.argmax(-1)
        pred_soft = soft_logits.argmax(-1)
        metrics = {
            "loss": float(total.item()),
            "direct_ce": float(direct_ce.item()),
            "joint_ce": float(joint_ce.item()),
            "visreg": float(vis_loss_raw.item()),
            "grpo": float(grpo_loss.item()),
            "safe": float(safe_loss.item()),
            "kl": float(kl_loss.item()),
            "direct_correct": int(pred_direct.eq(targets).sum().item()),
            "soft_correct": int(pred_soft.eq(targets).sum().item()),
            "queries": int(targets.numel()),
            "candidate_fraction": float(candidate_mask.float().mean().item()),
            "retain_prob": float(retain_prob.sum().item() / candidate_mask.float().sum().clamp_min(1.0).item()),
            **grpo_info,
        }
    return total, metrics


def run_episode(ep: Episode, *, compressor, policy, old_policy, visreg, rae, flow,
                transform, args, devices, shot_tau, cls_tau, training: bool, epoch: int):
    comp_device, flow_device, rae_device = devices
    support_img = load_images(ep.support_paths, transform)
    query_img = load_images(ep.query_paths, transform)
    n_way = len(ep.classes)
    n_shot = int(args.n_shot)
    qn = len(ep.query_paths)
    target = ep.query_labels.to(comp_device)

    all_img_rae = torch.cat([support_img, query_img], 0).to(rae_device, non_blocking=True)
    with torch.no_grad():
        latent = base.encode_rae_chunked(rae, all_img_rae, int(args.rae_chunk_size))
    ns = n_way*n_shot
    support_z = latent[:ns].detach()
    query_z = latent[ns:].detach()

    # Frozen Flow; all learning happens downstream in Compressor/policy.
    post = generate_post_states(flow, query_z, support_img, args, n_way, n_shot)
    geo = compute_episode_geometry(compressor, support_z, query_z, post, args, comp_device)
    direct_pair = geo["direct_pair"]
    steps = geo["q2step"]
    pair_count, t = steps.shape

    direct_pair_4 = direct_pair.reshape(qn, n_way, n_shot)
    direct_logits = class_logits_from_pair_scores(direct_pair_4, shot_tau, cls_tau)
    direct_ce = F.cross_entropy(direct_logits, target)

    candidate_mask = strict_record_mask(steps)
    stats = build_policy_statistics(
        steps, direct_pair, n_way, n_shot, shot_tau, cls_tau,
        eps=float(args.feature_eps), ratio_clip=float(args.feature_ratio_clip),
    )

    # One policy forward per episode for DDP correctness. The first half keeps the
    # Compressor graph for joint CE; the second half uses detached observables so
    # PPO/safe/KL update only the policy.
    q_joint = geo["query_repr_pair"]
    s_joint = geo["support_repr_pair"]
    z_joint = geo["step_repr"]
    stat_joint = stats
    pair_n = q_joint.shape[0]
    policy_logits_all = policy(
        torch.cat([q_joint, q_joint.detach()], dim=0),
        torch.cat([s_joint, s_joint.detach()], dim=0),
        torch.cat([z_joint, z_joint.detach()], dim=0),
        torch.cat([stat_joint, stat_joint.detach()], dim=0),
    )
    retain_logits = policy_logits_all[:pair_n]
    pol_det = policy_logits_all[pair_n:]

    retain_prob = torch.sigmoid(retain_logits) * candidate_mask.float()
    soft_pair = soft_q2mono_pair_score(direct_pair, steps, retain_prob, candidate_mask)
    soft_logits = class_logits_from_pair_scores(soft_pair.reshape(qn, n_way, n_shot), shot_tau, cls_tau)
    joint_ce = F.cross_entropy(soft_logits, target)

    # VISReg exact supplied semantics: each support class cloud and each query cloud
    # is one pre-L2 cloud embedding sample.  K support images are averaged within class.
    support_cloud = geo["support_image_repr"].reshape(n_way, n_shot, -1).mean(dim=1)
    query_cloud = geo["query_cloud"]
    vis_in_local = cloud_features_to_visreg_input(support_cloud, query_cloud)
    rank = int(getattr(args, "rank", 0))
    world_size = int(getattr(args, "world_size", 1))
    vis_in = ddp_global_visreg_input(vis_in_local.float(), world_size, rank)
    # All ranks evaluate the same global population. Remote embeddings are detached
    # on each rank, therefore compensate DDP's gradient averaging by world_size.
    vis_loss_raw = visreg(vis_in)
    vis_loss = vis_loss_raw * float(world_size) if training and world_size > 1 else vis_loss_raw

    zero = retain_logits.sum() * 0.0
    grpo_loss = zero
    safe_loss = zero
    kl_loss = zero
    grpo_info = {"reward": 0.0, "reward_std": 0.0, "adv_abs": 0.0, "ratio": 1.0, "clip_frac": 0.0}

    # EMA/reference policy receives the same detached Compressor-derived observables.
    with torch.no_grad():
        old_det = old_policy(
            geo["query_repr_pair"].detach(), geo["support_repr_pair"].detach(),
            geo["step_repr"].detach(), stats.detach()
        )

    if training and epoch >= int(args.grpo_warmup_epochs) and float(args.lambda_grpo) > 0:
        grpo_loss, grpo_info = group_relative_ppo(
            pol_det, old_det, candidate_mask, direct_pair.detach(), steps.detach(), target,
            n_way, n_shot, shot_tau.detach(), cls_tau.detach(), args
        )

    # Direct-safe policy-only hinge: if Direct is correct, soft retained policy should
    # not have larger per-query CE than Direct by more than safe_margin.
    det_prob = torch.sigmoid(pol_det) * candidate_mask.float()
    safe_pair = soft_q2mono_pair_score(direct_pair.detach(), steps.detach(), det_prob, candidate_mask)
    safe_logits = class_logits_from_pair_scores(safe_pair.reshape(qn, n_way, n_shot), shot_tau.detach(), cls_tau.detach())
    safe_ce_q = F.cross_entropy(safe_logits, target, reduction="none")
    direct_ce_q_det = F.cross_entropy(direct_logits.detach(), target, reduction="none")
    direct_correct = direct_logits.detach().argmax(-1).eq(target)
    if direct_correct.any():
        safe_loss = F.relu(safe_ce_q[direct_correct] - direct_ce_q_det[direct_correct] - float(args.safe_margin)).mean()

    kl_loss = bernoulli_kl(old_det, pol_det, candidate_mask)

    total = (
        float(args.lambda_direct_ce) * direct_ce
        + float(args.lambda_visreg) * vis_loss
        + float(args.lambda_joint_ce) * joint_ce
        + float(args.lambda_grpo) * grpo_loss
        + float(args.lambda_safe) * safe_loss
        + float(args.lambda_kl) * kl_loss
    )

    with torch.no_grad():
        pred_direct = direct_logits.argmax(-1)
        pred_soft = soft_logits.argmax(-1)
        metrics = {
            "loss": float(total.item()),
            "direct_ce": float(direct_ce.item()),
            "joint_ce": float(joint_ce.item()),
            "visreg": float(vis_loss_raw.item()),
            "grpo": float(grpo_loss.item()),
            "safe": float(safe_loss.item()),
            "kl": float(kl_loss.item()),
            "direct_correct": int(pred_direct.eq(target).sum().item()),
            "soft_correct": int(pred_soft.eq(target).sum().item()),
            "queries": int(target.numel()),
            "candidate_fraction": float(candidate_mask.float().mean().item()),
            "retain_prob": float(retain_prob.sum().item() / candidate_mask.float().sum().clamp_min(1.0).item()),
            **grpo_info,
        }
    return total, metrics


def aggregate_metrics(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    q = max(1.0, sum(x["queries"] for x in items))
    out: Dict[str, float] = {}
    per_query_keys = ["loss", "direct_ce", "joint_ce", "visreg", "grpo", "safe", "kl",
                      "candidate_fraction", "retain_prob", "reward", "reward_std", "adv_abs", "ratio", "clip_frac"]
    for k in per_query_keys:
        out[k] = sum(x[k] * x["queries"] for x in items) / q
    out["direct_acc"] = sum(x["direct_correct"] for x in items) / q
    out["soft_acc"] = sum(x["soft_correct"] for x in items) / q
    out["queries"] = q
    return out


def reduce_aggregated_metrics(local: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not local or not (dist.is_available() and dist.is_initialized()):
        return local
    keys = ["loss", "direct_ce", "joint_ce", "visreg", "grpo", "safe", "kl",
            "candidate_fraction", "retain_prob", "reward", "reward_std", "adv_abs", "ratio", "clip_frac"]
    q = float(local["queries"])
    vals = [local[k] * q for k in keys]
    vals += [local["direct_acc"] * q, local["soft_acc"] * q, q]
    t = torch.tensor(vals, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    total_q = max(float(t[-1].item()), 1.0)
    out = {k: float(t[i].item()) / total_q for i, k in enumerate(keys)}
    off = len(keys)
    out["direct_acc"] = float(t[off].item()) / total_q
    out["soft_acc"] = float(t[off+1].item()) / total_q
    out["queries"] = total_q
    return out


def run_epoch(pool, *, compressor, policy, old_policy, visreg, rae, flow, transform,
              args, devices, shot_tau, cls_tau, optimizer, epoch: int, training: bool) -> Dict[str, float]:
    compressor.train(training)
    policy.train(training)
    old_policy.eval()
    requested_global = int(args.episodes_per_epoch if training else args.val_episodes)
    world_size = int(getattr(args, "world_size", 1))
    rank = int(getattr(args, "rank", 0))
    local_episodes = requested_global // world_size
    if local_episodes < 1:
        raise ValueError(f"requested episodes={requested_global} is smaller than world_size={world_size}")
    used_global = local_episodes * world_size
    if is_main_process() and used_global != requested_global:
        print(f"[DDP WARN] requested {requested_global} episodes; using {used_global} "
              f"({local_episodes}/rank) so every rank performs the same number of backward steps.", flush=True)

    batch_size = max(1, int(args.episode_batch_size))
    n_batches = math.ceil(local_episodes / batch_size)
    rng = random.Random(int(args.seed) + epoch * 100003 + (0 if training else 99991) + rank * 10000019)
    records: List[Dict[str, float]] = []
    mode = "train" if training else "val"
    progress = None
    if is_main_process() and bool(args.show_progress):
        progress = tqdm(
            total=n_batches,
            desc=f"{mode} {epoch+1}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=True,
        )
        progress.set_postfix_str(
            f"ep/rank={local_episodes} bs={batch_size} global_bs<={batch_size*world_size}"
        )

    done = 0
    for bi in range(n_batches):
        current_bs = min(batch_size, local_episodes - done)
        episodes = [sample_episode(pool, args.n_way, args.n_shot, args.n_query, rng) for _ in range(current_bs)]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=bool(args.amp)):
            loss, m = run_episode_batch(
                episodes, compressor=compressor, policy=policy, old_policy=old_policy,
                visreg=visreg, rae=rae, flow=flow, transform=transform, args=args,
                devices=devices, shot_tau=shot_tau, cls_tau=cls_tau,
                training=training, epoch=epoch, progress=progress,
            )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(unwrap_model(compressor).parameters()) + list(unwrap_model(policy).parameters()),
                float(args.grad_clip),
            )
            optimizer.step()
            ema_update(old_policy, policy, float(args.ema_decay))
        records.append(m)
        done += current_bs

        should_log = ((bi + 1) % max(1, int(args.log_every)) == 0) or (bi + 1 == n_batches)
        if should_log:
            recent_n = min(max(1, int(args.log_every)), len(records))
            a = aggregate_metrics(records[-recent_n:])
            a = reduce_aggregated_metrics(a, devices[0])
            if is_main_process():
                status = (
                    f"loss={a['loss']:.3f} D={100*a['direct_acc']:.1f}% "
                    f"Q2M={100*a['soft_acc']:.1f}% vis={a['visreg']:.3f} "
                    f"grpo={a['grpo']:.3f} keep={a['retain_prob']:.2f}"
                )
                if progress is not None:
                    progress.set_postfix_str(status)
                else:
                    print(f"[{mode}] ep={epoch+1} batch={bi+1}/{n_batches} episodes={done}/{local_episodes} {status}", flush=True)
        if progress is not None:
            progress.update(1)

    if progress is not None:
        progress.close()
    return reduce_aggregated_metrics(aggregate_metrics(records), devices[0])


def save_checkpoint(path: str, *, compressor, policy, old_policy, optimizer, scheduler,
                    shot_tau, cls_tau, epoch: int, args, val_metrics: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "stage": STAGE,
        "epoch": int(epoch),
        "compressor": unwrap_model(compressor).state_dict(),
        # Compatibility alias for existing evaluators/loaders.
        "metric_encoder": unwrap_model(compressor).state_dict(),
        "q2mono_retention_policy": unwrap_model(policy).state_dict(),
        "ema_q2mono_retention_policy": old_policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "shot_tau": shot_tau.detach().cpu(),
        "cls_tau": cls_tau.detach().cpu(),
        "feature_names": list(FEATURE_NAMES),
        "args": vars(args),
        "val": val_metrics,
        "dit_checkpoint": args.dit_checkpoint,
        "m2_init_checkpoint": args.m2_checkpoint,
        "method": {
            "q2mono_candidate_rule": "strict running-record Query-to-Step distances",
            "pair_score": "Direct Query-to-Support + policy-retained Q2Mono Query-to-Step mean",
            "compressor": "existing MetricEncoder architecture; trainable, structure unchanged",
            "compressor_objectives": ["Direct CE", "VISReg", "soft-Q2Mono joint CE"],
            "policy_inputs": ["Query/Support/Step Compressor relations", "trajectory statistics", "classification statistics", "K-shot statistics"],
            "policy_objectives": ["soft-Q2Mono joint CE", "group-relative PPO/GRPO", "Direct-safe hinge", "EMA-policy KL"],
            "rl_gradient_to_compressor": False,
            "rae": "frozen",
            "flow": "frozen",
        },
    }, path)


def resume(path, compressor, policy, old_policy, optimizer, scheduler) -> int:
    ckpt = base._torch_load(path, "cpu")
    if ckpt.get("stage") != STAGE:
        raise ValueError(f"checkpoint stage mismatch: {ckpt.get('stage')} != {STAGE}")
    compressor.load_state_dict(ckpt.get("compressor", ckpt["metric_encoder"]), strict=True)
    policy.load_state_dict(ckpt["q2mono_retention_policy"], strict=True)
    old_policy.load_state_dict(ckpt["ema_q2mono_retention_policy"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["epoch"]) + 1


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed(args)
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank
    seed_everything(int(args.seed) + rank)
    comp_device, flow_device, rae_device = resolve_devices(args, local_rank)
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    barrier()

    if args.n_way < 2 or args.n_shot < 1 or args.n_query < 1:
        raise ValueError("invalid N-way/K-shot/Q-query")
    if args.grpo_group_size < 2:
        raise ValueError("grpo_group_size must be >=2")
    if args.episode_batch_size < 1:
        raise ValueError("episode_batch_size must be >=1")
    if not 0.0 <= args.policy_dropout < 1.0:
        raise ValueError("policy_dropout must be in [0,1)")

    # Initial Compressor checkpoint defines the existing Compressor/OT protocol.
    # Resolve it BEFORE the potentially expensive JSON-root scan.
    m2_ckpt = base._torch_load(args.m2_checkpoint, "cpu")
    if not isinstance(m2_ckpt, dict):
        raise ValueError("m2_checkpoint must contain a dict")
    resolve_pretrain_from_m2(args, m2_ckpt)

    # Data.
    if is_main_process():
        print(f"[data] scanning train JSON root: {args.train_jsons_root}", flush=True)
    full_train = load_class_json_root(
        args.train_jsons_root, show_progress=is_main_process() and bool(args.show_progress), desc="train JSONs"
    )
    if args.val_jsons_root:
        train_pool = full_train
        if is_main_process():
            print(f"[data] scanning val JSON root: {args.val_jsons_root}", flush=True)
        val_pool = load_class_json_root(
            args.val_jsons_root, show_progress=is_main_process() and bool(args.show_progress), desc="val JSONs"
        )
    else:
        train_pool, val_pool = split_pool(full_train, float(args.val_fraction), int(args.seed))
    if is_main_process():
        print(f"[data] train classes={len(train_pool)} val classes={len(val_pool)}")

    train_transform = base.build_transform(args.image_size, augment=args.augment)
    val_transform = base.build_transform(args.image_size, augment=False)

    # Frozen RAE.
    torch.cuda.set_device(rae_device)
    rae = base.build_rae(args.project_root, rae_device)
    rae.eval()
    for p in rae.parameters():
        p.requires_grad_(False)

    # Frozen Flow.
    torch.cuda.set_device(flow_device)
    flow = base.FrozenFlowRuntime(
        project_root=args.project_root,
        checkpoint=args.dit_checkpoint,
        device=flow_device,
        dit_variant=args.dit_variant,
        num_classes=args.num_classes,
        image_size=args.image_size,
        use_ema=args.use_ema,
        hint_signal_ratio=args.hint_signal_ratio,
        allow_condition_mismatch=args.allow_condition_mismatch,
        amp=args.amp,
    )
    args.hint_signal_ratio = flow.hint_signal_ratio

    # Trainable Compressor: SAME MetricEncoder/Compressor architecture and checkpoint state.
    torch.cuda.set_device(comp_device)
    compressor = MetricEncoder(
        in_dim=1024, hidden_dim=args.hidden_dim, out_dim=args.metric_dim, dropout=args.dropout
    ).to(comp_device)
    shot_tau, cls_tau = base.load_m2_metric_and_temperatures(compressor, m2_ckpt, comp_device)
    if float(args.shot_tau_override) > 0:
        shot_tau = torch.tensor(float(args.shot_tau_override), device=comp_device)
    del m2_ckpt
    compressor.train()
    for p in compressor.parameters():
        p.requires_grad_(True)

    policy = Q2MonoRetentionPolicy(
        repr_dim=args.metric_dim,
        stat_dim=len(FEATURE_NAMES),
        hidden_dim=args.policy_hidden_dim,
        relation_dim=args.policy_relation_dim,
        condition_dim=args.policy_condition_dim,
        dropout=args.policy_dropout,
        retain_init_prob=args.retain_init_prob,
    ).to(comp_device)
    old_policy = copy.deepcopy(policy).eval().requires_grad_(False)

    visreg = VISReg(
        num_projections=args.visreg_num_projections,
        scale_weight=args.visreg_scale_weight,
        shape_weight=args.visreg_shape_weight,
        center_weight=args.visreg_center_weight,
        projection_chunk_size=args.visreg_projection_chunk_size,
    ).to(comp_device)

    optimizer = torch.optim.AdamW([
        {"params": compressor.parameters(), "lr": float(args.compressor_lr)},
        {"params": policy.parameters(), "lr": float(args.policy_lr)},
    ], weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(args.epochs)), eta_min=min(args.compressor_lr, args.policy_lr) * 0.05
    )

    start_epoch = 0
    if args.resume_checkpoint:
        start_epoch = resume(args.resume_checkpoint, compressor, policy, old_policy, optimizer, scheduler)

    # Wrap trainable modules only after checkpoint restoration. Frozen RAE/Flow and
    # EMA reference policy remain rank-local replicas.
    if world_size > 1:
        compressor = DDP(compressor, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        policy = DDP(policy, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    if is_main_process():
        print("="*100)
        print("[Q2Mono-GRPO v1 DDP PRETRAIN]")
        print(f"world size / GPUs      : {world_size}")
        print(f"episode batch / rank   : {args.episode_batch_size}")
        print(f"max global ep batch    : {args.episode_batch_size * world_size}")
        print(f"Compressor init        : {args.m2_checkpoint}")
        print("Compressor architecture: unchanged existing MetricEncoder/Compressor")
        print("RAE / Flow             : frozen / frozen; full replica per rank/local GPU")
        print("Compressor objectives  : Direct CE + global-population VISReg + joint soft-Q2Mono CE")
        print("Policy objectives      : joint soft-Q2Mono CE + group-relative PPO + safe + EMA KL")
        print(f"Q2Mono features        : {FEATURE_NAMES}")
        print(f"train/val classes      : {len(train_pool)} / {len(val_pool)}")
        print("="*100, flush=True)

    best = -1.0
    for epoch in range(start_epoch, int(args.epochs)):
        train_m = run_epoch(
            train_pool, compressor=compressor, policy=policy, old_policy=old_policy,
            visreg=visreg, rae=rae, flow=flow, transform=train_transform,
            args=args, devices=(comp_device, flow_device, rae_device), shot_tau=shot_tau,
            cls_tau=cls_tau, optimizer=optimizer, epoch=epoch, training=True,
        )
        with torch.no_grad():
            val_m = run_epoch(
                val_pool, compressor=compressor, policy=policy, old_policy=old_policy,
                visreg=visreg, rae=rae, flow=flow, transform=val_transform,
                args=args, devices=(comp_device, flow_device, rae_device), shot_tau=shot_tau,
                cls_tau=cls_tau, optimizer=None, epoch=epoch, training=False,
            )
        scheduler.step()

        if is_main_process():
            print(
                f"[epoch {epoch+1}/{args.epochs}] "
                f"train direct={100*train_m['direct_acc']:.2f}% softQ2M={100*train_m['soft_acc']:.2f}% | "
                f"val direct={100*val_m['direct_acc']:.2f}% softQ2M={100*val_m['soft_acc']:.2f}% "
                f"loss={val_m['loss']:.4f}", flush=True,
            )

            last_path = os.path.join(args.output_dir, "q2mono_grpo_last.pt")
            save_checkpoint(last_path, compressor=compressor, policy=policy, old_policy=old_policy,
                            optimizer=optimizer, scheduler=scheduler, shot_tau=shot_tau, cls_tau=cls_tau,
                            epoch=epoch, args=args, val_metrics=val_m)
            if val_m["soft_acc"] > best:
                best = val_m["soft_acc"]
                best_path = os.path.join(args.output_dir, "q2mono_grpo_best.pt")
                save_checkpoint(best_path, compressor=compressor, policy=policy, old_policy=old_policy,
                                optimizer=optimizer, scheduler=scheduler, shot_tau=shot_tau, cls_tau=cls_tau,
                                epoch=epoch, args=args, val_metrics=val_m)
                print(f"[best] soft-Q2Mono val acc={100*best:.2f}% -> {best_path}")
        barrier()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
