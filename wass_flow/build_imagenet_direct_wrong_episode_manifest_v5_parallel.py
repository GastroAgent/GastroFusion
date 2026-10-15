#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-mine ONLY Direct-misclassified fixed ImageNet episodes into a manifest.

Definition of a hard episode
----------------------------
A candidate N-way K-shot episode is accepted IFF frozen M2 Direct predicts a
wrong class.  No "hard-correct" or "easy-correct" examples are collected.

Direct class distance
---------------------
For query q, class c and its K concrete support images, first compute the exact
M2 Dynamic-SK + Sinkhorn query-to-support distances d_{c,k}.  Then aggregate
K-shot supports exactly as the v10c Direct branch does:

    D_c = -tau_shot * [ logsumexp_k(-d_{c,k}/tau_shot) - log(K) ]

Lower is better.  Let y be GT and

    margin_dist = min_{c != y} D_c - D_y.

An episode is saved only when margin_dist < 0, equivalently Direct argmin(D_c)
is not y.

Mining strategy
---------------
For each query (multiple queries are sampled/encoded together in one attempt batch):
  1) randomly screen candidate_way classes including GT;
  2) for each class sample support_repeats concrete K-shot realizations;
  3) GT keeps the realization with the LARGEST class distance (hard GT support);
  4) every wrong class keeps the realization with the SMALLEST class distance
     (most confusing support realization);
  5) take the N-1 wrong classes with smallest selected distances;
  6) recompute/check the final N-way Direct decision and save only if wrong.

The main manifest contains only episode data.  A sidecar stats JSON records the
actual Direct distances and margins so every collected episode is auditable.

This script intentionally does NOT invoke Flow/DiT.

Parallel batching notes
-----------------------
``attempt_batch_size`` independent candidate episodes are prepared together.

The expensive Direct computation is truly vectorized on ONE GPU:

    query tokens   : [B, 1, P, D]
    support tokens : [B, C*R*K, P, D]
        -> batched Dynamic-SK
        -> batched Sinkhorn
        -> distances [B, C, R, K]

The leading B dimension keeps every episode isolated; there is no cross-episode
query/support matching.  Final N-way distances are gathered from the already exact
screening pair distances, so the old redundant second Sinkhorn pass is removed.

Image loading is also parallelized on CPU with a ThreadPoolExecutor, while RAE and
MetricEncoder remain chunked GPU batches for memory safety.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

# import debugpy
# try:
#     print("Waiting for debugger attach")
#     # 5678 is the default attach port in the VS Code debug configurations.
#     debugpy.listen(("localhost", 9505))
#     # debugpy.listen(("172.16.0.108", 9504))
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Build fixed Direct-WRONG ImageNet episode manifest with explicit M2 OT distances"
    )

    # Existing v10c is used as the single source of truth for RAE/M2/OT helpers.
    p.add_argument("--v10c_script", required=True,
                   help="Path to pretrain_f1_flow_trajectory_pairwise_corrector_v10c_*.py")
    p.add_argument("--train_class_json_root", required=True)
    p.add_argument("--project_root", required=True)
    p.add_argument("--m2_checkpoint", required=True)

    p.add_argument("--output_manifest", required=True)
    p.add_argument("--output_stats", default=None)

    p.add_argument("--n_way", type=int, default=12)
    p.add_argument("--n_shot", type=int, default=5)
    p.add_argument("--num_hard_episodes", type=int, default=1000)

    # Mining strength.
    p.add_argument("--candidate_way", type=int, default=32,
                   help="Number of classes screened before selecting final N-way")
    p.add_argument("--support_repeats", type=int, default=4,
                   help="Concrete K-shot realizations screened per candidate class")
    p.add_argument("--max_total_attempts", type=int, default=0,
                   help="0 => max(1000, 100 * num_hard_episodes)")
    p.add_argument("--attempt_batch_size", type=int, default=8,
                   help="Independent candidate episodes vectorized together on one GPU")
    p.add_argument("--ot_attempt_chunk_size", type=int, default=0,
                   help="0 = all attempt_batch_size episodes in one batched OT call; "
                        "set e.g. 4 only as an OOM fallback")

    # Exact Direct/M2 settings.  None values are resolved from the M2 checkpoint.
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--rae_chunk_size", type=int, default=32,
                   help="Images per frozen RAE forward")
    p.add_argument("--metric_chunk_size", type=int, default=64,
                   help="Latents per frozen MetricEncoder forward; <=0 uses the whole RAE chunk")
    p.add_argument("--image_workers", type=int, default=8,
                   help="CPU threads for parallel PIL image loading/transform")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--metric_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--ot_pool_factor", type=int, default=None)
    p.add_argument("--sinkhorn_reg", type=float, default=None)
    p.add_argument("--sinkhorn_iter", type=int, default=None)
    p.add_argument("--sk_weight_eps", type=float, default=None)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=20,
                   help="Periodically flush accepted records to disk")
    return p.parse_args()


def load_python_module(path: str, module_name: str = "f1_v10c_ref"):
    path_obj = Path(path).resolve()
    if not path_obj.is_file():
        raise FileNotFoundError(path_obj)
    # v10c imports project-local packages relative to its parent tree.
    for d in (path_obj.parent, path_obj.parent.parent):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(module_name, str(path_obj))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import v10c script: {path_obj}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_rgb_tensor(path: str, transform) -> torch.Tensor:
    with Image.open(path) as im:
        return transform(im.convert("RGB"))


def sample_kshot(
    paths: list[str],
    k: int,
    rng: random.Random,
    *,
    exclude: str | None = None,
) -> list[str]:
    cand = [str(p) for p in paths if str(p) != str(exclude)]
    if len(cand) < k:
        raise RuntimeError(
            f"not enough support images after query exclusion: {len(cand)} < {k}"
        )
    return rng.sample(cand, k)



@torch.no_grad()
def batched_pairwise_shot_distances(
    *,
    v10: Any,
    support_raw: torch.Tensor,
    support_norm: torch.Tensor,
    query_raw: torch.Tensor,
    query_norm: torch.Tensor,
    sk_weight_eps: float,
    sinkhorn_reg: float,
    sinkhorn_iter: int,
    attempt_chunk_size: int = 0,
) -> torch.Tensor:
    """Fully vectorized Dynamic-SK + Sinkhorn over independent episodes.

    Parameters
    ----------
    support_raw/support_norm:
        [B, S, P, D]
    query_raw/query_norm:
        [B, Q, P, D], here Q=1

    Returns
    -------
    distance:
        [B, Q, S]

    Important
    ---------
    The batch axis B is a leading broadcast dimension supported natively by
    ``compute_pairwise_dynamic_sk_weights`` and ``pairwise_sinkhorn_from_weights``.
    Therefore episode b can only compare with supports from episode b; no Cartesian
    mixing across B is created.
    """
    if support_raw.ndim != 4 or query_raw.ndim != 4:
        raise ValueError(
            "batched OT expects support=[B,S,P,D], query=[B,Q,P,D], "
            f"got support={tuple(support_raw.shape)}, query={tuple(query_raw.shape)}"
        )
    if support_raw.shape[0] != query_raw.shape[0]:
        raise ValueError("support/query batch size mismatch")
    if support_norm.shape != support_raw.shape or query_norm.shape != query_raw.shape:
        raise ValueError("raw/norm shape mismatch")

    bsz = int(query_raw.shape[0])
    chunk = int(attempt_chunk_size)
    if chunk <= 0:
        chunk = bsz

    dist_parts: list[torch.Tensor] = []
    for b0 in range(0, bsz, chunk):
        b1 = min(b0 + chunk, bsz)
        qr = query_raw[b0:b1]
        qn = query_norm[b0:b1]
        sr = support_raw[b0:b1]
        sn = support_norm[b0:b1]

        sk = v10.compute_pairwise_dynamic_sk_weights(
            qr, sr, weight_eps=sk_weight_eps
        )
        ot = v10.pairwise_sinkhorn_from_weights(
            qn,
            sn,
            sk.query,
            sk.support,
            reg=sinkhorn_reg,
            n_iter=sinkhorn_iter,
            normalize_cost_points=False,
        )
        dist_parts.append(ot.distance.float())

    return torch.cat(dist_parts, dim=0)


@torch.no_grad()
def encode_paths_to_metric_tokens(
    *,
    v10: Any,
    image_paths: list[str],
    transform,
    rae,
    metric,
    device: torch.device,
    rae_chunk_size: int,
    metric_chunk_size: int,
    ot_pool_factor: int,
    image_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parallel CPU loading + chunked frozen RAE/MetricEncoder GPU encoding."""
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    rc = max(1, int(rae_chunk_size))
    mc = int(metric_chunk_size)
    workers = max(0, int(image_workers))

    raw_parts: list[torch.Tensor] = []
    norm_parts: list[torch.Tensor] = []

    executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in range(0, len(image_paths), rc):
            paths_chunk = image_paths[start:start + rc]

            if executor is None:
                image_list = [load_rgb_tensor(p, transform) for p in paths_chunk]
            else:
                # executor.map preserves path order, which is essential for episode layout.
                image_list = list(
                    executor.map(
                        lambda p: load_rgb_tensor(p, transform),
                        paths_chunk,
                    )
                )

            images_cpu = torch.stack(image_list, dim=0)
            # Pinned transfer helps hide part of H2D latency when CUDA permits it.
            if device.type == "cuda":
                try:
                    images_cpu = images_cpu.pin_memory()
                except RuntimeError:
                    pass
            images = images_cpu.to(device, non_blocking=True)

            latent = rae.encode(images)
            if latent.ndim != 4 or latent.shape[1:] != (1024, 16, 16):
                raise RuntimeError(
                    f"RAE output must be (B,1024,16,16), got {tuple(latent.shape)}"
                )

            step = latent.shape[0] if mc <= 0 else max(1, mc)
            for m0 in range(0, latent.shape[0], step):
                feat = metric(latent[m0:m0 + step])
                raw, norm = v10._pool_raw(feat.raw, ot_pool_factor)
                raw_parts.append(raw.detach())
                norm_parts.append(norm.detach())

            del image_list, images_cpu, images, latent
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return torch.cat(raw_parts, dim=0), torch.cat(norm_parts, dim=0)


def sample_attempt_plan(
    *,
    pool: dict[str, list[str]],
    labels_all: list[str],
    rng: random.Random,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Sample one candidate episode plan without running any model."""
    gt_label = rng.choice(labels_all)
    query_path = rng.choice(pool[gt_label])

    wrong_labels_all = [x for x in labels_all if x != gt_label]
    wrong_labels = rng.sample(wrong_labels_all, args.candidate_way - 1)

    candidate_labels = [gt_label] + wrong_labels
    rng.shuffle(candidate_labels)
    gt_screen_index = candidate_labels.index(gt_label)

    support_sets: list[list[list[str]]] = []
    flat_screen_paths: list[str] = []
    for ci, label in enumerate(candidate_labels):
        reps: list[list[str]] = []
        for _ in range(args.support_repeats):
            one = sample_kshot(
                pool[label],
                args.n_shot,
                rng,
                exclude=query_path if ci == gt_screen_index else None,
            )
            reps.append(one)
            flat_screen_paths.extend(one)
        support_sets.append(reps)

    return {
        "gt_label": gt_label,
        "query_path": query_path,
        "candidate_labels": candidate_labels,
        "gt_screen_index": gt_screen_index,
        "support_sets": support_sets,
        "flat_screen_paths": flat_screen_paths,
    }


@torch.no_grad()
def mine_attempt_batch(
    *,
    v10: Any,
    pool: dict[str, list[str]],
    labels_all: list[str],
    rng: random.Random,
    transform,
    rae,
    metric,
    device: torch.device,
    shot_tau: torch.Tensor,
    args: argparse.Namespace,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Mine B independent episodes with one batched Dynamic-SK/Sinkhorn stream."""
    bsz = int(batch_size)
    if bsz < 1:
        raise ValueError("batch_size must be >=1")

    plans = [
        sample_attempt_plan(pool=pool, labels_all=labels_all, rng=rng, args=args)
        for _ in range(bsz)
    ]

    C = int(args.candidate_way)
    R = int(args.support_repeats)
    K = int(args.n_shot)
    N = int(args.n_way)
    S = C * R * K
    block_size = S + 1  # screened supports + one query

    # ------------------------------------------------------------
    # 1) Build batch path layout and deduplicate paths.
    #    Transform is deterministic (augment=False), so one path needs one encoding.
    # ------------------------------------------------------------
    flat_paths: list[str] = []
    for plan in plans:
        if len(plan["flat_screen_paths"]) != S:
            raise RuntimeError("screen support path count mismatch")
        flat_paths.extend(plan["flat_screen_paths"])
        flat_paths.append(plan["query_path"])

    path_to_unique: dict[str, int] = {}
    unique_paths: list[str] = []
    inverse: list[int] = []
    for p in flat_paths:
        p = str(p)
        idx = path_to_unique.get(p)
        if idx is None:
            idx = len(unique_paths)
            path_to_unique[p] = idx
            unique_paths.append(p)
        inverse.append(idx)

    unique_raw, unique_norm = encode_paths_to_metric_tokens(
        v10=v10,
        image_paths=unique_paths,
        transform=transform,
        rae=rae,
        metric=metric,
        device=device,
        rae_chunk_size=args.rae_chunk_size,
        metric_chunk_size=args.metric_chunk_size,
        ot_pool_factor=args.ot_pool_factor,
        image_workers=args.image_workers,
    )

    inverse_t = torch.tensor(inverse, dtype=torch.long, device=device)
    all_raw = unique_raw.index_select(0, inverse_t)
    all_norm = unique_norm.index_select(0, inverse_t)

    expected = bsz * block_size
    if int(all_raw.shape[0]) != expected:
        raise RuntimeError(
            f"metric token count mismatch: got {all_raw.shape[0]}, expected {expected}"
        )

    P, D = int(all_raw.shape[-2]), int(all_raw.shape[-1])
    raw_block = all_raw.reshape(bsz, block_size, P, D)
    norm_block = all_norm.reshape(bsz, block_size, P, D)

    support_raw = raw_block[:, :S]     # [B,S,P,D]
    support_norm = norm_block[:, :S]
    query_raw = raw_block[:, S:S+1]    # [B,1,P,D]
    query_norm = norm_block[:, S:S+1]

    # ------------------------------------------------------------
    # 2) TRUE batched Dynamic-SK + Sinkhorn.
    #    One tensor program handles B independent episodes in parallel.
    # ------------------------------------------------------------
    screen_pair_distance = batched_pairwise_shot_distances(
        v10=v10,
        support_raw=support_raw,
        support_norm=support_norm,
        query_raw=query_raw,
        query_norm=query_norm,
        sk_weight_eps=args.sk_weight_eps,
        sinkhorn_reg=args.sinkhorn_reg,
        sinkhorn_iter=args.sinkhorn_iter,
        attempt_chunk_size=args.ot_attempt_chunk_size,
    )[:, 0]  # [B,S]

    shot_distance = screen_pair_distance.reshape(bsz, C, R, K)
    tau = shot_tau.to(device).float().clamp_min(1e-8)
    set_distance = -tau * (
        torch.logsumexp(-shot_distance / tau, dim=-1) - math.log(float(K))
    )  # [B,C,R]

    # ------------------------------------------------------------
    # 3) Vectorized hard-positive / confusing-negative selection.
    # ------------------------------------------------------------
    bidx = torch.arange(bsz, device=device)
    gt_idx = torch.tensor(
        [int(p["gt_screen_index"]) for p in plans],
        dtype=torch.long,
        device=device,
    )

    # GT chooses the largest class distance across repeats.
    gt_repeat = set_distance[bidx, gt_idx].argmax(dim=-1)  # [B]
    gt_hard_distance = set_distance[bidx, gt_idx, gt_repeat]

    # Wrong classes choose the smallest distance across repeats.
    wrong_best_distance, wrong_repeat = set_distance.min(dim=-1)  # [B,C], [B,C]

    selected_class_distance = wrong_best_distance.clone()
    selected_class_distance[bidx, gt_idx] = gt_hard_distance

    # Select N-1 most confusing wrong classes.
    wrong_rank_distance = selected_class_distance.clone()
    wrong_rank_distance[bidx, gt_idx] = float("inf")
    chosen_wrong = torch.topk(
        wrong_rank_distance,
        k=N - 1,
        dim=-1,
        largest=False,
        sorted=True,
    ).indices  # [B,N-1]

    chosen_candidate_pre = torch.cat(
        [gt_idx.unsqueeze(1), chosen_wrong],
        dim=1,
    )  # [B,N]

    chosen_repeat_pre = wrong_repeat.gather(1, chosen_candidate_pre)
    chosen_repeat_pre[:, 0] = gt_repeat

    # ------------------------------------------------------------
    # 4) Randomize final candidate order independently per episode.
    # ------------------------------------------------------------
    final_orders_py: list[list[int]] = []
    for _ in range(bsz):
        order = list(range(N))
        rng.shuffle(order)
        final_orders_py.append(order)
    final_order = torch.tensor(final_orders_py, dtype=torch.long, device=device)

    final_candidate_idx = chosen_candidate_pre.gather(1, final_order)  # [B,N]
    final_repeat_idx = chosen_repeat_pre.gather(1, final_order)       # [B,N]

    gt_index = (final_candidate_idx == gt_idx.unsqueeze(1)).to(torch.long).argmax(dim=1)

    # ------------------------------------------------------------
    # 5) NO second Sinkhorn pass.
    #
    # Dynamic-SK and Sinkhorn are pair-conditioned only.  The exact distance of
    # q to a concrete support image is independent of the other candidate classes.
    # Therefore selected final N-way shot distances can be gathered exactly from
    # the screening tensor [B,C,R,K].
    # ------------------------------------------------------------
    gather_b = bidx.unsqueeze(1).expand(bsz, N)
    final_shot_distance = shot_distance[
        gather_b,
        final_candidate_idx,
        final_repeat_idx,
    ]  # [B,N,K]

    final_distance = -tau * (
        torch.logsumexp(-final_shot_distance / tau, dim=-1)
        - math.log(float(K))
    )  # [B,N]

    direct_pred = final_distance.argmin(dim=-1)
    gt_distance = final_distance.gather(1, gt_index.unsqueeze(1)).squeeze(1)

    wrong_masked = final_distance.clone()
    wrong_masked[bidx, gt_index] = float("inf")
    best_wrong_distance, _ = wrong_masked.min(dim=-1)
    margin_dist = best_wrong_distance - gt_distance

    accepted = direct_pred.ne(gt_index) & margin_dist.lt(0.0)

    # Move only compact audit tensors back to CPU.
    final_candidate_cpu = final_candidate_idx.cpu()
    final_repeat_cpu = final_repeat_idx.cpu()
    gt_index_cpu = gt_index.cpu()
    direct_pred_cpu = direct_pred.cpu()
    gt_repeat_cpu = gt_repeat.cpu()
    final_distance_cpu = final_distance.cpu()
    final_shot_cpu = final_shot_distance.cpu()
    gt_distance_cpu = gt_distance.cpu()
    best_wrong_cpu = best_wrong_distance.cpu()
    margin_cpu = margin_dist.cpu()
    accepted_cpu = accepted.cpu()

    results: list[dict[str, Any]] = []
    for bi, plan in enumerate(plans):
        cand_idx_row = [int(x) for x in final_candidate_cpu[bi].tolist()]
        repeat_row = [int(x) for x in final_repeat_cpu[bi].tolist()]

        labels_final = [plan["candidate_labels"][ci] for ci in cand_idx_row]
        supports_final = [
            plan["support_sets"][ci][ri]
            for ci, ri in zip(cand_idx_row, repeat_row)
        ]
        gt_local = int(gt_index_cpu[bi].item())
        pred_local = int(direct_pred_cpu[bi].item())
        support_dirs = [str(Path(x[0]).parent) for x in supports_final]

        results.append(
            {
                "accepted": bool(accepted_cpu[bi].item()),
                "record": {
                    "query_path": str(plan["query_path"]),
                    "query_label": str(plan["gt_label"]),
                    "candidate_labels": labels_final,
                    "support_dirs": support_dirs,
                    "support_paths": supports_final,
                    "source_dataset": "imagenet_train_direct_wrong_mined",
                },
                "stats": {
                    "gt_index": gt_local,
                    "direct_pred_index": pred_local,
                    "direct_pred_label": labels_final[pred_local],
                    "gt_distance": float(gt_distance_cpu[bi].item()),
                    "best_wrong_distance": float(best_wrong_cpu[bi].item()),
                    "distance_margin": float(margin_cpu[bi].item()),
                    "candidate_labels": labels_final,
                    "class_distances": [
                        float(x) for x in final_distance_cpu[bi].tolist()
                    ],
                    "shot_distances": [
                        [float(x) for x in final_shot_cpu[bi, ci].tolist()]
                        for ci in range(N)
                    ],
                    "candidate_way_screened": C,
                    "support_repeats": R,
                    "gt_support_repeat": int(gt_repeat_cpu[bi].item()),
                    "definition": (
                        "accepted iff Direct prediction is wrong after randomized "
                        "candidate ordering and best_wrong_distance - gt_distance < 0"
                    ),
                    "gt_screen_index": int(plan["gt_screen_index"]),
                    "attempt_batch_size": int(bsz),
                    "attempt_batch_slot": int(bi),
                    "parallel_ot": True,
                    "final_ot_reused_from_screen": True,
                    "unique_paths_encoded": int(len(unique_paths)),
                    "raw_paths_requested": int(len(flat_paths)),
                },
            }
        )

    del unique_raw, unique_norm, all_raw, all_norm
    return results

def atomic_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()

    if args.n_way < 2:
        raise ValueError("n_way must be >= 2")
    if args.n_shot < 1:
        raise ValueError("n_shot must be >= 1")
    if args.candidate_way < args.n_way:
        raise ValueError("candidate_way must be >= n_way")
    if args.support_repeats < 1:
        raise ValueError("support_repeats must be >= 1")
    if args.num_hard_episodes < 1:
        raise ValueError("num_hard_episodes must be >= 1")
    if args.attempt_batch_size < 1:
        raise ValueError("attempt_batch_size must be >= 1")
    if args.rae_chunk_size < 1:
        raise ValueError("rae_chunk_size must be >= 1")
    if args.ot_attempt_chunk_size < 0:
        raise ValueError("ot_attempt_chunk_size must be >= 0")
    if args.image_workers < 0:
        raise ValueError("image_workers must be >= 0")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for RAE + M2 Direct mining")
    torch.cuda.set_device(device)

    v10 = load_python_module(args.v10c_script)

    # ------------------------------------------------------------
    # Resolve M2 settings exactly as v10c does.
    # ------------------------------------------------------------
    m2_ckpt = v10._torch_load(args.m2_checkpoint, "cpu")
    m2_args = SimpleNamespace(
        n_way=args.n_way,
        n_shot=args.n_shot,
        sinkhorn_reg=args.sinkhorn_reg,
        sinkhorn_iter=args.sinkhorn_iter,
        sk_weight_eps=args.sk_weight_eps,
        ot_pool_factor=args.ot_pool_factor,
    )
    v10.resolve_from_m2(m2_args, m2_ckpt)
    args.sinkhorn_reg = m2_args.sinkhorn_reg
    args.sinkhorn_iter = m2_args.sinkhorn_iter
    args.sk_weight_eps = m2_args.sk_weight_eps
    args.ot_pool_factor = m2_args.ot_pool_factor

    # ------------------------------------------------------------
    # ImageNet class pool.
    # filter_episode_classes removes classes that cannot support K-shot+query.
    # ------------------------------------------------------------
    pool = v10.filter_episode_classes(
        v10.load_class_pool(None, args.train_class_json_root),
        n_shot=args.n_shot,
        n_query=1,
    )
    labels_all = sorted(pool.keys())
    if len(labels_all) < args.candidate_way:
        raise RuntimeError(
            f"usable classes={len(labels_all)} < candidate_way={args.candidate_way}"
        )

    transform = v10.build_transform(args.image_size, augment=False)
    rae = v10.build_rae(args.project_root, device)
    if hasattr(rae, "eval"):
        rae.eval()
    if hasattr(rae, "parameters"):
        for p in rae.parameters():
            p.requires_grad_(False)

    metric = v10.MetricEncoder(
        in_dim=1024,
        hidden_dim=args.hidden_dim,
        out_dim=args.metric_dim,
        dropout=args.dropout,
    ).to(device).eval()
    for p in metric.parameters():
        p.requires_grad_(False)

    # Load only the frozen metric state.  This miner accepts both native M2 and
    # M0-B checkpoints; M0-B intentionally has no episodic temperatures.
    metric_state = v10.extract_metric_state(m2_ckpt)
    missing, unexpected = metric.load_state_dict(metric_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "MetricEncoder checkpoint mismatch.\n"
            f"missing={missing[:20]}\nunexpected={unexpected[:20]}"
        )

    # Project convention for Direct K-shot aggregation.
    shot_tau = torch.tensor(0.10, dtype=torch.float32, device=device)

    # cls_tau never affects argmin class distance in this mining script.
    if m2_ckpt.get("log_cls_tau") is not None:
        cls_tau = torch.as_tensor(
            m2_ckpt["log_cls_tau"], dtype=torch.float32, device=device
        ).reshape(()).exp()
    elif m2_ckpt.get("cls_tau") is not None:
        cls_tau = torch.as_tensor(
            m2_ckpt["cls_tau"], dtype=torch.float32, device=device
        ).reshape(())
    else:
        cls_tau = torch.tensor(0.10, dtype=torch.float32, device=device)

    output_manifest = Path(args.output_manifest)
    output_stats = (
        Path(args.output_stats)
        if args.output_stats
        else output_manifest.with_name(output_manifest.stem + "_stats.json")
    )

    max_attempts = int(args.max_total_attempts)
    if max_attempts <= 0:
        max_attempts = max(1000, 100 * int(args.num_hard_episodes))

    accepted_records: list[dict[str, Any]] = []
    accepted_stats: list[dict[str, Any]] = []
    attempts = 0

    print("=" * 110)
    print("[Direct-WRONG hard episode mining]")
    print(f"seed                 = {args.seed}")
    print(f"target hard episodes = {args.num_hard_episodes}")
    print(f"episode              = {args.n_way}-way {args.n_shot}-shot 1-query")
    print(f"screen               = candidate_way={args.candidate_way}, support_repeats={args.support_repeats}")
    print(f"attempt batch        = {args.attempt_batch_size}")
    print(f"OT attempt chunk     = {args.ot_attempt_chunk_size} (0 = full batch)")
    print(f"RAE/metric chunks    = {args.rae_chunk_size}/{args.metric_chunk_size}")
    print(f"CPU image workers    = {args.image_workers}")
    print("parallel Direct OT    = TRUE (batch dimension is episode-local)")
    print("final N-way OT        = gathered from exact screening pair distances")
    print(f"shot_tau             = {float(shot_tau):.6f}")
    print(f"cls_tau              = {float(cls_tau):.6f} (not needed for distance argmin)")
    print(f"OT                   = pool={args.ot_pool_factor}, reg={args.sinkhorn_reg}, iter={args.sinkhorn_iter}, eps={args.sk_weight_eps}")
    print("hard definition      = Direct prediction WRONG only")
    print("accept condition      = best_wrong_distance - gt_distance < 0")
    print("=" * 110)

    pbar = tqdm(total=args.num_hard_episodes, desc="accepted-direct-wrong")
    last_margin = None

    while len(accepted_records) < args.num_hard_episodes and attempts < max_attempts:
        batch_size = min(
            int(args.attempt_batch_size),
            max_attempts - attempts,
        )
        batch_attempt_start = attempts + 1

        mined_batch = mine_attempt_batch(
            v10=v10,
            pool=pool,
            labels_all=labels_all,
            rng=rng,
            transform=transform,
            rae=rae,
            metric=metric,
            device=device,
            shot_tau=shot_tau,
            args=args,
            batch_size=batch_size,
        )
        attempts += batch_size

        for bi, mined in enumerate(mined_batch):
            attempt_index = batch_attempt_start + bi
            last_margin = float(mined["stats"]["distance_margin"])

            if not mined["accepted"]:
                continue
            if len(accepted_records) >= args.num_hard_episodes:
                break

            record_id = len(accepted_records)
            rec = mined["record"]
            rec["source_query_id"] = record_id
            rec["id"] = record_id
            accepted_records.append(rec)

            st = mined["stats"]
            st.update(
                {
                    "id": record_id,
                    "query_path": rec["query_path"],
                    "query_label": rec["query_label"],
                    "seed": int(args.seed),
                    "attempt_index": int(attempt_index),
                }
            )
            accepted_stats.append(st)
            pbar.update(1)

            if not (
                st["direct_pred_index"] != st["gt_index"]
                and st["distance_margin"] < 0.0
            ):
                raise AssertionError(f"invalid accepted hard episode: {st}")

            if args.save_every > 0 and len(accepted_records) % args.save_every == 0:
                atomic_json_dump(accepted_records, output_manifest)
                atomic_json_dump(
                    {
                        "summary": {
                            "complete": False,
                            "seed": args.seed,
                            "accepted": len(accepted_records),
                            "target": args.num_hard_episodes,
                            "attempts": attempts,
                            "accept_rate": len(accepted_records) / max(attempts, 1),
                            "attempt_batch_size": int(args.attempt_batch_size),
                        },
                        "records": accepted_stats,
                    },
                    output_stats,
                )

        if args.log_every > 0 and (
            attempts % args.log_every < batch_size
            or len(accepted_records) >= args.num_hard_episodes
        ):
            rate = len(accepted_records) / max(attempts, 1)
            postfix = {
                "attempts": attempts,
                "accept_rate": f"{100.0*rate:.2f}%",
                "batch": batch_size,
            }
            if accepted_stats:
                margins_now = [x["distance_margin"] for x in accepted_stats]
                postfix["margin_mean"] = f"{sum(margins_now)/len(margins_now):+.4f}"
            elif last_margin is not None:
                postfix["last_margin"] = f"{last_margin:+.4f}"
            pbar.set_postfix(**postfix)

    pbar.close()

    if len(accepted_records) < args.num_hard_episodes:
        print(
            f"[WARN] reached max_total_attempts={max_attempts}; "
            f"collected {len(accepted_records)}/{args.num_hard_episodes}."
        )

    margins = [x["distance_margin"] for x in accepted_stats]
    summary = {
        "complete": len(accepted_records) >= args.num_hard_episodes,
        "definition": "Direct-WRONG only",
        "accept_condition": "best_wrong_distance - gt_distance < 0 and Direct argmin != GT",
        "seed": int(args.seed),
        "accepted": len(accepted_records),
        "target": int(args.num_hard_episodes),
        "attempts": int(attempts),
        "accept_rate": len(accepted_records) / max(attempts, 1),
        "n_way": int(args.n_way),
        "n_shot": int(args.n_shot),
        "candidate_way": int(args.candidate_way),
        "support_repeats": int(args.support_repeats),
        "attempt_batch_size": int(args.attempt_batch_size),
        "ot_attempt_chunk_size": int(args.ot_attempt_chunk_size),
        "rae_chunk_size": int(args.rae_chunk_size),
        "metric_chunk_size": int(args.metric_chunk_size),
        "image_workers": int(args.image_workers),
        "parallel_ot": True,
        "final_ot_reused_from_screen": True,
        "distance_margin_mean": (
            sum(margins) / len(margins) if margins else None
        ),
        "distance_margin_min": min(margins) if margins else None,
        "distance_margin_max": max(margins) if margins else None,
        "m2_checkpoint": str(args.m2_checkpoint),
        "train_class_json_root": str(args.train_class_json_root),
        "v10c_script": str(args.v10c_script),
        "shot_tau": float(shot_tau),
        "cls_tau": float(cls_tau),
        "ot_pool_factor": int(args.ot_pool_factor),
        "sinkhorn_reg": float(args.sinkhorn_reg),
        "sinkhorn_iter": int(args.sinkhorn_iter),
        "sk_weight_eps": float(args.sk_weight_eps),
    }

    atomic_json_dump(accepted_records, output_manifest)
    atomic_json_dump({"summary": summary, "records": accepted_stats}, output_stats)

    print("\n[DONE]")
    print("manifest =", output_manifest)
    print("stats    =", output_stats)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
