#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect ONLY Direct-misclassified fixed ImageNet episodes into a manifest.

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
For each query:
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
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
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

    # Exact Direct/M2 settings.  None values are resolved from the M2 checkpoint.
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--rae_chunk_size", type=int, default=32)
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
    p.add_argument("--save_every", type=int, default=50,
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
def exact_direct_class_distances(
    *,
    v10: Any,
    metric,
    metric_device: torch.device,
    support_z: torch.Tensor,
    query_z: torch.Tensor,
    n_classes: int,
    n_shot: int,
    shot_tau: torch.Tensor,
    ot_pool_factor: int,
    sk_weight_eps: float,
    sinkhorn_reg: float,
    sinkhorn_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact M2 Direct class distances and per-shot OT distances.

    support_z layout must be class-major and K-shot contiguous:
      [class0 shot0..K-1, class1 shot0..K-1, ...]

    Returns
    -------
    class_distance : [Q, C]   lower is better
    shot_distance  : [Q, C, K]
    """
    n_support = int(support_z.shape[0])
    n_query = int(query_z.shape[0])
    if n_support != int(n_classes * n_shot):
        raise ValueError(
            f"support count mismatch: got {n_support}, expected {n_classes*n_shot}"
        )

    s_feat = metric(support_z.to(metric_device, non_blocking=True))
    q_feat = metric(query_z.to(metric_device, non_blocking=True))

    s_raw, s_norm = v10._pool_raw(s_feat.raw, ot_pool_factor)
    q_raw, q_norm = v10._pool_raw(q_feat.raw, ot_pool_factor)

    sk = v10.compute_pairwise_dynamic_sk_weights(
        q_raw, s_raw, weight_eps=sk_weight_eps
    )
    ot = v10.pairwise_sinkhorn_from_weights(
        q_norm,
        s_norm,
        sk.query,
        sk.support,
        reg=sinkhorn_reg,
        n_iter=sinkhorn_iter,
        normalize_cost_points=False,
    )

    shot_distance = ot.distance.float().reshape(n_query, n_classes, n_shot)
    tau = shot_tau.to(metric_device).float().clamp_min(1e-8)

    # EXACT K-shot aggregation used by v10c Direct.
    class_distance = -tau * (
        torch.logsumexp(-shot_distance / tau, dim=-1) - math.log(float(n_shot))
    )
    return class_distance, shot_distance


@torch.no_grad()
def mine_one_attempt(
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
) -> dict[str, Any]:
    # ------------------------------------------------------------
    # 1. Query + candidate classes.
    # ------------------------------------------------------------
    gt_label = rng.choice(labels_all)
    query_path = rng.choice(pool[gt_label])

    wrong_labels_all = [x for x in labels_all if x != gt_label]
    wrong_labels = rng.sample(wrong_labels_all, args.candidate_way - 1)

    # IMPORTANT: GT has NO fixed candidate position, even during screening.
    candidate_labels = [gt_label] + wrong_labels
    rng.shuffle(candidate_labels)
    gt_screen_index = candidate_labels.index(gt_label)

    # ------------------------------------------------------------
    # 2. Sample R concrete K-shot support realizations per class.
    #    Flatten in [class, repeat, shot] order.
    # ------------------------------------------------------------
    support_sets: list[list[list[str]]] = []
    support_images: list[torch.Tensor] = []

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
            support_images.extend(load_rgb_tensor(p, transform) for p in one)
        support_sets.append(reps)

    query_img = load_rgb_tensor(query_path, transform).unsqueeze(0)
    support_img = torch.stack(support_images, dim=0)
    image_batch = torch.cat([support_img, query_img], dim=0).to(
        device, non_blocking=True
    )

    # One frozen RAE pass for all screened supports + query.
    z = v10.encode_rae_chunked(rae, image_batch, args.rae_chunk_size)
    support_z = z[:-1].detach()
    query_z = z[-1:].detach()

    # ------------------------------------------------------------
    # 3. Every (class, repeat) is temporarily treated as one K-shot
    #    class.  We explicitly get its class DISTANCE, not logits.
    # ------------------------------------------------------------
    n_screen_sets = args.candidate_way * args.support_repeats
    set_distance, _ = exact_direct_class_distances(
        v10=v10,
        metric=metric,
        metric_device=device,
        support_z=support_z,
        query_z=query_z,
        n_classes=n_screen_sets,
        n_shot=args.n_shot,
        shot_tau=shot_tau,
        ot_pool_factor=args.ot_pool_factor,
        sk_weight_eps=args.sk_weight_eps,
        sinkhorn_reg=args.sinkhorn_reg,
        sinkhorn_iter=args.sinkhorn_iter,
    )
    set_distance = set_distance[0].reshape(
        args.candidate_way, args.support_repeats
    )

    # GT: choose largest distance (hardest concrete positive support set).
    gt_repeat = int(torch.argmax(set_distance[gt_screen_index]).item())

    # Wrong class: choose smallest distance (most confusing concrete support set).
    wrong_repeat: dict[int, int] = {}
    selected_distance_by_candidate: dict[int, float] = {
        gt_screen_index: float(set_distance[gt_screen_index, gt_repeat].item())
    }
    for ci in range(args.candidate_way):
        if ci == gt_screen_index:
            continue
        ri = int(torch.argmin(set_distance[ci]).item())
        wrong_repeat[ci] = ri
        selected_distance_by_candidate[ci] = float(set_distance[ci, ri].item())

    # ------------------------------------------------------------
    # 4. Select final N-1 most confusing wrong classes by distance.
    # ------------------------------------------------------------
    ranked_wrong = sorted(
        [ci for ci in range(args.candidate_way) if ci != gt_screen_index],
        key=lambda ci: selected_distance_by_candidate[ci],
    )
    chosen_wrong = ranked_wrong[: args.n_way - 1]
    chosen_candidate_idx = [gt_screen_index] + chosen_wrong

    final_labels = [candidate_labels[i] for i in chosen_candidate_idx]
    final_supports: list[list[str]] = []
    for ci in chosen_candidate_idx:
        if ci == gt_screen_index:
            ri = gt_repeat
        else:
            ri = wrong_repeat[ci]
        final_supports.append(support_sets[ci][ri])

    # Randomize FINAL candidate order BEFORE the acceptance Direct calculation.
    final_order = list(range(args.n_way))
    rng.shuffle(final_order)
    labels_final = [final_labels[i] for i in final_order]
    supports_final = [final_supports[i] for i in final_order]
    gt_index = labels_final.index(gt_label)

    # ------------------------------------------------------------
    # 5. IMPORTANT: recompute exact final N-way Direct distances from
    #    the concrete selected supports AFTER final candidate shuffling.
    #    This is the acceptance truth; GT has no privileged position.
    # ------------------------------------------------------------
    final_support_images = torch.stack(
        [
            load_rgb_tensor(p, transform)
            for shot_paths in supports_final
            for p in shot_paths
        ],
        dim=0,
    ).to(device, non_blocking=True)
    final_query_img = load_rgb_tensor(query_path, transform).unsqueeze(0).to(
        device, non_blocking=True
    )
    final_all = torch.cat([final_support_images, final_query_img], dim=0)
    final_z = v10.encode_rae_chunked(rae, final_all, args.rae_chunk_size)
    final_support_z = final_z[:-1].detach()
    final_query_z = final_z[-1:].detach()

    final_distance, final_shot_distance = exact_direct_class_distances(
        v10=v10,
        metric=metric,
        metric_device=device,
        support_z=final_support_z,
        query_z=final_query_z,
        n_classes=args.n_way,
        n_shot=args.n_shot,
        shot_tau=shot_tau,
        ot_pool_factor=args.ot_pool_factor,
        sk_weight_eps=args.sk_weight_eps,
        sinkhorn_reg=args.sinkhorn_reg,
        sinkhorn_iter=args.sinkhorn_iter,
    )
    final_distance = final_distance[0]  # [N]
    final_shot_distance = final_shot_distance[0]  # [N,K]

    gt_distance = float(final_distance[gt_index].item())
    wrong_indices = [i for i in range(args.n_way) if i != gt_index]
    best_wrong_local = min(
        wrong_indices,
        key=lambda i: float(final_distance[i].item()),
    )
    best_wrong_distance = float(final_distance[best_wrong_local].item())
    margin_dist = best_wrong_distance - gt_distance
    direct_pred_after_shuffle = int(torch.argmin(final_distance).item())

    # Strict definition requested by user: only Direct-WRONG is hard.
    is_direct_wrong = (
        direct_pred_after_shuffle != gt_index
        and margin_dist < 0.0
    )

    distances_final = [float(x) for x in final_distance.tolist()]
    shot_distances_final = [
        [float(x) for x in final_shot_distance[i].tolist()]
        for i in range(args.n_way)
    ]

    support_dirs = [str(Path(x[0]).parent) for x in supports_final]

    return {
        "accepted": bool(is_direct_wrong),
        "record": {
            "query_path": str(query_path),
            "query_label": str(gt_label),
            "candidate_labels": labels_final,
            "support_dirs": support_dirs,
            "support_paths": supports_final,
            "source_dataset": "imagenet_train_direct_wrong_mined",
        },
        "stats": {
            "gt_index": int(gt_index),
            "direct_pred_index": int(direct_pred_after_shuffle),
            "direct_pred_label": labels_final[direct_pred_after_shuffle],
            "gt_distance": float(gt_distance),
            "best_wrong_distance": float(best_wrong_distance),
            "distance_margin": float(margin_dist),
            "candidate_labels": labels_final,
            "class_distances": distances_final,
            "shot_distances": shot_distances_final,
            "candidate_way_screened": int(args.candidate_way),
            "support_repeats": int(args.support_repeats),
            "gt_support_repeat": int(gt_repeat),
            "definition": "accepted iff Direct prediction is wrong after randomized candidate ordering and best_wrong_distance - gt_distance < 0",
            "gt_screen_index": int(gt_screen_index),
        },
    }


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

    shot_tau, cls_tau = v10.load_m2_metric_and_temperatures(
        metric, m2_ckpt, device
    )
    shot_tau = torch.tensor(0.10, dtype=torch.float32, device=device)
    # cls_tau is not needed to decide argmin distance; keep it only for audit.

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
    print(f"shot_tau             = {float(shot_tau):.6f}")
    print(f"cls_tau              = {float(cls_tau):.6f} (not needed for distance argmin)")
    print(f"OT                   = pool={args.ot_pool_factor}, reg={args.sinkhorn_reg}, iter={args.sinkhorn_iter}, eps={args.sk_weight_eps}")
    print("hard definition      = Direct prediction WRONG only")
    print("accept condition      = best_wrong_distance - gt_distance < 0")
    print("=" * 110)

    pbar = tqdm(total=args.num_hard_episodes, desc="accepted-direct-wrong")

    while len(accepted_records) < args.num_hard_episodes and attempts < max_attempts:
        attempts += 1
        mined = mine_one_attempt(
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
        )

        if not mined["accepted"]:
            if args.log_every > 0 and attempts % args.log_every == 0:
                rate = len(accepted_records) / max(attempts, 1)
                pbar.set_postfix(
                    attempts=attempts,
                    accept_rate=f"{100.0*rate:.2f}%",
                    last_margin=f"{mined['stats']['distance_margin']:+.4f}",
                )
            continue

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
                "attempt_index": int(attempts),
            }
        )
        accepted_stats.append(st)
        pbar.update(1)

        # Safety assertion: every saved episode MUST actually be Direct-wrong.
        if not (
            st["direct_pred_index"] != st["gt_index"]
            and st["distance_margin"] < 0.0
        ):
            raise AssertionError(f"invalid accepted hard episode: {st}")

        if args.log_every > 0 and len(accepted_records) % args.log_every == 0:
            rate = len(accepted_records) / max(attempts, 1)
            margins = [x["distance_margin"] for x in accepted_stats]
            pbar.set_postfix(
                attempts=attempts,
                accept_rate=f"{100.0*rate:.2f}%",
                margin_mean=f"{sum(margins)/len(margins):+.4f}",
            )

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
                    },
                    "records": accepted_stats,
                },
                output_stats,
            )

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
