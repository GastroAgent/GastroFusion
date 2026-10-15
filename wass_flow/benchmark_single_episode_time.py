#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-episode inference-time benchmark for
Direct + Q2Last + Q2MonoMean + Fuse-A/B/C.

Timing policy
-------------
EXCLUDED:
  * checkpoint/model loading
  * dataset scan and episode sampling
  * disk image I/O and image preprocessing

INCLUDED (timed core):
  * RAE encoding of one query + all K-shot supports
  * support-conditioned Flow trajectory generation
  * Direct Dynamic-SK
  * per-step Dynamic-SK used by Q2Last/Q2MonoMean
  * K-shot aggregation
  * Fuse-A/B/C and final argmin/argmax predictions

The script measures ONE episode after one untimed CUDA warm-up.
Run it from the same directory as:
  eval_q2l_direct_q2last_q2mono_scorefuse_maxmin_testtrain_reduced_v6.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

import eval_q2l_direct_q2last_q2mono_scorefuse_maxmin_testtrain_reduced_v6 as E


# =============================================================================
# BENCHMARK CONFIG
# =============================================================================
# Default = a standard 8-way 5-shot 1-query episode.
BENCH_HOSPITAL = "华西"
BENCH_PART = "食管"              # 食管/胃 -> nominal 8-way; 十二指肠 -> nominal 7-way
BENCH_METRIC_NAME = "m2_食管"    # must be a key in E.Q2L_CHECKPOINTS
BENCH_EPISODE_INDEX = 0

# Warm-up is NOT reported in timing. Keep >=1 to avoid first-call CUDA overhead.
WARMUP_RUNS = 1

# Save a compact machine-readable result next to the script.
OUTPUT_JSON = "single_episode_time_result.json"


# =============================================================================
# HELPERS
# =============================================================================
def load_one_metric(
    name: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, E.Q2LRuntimeConfig, Dict[str, Any]]:
    """Load exactly one MetricEncoder and its own runtime config.

    Important: the original evaluator can hold several organ/checkpoint entries.
    For a single-model timing benchmark, we restore the config from the selected
    checkpoint itself instead of inheriting a common config from another entry.
    """
    if name not in E.Q2L_CHECKPOINTS:
        raise KeyError(
            f"BENCH_METRIC_NAME={name!r} is not in Q2L_CHECKPOINTS. "
            f"Available: {list(E.Q2L_CHECKPOINTS)}"
        )

    path = E.Q2L_CHECKPOINTS[name]
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Metric checkpoint missing: {path}")

    ckpt = E.torch_load(path)
    if not isinstance(ckpt, dict):
        raise TypeError("Metric checkpoint must be a dict")

    a = E.as_args_dict(ckpt.get("args", {}))
    hidden = int(a.get("hidden_dim", 256))
    metric_dim = int(a.get("metric_dim", 256))
    dropout = float(a.get("dropout", 0.0))

    metric = E.MetricEncoder(1024, hidden, metric_dim, dropout=dropout)
    metric.load_state_dict(E.extract_metric_state(ckpt), strict=True)
    metric = metric.to(device).eval().requires_grad_(False)

    cfg = E.runtime_config_from_checkpoint(ckpt)
    meta = {
        "name": name,
        "path": path,
        "epoch": ckpt.get("epoch"),
        "checkpoint_kind": ckpt.get("checkpoint_kind"),
        "stage": ckpt.get("stage"),
        "hidden_dim": hidden,
        "metric_dim": metric_dim,
        "dropout": dropout,
    }
    return metric, cfg, meta


def build_one_episode(
    *,
    hospital: str,
    part: str,
    transform,
    device: torch.device,
) -> Tuple[E.EpisodePlan, torch.Tensor, torch.Tensor, int, int, int]:
    """Prepare exactly one matched episode and preload its image tensors.

    Image reading/preprocessing happens here, BEFORE the timed region.
    """
    requested_way = int(E.ORGAN_N_WAY.get(part, 8))

    query_pool, support_pool, candidate_labels, query_labels, _, _ = E.prepare_task_pools(
        hospital,
        part,
        eval_setting=E.EVAL_SETTING,
        n_way=requested_way,
        n_shot=E.N_SHOT,
    )

    # Match the original evaluator's effective-way policy.
    effective_way = (
        len(candidate_labels)
        if E.EVAL_SETTING == "test_train"
        else min(requested_way, len(candidate_labels))
    )

    task_seed = (
        E.SEED
        + sum(ord(c) for c in hospital) * 1009
        + sum(ord(c) for c in part) * 9176
    )

    n_needed = BENCH_EPISODE_INDEX + 1
    plans = E.generate_episode_plans(
        query_pool,
        support_pool,
        candidate_labels,
        query_labels,
        n_way=effective_way,
        n_shot=E.N_SHOT,
        n_episodes=n_needed,
        seed=task_seed,
        eval_setting=E.EVAL_SETTING,
    )
    ep = plans[BENCH_EPISODE_INDEX]

    support_flat: List[str] = []
    for c in ep.candidate_labels:
        support_flat.extend(ep.support_paths[c])

    # Deliberately outside the timed region.
    support_images = E.load_images(support_flat, transform).to(device, non_blocking=True)
    query_image = E.load_images([ep.query_path], transform).to(device, non_blocking=True)
    torch.cuda.synchronize(device)

    actual_way = len(ep.candidate_labels)
    actual_shot = E.N_SHOT
    n_support = len(support_flat)

    if n_support != actual_way * actual_shot:
        raise RuntimeError(
            f"Unexpected support count: got {n_support}, "
            f"expected way*shot={actual_way}*{actual_shot}"
        )

    return ep, support_images, query_image, actual_way, actual_shot, task_seed


@torch.inference_mode()
def infer_one_episode(
    *,
    ep: E.EpisodePlan,
    support_images: torch.Tensor,
    query_image: torch.Tensor,
    actual_way: int,
    actual_shot: int,
    task_seed: int,
    rae: torch.nn.Module,
    flow: E.FrozenFlowRuntime,
    metric: torch.nn.Module,
    cfg: E.Q2LRuntimeConfig,
) -> Dict[str, Any]:
    """Run the exact expensive path needed for all reported matched scores."""

    # -------------------------------------------------------------------------
    # 1) RAE encode: all K-shot supports + one query
    # -------------------------------------------------------------------------
    all_images = torch.cat([support_images, query_image], dim=0)
    all_z = E.encode_rae_chunked(rae, all_images, E.RAE_CHUNK_SIZE)
    support_z = all_z[:-1]
    query_z = all_z[-1:]

    # -------------------------------------------------------------------------
    # 2) One support-conditioned Flow trajectory for every query-support pair
    # -------------------------------------------------------------------------
    n_pairs = actual_way * actual_shot
    q_pairs = query_z.expand(n_pairs, -1, -1, -1).contiguous()
    hint_seed = int(task_seed + 8_000_003 + BENCH_EPISODE_INDEX * 97_003)
    post_states = E.generate_post_states_for_pairs(
        flow,
        q_pairs,
        support_images,
        cfg,
        hint_seed=hint_seed,
    )

    # -------------------------------------------------------------------------
    # 3) Direct + per-step Dynamic-SK
    #    q2mono_per_step is shared by Q2Last and Q2MonoMean.
    # -------------------------------------------------------------------------
    direct_pair = E.aligned_pair_distances(
        metric, query_z, support_z, cfg
    ).reshape(actual_way, actual_shot)

    q2mono_pair_flat, q2mono_per_step_flat = E.q2mono_mean_pair_distances(
        metric,
        query_z,
        post_states,
        direct_pair.reshape(-1),
        cfg,
    )
    q2mono_pair = q2mono_pair_flat.reshape(actual_way, actual_shot)
    q2mono_per_step = q2mono_per_step_flat.reshape(
        actual_way, actual_shot, cfg.num_steps
    )

    # -------------------------------------------------------------------------
    # 4) K-shot aggregation + predictions + fusion
    # -------------------------------------------------------------------------
    direct_class = E.aggregate_kshot(direct_pair)
    q2last_pair = E.query2last_from_steps(q2mono_per_step)
    q2last_class = E.aggregate_kshot(q2last_pair)
    q2mono_class = E.aggregate_kshot(q2mono_pair)

    direct_idx = int(direct_class.argmin().item())
    q2last_idx = int(q2last_class.argmin().item())
    q2mono_idx = int(q2mono_class.argmin().item())

    score_dict = {
        "direct": direct_class,
        "q2last": q2last_class,
        "q2mono": q2mono_class,
    }
    fuse_preds = {
        mode: E.score_fusion_predictions(score_dict, mode)
        for mode in E.FUSION_MODES
    }

    return {
        "gt_label": ep.query_label,
        "direct_pred": ep.candidate_labels[direct_idx],
        "q2last_pred": ep.candidate_labels[q2last_idx],
        "q2mono_pred": ep.candidate_labels[q2mono_idx],
        "fuse_a_pred": ep.candidate_labels[fuse_preds["dist_sum"]],
        "fuse_b_pred": ep.candidate_labels[fuse_preds["soft_vote"]],
        "fuse_c_pred": ep.candidate_labels[fuse_preds["max_then_min"]],
        "direct_correct": ep.candidate_labels[direct_idx] == ep.query_label,
        "q2last_correct": ep.candidate_labels[q2last_idx] == ep.query_label,
        "q2mono_correct": ep.candidate_labels[q2mono_idx] == ep.query_label,
        "fuse_a_correct": ep.candidate_labels[fuse_preds["dist_sum"]] == ep.query_label,
        "fuse_b_correct": ep.candidate_labels[fuse_preds["soft_vote"]] == ep.query_label,
        "fuse_c_correct": ep.candidate_labels[fuse_preds["max_then_min"]] == ep.query_label,
    }


def main() -> None:
    E.setup_seed(E.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # =========================================================================
    # MODEL LOADING -- EXCLUDED FROM TIMING
    # =========================================================================
    print("=" * 100)
    print("[1/3] Loading models/checkpoints -- NOT TIMED")

    metric, cfg, metric_meta = load_one_metric(BENCH_METRIC_NAME, device)
    if cfg.ot_pool_factor < 1:
        raise ValueError("ot_pool_factor must be >= 1")
    if not os.path.isfile(cfg.dit_checkpoint):
        raise FileNotFoundError(f"DiT checkpoint missing: {cfg.dit_checkpoint}")

    rae = E.build_rae(E.PROJECT_ROOT, device)
    flow = E.FrozenFlowRuntime(project_root=E.PROJECT_ROOT, cfg=cfg, device=device)
    transform = E.build_transform(E.IMAGE_SIZE)
    torch.cuda.synchronize(device)

    # =========================================================================
    # EPISODE CREATION + IMAGE I/O -- EXCLUDED FROM TIMING
    # =========================================================================
    print("[2/3] Preparing one episode/images -- NOT TIMED")
    ep, support_images, query_image, actual_way, actual_shot, task_seed = build_one_episode(
        hospital=BENCH_HOSPITAL,
        part=BENCH_PART,
        transform=transform,
        device=device,
    )

    n_support = actual_way * actual_shot
    print("-" * 100)
    print(f"Hospital / organ     : {BENCH_HOSPITAL} / {BENCH_PART}")
    print(f"Evaluation setting   : {E.EVAL_SETTING}")
    print(f"Episode              : {actual_way}-way {actual_shot}-shot {E.N_QUERY}-query")
    print(f"Input images         : 1 query + {n_support} supports = {n_support + 1} images")
    print(f"Metric checkpoint    : {BENCH_METRIC_NAME}")
    print(f"ODE                   : {cfg.num_steps} steps, solver={cfg.solver}")
    print(f"Flow pair chunk      : {cfg.pair_chunk_size}")
    print(f"AMP                   : {cfg.amp}")
    print(
        f"Sinkhorn              : reg={cfg.sinkhorn_reg}, "
        f"iter={cfg.sinkhorn_iter}, weight_eps={cfg.sk_weight_eps}, "
        f"pool={cfg.ot_pool_factor}"
    )
    print(f"K-shot aggregation   : {E.SHOT_AGG}" + (f", tau={E.SHOT_TAU}" if E.SHOT_AGG == "softmin" else ""))
    print("-" * 100)

    # =========================================================================
    # WARM-UP -- EXCLUDED FROM REPORTED TIMING
    # =========================================================================
    for i in range(WARMUP_RUNS):
        _ = infer_one_episode(
            ep=ep,
            support_images=support_images,
            query_image=query_image,
            actual_way=actual_way,
            actual_shot=actual_shot,
            task_seed=task_seed,
            rae=rae,
            flow=flow,
            metric=metric,
            cfg=cfg,
        )
        torch.cuda.synchronize(device)
        print(f"Warm-up {i + 1}/{WARMUP_RUNS} finished -- NOT TIMED")

    # =========================================================================
    # ONE MEASURED EPISODE
    # =========================================================================
    print("[3/3] Measuring ONE episode")

    # CUDA is asynchronous; synchronize immediately before and after timing.
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    pred = infer_one_episode(
        ep=ep,
        support_images=support_images,
        query_image=query_image,
        actual_way=actual_way,
        actual_shot=actual_shot,
        task_seed=task_seed,
        rae=rae,
        flow=flow,
        metric=metric,
        cfg=cfg,
    )

    torch.cuda.synchronize(device)
    elapsed_sec = time.perf_counter() - t0

    result = {
        "timing_scope": {
            "model_loading_included": False,
            "episode_sampling_included": False,
            "disk_image_io_included": False,
            "image_preprocessing_included": False,
            "included": [
                "RAE encoding",
                "support-conditioned Flow trajectory",
                "Direct Dynamic-SK",
                "Q2Last/Q2MonoMean per-step Dynamic-SK",
                "K-shot aggregation",
                "Fuse-A/B/C",
                "final predictions",
            ],
            "warmup_runs_not_timed": WARMUP_RUNS,
            "measured_episodes": 1,
        },
        "protocol": {
            "hospital": BENCH_HOSPITAL,
            "organ": BENCH_PART,
            "eval_setting": E.EVAL_SETTING,
            "episode_index": BENCH_EPISODE_INDEX,
            "n_way": actual_way,
            "n_shot": actual_shot,
            "n_query": E.N_QUERY,
            "n_support_images": n_support,
            "n_total_input_images": n_support + E.N_QUERY,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(device),
            "num_steps": cfg.num_steps,
            "solver": cfg.solver,
            "pair_chunk_size": cfg.pair_chunk_size,
            "amp": cfg.amp,
            "ot_pool_factor": cfg.ot_pool_factor,
            "sinkhorn_reg": cfg.sinkhorn_reg,
            "sinkhorn_iter": cfg.sinkhorn_iter,
            "sk_weight_eps": cfg.sk_weight_eps,
            "shot_aggregation": E.SHOT_AGG,
            "shot_tau": E.SHOT_TAU if E.SHOT_AGG == "softmin" else None,
        },
        "checkpoint": metric_meta,
        "prediction": pred,
        "single_episode_inference_sec": elapsed_sec,
    }

    out_path = Path(__file__).resolve().parent / OUTPUT_JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("[RESULT]")
    print(f"Protocol              : {actual_way}-way {actual_shot}-shot {E.N_QUERY}-query")
    print(f"Supports              : {n_support}")
    print(f"ODE                   : {cfg.num_steps} steps / {cfg.solver}")
    print(
        f"Sinkhorn              : reg={cfg.sinkhorn_reg}, "
        f"iter={cfg.sinkhorn_iter}, eps={cfg.sk_weight_eps}"
    )
    print(f"Single-episode time   : {elapsed_sec:.6f} s")
    print(f"GT                    : {pred['gt_label']}")
    print(
        "Predictions           : "
        f"Direct={pred['direct_pred']} | "
        f"Q2Last={pred['q2last_pred']} | "
        f"Q2Mono={pred['q2mono_pred']} | "
        f"FuseA={pred['fuse_a_pred']} | "
        f"FuseB={pred['fuse_b_pred']} | "
        f"FuseC={pred['fuse_c_pred']}"
    )
    print(f"Saved JSON            : {out_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
