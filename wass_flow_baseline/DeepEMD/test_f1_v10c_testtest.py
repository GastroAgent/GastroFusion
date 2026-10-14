#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate an F1-v10c checkpoint under the strict test-test protocol.

Protocol
--------
* Query and support come from the same test class pool.
* One N-way / K-shot / 1-query episode is sampled at a time.
* The exact query image path is ALWAYS excluded from every support candidate set.
* A class is eligible only if it contains >=2 distinct images.
* After query exclusion, K-shot uses sampling WITHOUT replacement when possible,
  otherwise WITH replacement from the remaining images.
* By default all eligible classes are preserved as the candidate set (dynamic-way).
  Set --n_way > 0 to cap the candidate way; if fewer classes exist it is clamped.

The evaluator reuses the pretraining module's frozen RAE/M2/Flow distance pipeline,
V10 trajectory features, and PairwiseFlowTrajectoryCorrector implementation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


PRETRAIN_BASENAME = "pretrain_f1_flow_trajectory_pairwise_corrector_v10c_classjson_torchrun_ddp_batch.py"


@dataclass
class EpisodeSpec:
    classes: list[str]
    support_paths: list[list[str]]  # N x K
    query_path: str
    query_class: str
    target_index: int
    support_used_replacement: list[bool]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("F1-v10c strict test-test evaluation")
    p.add_argument("--pretrain_py", type=str, default=None,
                   help=f"Path to {PRETRAIN_BASENAME}; default: sibling file next to this script")
    p.add_argument("--f1_checkpoint", type=str, required=True)
    p.add_argument("--test_class_json_root", type=str, required=True,
                   help="Test/query/support class JSON root or class-pool path accepted by load_class_pool().")
    p.add_argument("--output_json", type=str, required=True)

    # Optional upstream overrides. By default use paths saved in the F1 checkpoint.
    p.add_argument("--project_root", type=str, default=None)
    p.add_argument("--m2_checkpoint", type=str, default=None)
    p.add_argument("--dit_checkpoint", type=str, default=None)

    # Episode protocol. n_way<=0 means preserve all eligible classes.
    p.add_argument("--n_way", type=int, default=0)
    p.add_argument("--n_shot", type=int, default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--episode_batch_size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # Runtime overrides; otherwise copied from checkpoint training args.
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--solver", type=str, default=None, choices=["euler", "heun"])
    p.add_argument("--pair_chunk_size", type=int, default=None)
    p.add_argument("--rae_chunk_size", type=int, default=None)
    p.add_argument("--correction_topk", type=int, default=None)
    p.add_argument("--metric_device", type=str, default="cuda:0")
    p.add_argument("--flow_device", type=str, default="cuda:0")
    p.add_argument("--rae_device", type=str, default="cuda:0")
    p.add_argument("--save_episode_details", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="Override Flow AMP. Default uses training checkpoint setting.")
    return p.parse_args()


def _load_pretrain_module(path: Optional[str]):
    if path is None:
        path = str(Path(__file__).resolve().with_name(PRETRAIN_BASENAME))
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"pretrain module not found: {p}\n"
            f"Pass --pretrain_py /path/to/{PRETRAIN_BASENAME}"
        )
    spec = importlib.util.spec_from_file_location("f1_v10c_pretrain_runtime", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pretrain module: {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, str(p)


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _unique_paths(paths: list[str]) -> list[str]:
    out, seen = [], set()
    for x in paths:
        s = str(x)
        key = os.path.realpath(os.path.abspath(os.path.expanduser(s)))
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def normalize_pool(pool: dict[str, list[str]]) -> dict[str, list[str]]:
    return {str(k): _unique_paths([str(x) for x in v]) for k, v in pool.items() if v}


def _same_path(a: str, b: str) -> bool:
    return os.path.realpath(os.path.abspath(os.path.expanduser(a))) == os.path.realpath(
        os.path.abspath(os.path.expanduser(b))
    )


def sample_k(paths: list[str], k: int, rng: random.Random) -> tuple[list[str], bool]:
    if k < 1:
        raise ValueError("n_shot must be >=1")
    if not paths:
        raise ValueError("cannot sample support from an empty class")
    if len(paths) >= k:
        return rng.sample(paths, k), False
    return [rng.choice(paths) for _ in range(k)], True


def effective_way(requested: int, n_eligible: int) -> int:
    if requested == 1:
        raise ValueError("n_way=1 is invalid for classification; use <=0 for all classes or >=2")
    if n_eligible < 2:
        raise RuntimeError(f"need at least 2 eligible classes, got {n_eligible}")
    if requested <= 0:
        return n_eligible
    return min(int(requested), n_eligible)


def build_testtest_episode(
    pool: dict[str, list[str]], *, eligible_classes: list[str], n_way: int, n_shot: int,
    rng: random.Random,
) -> EpisodeSpec:
    if n_way == len(eligible_classes):
        classes = list(eligible_classes)
        rng.shuffle(classes)
    else:
        classes = rng.sample(eligible_classes, n_way)
        rng.shuffle(classes)

    query_class = rng.choice(classes)
    query_path = rng.choice(pool[query_class])
    support_rows: list[list[str]] = []
    replacement_flags: list[bool] = []

    for c in classes:
        # Strict defensive exclusion for ALL candidate classes, not only GT.
        candidates = [p for p in pool[c] if not _same_path(p, query_path)]
        if not candidates:
            raise RuntimeError(
                f"test-test exclusion left no support for class={c!r}, query={query_path}"
            )
        row, used_replacement = sample_k(candidates, n_shot, rng)
        if any(_same_path(p, query_path) for p in row):
            raise AssertionError("query leaked into test-test support")
        support_rows.append(row)
        replacement_flags.append(used_replacement)

    target_index = classes.index(query_class)
    return EpisodeSpec(classes, support_rows, query_path, query_class, target_index, replacement_flags)


def _open_transform(path: str, transform) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return transform(im)


def load_episode_tensors(spec: EpisodeSpec, transform) -> tuple[torch.Tensor, torch.Tensor]:
    support_flat = [p for row in spec.support_paths for p in row]
    support = torch.stack([_open_transform(p, transform) for p in support_flat], dim=0)
    query = torch.stack([_open_transform(spec.query_path, transform)], dim=0)
    return support, query


def _cfg(saved: dict[str, Any], name: str, default: Any) -> Any:
    value = saved.get(name, default)
    return default if value is None else value


def make_runtime_args(cli: argparse.Namespace, saved: dict[str, Any], *, n_way: int, n_shot: int) -> SimpleNamespace:
    # Only attributes used by the frozen distance pipeline / V10 feature builder are required.
    return SimpleNamespace(
        n_way=int(n_way), n_shot=int(n_shot), n_query=1,
        image_size=int(_cfg(saved, "image_size", 256)),
        hidden_dim=int(_cfg(saved, "hidden_dim", 256)),
        metric_dim=int(_cfg(saved, "metric_dim", 256)),
        dropout=float(_cfg(saved, "dropout", 0.0)),
        ot_pool_factor=int(_cfg(saved, "ot_pool_factor", 1)),
        sinkhorn_reg=float(_cfg(saved, "sinkhorn_reg", 0.1)),
        sinkhorn_iter=int(_cfg(saved, "sinkhorn_iter", 30)),
        sk_weight_eps=float(_cfg(saved, "sk_weight_eps", 1e-3)),
        num_steps=int(cli.num_steps if cli.num_steps is not None else _cfg(saved, "num_steps", 8)),
        solver=str(cli.solver if cli.solver is not None else _cfg(saved, "solver", "heun")),
        pair_chunk_size=int(cli.pair_chunk_size if cli.pair_chunk_size is not None else _cfg(saved, "pair_chunk_size", 2)),
        rae_chunk_size=int(cli.rae_chunk_size if cli.rae_chunk_size is not None else _cfg(saved, "rae_chunk_size", 8)),
        stochastic_hint=bool(_cfg(saved, "stochastic_hint", True)),
        pure_noise_hint=bool(_cfg(saved, "pure_noise_hint", False)),
        trajectory_score_mode="step_to_support",
        feature_eps=float(_cfg(saved, "feature_eps", 1e-6)),
        feature_ratio_clip=float(_cfg(saved, "feature_ratio_clip", 10.0)),
        correction_topk=int(cli.correction_topk if cli.correction_topk is not None else _cfg(saved, "correction_topk", 4)),
        dit_variant=str(_cfg(saved, "dit_variant", "L")),
        num_classes=int(_cfg(saved, "num_classes", 1000)),
        use_ema=bool(_cfg(saved, "use_ema", True)),
        hint_signal_ratio=saved.get("hint_signal_ratio", None),
        allow_condition_mismatch=bool(_cfg(saved, "allow_condition_mismatch", False)),
        amp=bool(cli.amp if cli.amp is not None else _cfg(saved, "amp", True)),
        utility_hidden_dim=int(_cfg(saved, "utility_hidden_dim", 384)),
        temporal_attn_heads=int(_cfg(saved, "temporal_attn_heads", 8)),
        temporal_attn_layers=int(_cfg(saved, "temporal_attn_layers", 3)),
        temporal_attn_ffn_dim=int(_cfg(saved, "temporal_attn_ffn_dim", 768)),
        class_attn_heads=int(_cfg(saved, "class_attn_heads", 8)),
        class_attn_layers=int(_cfg(saved, "class_attn_layers", 3)),
        class_attn_ffn_dim=int(_cfg(saved, "class_attn_ffn_dim", 768)),
        pairwise_hidden_dim=int(_cfg(saved, "pairwise_hidden_dim", 512)),
        budget_mlp_dim=int(_cfg(saved, "budget_mlp_dim", 256)),
        utility_head_dropout=float(_cfg(saved, "utility_head_dropout", 0.05)),
        max_correction_budget=float(_cfg(saved, "max_correction_budget", 4.0)),
        correction_budget_init=float(_cfg(saved, "correction_budget_init", 0.05)),
    )


def ci95(correct: int, total: int) -> float:
    if total <= 1:
        return 0.0
    p = float(correct) / float(total)
    return 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / float(total))


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    v = sum((x - m) ** 2 for x in values) / len(values)
    return m, math.sqrt(max(v, 0.0))


def _validate_device(d: torch.device):
    if d.type != "cuda":
        raise RuntimeError("This evaluator expects CUDA for the Flow runtime")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    idx = 0 if d.index is None else d.index
    if idx >= torch.cuda.device_count():
        raise RuntimeError(f"requested {d}, visible CUDA count={torch.cuda.device_count()}")


def load_runtime(cli: argparse.Namespace, mod, ckpt: dict[str, Any], *, n_way: int, n_shot: int):
    saved = dict(ckpt.get("args") or {})
    args = make_runtime_args(cli, saved, n_way=n_way, n_shot=n_shot)

    project_root = cli.project_root or saved.get("project_root")
    m2_checkpoint = cli.m2_checkpoint or ckpt.get("m2_checkpoint") or saved.get("m2_checkpoint")
    dit_checkpoint = cli.dit_checkpoint or ckpt.get("dit_checkpoint") or saved.get("dit_checkpoint")
    if not project_root:
        raise ValueError("project_root missing; pass --project_root")
    if not m2_checkpoint:
        raise ValueError("m2_checkpoint missing; pass --m2_checkpoint")
    if not dit_checkpoint:
        raise ValueError("dit_checkpoint missing; pass --dit_checkpoint")

    metric_device = torch.device(cli.metric_device)
    flow_device = torch.device(cli.flow_device)
    rae_device = torch.device(cli.rae_device)
    for d in (metric_device, flow_device, rae_device):
        _validate_device(d)

    # RAE
    torch.cuda.set_device(rae_device)
    rae = mod.build_rae(str(project_root), rae_device)
    if hasattr(rae, "eval"):
        rae.eval()
    if hasattr(rae, "parameters"):
        for p in rae.parameters():
            p.requires_grad_(False)

    # Flow / conditioning vision encoder.
    torch.cuda.set_device(flow_device)
    flow = mod.FrozenFlowRuntime(
        project_root=str(project_root), checkpoint=str(dit_checkpoint), device=flow_device,
        dit_variant=args.dit_variant, num_classes=args.num_classes, image_size=args.image_size,
        use_ema=args.use_ema, hint_signal_ratio=args.hint_signal_ratio,
        allow_condition_mismatch=args.allow_condition_mismatch, amp=args.amp,
    )
    args.hint_signal_ratio = flow.hint_signal_ratio

    # Frozen M2 metric. Prefer the exact metric state embedded in the F1 checkpoint.
    torch.cuda.set_device(metric_device)
    metric = mod.MetricEncoder(
        in_dim=1024, hidden_dim=args.hidden_dim, out_dim=args.metric_dim, dropout=args.dropout
    ).to(metric_device)
    metric_state = ckpt.get("metric_encoder")
    if metric_state is not None:
        missing, unexpected = metric.load_state_dict(metric_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"F1 checkpoint metric mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}")
    else:
        m2_ckpt = _torch_load(str(m2_checkpoint))
        mod.load_m2_metric_and_temperatures(metric, m2_ckpt, metric_device)
    metric.eval()
    for p in metric.parameters():
        p.requires_grad_(False)

    # Temperatures saved by F1 training are authoritative.
    if "shot_tau" in ckpt:
        shot_tau = torch.as_tensor(ckpt["shot_tau"], device=metric_device, dtype=torch.float32).reshape(())
    else:
        shot_tau = torch.tensor(0.10, device=metric_device, dtype=torch.float32)
    if "cls_tau" in ckpt:
        cls_tau = torch.as_tensor(ckpt["cls_tau"], device=metric_device, dtype=torch.float32).reshape(())
    else:
        m2_ckpt = _torch_load(str(m2_checkpoint))
        _, cls_tau = mod.load_m2_metric_and_temperatures(metric, m2_ckpt, metric_device)

    # V10 corrector.
    corrector = mod.PairwiseFlowTrajectoryCorrector(
        class_feature_dim=len(mod.V10_CLASS_FEATURE_NAMES), repr_dim=args.metric_dim,
        hidden_dim=args.utility_hidden_dim,
        temporal_heads=args.temporal_attn_heads, temporal_layers=args.temporal_attn_layers,
        temporal_ffn_dim=args.temporal_attn_ffn_dim,
        class_heads=args.class_attn_heads, class_layers=args.class_attn_layers,
        class_ffn_dim=args.class_attn_ffn_dim,
        pair_hidden_dim=args.pairwise_hidden_dim, budget_mlp_dim=args.budget_mlp_dim,
        dropout=args.utility_head_dropout,
        max_budget=args.max_correction_budget, budget_init=args.correction_budget_init,
    ).to(metric_device)
    state = ckpt.get("pairwise_flow_trajectory_corrector")
    if state is None:
        raise KeyError("F1 checkpoint lacks pairwise_flow_trajectory_corrector")
    corrector.load_state_dict(state, strict=True)
    corrector.eval()
    for p in corrector.parameters():
        p.requires_grad_(False)

    transform = mod.build_transform(args.image_size, augment=False)
    return args, metric, flow, rae, corrector, shot_tau, cls_tau, transform, metric_device, flow_device, rae_device, {
        "project_root": str(project_root),
        "m2_checkpoint": str(m2_checkpoint),
        "dit_checkpoint": str(dit_checkpoint),
    }


def evaluate_specs(
    specs: list[EpisodeSpec], *, mod, args, metric, flow, rae, corrector,
    shot_tau, cls_tau, transform, metric_device, rae_device,
    episode_batch_size: int, seed: int, save_episode_details: bool,
) -> dict[str, Any]:
    n_steps = int(args.num_steps)
    total = direct_correct = final_correct = 0
    step_correct = [0] * (n_steps + 1)
    oracle_correct = rescue = degradation = changed = 0
    gt_in_topk = wrong_gt_in_topk = direct_wrong = 0
    budgets: list[float] = []
    adjustment_abs: list[float] = []
    adjustment_max: list[float] = []
    temporal_sum = [0.0] * (n_steps + 1)
    temporal_count = 0
    details: list[dict[str, Any]] = []

    rng = random.Random(seed + 91_733)
    bs = max(int(episode_batch_size), 1)

    for start in tqdm(range(0, len(specs), bs), desc="F1-v10c eval"):
        batch_specs = specs[start:start + bs]
        support_imgs, query_imgs = [], []
        for spec in batch_specs:
            s, q = load_episode_tensors(spec, transform)
            support_imgs.append(s)
            query_imgs.append(q)
        support_img_cpu_batch = torch.stack(support_imgs, dim=0)
        query_img_cpu_batch = torch.stack(query_imgs, dim=0)
        current_bs = len(batch_specs)
        ns = int(args.n_way * args.n_shot)

        all_img = torch.cat([support_img_cpu_batch, query_img_cpu_batch], dim=1)
        flat = all_img.reshape(current_bs * (ns + 1), *all_img.shape[2:]).to(rae_device, non_blocking=True)
        with torch.no_grad():
            z = mod.encode_rae_chunked(rae, flat, args.rae_chunk_size)
        z = z.reshape(current_bs, ns + 1, *z.shape[1:])
        support_z_batch = z[:, :ns].detach()
        query_z_batch = z[:, ns:].detach()
        target = torch.tensor([x.target_index for x in batch_specs], device=metric_device, dtype=torch.long)
        hint_seed = rng.randrange(0, 2**31 - 1)

        with torch.no_grad():
            if current_bs == 1:
                support_distances, _, aux = mod.compute_episode_distances(
                    metric=metric, metric_device=metric_device, flow=flow,
                    support_z=support_z_batch[0], query_z=query_z_batch[0],
                    support_images_cpu=support_img_cpu_batch[0], episode_n_way=args.n_way,
                    args=args, hint_seed=hint_seed,
                )
            else:
                support_distances, _, aux = mod.compute_episode_batch_distances(
                    metric=metric, metric_device=metric_device, flow=flow,
                    support_z_batch=support_z_batch, query_z_batch=query_z_batch,
                    support_images_cpu_batch=support_img_cpu_batch, episode_n_way=args.n_way,
                    args=args, hint_seed=hint_seed,
                )
            support_scores, support_step_logits = mod.shot_softmin_class_scores(
                support_distances, shot_tau=shot_tau, cls_tau=cls_tau
            )
            direct_logits = mod.direct_support_logits(support_distances, shot_tau=shot_tau, cls_tau=cls_tau)
            class_features = mod.build_v10_class_trajectory_features(
                support_distances, support_scores, support_step_logits, direct_logits,
                eps=args.feature_eps, ratio_clip=args.feature_ratio_clip,
            )
            out = corrector(
                class_features, aux["support_class_repr"], aux["direct_query_repr"], aux["step_class_repr"],
                direct_logits, correction_topk=args.correction_topk,
            )

        final_logits = out["final_logits"]
        d_pred = direct_logits.argmax(dim=-1)
        f_pred = final_logits.argmax(dim=-1)
        d_ok = d_pred.eq(target)
        f_ok = f_pred.eq(target)
        step_preds = support_step_logits.argmax(dim=-1)  # B,V
        step_ok = step_preds.eq(target[:, None])
        oracle = step_ok.any(dim=1) | d_ok
        topk_gt = out["topk_mask"].gather(1, target[:, None]).squeeze(1)

        bsz = current_bs
        total += bsz
        direct_correct += int(d_ok.sum().item())
        final_correct += int(f_ok.sum().item())
        oracle_correct += int(oracle.sum().item())
        rescue += int(((~d_ok) & f_ok).sum().item())
        degradation += int((d_ok & (~f_ok)).sum().item())
        changed += int(f_pred.ne(d_pred).sum().item())
        direct_wrong += int((~d_ok).sum().item())
        gt_in_topk += int(topk_gt.sum().item())
        wrong_gt_in_topk += int(((~d_ok) & topk_gt).sum().item())
        for t in range(n_steps + 1):
            step_correct[t] += int(step_ok[:, t].sum().item())

        budgets.extend(out["budget"].detach().cpu().float().tolist())
        adjustment_abs.extend(out["adjustment"].abs().mean(dim=-1).detach().cpu().float().tolist())
        adjustment_max.extend(out["adjustment"].abs().amax(dim=-1).detach().cpu().float().tolist())
        ta = out["temporal_attention"].mean(dim=1).detach().cpu().float()  # B,V
        for row in ta.tolist():
            for t, value in enumerate(row):
                temporal_sum[t] += float(value)
            temporal_count += 1

        if save_episode_details:
            d_probs = F.softmax(direct_logits.float(), dim=-1).detach().cpu()
            f_probs = F.softmax(final_logits.float(), dim=-1).detach().cpu()
            topk_idx = out["topk_idx"].detach().cpu().tolist()
            adj = out["adjustment"].detach().cpu().float().tolist()
            budget_cpu = out["budget"].detach().cpu().float().tolist()
            ta_cpu = out["temporal_attention"].detach().cpu().float().tolist()
            d_pred_cpu = d_pred.detach().cpu().tolist()
            f_pred_cpu = f_pred.detach().cpu().tolist()
            step_pred_cpu = step_preds.detach().cpu().tolist()
            for i, spec in enumerate(batch_specs):
                details.append({
                    "episode_index": start + i,
                    "query_path": spec.query_path,
                    "query_class": spec.query_class,
                    "candidate_labels": spec.classes,
                    "support_paths": {c: row for c, row in zip(spec.classes, spec.support_paths)},
                    "support_used_replacement": {c: flag for c, flag in zip(spec.classes, spec.support_used_replacement)},
                    "target_index": spec.target_index,
                    "direct_prediction": spec.classes[d_pred_cpu[i]],
                    "final_prediction": spec.classes[f_pred_cpu[i]],
                    "step_predictions": [spec.classes[j] for j in step_pred_cpu[i]],
                    "direct_correct": bool(d_pred_cpu[i] == spec.target_index),
                    "final_correct": bool(f_pred_cpu[i] == spec.target_index),
                    "rescue": bool(d_pred_cpu[i] != spec.target_index and f_pred_cpu[i] == spec.target_index),
                    "degradation": bool(d_pred_cpu[i] == spec.target_index and f_pred_cpu[i] != spec.target_index),
                    "direct_probabilities": {c: float(v) for c, v in zip(spec.classes, d_probs[i].tolist())},
                    "final_probabilities": {c: float(v) for c, v in zip(spec.classes, f_probs[i].tolist())},
                    "topk_labels": [spec.classes[j] for j in topk_idx[i]],
                    "budget": float(budget_cpu[i]),
                    "adjustment": {c: float(v) for c, v in zip(spec.classes, adj[i])},
                    "temporal_attention": {
                        c: [float(v) for v in row] for c, row in zip(spec.classes, ta_cpu[i])
                    },
                })

        del support_img_cpu_batch, query_img_cpu_batch, support_z_batch, query_z_batch
        del support_distances, support_scores, support_step_logits, direct_logits, class_features, out, final_logits

    budget_mean, budget_std = _mean_std(budgets)
    adj_mean, _ = _mean_std(adjustment_abs)
    adj_max_mean, _ = _mean_std(adjustment_max)
    return {
        "n_queries": total,
        "direct_correct": direct_correct,
        "direct_accuracy_percent": 100.0 * direct_correct / max(total, 1),
        "direct_accuracy_ci95_percent": 100.0 * ci95(direct_correct, total),
        "final_correct": final_correct,
        "final_accuracy_percent": 100.0 * final_correct / max(total, 1),
        "final_accuracy_ci95_percent": 100.0 * ci95(final_correct, total),
        "gain_vs_direct_pp": 100.0 * (final_correct - direct_correct) / max(total, 1),
        "per_step_accuracy_percent": [100.0 * x / max(total, 1) for x in step_correct],
        "oracle_any_step_accuracy_percent": 100.0 * oracle_correct / max(total, 1),
        "rescue": rescue,
        "degradation": degradation,
        "net_rescue": rescue - degradation,
        "prediction_changed": changed,
        "override_rate_percent": 100.0 * changed / max(total, 1),
        "override_precision_percent": 100.0 * rescue / max(changed, 1),
        "gt_in_topk_rate_percent": 100.0 * gt_in_topk / max(total, 1),
        "direct_wrong_gt_in_topk_rate_percent": 100.0 * wrong_gt_in_topk / max(direct_wrong, 1),
        "budget_mean": budget_mean,
        "budget_std": budget_std,
        "adjustment_abs_mean": adj_mean,
        "adjustment_max_mean": adj_max_mean,
        "temporal_attention_mean_per_step": [x / max(temporal_count, 1) for x in temporal_sum],
        "episode_details": details if save_episode_details else None,
    }


def run_testtest(cli: argparse.Namespace) -> dict[str, Any]:
    mod, pretrain_path = _load_pretrain_module(cli.pretrain_py)
    ckpt = _torch_load(cli.f1_checkpoint)
    if not isinstance(ckpt, dict):
        raise ValueError("F1 checkpoint must be a dict")
    if hasattr(mod, "F1V10_STAGE") and ckpt.get("stage") != mod.F1V10_STAGE:
        raise ValueError(f"checkpoint stage={ckpt.get('stage')!r} is not F1-v10c stage={mod.F1V10_STAGE!r}")
    if ckpt.get("class_feature_names") not in (None, list(mod.V10_CLASS_FEATURE_NAMES)):
        raise ValueError("checkpoint class_feature_names do not match V10 evaluator")

    raw_pool = mod.load_class_pool(None, cli.test_class_json_root)
    pool = normalize_pool(raw_pool)
    eligible = sorted([c for c, xs in pool.items() if len(xs) >= 2])
    rejected = {c: len(xs) for c, xs in pool.items() if len(xs) < 2}
    n_way = effective_way(cli.n_way, len(eligible))

    saved = dict(ckpt.get("args") or {})
    n_shot = int(cli.n_shot if cli.n_shot is not None else _cfg(saved, "n_shot", 5))
    if n_shot < 1:
        raise ValueError("n_shot must be >=1")
    if cli.episodes < 1:
        raise ValueError("episodes must be >=1")

    print("=" * 120)
    print("[F1-v10c TEST-TEST] strict same-test-pool evaluation")
    print(f"pool={cli.test_class_json_root}")
    print(f"eligible classes={len(eligible)}, rejected singleton/empty={len(rejected)}, effective n_way={n_way}")
    print(f"episode={n_way}-way {n_shot}-shot 1-query x {cli.episodes}")
    print("support rule=exact query excluded; without replacement if enough else with replacement")
    print("=" * 120)

    rng = random.Random(cli.seed)
    specs = [
        build_testtest_episode(pool, eligible_classes=eligible, n_way=n_way, n_shot=n_shot, rng=rng)
        for _ in range(cli.episodes)
    ]
    # Absolute safety check before any GPU work.
    for spec in specs:
        for row in spec.support_paths:
            if any(_same_path(p, spec.query_path) for p in row):
                raise AssertionError(f"query-support leakage detected: {spec.query_path}")

    runtime = load_runtime(cli, mod, ckpt, n_way=n_way, n_shot=n_shot)
    args, metric, flow, rae, corrector, shot_tau, cls_tau, transform, metric_device, flow_device, rae_device, paths = runtime
    metrics = evaluate_specs(
        specs, mod=mod, args=args, metric=metric, flow=flow, rae=rae, corrector=corrector,
        shot_tau=shot_tau, cls_tau=cls_tau, transform=transform,
        metric_device=metric_device, rae_device=rae_device,
        episode_batch_size=cli.episode_batch_size, seed=cli.seed,
        save_episode_details=cli.save_episode_details,
    )

    result = {
        "stage": "f1-v10c-eval-test-test",
        "protocol": "test-test",
        "query_source": str(cli.test_class_json_root),
        "support_source": str(cli.test_class_json_root),
        "candidate_rule": "preserve eligible test classes; class requires >=2 distinct images; optional n_way cap",
        "support_sampling": "exclude exact query path; without_replacement_if_enough_else_with_replacement",
        "query_support_exact_overlap_allowed": False,
        "eligible_classes": eligible,
        "rejected_classes_lt2_images": rejected,
        "n_way_requested": int(cli.n_way),
        "n_way_eval": int(n_way),
        "n_shot": int(n_shot),
        "n_query": 1,
        "episodes_requested": int(cli.episodes),
        "seed": int(cli.seed),
        "num_steps": int(args.num_steps),
        "solver": args.solver,
        "correction_topk": int(args.correction_topk),
        "shot_tau": float(shot_tau.item()),
        "cls_tau": float(cls_tau.item()),
        "f1_checkpoint": str(cli.f1_checkpoint),
        "pretrain_py": pretrain_path,
        **paths,
        "metrics": metrics,
    }
    out = Path(cli.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "episode_details"}, ensure_ascii=False, indent=2))
    print(f"[saved] {out}")
    return result


def main() -> None:
    cli = parse_args()
    random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.seed)
    run_testtest(cli)


if __name__ == "__main__":
    main()
