#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean M2 Direct + F1-v5 adaptive Flow + rule router evaluator (Test-Train).

Query is external/test and support is internal/train. Only Direct, F1 adaptive Flow,
and their original no-GT four-rule router are evaluated.

shot_tau is ALWAYS the constant 0.1 and is NEVER read from any checkpoint.
"""

from __future__ import annotations

PATCH_TAG = "20260827_m2_f1_flow_router_testtrain_clean_v1"

import wass_flow.eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_testtrain_m2_dit as base

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import torch

import pretrain_f1_support_conditioned_lstm_v5_classjson_torchrun_ddp_batch as f1
from models.metric_encoder import MetricEncoder

ROUTER_SOFTMAX_TEMPERATURE = 1.0
ROUTER_MIN_PASSED_RULES = 3
ROUTER_REQUIRE_PAIRWISE_DOMINANCE = True
ROUTER_MARGIN_GAP = 0.0
ROUTER_ENTROPY_GAP = 0.0
ROUTER_PAIRWISE_DOMINANCE_GAP = 0.0
ROUTER_TOP1_PROB_GAP = 0.0
ROUTER_EPS = 1e-12
FIXED_SHOT_TAU = 0.10


def parse_kv(items: list[str], *, name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{name} must use KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not value:
            raise ValueError(f"invalid {name}: {item!r}")
        out[key] = value
    return out


def _device(text: str) -> torch.device:
    d = torch.device(text)
    if d.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable, requested {d}")
        idx = 0 if d.index is None else int(d.index)
        if idx >= torch.cuda.device_count():
            raise RuntimeError(f"requested {d}, visible CUDA count={torch.cuda.device_count()}")
    return d


def _load_cls_tau_only(m2_ckpt: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Load ONLY cls_tau from M2. shot_tau is never read from the checkpoint."""
    if m2_ckpt.get("log_cls_tau") is not None:
        return torch.as_tensor(
            m2_ckpt["log_cls_tau"], dtype=torch.float32, device=device
        ).reshape(()).exp().detach()
    if m2_ckpt.get("cls_tau") is not None:
        return torch.as_tensor(
            m2_ckpt["cls_tau"], dtype=torch.float32, device=device
        ).reshape(()).detach()
    raise KeyError("M2 checkpoint lacks log_cls_tau/cls_tau")


def load_bundle(
    f1_path: str,
    *,
    m2_path: str,
    cli: argparse.Namespace,
    metric_device: torch.device,
) -> dict[str, Any]:
    # 1) F1-v5: only TemporalDecayHead + its architecture/config.
    f1_ckpt = f1._torch_load(f1_path, "cpu")
    if not isinstance(f1_ckpt, dict):
        raise ValueError(f"F1 checkpoint must be a dict: {f1_path}")
    if f1_ckpt.get("temporal_decay_head") is None:
        raise KeyError(f"F1 checkpoint lacks temporal_decay_head: {f1_path}")

    feature_names = list(f1_ckpt.get("feature_names") or [])
    if feature_names and feature_names != list(f1.FEATURE_NAMES):
        raise RuntimeError(
            "F1 feature definition mismatch.\n"
            f"checkpoint={feature_names}\ncurrent={list(f1.FEATURE_NAMES)}"
        )

    args = base.build_runtime_args(cli, f1_ckpt)
    head = f1.TemporalDecayHead(
        stat_feature_dim=len(f1.FEATURE_NAMES),
        repr_dim=args.metric_dim,
        hidden_dim=args.utility_hidden_dim,
        set_dim=args.utility_proj_dim,
        condition_dim=args.utility_condition_dim,
        head_dropout=args.utility_head_dropout,
        gamma_min=args.gamma_min,
        gamma_init=args.gamma_init,
    ).to(metric_device)
    head.load_state_dict(f1_ckpt["temporal_decay_head"], strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    # 2) Downstream M2 MetricEncoder + cls_tau.
    # IMPORTANT: shot_tau is NOT loaded from M2/F1 checkpoint. It is always 0.1.
    m2_ckpt = f1._torch_load(m2_path, "cpu")
    if not isinstance(m2_ckpt, dict):
        raise ValueError(f"M2 checkpoint must be a dict: {m2_path}")
    base._apply_m2_runtime_overrides(args, m2_ckpt)

    metric = MetricEncoder(
        in_dim=1024,
        hidden_dim=args.hidden_dim,
        out_dim=args.metric_dim,
        dropout=args.dropout,
    ).to(metric_device)
    metric_state = f1.extract_metric_state(m2_ckpt)
    missing, unexpected = metric.load_state_dict(metric_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "M2 MetricEncoder mismatch.\n"
            f"missing={missing[:20]}\nunexpected={unexpected[:20]}"
        )
    metric.eval()
    for p in metric.parameters():
        p.requires_grad_(False)

    shot_tau = torch.tensor(FIXED_SHOT_TAU, dtype=torch.float32, device=metric_device)
    cls_tau = _load_cls_tau_only(m2_ckpt, metric_device)

    return {
        "f1_path": f1_path,
        "m2_path": m2_path,
        "f1_ckpt": f1_ckpt,
        "m2_ckpt": m2_ckpt,
        "args": args,
        "metric": metric,
        "head": head,
        "shot_tau": shot_tau,
        "cls_tau": cls_tau,
    }


def _router_softmax_from_distance(scores: dict[str, float]) -> dict[str, float]:
    if len(scores) < 2:
        return {}
    classes = sorted(scores)
    vals = [float(scores[c]) for c in classes]
    mean_v = sum(vals) / len(vals)
    var_v = sum((v - mean_v) ** 2 for v in vals) / len(vals)
    std_v = math.sqrt(max(var_v, 1e-12))
    temp = max(float(ROUTER_SOFTMAX_TEMPERATURE), 1e-8)
    logits = [-(v - mean_v) / std_v / temp for v in vals]
    m = max(logits)
    exps = [math.exp(max(min(x - m, 60.0), -60.0)) for x in logits]
    den = sum(exps)
    return {c: exps[i] / den for i, c in enumerate(classes)} if den > 0 else {}


def _router_entropy(prob: dict[str, float]) -> float:
    if len(prob) < 2:
        return float("nan")
    h = -sum(p * math.log(max(p, 1e-12)) for p in prob.values())
    return h / math.log(len(prob))


def _router_top2(prob: dict[str, float]):
    ordered = sorted(prob.items(), key=lambda kv: kv[1], reverse=True)
    if len(ordered) < 2:
        raise ValueError("Router requires >=2 candidate classes")
    pred, p1 = ordered[0]
    p2 = ordered[1][1]
    return pred, float(p1), float(p2), float(p1 - p2)


def _router_js(p: dict[str, float], q: dict[str, float]) -> float:
    classes = sorted(set(p) & set(q))
    if len(classes) < 2:
        return float("nan")
    m = {c: 0.5 * (p[c] + q[c]) for c in classes}

    def kl(a, b):
        return sum(
            max(a[c], 1e-12)
            * (math.log(max(a[c], 1e-12)) - math.log(max(b[c], 1e-12)))
            for c in classes
        )

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def route_without_gt(direct_scores: dict[str, float], flow_scores: dict[str, float]) -> dict[str, Any]:
    """Same four-rule no-GT Direct-safe router as the original eval_m2 router."""
    direct_prob = _router_softmax_from_distance(direct_scores)
    flow_prob = _router_softmax_from_distance(flow_scores)
    if not direct_prob or not flow_prob:
        raise ValueError("cannot construct router probabilities from class distances")

    direct_pred, direct_p1, direct_p2, direct_margin = _router_top2(direct_prob)
    flow_pred, flow_p1, flow_p2, flow_margin = _router_top2(flow_prob)
    direct_entropy = _router_entropy(direct_prob)
    flow_entropy = _router_entropy(flow_prob)
    agree = direct_pred == flow_pred

    flow_reject_direct = flow_prob[flow_pred] - flow_prob.get(direct_pred, 0.0)
    direct_reject_flow = direct_prob[direct_pred] - direct_prob.get(flow_pred, 0.0)
    pairwise_dominance = flow_reject_direct - direct_reject_flow

    rules = {
        "flow_margin_better": flow_margin > direct_margin + ROUTER_MARGIN_GAP + ROUTER_EPS,
        "flow_entropy_lower": flow_entropy < direct_entropy - ROUTER_ENTROPY_GAP - ROUTER_EPS,
        "flow_pairwise_dominant": pairwise_dominance > ROUTER_PAIRWISE_DOMINANCE_GAP + ROUTER_EPS,
        "flow_top1_prob_higher": flow_p1 > direct_p1 + ROUTER_TOP1_PROB_GAP + ROUTER_EPS,
    }
    passed_rules = sum(int(v) for v in rules.values())
    override = (
        (not agree)
        and passed_rules >= ROUTER_MIN_PASSED_RULES
        and ((not ROUTER_REQUIRE_PAIRWISE_DOMINANCE) or rules["flow_pairwise_dominant"])
    )
    final_pred = flow_pred if override else direct_pred
    return {
        "direct_pred": direct_pred,
        "flow_pred": flow_pred,
        "final_pred": final_pred,
        "agree": bool(agree),
        "override": bool(override),
        "passed_rules": int(passed_rules),
        "router_score": passed_rules / float(len(rules)),
        "rules": rules,
        "direct_margin": direct_margin,
        "direct_entropy": direct_entropy,
        "flow_margin": flow_margin,
        "flow_entropy": flow_entropy,
        "pairwise_dominance": pairwise_dominance,
        "js_divergence": _router_js(direct_prob, flow_prob),
        "direct_prob": direct_prob,
        "flow_prob": flow_prob,
    }


def _ci95(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    p = correct / total
    return 100.0 * 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / total)


@torch.inference_mode()
def evaluate_dataset(
    *,
    plans,
    bundle: dict[str, Any],
    rae,
    flow: f1.FrozenFlowRuntime,
    transform,
    metric_device: torch.device,
    rae_device: torch.device,
    cli: argparse.Namespace,
    dataset_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    args = bundle["args"]
    metric: MetricEncoder = bundle["metric"]
    head: f1.TemporalDecayHead = bundle["head"]
    shot_tau: torch.Tensor = bundle["shot_tau"]
    cls_tau: torch.Tensor = bundle["cls_tau"]

    counters = {
        "total": 0,
        "direct_correct": 0,
        "flow_correct": 0,
        "final_correct": 0,
        "oracle_correct": 0,
        "agree": 0,
        "override": 0,
        "override_correct": 0,
        "rescue": 0,
        "degradation": 0,
        "oracle_rescuable": 0,
    }
    rule_pass = {
        "flow_margin_better": 0,
        "flow_entropy_lower": 0,
        "flow_pairwise_dominant": 0,
        "flow_top1_prob_higher": 0,
    }
    gamma_sum = [0.0 for _ in range(args.num_steps)]
    weight_sum = [0.0 for _ in range(args.num_steps + 1)]
    passed_rule_sum = 0
    details: list[dict[str, Any]] = []

    ep_bs = max(int(cli.episode_batch_size), 1)
    n_batches = int(math.ceil(len(plans) / ep_bs))

    for batch_idx in range(n_batches):
        batch = plans[batch_idx * ep_bs : (batch_idx + 1) * ep_bs]
        current_bs = len(batch)
        n_way = len(batch[0].classes)
        if any(len(ep.classes) != n_way for ep in batch):
            raise RuntimeError("mixed n_way inside one eval batch")

        support_imgs = []
        query_imgs = []
        for ep in batch:
            s, q = base.load_episode_images(ep, transform)
            support_imgs.append(s)
            query_imgs.append(q)

        support_cpu = torch.stack(support_imgs, dim=0)
        query_cpu = torch.stack(query_imgs, dim=0)
        ns = n_way * int(args.n_shot)
        if support_cpu.shape[1] != ns:
            raise RuntimeError(f"support count={support_cpu.shape[1]}, expected {ns}")

        all_cpu = torch.cat([support_cpu, query_cpu], dim=1)
        flat = all_cpu.reshape(
            current_bs * (ns + args.n_query), *all_cpu.shape[2:]
        ).to(rae_device, non_blocking=True)
        latent_flat = f1.encode_rae_chunked(rae, flat, args.rae_chunk_size).detach()
        latent = latent_flat.reshape(
            current_bs, ns + args.n_query, *latent_flat.shape[1:]
        )
        support_z = latent[:, :ns]
        query_z = latent[:, ns:]

        hint_seed = int(dataset_seed + batch_idx * 1_000_003 + 97) % (2**31 - 1)
        if current_bs == 1:
            support_distances, selected_distances, aux = f1.compute_episode_distances(
                metric=metric,
                metric_device=metric_device,
                flow=flow,
                support_z=support_z[0],
                query_z=query_z[0],
                support_images_cpu=support_cpu[0],
                episode_n_way=n_way,
                args=args,
                hint_seed=hint_seed,
            )
        else:
            support_distances, selected_distances, aux = f1.compute_episode_batch_distances(
                metric=metric,
                metric_device=metric_device,
                flow=flow,
                support_z_batch=support_z,
                query_z_batch=query_z,
                support_images_cpu_batch=support_cpu,
                episode_n_way=n_way,
                args=args,
                hint_seed=hint_seed,
            )

        # M2 Direct. shot_tau is the hard-coded constant 0.1.
        direct_logits = f1.direct_support_logits(
            support_distances, shot_tau=shot_tau, cls_tau=cls_tau
        )

        # F1 Flow: observable features -> TemporalDecayHead -> gamma_t -> state weights
        # -> weighted trajectory pair distances -> K-shot soft-min.
        _, step_logits = f1.shot_softmin_class_scores(
            selected_distances, shot_tau=shot_tau, cls_tau=cls_tau
        )
        features = f1.build_observable_step_features(
            selected_distances,
            step_logits,
            direct_logits,
            eps=args.feature_eps,
            ratio_clip=args.feature_ratio_clip,
        )
        _, gamma = head(
            features,
            aux["support_class_repr"],
            aux["direct_query_repr"],
            aux["step_class_repr"],
        )
        state_weights = f1.decay_gamma_to_state_weights(gamma)
        _, flow_class_score, flow_logits = f1.weighted_pair_class_logits(
            selected_distances,
            state_weights,
            shot_tau=shot_tau,
            cls_tau=cls_tau,
        )

        # Direct class score in the same lower-is-better scale.
        direct_class_score = -direct_logits.float() * cls_tau.float()

        qn = int(direct_logits.shape[0])
        for t in range(args.num_steps):
            gamma_sum[t] += float(gamma[:, t].sum().item())
        for t in range(args.num_steps + 1):
            weight_sum[t] += float(state_weights[:, t].sum().item())

        for local_i, ep in enumerate(batch):
            # Current aligned protocol has n_query=1; keep indexing explicit.
            for qlocal in range(int(args.n_query)):
                qi = local_i * int(args.n_query) + qlocal
                labels = list(ep.classes)
                gt = labels[int(ep.query_label)]
                direct_scores = {
                    c: float(direct_class_score[qi, j].item())
                    for j, c in enumerate(labels)
                }
                flow_scores = {
                    c: float(flow_class_score[qi, j].item())
                    for j, c in enumerate(labels)
                }
                routed = route_without_gt(direct_scores, flow_scores)

                d_ok = routed["direct_pred"] == gt
                f_ok = routed["flow_pred"] == gt
                r_ok = routed["final_pred"] == gt
                oracle_ok = d_ok or f_ok
                rescue = (not d_ok) and r_ok
                degradation = d_ok and (not r_ok)
                oracle_rescuable = (not d_ok) and f_ok

                counters["total"] += 1
                counters["direct_correct"] += int(d_ok)
                counters["flow_correct"] += int(f_ok)
                counters["final_correct"] += int(r_ok)
                counters["oracle_correct"] += int(oracle_ok)
                counters["agree"] += int(routed["agree"])
                counters["override"] += int(routed["override"])
                counters["override_correct"] += int(routed["override"] and r_ok)
                counters["rescue"] += int(rescue)
                counters["degradation"] += int(degradation)
                counters["oracle_rescuable"] += int(oracle_rescuable)
                passed_rule_sum += int(routed["passed_rules"])
                for key, passed in routed["rules"].items():
                    rule_pass[key] += int(passed)

                if cli.save_episode_details:
                    details.append({
                        "episode_index": batch_idx * ep_bs + local_i,
                        "query_index": qlocal,
                        "query_path": ep.query_paths[qlocal],
                        "gt": gt,
                        "candidate_labels": labels,
                        "support_paths_by_class": {
                            c: list(ep.support_paths[j]) for j, c in enumerate(labels)
                        },
                        "direct_pred": routed["direct_pred"],
                        "flow_pred": routed["flow_pred"],
                        "final_pred": routed["final_pred"],
                        "direct_correct": bool(d_ok),
                        "flow_correct": bool(f_ok),
                        "final_correct": bool(r_ok),
                        "override": routed["override"],
                        "is_rescue": bool(rescue),
                        "is_degradation": bool(degradation),
                        "passed_rules": routed["passed_rules"],
                        "rules": routed["rules"],
                        "direct_margin": routed["direct_margin"],
                        "direct_entropy": routed["direct_entropy"],
                        "flow_margin": routed["flow_margin"],
                        "flow_entropy": routed["flow_entropy"],
                        "pairwise_dominance": routed["pairwise_dominance"],
                        "direct_class_distances": direct_scores,
                        "flow_class_distances": flow_scores,
                        "gamma": [float(x) for x in gamma[qi].detach().cpu().tolist()],
                        "state_weights": [float(x) for x in state_weights[qi].detach().cpu().tolist()],
                    })

        if cli.log_every > 0 and (batch_idx + 1) % int(cli.log_every) == 0:
            print(
                f"  [eval {batch_idx+1:04d}/{n_batches}] "
                f"gamma_mean={float(gamma.mean().item()):.4f} "
                f"w_last={float(state_weights[:, -1].mean().item()):.4f}"
            )

        del support_cpu, query_cpu, all_cpu, flat, latent_flat, latent
        del support_z, query_z, support_distances, selected_distances
        del direct_logits, direct_class_score, step_logits, features
        del gamma, state_weights, flow_class_score, flow_logits

    total = counters["total"]
    denom = max(total, 1)
    pct = lambda x: 100.0 * x / denom
    override_n = counters["override"]
    oracle_rescuable = counters["oracle_rescuable"]

    metrics = {
        "n_queries": total,
        "direct_correct": counters["direct_correct"],
        "flow_correct": counters["flow_correct"],
        "final_correct": counters["final_correct"],
        "oracle_correct": counters["oracle_correct"],
        "direct_accuracy_percent": pct(counters["direct_correct"]),
        "flow_accuracy_percent": pct(counters["flow_correct"]),
        "final_accuracy_percent": pct(counters["final_correct"]),
        "oracle_direct_flow_accuracy_percent": pct(counters["oracle_correct"]),
        "direct_ci95_percent": _ci95(counters["direct_correct"], total),
        "flow_ci95_percent": _ci95(counters["flow_correct"], total),
        "final_ci95_percent": _ci95(counters["final_correct"], total),
        "oracle_ci95_percent": _ci95(counters["oracle_correct"], total),
        "gain_vs_direct_pp": pct(counters["final_correct"] - counters["direct_correct"]),
        "oracle_gain_vs_direct_pp": pct(counters["oracle_correct"] - counters["direct_correct"]),
        "agree_count": counters["agree"],
        "agree_rate_percent": pct(counters["agree"]),
        "override_count": override_n,
        "override_rate_percent": pct(override_n),
        "override_precision_percent": (
            100.0 * counters["override_correct"] / override_n if override_n else 0.0
        ),
        "rescue_count": counters["rescue"],
        "degradation_count": counters["degradation"],
        "net_rescue": counters["rescue"] - counters["degradation"],
        "oracle_rescuable_count": oracle_rescuable,
        "rescue_recall_percent": (
            100.0 * counters["rescue"] / oracle_rescuable if oracle_rescuable else 0.0
        ),
        "mean_passed_rules": passed_rule_sum / denom,
        "rule_pass_count": rule_pass,
        "rule_pass_rate_percent": {k: 100.0 * v / denom for k, v in rule_pass.items()},
        "gamma_t_mean": [x / denom for x in gamma_sum],
        "state_weight_t_mean": [x / denom for x in weight_sum],
        "shot_tau": FIXED_SHOT_TAU,
        "router_config": {
            "softmax_temperature": ROUTER_SOFTMAX_TEMPERATURE,
            "min_passed_rules": ROUTER_MIN_PASSED_RULES,
            "require_pairwise_dominance": ROUTER_REQUIRE_PAIRWISE_DOMINANCE,
            "margin_gap": ROUTER_MARGIN_GAP,
            "entropy_gap": ROUTER_ENTROPY_GAP,
            "pairwise_dominance_gap": ROUTER_PAIRWISE_DOMINANCE_GAP,
            "top1_prob_gap": ROUTER_TOP1_PROB_GAP,
        },
        "direct_score": "M2 Direct D(z0,support) -> K-shot soft-min",
        "flow_score": "F1-v5 predicted gamma_t -> state_weights -> weighted trajectory -> K-shot soft-min",
    }
    return metrics, details


def macro_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    keys = [
        "direct_accuracy_percent",
        "flow_accuracy_percent",
        "final_accuracy_percent",
        "oracle_direct_flow_accuracy_percent",
        "gain_vs_direct_pp",
        "override_rate_percent",
    ]
    out = {key: sum(float(r["metrics"][key]) for r in results) / len(results) for key in keys}
    out["datasets"] = len(results)
    out["net_rescue_total"] = sum(int(r["metrics"]["net_rescue"]) for r in results)
    return out


def write_csv(output_json: Path, results: list[dict[str, Any]]) -> Path:
    csv_path = output_json.with_suffix(".csv")
    fields = [
        "hospital", "organ", "n_way_eval", "n_shot", "n_queries",
        "direct_accuracy_percent", "flow_accuracy_percent", "final_accuracy_percent",
        "oracle_direct_flow_accuracy_percent", "gain_vs_direct_pp",
        "override_count", "override_rate_percent", "rescue_count",
        "degradation_count", "net_rescue", "rescue_recall_percent",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields)
        writer.writeheader()
        for r in results:
            m = r["metrics"]
            writer.writerow({
                "hospital": r.get("hospital"),
                "organ": r.get("organ"),
                "n_way_eval": r.get("n_way_eval"),
                "n_shot": r.get("n_shot"),
                "n_queries": m["n_queries"],
                "direct_accuracy_percent": m["direct_accuracy_percent"],
                "flow_accuracy_percent": m["flow_accuracy_percent"],
                "final_accuracy_percent": m["final_accuracy_percent"],
                "oracle_direct_flow_accuracy_percent": m["oracle_direct_flow_accuracy_percent"],
                "gain_vs_direct_pp": m["gain_vs_direct_pp"],
                "override_count": m["override_count"],
                "override_rate_percent": m["override_rate_percent"],
                "rescue_count": m["rescue_count"],
                "degradation_count": m["degradation_count"],
                "net_rescue": m["net_rescue"],
                "rescue_recall_percent": m["rescue_recall_percent"],
            })
    return csv_path

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("M2 Direct + F1 adaptive Flow router | Test-Train")
    p.add_argument("--query_base", type=str, required=True)
    p.add_argument("--organ_support", nargs="+", required=True)
    p.add_argument("--hospitals", nargs="+", default=["自贡", "苍溪", "南充"])
    p.add_argument("--expected_hospitals", type=int, default=3)
    p.add_argument("--organs", nargs="+", default=["食管", "胃", "十二指肠"])
    p.add_argument("--expected_organs", type=int, default=3)
    p.add_argument("--f1_checkpoint", type=str, default=None)
    p.add_argument("--organ_f1_checkpoint", nargs="*", default=[])
    p.add_argument("--m2_checkpoint", type=str, default=None)
    p.add_argument("--organ_m2_checkpoint", nargs="*", default=[])
    p.add_argument("--project_root", type=str, default=None)
    p.add_argument("--dit_checkpoint", type=str, default=None)
    p.add_argument("--trajectory_score_mode", choices=["step_to_support", "step_to_last"], default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_way", type=int, default=None)
    p.add_argument("--n_shot", type=int, default=5)
    p.add_argument("--n_query", type=int, default=1)
    p.add_argument("--episode_batch_size", type=int, default=4)
    p.add_argument("--pair_batch_size", type=int, default=8)
    p.add_argument("--rae_chunk_size", type=int, default=32)
    p.add_argument("--metric_device", type=str, default="cuda:0")
    p.add_argument("--flow_device", type=str, default="cuda:0")
    p.add_argument("--rae_device", type=str, default="cuda:0")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--stochastic_hint", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--pure_noise_hint", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--save_episode_details", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--log_cuda_memory", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    cli = parse_args()
    if cli.episodes < 1 or cli.n_shot < 1 or cli.n_query < 1:
        raise ValueError("episodes/n_shot/n_query must be >=1")
    if cli.episode_batch_size < 1 or cli.pair_batch_size < 1 or cli.rae_chunk_size < 1:
        raise ValueError("batch/chunk sizes must be >=1")

    base.setup_seed(cli.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    metric_device = _device(cli.metric_device)
    flow_device = _device(cli.flow_device)
    rae_device = _device(cli.rae_device)

    support_map = parse_kv(cli.organ_support, name="--organ_support")
    sources = base.resolve_sources(cli, support_map)
    f1_map = parse_kv(cli.organ_f1_checkpoint, name="--organ_f1_checkpoint")
    m2_map = parse_kv(cli.organ_m2_checkpoint, name="--organ_m2_checkpoint")

    required_pairs: list[tuple[str, str]] = []
    for src in sources:
        f1_path = base.checkpoint_for_source(src, organ_map=f1_map, fallback=cli.f1_checkpoint)
        m2_path = base.m2_checkpoint_for_source(src, organ_map=m2_map, fallback=cli.m2_checkpoint)
        pair = (f1_path, m2_path)
        if pair not in required_pairs:
            required_pairs.append(pair)
        if not Path(f1_path).is_file():
            raise FileNotFoundError(f"F1 checkpoint not found: {f1_path}")
        if not Path(m2_path).is_file():
            raise FileNotFoundError(f"M2 checkpoint not found: {m2_path}")

    bundles = {
        pair: load_bundle(pair[0], m2_path=pair[1], cli=cli, metric_device=metric_device)
        for pair in required_pairs
    }
    first_args = bundles[required_pairs[0]]["args"]
    for pair, bundle in bundles.items():
        a = bundle["args"]
        if not (
            a.project_root == first_args.project_root
            and a.dit_checkpoint == first_args.dit_checkpoint
            and a.image_size == first_args.image_size
            and a.dit_variant == first_args.dit_variant
            and a.num_steps == first_args.num_steps
            and a.solver == first_args.solver
        ):
            raise RuntimeError(f"F1/M2 runtime mismatch at pair={pair}")

    transform = f1.build_transform(first_args.image_size, augment=False)
    if rae_device.type == "cuda":
        torch.cuda.set_device(rae_device)
    rae = f1.build_rae(first_args.project_root, rae_device).eval()
    for p in rae.parameters():
        p.requires_grad_(False)

    if flow_device.type == "cuda":
        torch.cuda.set_device(flow_device)
    flow = f1.FrozenFlowRuntime(
        project_root=first_args.project_root,
        checkpoint=first_args.dit_checkpoint,
        device=flow_device,
        dit_variant=first_args.dit_variant,
        num_classes=first_args.num_classes,
        image_size=first_args.image_size,
        use_ema=first_args.use_ema,
        hint_signal_ratio=first_args.hint_signal_ratio,
        allow_condition_mismatch=first_args.allow_condition_mismatch,
        amp=first_args.amp,
    )
    for bundle in bundles.values():
        bundle["args"].hint_signal_ratio = flow.hint_signal_ratio

    print("=" * 110)
    print("[M2 DIRECT + F1 ADAPTIVE FLOW ROUTER | TEST-TRAIN]")
    print(f"shot_tau = {FIXED_SHOT_TAU} (hard-coded; checkpoint value ignored/not read)")
    print("Flow = F1 gamma_t -> state_weights -> trajectory score")
    print(f"DiT = {first_args.dit_checkpoint}")
    print("=" * 110)

    output = Path(cli.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for src in sources:
        f1_path = base.checkpoint_for_source(src, organ_map=f1_map, fallback=cli.f1_checkpoint)
        m2_path = base.m2_checkpoint_for_source(src, organ_map=m2_map, fallback=cli.m2_checkpoint)
        bundle = bundles[(f1_path, m2_path)]
        args = bundle["args"]
        query_pool, support_pool, labels = base.eligible_test_train_pools(
            src["query_path"], src["support_path"],
            n_shot=cli.n_shot, n_query=cli.n_query,
        )
        n_way = len(labels) if cli.n_way is None else min(int(cli.n_way), len(labels))
        dataset_seed = base.task_seed(cli.seed, src.get("hospital"), src.get("organ"))
        plans = base.generate_episodes(
            query_pool, support_pool, labels,
            n_way=n_way, n_shot=cli.n_shot, n_query=cli.n_query,
            episodes=cli.episodes, seed=dataset_seed,
        )
        tag = f"{src.get('hospital')}-{src.get('organ')}"
        print(f"\n[DATASET] {tag} | {n_way}-way {cli.n_shot}-shot | seed={dataset_seed}")
        metrics, details = evaluate_dataset(
            plans=plans, bundle=bundle, rae=rae, flow=flow, transform=transform,
            metric_device=metric_device, rae_device=rae_device,
            cli=cli, dataset_seed=dataset_seed,
        )
        print(
            f"RESULT {tag} | Direct={metrics['direct_accuracy_percent']:.2f}% "
            f"| F1-Flow={metrics['flow_accuracy_percent']:.2f}% "
            f"| Final={metrics['final_accuracy_percent']:.2f}% "
            f"| Oracle={metrics['oracle_direct_flow_accuracy_percent']:.2f}% "
            f"| R/D/N={metrics['rescue_count']}/{metrics['degradation_count']}/{metrics['net_rescue']}"
        )
        print("  gamma_t = " + str([round(x, 4) for x in metrics["gamma_t_mean"]]))
        print("  state_w = " + str([round(x, 4) for x in metrics["state_weight_t_mean"]]))

        one = {
            "stage": "m2-f1-flow-router-testtrain-v1",
            "hospital": src.get("hospital"),
            "organ": src.get("organ"),
            "query_path": src["query_path"],
            "support_path": src["support_path"],
            "f1_checkpoint": f1_path,
            "m2_checkpoint": m2_path,
            "dit_checkpoint": args.dit_checkpoint,
            "trajectory_score_mode": args.trajectory_score_mode,
            "shot_tau": FIXED_SHOT_TAU,
            "seed": dataset_seed,
            "n_way_eval": n_way,
            "n_shot": cli.n_shot,
            "n_query": cli.n_query,
            "episodes": cli.episodes,
            "num_steps": args.num_steps,
            "solver": args.solver,
            "metrics": metrics,
        }
        if cli.save_episode_details:
            one["episode_details"] = details
        results.append(one)

    summary = {
        "stage": "m2-f1-flow-router-testtrain-v1",
        "dataset_count": len(results),
        "shot_tau": FIXED_SHOT_TAU,
        "shot_tau_source": "hard-coded constant; never loaded from checkpoint",
        "flow_definition": "F1-v5 gamma_t -> state_weights -> weighted trajectory score",
        "macro": macro_summary(results),
        "datasets": results,
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = write_csv(output, results)
    print(f"\nSaved JSON -> {output}")
    print(f"Saved CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
