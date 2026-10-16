#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline flat-JSON evaluator using the SAME adaptive F1-v5 inference core as the
new TEST-TEST F1-v5 evaluator.

Purpose
-------
Keep the existing pre-generated query/support flat JSON protocol used by the
three run_batch_* scripts, but replace the old fixed trajectory aggregation
with the learned adaptive F1-v5 branch:

    support-conditioned LSTM -> gamma_t -> monotonic state weights w_t
    -> weighted pair score -> K-shot soft-min -> class prediction

The model implementation is imported from the fixed TEST-TEST eval module, so the
F1 head, feature definition, downstream-M2 reload logic, RAE runtime and
DiT/Flow runtime are not reimplemented here.

Primary output = adaptive(F1-v5).
Direct / endpoint / uniform / fixed_decay are diagnostic branches only.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import random
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import torch

SCRIPT_VERSION = "2026-09-17-flatjson-adaptive-f1v5-testtest-v4-exactjson-dynamick-directtracefix"
PROTOCOL = "TEST-TEST"
EVAL_MODULE = "eval_f1_support_conditioned_lstm_v5_pretrain_m2_aligned_m2_dit"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------

def _json_safe(x: Any) -> Any:
    if x is None or isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if torch.is_tensor(x):
        if x.numel() == 1:
            return _json_safe(x.item())
        return [_json_safe(v) for v in x.detach().cpu().tolist()]
    if isinstance(x, Mapping):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return str(x)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_flat_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        rows = None
        for key in ("results", "records", "data", "items"):
            if isinstance(obj.get(key), list):
                rows = obj[key]
                break
        if rows is None:
            raise ValueError("JSON dict must contain a list in results/records/data/items")
    else:
        raise TypeError(f"unsupported JSON root type: {type(obj).__name__}")
    return [dict(r) for r in rows if isinstance(r, dict)]


def first_present(rec: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        v = rec.get(key)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, (list, tuple)) and not v:
            continue
        return v
    return None


def first_path(v: Any) -> Optional[str]:
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def resolve_path(v: Any, base_dir: Path) -> Optional[str]:
    s = first_path(v)
    if s is None:
        return None
    p = Path(s).expanduser()
    if not p.is_absolute():
        q = (base_dir / p).resolve()
        if q.exists():
            p = q
    return str(p)


@dataclass
class PairSpec:
    row_index: int
    query_path: str
    support_path: str
    matched_class: str
    ground_truth: Optional[str]


def parse_pairs(rows: Sequence[Mapping[str, Any]], data_path: str) -> list[PairSpec]:
    base = Path(data_path).expanduser().resolve().parent
    out: list[PairSpec] = []
    skipped = Counter()
    for i, rec in enumerate(rows):
        q = resolve_path(first_present(rec, ("x0", "x0_path", "target_image", "query_path", "query_image", "image_path")), base)
        s = resolve_path(first_present(rec, ("x1", "x1_path", "support_image", "support_path")), base)
        c = first_present(rec, ("label_B", "matched_class", "support_label", "class_name"))
        g = first_present(rec, ("label_A", "ground_truth", "ground_truth_name", "query_label", "label"))
        if q is None or s is None or c is None:
            skipped["missing_query_support_or_class"] += 1
            continue
        out.append(PairSpec(i, str(q), str(s), str(c), None if g is None else str(g)))
    if skipped:
        print(f"[input] skipped={dict(skipped)}")
    return out


@dataclass
class QuerySpec:
    query_path: str
    ground_truth: Optional[str]
    classes: list[str]
    supports: list[list[str]]
    selected_row_indices: list[list[int]]
    source_counts: dict[str, int]
    effective_n_shot: int


def build_query_specs(
    pairs: Sequence[PairSpec],
    *,
    n_shot: int,
    allow_shot_adjustment: bool = False,
) -> tuple[list[QuerySpec], dict[str, Any]]:
    """Build episodes from the EXISTING flat JSON without silently resampling.

    Default behaviour (allow_shot_adjustment=False):
      * keep every valid support row already present for each query x class;
      * do not truncate to --n_shot;
      * do not repeat supports to reach --n_shot;
      * require only that, within one query, all candidate classes have the same
        support count so the reference F1-v5 evaluator can form a rectangular
        [n_way, K] episode.

    Therefore --n_shot is only the requested/reference K in exact-JSON mode.
    The actual runtime K is inferred independently for every query.  Episodes are
    later bucketed by (n_way, effective_n_shot), and args.n_shot is set to the
    corresponding effective K immediately before inference.

    Debug behaviour (allow_shot_adjustment=True) retains the historical
    first-K/deterministic-repeat fallback to force every class to --n_shot.
    """
    grouped: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for p in pairs:
        g = grouped.setdefault(
            p.query_path,
            {"gt": p.ground_truth, "classes": OrderedDict()},
        )
        if g["gt"] is None and p.ground_truth is not None:
            g["gt"] = p.ground_truth
        g["classes"].setdefault(p.matched_class, []).append((p.support_path, p.row_index))

    queries: list[QuerySpec] = []
    count_dist = Counter()
    effective_k_dist = Counter()
    adjusted_groups = 0
    excluded_self = 0

    for q_index, (qpath, g) in enumerate(grouped.items()):
        classes: list[str] = []
        supports: list[list[str]] = []
        selected_rows: list[list[int]] = []
        source_counts: dict[str, int] = {}

        for cls, entries0 in g["classes"].items():
            # Defensive leakage guard: test-test requires query not to appear in
            # its own support set.  Apart from this exact self-path exclusion,
            # default exact-JSON mode preserves every existing support row.
            entries = [(p, idx) for p, idx in entries0 if os.path.realpath(p) != os.path.realpath(qpath)]
            excluded_self += len(entries0) - len(entries)
            if not entries:
                raise RuntimeError(f"query={qpath} class={cls}: no support remains after query-path exclusion")

            source_counts[str(cls)] = len(entries)
            count_dist[len(entries)] += 1

            if not allow_shot_adjustment:
                # Exact-JSON mode: use the complete pre-existing support set.
                chosen = list(entries)
            elif len(entries) == n_shot:
                chosen = list(entries)
            elif len(entries) > n_shot:
                chosen = list(entries[:n_shot])
                adjusted_groups += 1
            else:
                chosen = [entries[i % len(entries)] for i in range(n_shot)]
                adjusted_groups += 1

            classes.append(str(cls))
            supports.append([p for p, _ in chosen])
            selected_rows.append([idx for _, idx in chosen])

        if len(classes) < 2:
            raise RuntimeError(f"query={qpath}: fewer than 2 candidate classes: {classes}")

        per_class_k = [len(x) for x in supports]
        unique_k = sorted(set(per_class_k))
        if len(unique_k) != 1:
            count_map = {classes[i]: per_class_k[i] for i in range(len(classes))}
            raise RuntimeError(
                f"query={qpath}: existing JSON has unequal support counts across "
                f"candidate classes after self-exclusion: {count_map}. "
                "Exact-JSON mode will not truncate/repeat supports. The reference "
                "F1-v5 evaluator requires one rectangular K per query. Fix the "
                "upstream flat JSON, or explicitly use --allow_shot_adjustment if "
                "you accept first-K/repeat behaviour."
            )

        effective_k = int(unique_k[0])
        effective_k_dist[effective_k] += 1

        queries.append(QuerySpec(
            query_path=qpath,
            ground_truth=g["gt"],
            classes=classes,
            supports=supports,
            selected_row_indices=selected_rows,
            source_counts=source_counts,
            effective_n_shot=effective_k,
        ))

    diag = {
        "n_queries": len(queries),
        "requested_n_shot_cli": int(n_shot),
        "effective_n_shot_distribution_over_queries": {
            str(k): int(v) for k, v in sorted(effective_k_dist.items())
        },
        "query_class_source_row_count_distribution": {
            str(k): int(v) for k, v in sorted(count_dist.items())
        },
        "query_class_groups_adjusted_to_n_shot": int(adjusted_groups),
        "exact_query_support_self_rows_excluded": int(excluded_self),
        "selection_rule": (
            "all_existing_json_support_rows_no_truncate_no_repeat"
            if not allow_shot_adjustment
            else "force_requested_n_shot_first_K_or_deterministic_repeat"
        ),
        "allow_shot_adjustment": bool(allow_shot_adjustment),
    }
    return queries, diag


# -----------------------------------------------------------------------------
# Score helpers
# -----------------------------------------------------------------------------

def softmin_kshot(d: torch.Tensor, tau: torch.Tensor | float) -> torch.Tensor:
    # d: (..., K).  Mean-log-exp form; with fixed K this has the same ranking as
    # the usual logsumexp soft-min while preserving the distance scale better.
    t = torch.as_tensor(tau, device=d.device, dtype=d.dtype).clamp_min(1e-8)
    k = max(int(d.shape[-1]), 1)
    return -t * (torch.logsumexp(-d / t, dim=-1) - math.log(float(k)))


def ranking(class_distances: Mapping[str, float]) -> list[dict[str, Any]]:
    r = [{"class": str(k), "distance": float(v)} for k, v in class_distances.items() if math.isfinite(float(v))]
    r.sort(key=lambda x: x["distance"])
    return r


def pack_branch(classes: Sequence[str], dist: torch.Tensor) -> tuple[OrderedDict[str, float], list[dict[str, Any]], str, Optional[str], float]:
    vals = [float(x) for x in dist.detach().cpu().tolist()]
    cmap: "OrderedDict[str, float]" = OrderedDict((str(c), vals[i]) for i, c in enumerate(classes))
    r = ranking(cmap)
    top1 = r[0]["class"] if r else "未知"
    top2 = r[1]["class"] if len(r) > 1 else None
    gap = float(r[1]["distance"] - r[0]["distance"]) if len(r) > 1 else 999.0
    return cmap, r, top1, top2, gap


def mean_lists(items: Sequence[Sequence[float]]) -> list[float]:
    if not items:
        return []
    n = len(items[0])
    return [sum(float(v[i]) for v in items) / len(items) for i in range(n)]


# -----------------------------------------------------------------------------
# CLI / runtime
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Flat-JSON adaptive F1-v5 offline evaluator")
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--eval_code_dir", required=True, help="directory containing the new eval_f1_*.py files")
    p.add_argument("--project_root", required=True)
    p.add_argument("--f1_checkpoint", required=True)
    p.add_argument("--m2_checkpoint", required=True)
    p.add_argument("--dit_checkpoint", required=True)
    p.add_argument("--n_shot", type=int, default=5)
    p.add_argument("--n_query", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fixed_decay", type=float, default=0.90)
    p.add_argument("--trajectory_score_mode", choices=["step_to_support", "step_to_last"], default=None)
    p.add_argument("--episode_batch_size", type=int, default=4)
    p.add_argument("--pair_batch_size", type=int, default=16)
    p.add_argument("--rae_chunk_size", type=int, default=32)
    p.add_argument("--metric_device", default="cuda:0")
    p.add_argument("--flow_device", default="cuda:0")
    p.add_argument("--rae_device", default="cuda:0")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--stochastic_hint", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--pure_noise_hint", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument(
        "--allow_shot_adjustment", action="store_true",
        help=(
            "debug only: force every query-class to --n_shot using first-K/repeat. "
            "Default keeps all existing flat-JSON support rows and infers K per query."
        ),
    )
    return p.parse_args()


def device(s: str) -> torch.device:
    d = torch.device(s)
    if d.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {s}")
    return d


def build_eval_cli(a: argparse.Namespace) -> SimpleNamespace:
    # Fields consumed by the reference eval module's build_runtime_args/load_bundle.
    return SimpleNamespace(
        project_root=a.project_root,
        dit_checkpoint=a.dit_checkpoint,
        rae_chunk_size=int(a.rae_chunk_size),
        pair_batch_size=int(a.pair_batch_size),
        stochastic_hint=a.stochastic_hint,
        pure_noise_hint=a.pure_noise_hint,
        amp=a.amp,
        trajectory_score_mode=a.trajectory_score_mode,
        n_shot=int(a.n_shot),
        n_query=int(a.n_query),
        fixed_decay=float(a.fixed_decay),
        save_episode_details=False,
        log_every=int(a.log_every),
    )


# -----------------------------------------------------------------------------
# Main inference
# -----------------------------------------------------------------------------

def main() -> None:
    a = parse_args()
    if a.n_shot < 1 or a.n_query != 1:
        raise ValueError("this flat-JSON service requires n_shot>=1 and n_query=1")
    if not (0.0 < float(a.fixed_decay) <= 1.0):
        raise ValueError("fixed_decay must be in (0,1]")

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    code_dir = str(Path(a.eval_code_dir).expanduser().resolve())
    for pth in (code_dir, str(Path(a.project_root).expanduser().resolve()), str(Path(a.project_root).expanduser().resolve() / "fsl_multi_vae")):
        if pth not in sys.path:
            sys.path.insert(0, pth)

    ev = importlib.import_module(EVAL_MODULE)
    metric_device = device(a.metric_device)
    flow_device = device(a.flow_device)
    rae_device = device(a.rae_device)
    cli = build_eval_cli(a)

    for p in (a.data_path, a.f1_checkpoint, a.m2_checkpoint, a.dit_checkpoint):
        if not Path(p).is_file():
            raise FileNotFoundError(p)

    raw_rows = load_flat_json(a.data_path)
    pairs = parse_pairs(raw_rows, a.data_path)
    if not pairs:
        raise RuntimeError("no valid query/support pairs found")
    queries, support_diag = build_query_specs(
        pairs, n_shot=int(a.n_shot), allow_shot_adjustment=bool(a.allow_shot_adjustment)
    )

    # Exact reference loading order from the new eval files:
    # 1) F1-v5 TemporalDecayHead/config -> 2) downstream M2 reload
    # 3) RAE -> 4) explicit DiT/Flow reload.
    bundle = ev.load_bundle(
        a.f1_checkpoint,
        m2_path=a.m2_checkpoint,
        cli=cli,
        metric_device=metric_device,
    )
    args = bundle["args"]
    metric = bundle["metric"]
    head = bundle["head"]
    shot_tau = bundle["shot_tau"]
    cls_tau = bundle["cls_tau"]

    transform = ev.f1.build_transform(args.image_size, augment=False)

    if rae_device.type == "cuda":
        torch.cuda.set_device(rae_device)
    rae = ev.f1.build_rae(args.project_root, rae_device)
    rae.eval()
    for p in rae.parameters():
        p.requires_grad_(False)

    if flow_device.type == "cuda":
        torch.cuda.set_device(flow_device)
    flow = ev.f1.FrozenFlowRuntime(
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

    print("=" * 118)
    print("[ADAPTIVE F1-v5 FLAT-JSON OFFLINE EVAL]")
    print(f"version              = {SCRIPT_VERSION}")
    print(f"data_path            = {a.data_path}")
    print(f"protocol             = {PROTOCOL}")
    print(f"eval_module          = {EVAL_MODULE}")
    print(f"F1 checkpoint        = {a.f1_checkpoint}")
    print(f"M2 checkpoint        = {a.m2_checkpoint}")
    print(f"DiT checkpoint       = {a.dit_checkpoint}")
    print("load order           = F1-v5 -> M2 reload -> RAE -> DiT/Flow reload")
    print(f"trajectory score mode= {args.trajectory_score_mode}")
    print(f"requested n_shot     = {a.n_shot} (exact-JSON mode infers runtime K per query)")
    print(f"shot_tau             = {float(shot_tau.item()):.6f}")
    print(f"fixed_decay(diag)    = {a.fixed_decay}")
    print(f"support diagnostics  = {support_diag}")
    print("PRIMARY              = adaptive(F1-v5)")
    print("DIAGNOSTICS          = direct / endpoint / uniform / fixed_decay")
    print("=" * 118)

    # Prepare one EvalEpisode per pre-existing query.  Candidate order is the
    # first-seen class order in that query's flat JSON.
    episodes = []
    for q in queries:
        gt_idx = q.classes.index(q.ground_truth) if q.ground_truth in q.classes else 0
        episodes.append(ev.EvalEpisode(
            classes=list(q.classes),
            query_label=int(gt_idx),
            support_paths=[list(x) for x in q.supports],
            query_paths=[q.query_path],
        ))

    # Bucket by BOTH n_way and the effective K inferred from the existing JSON.
    # This allows, for example, some queries to be 1-shot and others 5-shot while
    # still keeping every query/support set unchanged.
    buckets: "OrderedDict[tuple[int, int], list[int]]" = OrderedDict()
    for i, ep in enumerate(episodes):
        k_shot = int(queries[i].effective_n_shot)
        buckets.setdefault((len(ep.classes), k_shot), []).append(i)

    query_results: list[Optional[dict[str, Any]]] = [None] * len(queries)
    row_updates: dict[int, dict[str, Any]] = {}
    correct = Counter()
    evaluable = 0
    rescue = 0
    degradation = 0
    gamma_all: list[list[float]] = []
    weights_all: list[list[float]] = []

    branches = ("adaptive", "direct", "endpoint", "uniform", "fixed_decay")
    global_batch_id = 0

    with torch.inference_mode():
        for (n_way, k_shot), idxs in buckets.items():
            # The imported reference evaluator uses args.n_shot for reshaping.
            # Set it to the exact K already present in this bucket; no support is
            # sampled, truncated, or repeated in default mode.
            args.n_shot = int(k_shot)
            bs = max(int(a.episode_batch_size), 1)
            for start in range(0, len(idxs), bs):
                sub_idx = idxs[start:start + bs]
                batch = [episodes[i] for i in sub_idx]
                current_bs = len(batch)

                support_imgs = []
                query_imgs = []
                for ep in batch:
                    s, qimg = ev.load_episode_images(ep, transform)
                    support_imgs.append(s)
                    query_imgs.append(qimg)
                support_cpu = torch.stack(support_imgs, dim=0)
                query_cpu = torch.stack(query_imgs, dim=0)

                ns = n_way * int(k_shot)
                if support_cpu.shape[1] != ns:
                    raise RuntimeError(
                        f"support count={support_cpu.shape[1]}, expected "
                        f"n_way*effective_n_shot={n_way}*{k_shot}={ns}"
                    )

                all_cpu = torch.cat([support_cpu, query_cpu], dim=1)
                flat = all_cpu.reshape(current_bs * (ns + args.n_query), *all_cpu.shape[2:]).to(rae_device, non_blocking=True)
                latent_flat = ev.f1.encode_rae_chunked(rae, flat, args.rae_chunk_size).detach()
                latent = latent_flat.reshape(current_bs, ns + args.n_query, *latent_flat.shape[1:])
                support_z = latent[:, :ns]
                query_z = latent[:, ns:]

                hint_seed = int(a.seed + global_batch_id * 1_000_003 + 97) % (2**31 - 1)
                global_batch_id += 1
                if current_bs == 1:
                    support_distances, selected_distances, aux = ev.f1.compute_episode_distances(
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
                    support_distances, selected_distances, aux = ev.f1.compute_episode_batch_distances(
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

                step_class_scores, step_logits = ev.f1.shot_softmin_class_scores(
                    selected_distances, shot_tau=shot_tau, cls_tau=cls_tau
                )
                direct_logits = ev.f1.direct_support_logits(
                    support_distances, shot_tau=shot_tau, cls_tau=cls_tau
                )
                direct_class_scores = softmin_kshot(support_distances.float(), shot_tau)

                features = ev.f1.build_observable_step_features(
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
                state_weights = ev.f1.decay_gamma_to_state_weights(gamma)

                adaptive_pair, adaptive_class, adaptive_logits = ev.f1.weighted_pair_class_logits(
                    selected_distances,
                    state_weights,
                    shot_tau=shot_tau,
                    cls_tau=cls_tau,
                )

                qn = int(adaptive_logits.shape[0])
                v = int(selected_distances.shape[-1])
                uniform_w = torch.full((qn, v), 1.0 / float(v), device=metric_device)
                fixed_one = ev.normalized_fixed_weights(v, a.fixed_decay, metric_device)
                fixed_w = fixed_one.unsqueeze(0).expand(qn, -1)
                uniform_pair, uniform_class, uniform_logits = ev.f1.weighted_pair_class_logits(
                    selected_distances, uniform_w, shot_tau=shot_tau, cls_tau=cls_tau
                )
                fixed_pair, fixed_class, fixed_logits = ev.f1.weighted_pair_class_logits(
                    selected_distances, fixed_w, shot_tau=shot_tau, cls_tau=cls_tau
                )

                # Reference evaluator indexes step_logits[:, -1], therefore step
                # dimension is axis=1.  Use the matching class-score tensor when
                # available; otherwise recover an equivalent ranking from logits.
                if torch.is_tensor(step_class_scores) and step_class_scores.ndim >= 3:
                    endpoint_class = step_class_scores[:, -1]
                else:
                    endpoint_class = -step_logits[:, -1].float() * torch.as_tensor(cls_tau, device=metric_device)
                endpoint_logits = step_logits[:, -1]
                endpoint_pair = selected_distances[..., -1]

                # Use the reference logits as the authoritative class ordering.
                # In the F1 evaluator logits are distance logits (-score/cls_tau);
                # converting back avoids depending on the internal layout of the
                # auxiliary class-score return value while preserving the exact
                # prediction and class-gap scale up to the saved cls_tau.
                cls_tau_t = torch.as_tensor(cls_tau, device=metric_device, dtype=torch.float32).clamp_min(1e-8)
                branch_class = {
                    "adaptive": -adaptive_logits.float() * cls_tau_t,
                    "direct": -direct_logits.float() * cls_tau_t,
                    "endpoint": -endpoint_logits.float() * cls_tau_t,
                    "uniform": -uniform_logits.float() * cls_tau_t,
                    "fixed_decay": -fixed_logits.float() * cls_tau_t,
                }
                branch_logits = {
                    "adaptive": adaptive_logits,
                    "direct": direct_logits,
                    "endpoint": endpoint_logits,
                    "uniform": uniform_logits,
                    "fixed_decay": fixed_logits,
                }
                branch_pair = {
                    "adaptive": adaptive_pair,
                    "direct": support_distances,
                    "endpoint": endpoint_pair,
                    "uniform": uniform_pair,
                    "fixed_decay": fixed_pair,
                }

                for local_i, qi in enumerate(sub_idx):
                    qspec = queries[qi]
                    qres: dict[str, Any] = {
                        "query_index": qi,
                        "query_path": qspec.query_path,
                        "ground_truth": qspec.ground_truth,
                        "candidate_labels": list(qspec.classes),
                        "n_way": len(qspec.classes),
                        "n_shot": int(qspec.effective_n_shot),
                        "requested_n_shot_cli": int(a.n_shot),
                        "support_paths": qspec.supports,
                        "source_support_row_counts": qspec.source_counts,
                        "gamma": [float(x) for x in gamma[local_i].detach().cpu().tolist()],
                        "state_weights": [float(x) for x in state_weights[local_i].detach().cpu().tolist()],
                        "branches": {},
                    }
                    gamma_all.append(qres["gamma"])
                    weights_all.append(qres["state_weights"])

                    for name in branches:
                        cdist, rr, top1, top2, gap = pack_branch(qspec.classes, branch_class[name][local_i])
                        qres["branches"][name] = {
                            "class_distances": cdist,
                            "ranking": rr,
                            "top1": top1,
                            "top2": top2,
                            "gap": gap,
                            "correct": None if qspec.ground_truth is None else bool(top1 == qspec.ground_truth),
                        }

                    if qspec.ground_truth is not None and qspec.ground_truth in qspec.classes:
                        evaluable += 1
                        for name in branches:
                            correct[name] += int(qres["branches"][name]["top1"] == qspec.ground_truth)
                        d_ok = qres["branches"]["direct"]["top1"] == qspec.ground_truth
                        a_ok = qres["branches"]["adaptive"]["top1"] == qspec.ground_truth
                        rescue += int((not d_ok) and a_ok)
                        degradation += int(d_ok and (not a_ok))

                    # Pair-level values for the support rows selected into this
                    # K-shot episode.  This preserves traceability back to the flat JSON.
                    for class_slot in range(n_way):
                        for shot_slot in range(int(qspec.effective_n_shot)):
                            row_index = qspec.selected_row_indices[class_slot][shot_slot]
                            upd = row_updates.setdefault(row_index, {"selected_slots": []})
                            # ``support_distances`` keeps the complete to-support
                            # trajectory for each pair, i.e. its trailing dimension is
                            # the 9 Flow states (t=0..8).  The Direct branch is exactly
                            # the t=0 query-to-support distance, so indexing only
                            # [batch, class, shot] leaves a 9-element vector and cannot
                            # be converted with .item().  Record state 0 explicitly.
                            direct_pair_trace = branch_pair["direct"][local_i, class_slot, shot_slot]
                            if direct_pair_trace.numel() == 1:
                                direct_pair_value = float(direct_pair_trace.item())
                            else:
                                direct_pair_value = float(direct_pair_trace.reshape(-1)[0].item())

                            upd["selected_slots"].append({
                                "query_index": qi,
                                "class_slot": class_slot,
                                "shot_slot": shot_slot,
                                "adaptive_pair_distance": float(branch_pair["adaptive"][local_i, class_slot, shot_slot].item()),
                                "direct_pair_distance": direct_pair_value,
                                "endpoint_pair_distance": float(branch_pair["endpoint"][local_i, class_slot, shot_slot].item()),
                                "uniform_pair_distance": float(branch_pair["uniform"][local_i, class_slot, shot_slot].item()),
                                "fixed_decay_pair_distance": float(branch_pair["fixed_decay"][local_i, class_slot, shot_slot].item()),
                            })

                    query_results[qi] = qres

                if a.log_every > 0 and global_batch_id % int(a.log_every) == 0:
                    print(
                        f"[batch {global_batch_id}] n_way={n_way} k_shot={k_shot} size={current_bs} "
                        f"gamma_mean={float(gamma.mean().item()):.4f} "
                        f"w_last={float(state_weights[:, -1].mean().item()):.4f}"
                    )

                del support_cpu, query_cpu, all_cpu, flat, latent_flat, latent, support_z, query_z
                del support_distances, selected_distances, step_class_scores, step_logits, direct_logits
                del features, gamma, state_weights, adaptive_pair, adaptive_class, adaptive_logits
                del uniform_pair, uniform_class, uniform_logits, fixed_pair, fixed_class, fixed_logits

    final_results = [x for x in query_results if x is not None]
    if len(final_results) != len(queries):
        raise RuntimeError("internal error: missing query results")

    acc_percent = {
        name: (100.0 * correct[name] / evaluable if evaluable else 0.0)
        for name in branches
    }
    effective_k_values = [int(q.effective_n_shot) for q in queries]
    effective_k_dist = Counter(effective_k_values)
    unique_effective_k = sorted(effective_k_dist)

    summary = {
        "script_version": SCRIPT_VERSION,
        "primary_model": "adaptive_f1_v5",
        "primary_definition": "support-conditioned LSTM -> gamma_t -> monotonic state weights -> weighted pair score -> K-shot soft-min",
        "protocol": PROTOCOL,
        "eval_module": EVAL_MODULE,
        "data_path": a.data_path,
        "n_queries": len(final_results),
        "n_evaluable_queries": evaluable,
        # Backward-compatible n_shot field is populated only when all queries
        # share one exact K.  Otherwise inspect effective_n_shot_distribution.
        "n_shot": int(unique_effective_k[0]) if len(unique_effective_k) == 1 else None,
        "requested_n_shot_cli": int(a.n_shot),
        "effective_n_shot_distribution": {
            str(k): int(v) for k, v in sorted(effective_k_dist.items())
        },
        "fixed_decay_diagnostic": float(a.fixed_decay),
        "trajectory_score_mode": args.trajectory_score_mode,
        "accuracy_percent": acc_percent,
        "correct": {name: int(correct[name]) for name in branches},
        "rescue_vs_direct": int(rescue),
        "degradation_vs_direct": int(degradation),
        "net_rescue_vs_direct": int(rescue - degradation),
        "gamma_t_mean": mean_lists(gamma_all),
        "state_weight_mean": mean_lists(weights_all),
        "support_diagnostics": support_diag,
        "weights": {
            "f1_checkpoint": a.f1_checkpoint,
            "m2_checkpoint": a.m2_checkpoint,
            "dit_checkpoint": a.dit_checkpoint,
            "load_order": ["F1-v5 head/config", "downstream M2 reload", "RAE", "DiT/Flow reload"],
            "shot_tau_runtime": float(shot_tau.detach().cpu().item()),
            "cls_tau_runtime": float(torch.as_tensor(cls_tau).detach().cpu().item()),
        },
    }

    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "query_results.json", {"summary": summary, "results": final_results})
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "summary_adaptive_f1v5.json", summary)

    enriched_rows = []
    for i, row in enumerate(raw_rows):
        r = dict(row)
        if i in row_updates:
            r["adaptive_f1v5"] = row_updates[i]
        enriched_rows.append(r)
    save_json(output_dir / "result.json", enriched_rows)

    # Fusion-compatible view.  Adaptive F1-v5 is the trajectory/main FSL branch;
    # Direct remains separately exposed exactly as before.
    fusion_records = []
    uniform_records = []
    for qres in final_results:
        ad = qres["branches"]["adaptive"]
        dr = qres["branches"]["direct"]
        un = qres["branches"]["uniform"]

        def fusion_record(main_branch: Mapping[str, Any], score_key: str) -> dict[str, Any]:
            rr = main_branch["ranking"]
            rr_d = dr["ranking"]
            fsl_full_result = {
                "trajectory_score_key": score_key,
                "top1": main_branch["top1"],
                "top2": main_branch["top2"],
                "gap": main_branch["gap"],
                "distance": rr[0]["distance"] if rr else None,
                "ranking": rr,
                "class_min": main_branch["class_distances"],
                "class_min_distances": main_branch["class_distances"],
                "class_distances": main_branch["class_distances"],
                "top1_direct": dr["top1"],
                "top2_direct": dr["top2"],
                "gap_direct": dr["gap"],
                "distance_direct": rr_d[0]["distance"] if rr_d else None,
                "ranking_direct": rr_d,
                "class_min_direct": dr["class_distances"],
                "class_min_distances_direct": dr["class_distances"],
                "class_distances_direct": dr["class_distances"],
                "valid": bool(rr),
                "gamma": qres["gamma"],
                "state_weights": qres["state_weights"],
            }
            return {
                "image_id": os.path.basename(qres["query_path"]),
                "image_path": qres["query_path"],
                "target_image": qres["query_path"],
                "ground_truth": qres["ground_truth"],
                "ok": True,
                "class_distances": main_branch["class_distances"],
                "class_min_distances": main_branch["class_distances"],
                "class_distances_direct": dr["class_distances"],
                "class_min_distances_direct": dr["class_distances"],
                "results": [],
                "fsl_trajectory_score_key": score_key,
                "fsl_full_result": fsl_full_result,
            }

        fusion_records.append(fusion_record(ad, "adaptive_f1_v5"))
        uniform_records.append(fusion_record(un, "uniform_state_weighting"))

    fusion_adaptive = {
        "results": fusion_records,
        "summary": {
            "n_queries": len(fusion_records),
            "contains_traj": True,
            "contains_direct": True,
            "primary_model": "adaptive_f1_v5",
            "trajectory_score_key": "adaptive_f1_v5",
            "support_aggregation": "K-shot softmin from reference F1-v5 evaluator",
            "format": "offline_flatjson_adaptive_f1v5_fusion_compatible_v1",
        },
    }
    fusion_uniform = {
        "results": uniform_records,
        "summary": {
            "n_queries": len(uniform_records),
            "contains_traj": True,
            "contains_direct": True,
            "primary_model": "uniform_state_weighting_diagnostic",
            "trajectory_score_key": "uniform_state_weighting",
            "support_aggregation": "K-shot softmin from reference F1-v5 evaluator",
            "format": "offline_flatjson_adaptive_f1v5_fusion_compatible_v1",
        },
    }
    save_json(output_dir / "fsl_full_results_for_fusion_adaptive.json", fusion_adaptive)
    # Legacy alias: downstream code that previously consumed *_dynamic.json can
    # keep doing so, but the file explicitly declares adaptive_f1_v5 semantics.
    save_json(output_dir / "fsl_full_results_for_fusion_dynamic.json", fusion_adaptive)
    save_json(output_dir / "fsl_full_results_for_fusion_uniform.json", fusion_uniform)

    print("\n" + "=" * 118)
    print("[DONE] adaptive F1-v5 flat-JSON evaluation")
    print(f"Adaptive F1-v5 : {acc_percent['adaptive']:.4f}% ({correct['adaptive']}/{evaluable})")
    print(f"Direct          : {acc_percent['direct']:.4f}% ({correct['direct']}/{evaluable})")
    print(f"Endpoint        : {acc_percent['endpoint']:.4f}%")
    print(f"Uniform         : {acc_percent['uniform']:.4f}%")
    print(f"Fixed({a.fixed_decay:g})     : {acc_percent['fixed_decay']:.4f}%")
    print(f"Rescue/Degrade/Net vs Direct = {rescue}/{degradation}/{rescue-degradation}")
    print(f"gamma_t mean    = {[round(x, 5) for x in summary['gamma_t_mean']]}")
    print(f"state w mean    = {[round(x, 5) for x in summary['state_weight_mean']]}")
    print(f"output_dir      = {output_dir}")
    print("=" * 118)


if __name__ == "__main__":
    main()
