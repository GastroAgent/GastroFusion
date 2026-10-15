#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate an F1-v5 ImageNet-pretrained support-conditioned LSTM on class-folder datasets.

Data protocol (test-test, aligned to eval_no_f1_five_trajectory_baselines_organ_models_decay_m2_aligned.py):

    batch_root / hospital / organ / class / image

For every hospital-organ dataset, query and support are sampled from the SAME class-folder pool,
while the query image itself is excluded from every support set.  Candidate way uses all actual query classes by default. Classes are NOT dropped merely
for having fewer than K+1 images; after excluding the query, undersized support pools are sampled
with replacement. --n_way remains an explicit optional cap/override.

Weight loading order is explicit:
  1) load the F1-v5 TemporalDecayHead/config from the F1 checkpoint;
  2) RELOAD the requested downstream M2 MetricEncoder + shot_tau / cls_tau afterwards;
  3) build RAE;
  4) RELOAD the requested DiT/Flow runtime last.
The MetricEncoder / temperatures embedded in the F1 checkpoint are therefore NOT the final evaluation M2 weights.

Inference is deterministic with respect to the requested seed and batch layout.  There is NO
relative exploration, no random gamma perturbation, no GT feature, and no oracle decision in the
F1 forward path.

Main F1 score:
    support-conditioned LSTM -> gamma_t -> cumulative monotonic state weights w_t
    -> weighted pair score -> frozen K-shot soft-min -> prediction.

For diagnosis the script also reports Endpoint, Uniform trajectory weighting, and one optional
fixed exponential decay.  These baselines use the exact same frozen trajectory evidence.

NOTE(nodirect): Direct is NOT reported.  Its logits are still computed internally only because
the trained decay head's observable features depend on them (model-input dependency); they are
excluded from every output table/metric.
"""

from __future__ import annotations

PATCH_TAG = "20261014_f1_eval_nodirect_v1"

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image

import pretrain_f1_support_conditioned_lstm_v5_classjson_torchrun_ddp_batch as f1
from models.metric_encoder import MetricEncoder


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_HOSPITALS = ["华西", "自贡", "苍溪", "南充"]
DEFAULT_ORGANS = ["食管", "胃", "十二指肠"]


@dataclass
class EvalEpisode:
    classes: list[str]
    query_label: int
    support_paths: list[list[str]]
    query_paths: list[str]


@dataclass
class EvalAccumulator:
    queries: int
    episodes: int
    correct: dict[str, int]
    ce_sum: dict[str, float]
    rescue: int
    degradation: int
    oracle_any: int
    per_step_correct: list[int]
    gamma_sum: list[float]
    weight_sum: list[float]
    gamma_count: int
    episode_final_correct: list[int]


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "F1-v5 pretrained support-conditioned LSTM evaluator on hospital/organ/class/image pools"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--batch_root", type=str)
    src.add_argument("--data_root", type=str)

    p.add_argument("--hospitals", nargs="+", default=DEFAULT_HOSPITALS)
    p.add_argument("--expected_hospitals", type=int, default=4)
    p.add_argument("--organs", nargs="+", default=DEFAULT_ORGANS)
    p.add_argument("--expected_organs", type=int, default=3)

    # F1-v5 checkpoint selection.  Prefer per-organ checkpoints when ImageNet pretraining
    # was run once per downstream M2/organ.  --f1_checkpoint is a single fallback.
    p.add_argument("--f1_checkpoint", type=str, default=None)
    p.add_argument(
        "--organ_f1_checkpoint",
        nargs="*",
        default=[],
        help="Per-organ F1 checkpoints: 食管=/a.pt 胃=/b.pt 十二指肠=/c.pt",
    )

    # Downstream M2 is deliberately reloaded AFTER the F1-v5 head.
    # This prevents the frozen MetricEncoder / temperatures stored inside the F1
    # pretraining checkpoint from silently becoming the final evaluation M2.
    p.add_argument("--m2_checkpoint", type=str, default=None)
    p.add_argument(
        "--organ_m2_checkpoint",
        nargs="*",
        default=[],
        help="Per-organ downstream M2 checkpoints reloaded after F1: 食管=/a.pt 胃=/b.pt 十二指肠=/c.pt",
    )

    p.add_argument("--project_root", type=str, default=None)
    p.add_argument("--dit_checkpoint", type=str, default=None)
    p.add_argument("--trajectory_score_mode", choices=["step_to_support", "step_to_last"], default=None)

    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_n_way", type=int, default=8, help="Legacy compatibility only; ignored unless you explicitly use --n_way")
    p.add_argument("--n_way", type=int, default=None, help="Optional fixed cap/way override per dataset")
    p.add_argument("--n_shot", type=int, default=5)
    p.add_argument("--n_query", type=int, default=1)

    # Real inference batching.  Does not alter episode semantics.
    p.add_argument("--episode_batch_size", type=int, default=4)
    p.add_argument("--pair_batch_size", type=int, default=16)
    p.add_argument("--rae_chunk_size", type=int, default=32)

    p.add_argument("--metric_device", type=str, default="cuda:0")
    p.add_argument("--flow_device", type=str, default="cuda:0")
    p.add_argument("--rae_device", type=str, default="cuda:0")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--stochastic_hint", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--pure_noise_hint", action=argparse.BooleanOptionalAction, default=None)

    # Diagnostic fixed-decay baseline.  Learned F1 remains the main evaluated method.
    p.add_argument("--fixed_decay", type=float, default=0.90)
    p.add_argument("--save_episode_details", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--log_cuda_memory", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _device(text: str) -> torch.device:
    d = torch.device(text)
    if d.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable, requested {d}")
        idx = 0 if d.index is None else int(d.index)
        if idx >= torch.cuda.device_count():
            raise RuntimeError(f"requested {d}, visible CUDA count={torch.cuda.device_count()}")
    return d


def setup_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def task_seed(base_seed: int, hospital: Optional[str], organ: Optional[str]) -> int:
    if hospital is None or organ is None:
        return int(base_seed)
    return (
        int(base_seed)
        + sum(ord(c) for c in str(hospital)) * 1009
        + sum(ord(c) for c in str(organ)) * 9176
    )


def scan_class_folders(root: str | Path) -> dict[str, list[str]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"class-folder root not found: {root}")
    result: dict[str, list[str]] = {}
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        paths = sorted(str(p) for p in child.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        # Stable duplicate removal.
        seen: set[str] = set()
        unique: list[str] = []
        for path in paths:
            key = os.path.normcase(os.path.abspath(os.path.normpath(os.path.expanduser(path))))
            if key not in seen:
                seen.add(key)
                unique.append(path)
        if unique:
            result[child.name] = unique
    if not result:
        raise RuntimeError(f"no class images found under {root}")
    return result


def eligible_test_test_pool(root: str | Path, n_shot: int) -> tuple[dict[str, list[str]], list[str]]:
    """Keep every query class that can provide at least one non-query support image.

    For test-test, query and support come from the same pool.  We therefore require
    only two distinct images/class when n_query=1: one may be the query and at least
    one different image must remain as support.  If fewer than n_shot support images
    remain in an episode, generate_episodes() samples them with replacement.
    """
    pool = scan_class_folders(root)
    labels = [c for c in sorted(pool) if len(pool[c]) >= 2]
    singleton = {c: len(pool[c]) for c in sorted(pool) if len(pool[c]) < 2}
    if singleton:
        print(
            f"[WARN] singleton classes cannot be evaluated leakage-safely in test-test and are excluded: {singleton}"
        )
    if len(labels) < 2:
        raise RuntimeError(
            f"fewer than 2 leakage-safe test-test classes in {root}; "
            "each class needs at least 2 distinct images (query + one non-query support)"
        )
    return {c: pool[c] for c in labels}, labels


def generate_episodes(
    pool: dict[str, list[str]],
    labels: list[str],
    *,
    n_way: int,
    n_shot: int,
    n_query: int,
    episodes: int,
    seed: int,
) -> list[EvalEpisode]:
    if int(n_query) != 1:
        raise ValueError("This M2-aligned test-test evaluator currently requires n_query=1")
    rng = random.Random(int(seed))
    plans: list[EvalEpisode] = []
    for _ in range(int(episodes)):
        selected = rng.sample(labels, int(n_way))
        query_label_name = rng.choice(selected)
        query_path = rng.choice(pool[query_label_name])
        query_key = os.path.normcase(os.path.abspath(os.path.normpath(os.path.expanduser(query_path))))

        support_by_label: dict[str, list[str]] = {}
        for label in selected:
            candidates = []
            for path in pool[label]:
                key = os.path.normcase(os.path.abspath(os.path.normpath(os.path.expanduser(path))))
                if key != query_key:
                    candidates.append(path)
            if not candidates:
                raise RuntimeError(
                    f"class={label!r}: no non-query support remains after excluding query={query_path}"
                )
            if len(candidates) >= int(n_shot):
                support_by_label[label] = rng.sample(candidates, int(n_shot))
            else:
                # Preserve the class: use leakage-safe replacement sampling from
                # the remaining non-query images until exactly K supports are drawn.
                support_by_label[label] = rng.choices(candidates, k=int(n_shot))

        # Match the M2-aligned evaluator: randomly shuffle candidate slots after sampling.
        rng.shuffle(selected)
        plans.append(
            EvalEpisode(
                classes=list(selected),
                query_label=int(selected.index(query_label_name)),
                support_paths=[list(support_by_label[c]) for c in selected],
                query_paths=[str(query_path)],
            )
        )
    return plans


def load_episode_images(ep: EvalEpisode, transform) -> tuple[torch.Tensor, torch.Tensor]:
    support: list[torch.Tensor] = []
    for class_paths in ep.support_paths:
        for path in class_paths:
            with Image.open(path) as im:
                support.append(transform(im.convert("RGB")))
    query: list[torch.Tensor] = []
    for path in ep.query_paths:
        with Image.open(path) as im:
            query.append(transform(im.convert("RGB")))
    return torch.stack(support, dim=0), torch.stack(query, dim=0)


def resolve_sources(cli: argparse.Namespace) -> list[dict[str, Any]]:
    if cli.data_root:
        return [{"path": cli.data_root, "hospital": None, "organ": None}]
    hospitals = list(cli.hospitals)
    organs = list(cli.organs)
    if cli.expected_hospitals > 0 and len(hospitals) != int(cli.expected_hospitals):
        raise RuntimeError(f"expected {cli.expected_hospitals} hospitals, got {hospitals}")
    if cli.expected_organs > 0 and len(organs) != int(cli.expected_organs):
        raise RuntimeError(f"expected {cli.expected_organs} organs, got {organs}")
    out: list[dict[str, Any]] = []
    for hospital in hospitals:
        for organ in organs:
            path = Path(cli.batch_root) / hospital / organ
            if not path.is_dir():
                raise FileNotFoundError(f"dataset directory does not exist: {path}")
            out.append({"path": str(path), "hospital": hospital, "organ": organ})
    return out


def checkpoint_for_source(
    src: dict[str, Any],
    *,
    organ_map: dict[str, str],
    fallback: Optional[str],
) -> str:
    organ = src.get("organ")
    if organ is not None and organ in organ_map:
        return organ_map[organ]
    if fallback:
        return fallback
    raise ValueError(
        f"No F1-v5 checkpoint configured for organ={organ!r}. Use --organ_f1_checkpoint or --f1_checkpoint."
    )


def m2_checkpoint_for_source(
    src: dict[str, Any],
    *,
    organ_map: dict[str, str],
    fallback: Optional[str],
) -> str:
    organ = src.get("organ")
    if organ is not None and organ in organ_map:
        return organ_map[organ]
    if fallback:
        return fallback
    raise ValueError(
        f"No downstream M2 checkpoint configured for organ={organ!r}. "
        "Use --organ_m2_checkpoint or --m2_checkpoint."
    )


def _saved(ckpt: dict[str, Any], key: str, default: Any = None) -> Any:
    return (ckpt.get("args") or {}).get(key, default)


def build_runtime_args(cli: argparse.Namespace, ckpt: dict[str, Any]) -> SimpleNamespace:
    saved = dict(ckpt.get("args") or {})
    flow_cfg = dict(ckpt.get("flow_config") or {})
    ot_cfg = dict(ckpt.get("ot_config") or {})
    decay_cfg = dict(ckpt.get("decay_config") or {})

    args = SimpleNamespace(**saved)
    args.project_root = cli.project_root or ckpt.get("project_root") or saved.get("project_root")
    args.dit_checkpoint = cli.dit_checkpoint or ckpt.get("dit_checkpoint") or saved.get("dit_checkpoint")
    if not args.project_root or not args.dit_checkpoint:
        raise ValueError("project_root / dit_checkpoint missing from CLI and F1 checkpoint")

    args.image_size = int(saved.get("image_size", 256))
    args.rae_chunk_size = int(cli.rae_chunk_size)
    args.dit_variant = saved.get("dit_variant", "L")
    args.num_classes = int(saved.get("num_classes", 1000))
    args.num_steps = int(flow_cfg.get("num_steps", saved.get("num_steps", 8)))
    args.solver = flow_cfg.get("solver", saved.get("solver", "heun"))
    args.pair_chunk_size = int(cli.pair_batch_size)
    args.use_ema = bool(saved.get("use_ema", True))
    args.hint_signal_ratio = saved.get("hint_signal_ratio", None)
    args.stochastic_hint = bool(saved.get("stochastic_hint", True) if cli.stochastic_hint is None else cli.stochastic_hint)
    args.pure_noise_hint = bool(saved.get("pure_noise_hint", False) if cli.pure_noise_hint is None else cli.pure_noise_hint)
    args.allow_condition_mismatch = bool(saved.get("allow_condition_mismatch", False))
    args.amp = bool(saved.get("amp", True) if cli.amp is None else cli.amp)

    args.hidden_dim = int(saved.get("hidden_dim", 256))
    args.metric_dim = int(saved.get("metric_dim", 256))
    args.dropout = float(saved.get("dropout", 0.0))
    args.ot_pool_factor = int(ot_cfg.get("ot_pool_factor", saved.get("ot_pool_factor", 1)))
    args.sinkhorn_reg = float(ot_cfg.get("sinkhorn_reg", saved.get("sinkhorn_reg", 0.1)))
    args.sinkhorn_iter = int(ot_cfg.get("sinkhorn_iter", saved.get("sinkhorn_iter", 30)))
    args.sk_weight_eps = float(ot_cfg.get("sk_weight_eps", saved.get("sk_weight_eps", 1e-3)))

    args.feature_eps = float(saved.get("feature_eps", 1e-6))
    args.feature_ratio_clip = float(saved.get("feature_ratio_clip", 10.0))
    args.utility_proj_dim = int(decay_cfg.get("set_dim", saved.get("utility_proj_dim", 64)))
    args.utility_hidden_dim = int(decay_cfg.get("hidden_dim", saved.get("utility_hidden_dim", 64)))
    args.utility_condition_dim = int(decay_cfg.get("condition_dim", saved.get("utility_condition_dim", 32)))
    args.utility_head_dropout = float(decay_cfg.get("head_dropout", saved.get("utility_head_dropout", 0.10)))
    args.gamma_min = float(decay_cfg.get("gamma_min", saved.get("gamma_min", 0.70)))
    args.gamma_init = float(decay_cfg.get("gamma_init", saved.get("gamma_init", 0.95)))
    args.trajectory_score_mode = (
        cli.trajectory_score_mode
        or decay_cfg.get("score_mode")
        or saved.get("trajectory_score_mode", "step_to_support")
    )
    args.n_shot = int(cli.n_shot)
    args.n_query = int(cli.n_query)
    return args


def _apply_m2_runtime_overrides(args: SimpleNamespace, m2_ckpt: dict[str, Any]) -> None:
    """Use OT hyperparameters from the reloaded M2 when they are saved there.

    The F1 head architecture still comes from the F1 checkpoint; only the downstream
    metric/OT runtime is replaced here.
    """
    ot = dict(m2_ckpt.get("ot_config") or {})
    saved = dict(m2_ckpt.get("args") or {})

    def pick(name: str, current: Any) -> Any:
        if name in ot and ot[name] is not None:
            return ot[name]
        if name in saved and saved[name] is not None:
            return saved[name]
        return current

    args.ot_pool_factor = int(pick("ot_pool_factor", args.ot_pool_factor))
    args.sinkhorn_reg = float(pick("sinkhorn_reg", args.sinkhorn_reg))
    args.sinkhorn_iter = int(pick("sinkhorn_iter", args.sinkhorn_iter))
    args.sk_weight_eps = float(pick("sk_weight_eps", args.sk_weight_eps))


def load_bundle(
    path: str,
    *,
    m2_path: str,
    cli: argparse.Namespace,
    metric_device: torch.device,
) -> dict[str, Any]:
    # ------------------------------------------------------------------
    # LOAD ORDER 1/4: F1-v5 HEAD / F1 CONFIG ONLY.
    # ------------------------------------------------------------------
    ckpt = f1._torch_load(path, "cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"F1 checkpoint must be a dict: {path}")
    if ckpt.get("temporal_decay_head") is None:
        raise KeyError(f"F1 checkpoint lacks temporal_decay_head: {path}")

    feature_names = list(ckpt.get("feature_names") or [])
    if feature_names and feature_names != list(f1.FEATURE_NAMES):
        raise RuntimeError(
            "F1 feature definition mismatch between checkpoint and current pretrain module.\n"
            f"checkpoint={feature_names}\ncurrent={list(f1.FEATURE_NAMES)}"
        )

    args = build_runtime_args(cli, ckpt)
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
    head.load_state_dict(ckpt["temporal_decay_head"], strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # LOAD ORDER 2/4: DOWNSTREAM M2 RELOAD.
    # Do NOT use ckpt["metric_encoder"] / ckpt["shot_tau"] / ckpt["cls_tau"]
    # as the final evaluation metric state.
    # ------------------------------------------------------------------
    m2_ckpt = f1._torch_load(m2_path, "cpu")
    if not isinstance(m2_ckpt, dict):
        raise ValueError(f"M2 checkpoint must be a dict: {m2_path}")
    _apply_m2_runtime_overrides(args, m2_ckpt)

    metric = MetricEncoder(
        in_dim=1024,
        hidden_dim=args.hidden_dim,
        out_dim=args.metric_dim,
        dropout=args.dropout,
    ).to(metric_device)
    shot_tau, cls_tau = f1.load_m2_metric_and_temperatures(metric, m2_ckpt, metric_device)
    shot_tau = torch.tensor(0.10, dtype=torch.float32, device=metric_device)
    metric.eval()
    for p in metric.parameters():
        p.requires_grad_(False)

    return {
        "path": path,
        "m2_path": m2_path,
        "ckpt": ckpt,
        "m2_ckpt": m2_ckpt,
        "args": args,
        "metric": metric,
        "head": head,
        "shot_tau": shot_tau,
        "cls_tau": cls_tau,
    }


def normalized_fixed_weights(length: int, decay: float, device: torch.device) -> torch.Tensor:
    if not 0.0 < float(decay) <= 1.0:
        raise ValueError("fixed_decay must satisfy 0 < decay <= 1")
    idx = torch.arange(int(length), device=device, dtype=torch.float32)
    w = torch.pow(torch.tensor(float(decay), device=device), idx)
    return w / w.sum().clamp_min(1e-12)


def _ci95_binary(values: list[int]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    x = torch.tensor(values, dtype=torch.float64)
    return float(1.96 * x.std(unbiased=True).item() / math.sqrt(n))


def _empty_acc(num_steps: int) -> EvalAccumulator:
    keys = ["adaptive", "endpoint", "uniform", "fixed_decay"]
    return EvalAccumulator(
        queries=0,
        episodes=0,
        correct={k: 0 for k in keys},
        ce_sum={k: 0.0 for k in keys},
        rescue=0,
        degradation=0,
        oracle_any=0,
        per_step_correct=[0 for _ in range(num_steps + 1)],
        gamma_sum=[0.0 for _ in range(num_steps)],
        weight_sum=[0.0 for _ in range(num_steps + 1)],
        gamma_count=0,
        episode_final_correct=[],
    )


@torch.inference_mode()
def evaluate_dataset(
    *,
    plans: list[EvalEpisode],
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

    acc = _empty_acc(args.num_steps)
    details: list[dict[str, Any]] = []
    ep_bs = max(int(cli.episode_batch_size), 1)
    n_batches = int(math.ceil(len(plans) / ep_bs))

    for batch_idx in range(n_batches):
        batch = plans[batch_idx * ep_bs : (batch_idx + 1) * ep_bs]
        current_bs = len(batch)
        n_way = len(batch[0].classes)
        if any(len(ep.classes) != n_way for ep in batch):
            raise RuntimeError("mixed n_way inside one eval batch")

        support_imgs: list[torch.Tensor] = []
        query_imgs: list[torch.Tensor] = []
        targets: list[int] = []
        for ep in batch:
            s, q = load_episode_images(ep, transform)
            support_imgs.append(s)
            query_imgs.append(q)
            targets.append(int(ep.query_label))

        support_cpu = torch.stack(support_imgs, dim=0)
        query_cpu = torch.stack(query_imgs, dim=0)
        ns = n_way * int(args.n_shot)
        if support_cpu.shape[1] != ns:
            raise RuntimeError(f"support count={support_cpu.shape[1]}, expected {ns}")

        all_cpu = torch.cat([support_cpu, query_cpu], dim=1)
        flat = all_cpu.reshape(current_bs * (ns + args.n_query), *all_cpu.shape[2:]).to(
            rae_device, non_blocking=True
        )
        latent_flat = f1.encode_rae_chunked(rae, flat, args.rae_chunk_size).detach()
        latent = latent_flat.reshape(current_bs, ns + args.n_query, *latent_flat.shape[1:])
        support_z = latent[:, :ns]
        query_z = latent[:, ns:]
        target = torch.tensor(targets, dtype=torch.long, device=metric_device).repeat_interleave(args.n_query)

        # Fixed deterministic hint stream for this dataset/batch.
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

        _, step_logits = f1.shot_softmin_class_scores(
            selected_distances, shot_tau=shot_tau, cls_tau=cls_tau
        )
        direct_logits = f1.direct_support_logits(
            support_distances, shot_tau=shot_tau, cls_tau=cls_tau
        )
        features = f1.build_observable_step_features(
            selected_distances,
            step_logits,
            direct_logits,
            eps=args.feature_eps,
            ratio_clip=args.feature_ratio_clip,
        )
        raw, gamma = head(
            features,
            aux["support_class_repr"],
            aux["direct_query_repr"],
            aux["step_class_repr"],
        )
        state_weights = f1.decay_gamma_to_state_weights(gamma)
        _, _, adaptive_logits = f1.weighted_pair_class_logits(
            selected_distances,
            state_weights,
            shot_tau=shot_tau,
            cls_tau=cls_tau,
        )

        qn = int(target.numel())
        v = int(selected_distances.shape[-1])
        uniform_w = torch.full((qn, v), 1.0 / float(v), device=metric_device)
        fixed_one = normalized_fixed_weights(v, cli.fixed_decay, metric_device)
        fixed_w = fixed_one.unsqueeze(0).expand(qn, -1)
        _, _, uniform_logits = f1.weighted_pair_class_logits(
            selected_distances, uniform_w, shot_tau=shot_tau, cls_tau=cls_tau
        )
        _, _, fixed_logits = f1.weighted_pair_class_logits(
            selected_distances, fixed_w, shot_tau=shot_tau, cls_tau=cls_tau
        )

        logits_map = {
            "adaptive": adaptive_logits,
            "endpoint": step_logits[:, -1],
            "uniform": uniform_logits,
            "fixed_decay": fixed_logits,
        }
        pred_map = {k: x.argmax(dim=-1) for k, x in logits_map.items()}
        ok_map = {k: pred.eq(target) for k, pred in pred_map.items()}

        for key, logits in logits_map.items():
            acc.correct[key] += int(ok_map[key].sum().item())
            ce = F.cross_entropy(logits.float(), target, reduction="sum")
            acc.ce_sum[key] += float(ce.item())

        a_ok = ok_map["adaptive"]

        step_oks = []
        for t in range(args.num_steps + 1):
            ok = step_logits[:, t].argmax(dim=-1).eq(target)
            step_oks.append(ok)
            acc.per_step_correct[t] += int(ok.sum().item())
        oracle = torch.stack(step_oks, dim=1).any(dim=1)
        acc.oracle_any += int(oracle.sum().item())

        gamma_mean_q = gamma
        weight_mean_q = state_weights
        for t in range(args.num_steps):
            acc.gamma_sum[t] += float(gamma_mean_q[:, t].sum().item())
        for t in range(args.num_steps + 1):
            acc.weight_sum[t] += float(weight_mean_q[:, t].sum().item())
        acc.gamma_count += qn
        acc.queries += qn
        acc.episodes += current_bs
        acc.episode_final_correct.extend(int(x) for x in a_ok.detach().cpu().tolist())

        if cli.save_episode_details:
            for local_i, ep in enumerate(batch):
                idx = local_i * int(args.n_query)
                details.append({
                    "episode_index": batch_idx * ep_bs + local_i,
                    "classes": ep.classes,
                    "query_label": int(ep.query_label),
                    "query_path": ep.query_paths[0],
                    "support_paths": ep.support_paths,
                    "prediction": {k: int(vv[idx].item()) for k, vv in pred_map.items()},
                    "correct": {k: bool(vv[idx].item()) for k, vv in ok_map.items()},
                    "gamma": [float(x) for x in gamma[idx].detach().cpu().tolist()],
                    "state_weights": [float(x) for x in state_weights[idx].detach().cpu().tolist()],
                })

        if cli.log_every > 0 and (batch_idx + 1) % int(cli.log_every) == 0:
            print(
                f"  [eval batch {batch_idx+1:04d}/{n_batches}] "
                f"adaptive={int(a_ok.sum())} "
                f"gamma={float(gamma.mean().item()):.4f} "
                f"w_last={float(state_weights[:, -1].mean().item()):.4f}"
            )

        del support_cpu, query_cpu, all_cpu, flat, latent_flat, latent
        del support_z, query_z, support_distances, selected_distances, step_logits
        del direct_logits, features, raw, gamma, state_weights, adaptive_logits
        del uniform_logits, fixed_logits

    qn = max(acc.queries, 1)
    metrics: dict[str, Any] = {
        "queries": acc.queries,
        "episodes": acc.episodes,
        "accuracy": {k: acc.correct[k] / qn for k in acc.correct},
        "accuracy_percent": {k: 100.0 * acc.correct[k] / qn for k in acc.correct},
        "ce": {k: acc.ce_sum[k] / qn for k in acc.ce_sum},
        "adaptive_accuracy_ci95_percent": 100.0 * _ci95_binary(acc.episode_final_correct),
        "oracle_any_step_accuracy": acc.oracle_any / qn,
        "oracle_any_step_accuracy_percent": 100.0 * acc.oracle_any / qn,
        "per_step_accuracy": [x / qn for x in acc.per_step_correct],
        "per_step_accuracy_percent": [100.0 * x / qn for x in acc.per_step_correct],
        "gamma_t_mean": [x / qn for x in acc.gamma_sum],
        "state_weight_t_mean": [x / qn for x in acc.weight_sum],
        "fixed_decay": float(cli.fixed_decay),
        "fixed_decay_weights": normalized_fixed_weights(args.num_steps + 1, cli.fixed_decay, torch.device("cpu")).tolist(),
    }
    return metrics, details


def macro_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    keys = ["adaptive", "endpoint", "uniform", "fixed_decay"]
    return {
        "datasets": len(results),
        "macro_accuracy_percent": {
            k: sum(float(r["metrics"]["accuracy_percent"][k]) for r in results) / len(results)
            for k in keys
        },
    }


def write_csv(output_json: Path, results: list[dict[str, Any]]) -> Path:
    csv_path = output_json.with_suffix(".csv")
    fields = [
        "hospital", "organ", "source_path", "f1_checkpoint", "m2_checkpoint", "trajectory_score_mode",
        "n_way_eval", "n_shot", "n_query", "episodes",
        "adaptive_acc_percent", "adaptive_ci95_percent",
        "endpoint_acc_percent", "uniform_acc_percent", "fixed_decay_acc_percent",
        "oracle_any_step_acc_percent", "gamma_mean", "last_state_weight_mean",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            m = r["metrics"]
            w.writerow({
                "hospital": r.get("hospital"),
                "organ": r.get("organ"),
                "source_path": r.get("source_path"),
                "f1_checkpoint": r.get("f1_checkpoint"),
                "m2_checkpoint": r.get("m2_checkpoint"),
                "trajectory_score_mode": r.get("trajectory_score_mode"),
                "n_way_eval": r.get("n_way_eval"),
                "n_shot": r.get("n_shot"),
                "n_query": r.get("n_query"),
                "episodes": r.get("episodes"),
                "adaptive_acc_percent": m["accuracy_percent"]["adaptive"],
                "adaptive_ci95_percent": m["adaptive_accuracy_ci95_percent"],
                "endpoint_acc_percent": m["accuracy_percent"]["endpoint"],
                "uniform_acc_percent": m["accuracy_percent"]["uniform"],
                "fixed_decay_acc_percent": m["accuracy_percent"]["fixed_decay"],
                "oracle_any_step_acc_percent": m["oracle_any_step_accuracy_percent"],
                "gamma_mean": sum(m["gamma_t_mean"]) / max(len(m["gamma_t_mean"]), 1),
                "last_state_weight_mean": m["state_weight_t_mean"][-1],
            })
    return csv_path


def main() -> None:
    cli = parse_args()
    if cli.episodes < 1 or cli.n_shot < 1 or cli.n_query < 1:
        raise ValueError("episodes/n_shot/n_query must be >=1")
    if cli.episode_batch_size < 1 or cli.pair_batch_size < 1 or cli.rae_chunk_size < 1:
        raise ValueError("batch/chunk sizes must be >=1")
    if not 0.0 < float(cli.fixed_decay) <= 1.0:
        raise ValueError("fixed_decay must be in (0,1]")

    setup_seed(cli.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    metric_device = _device(cli.metric_device)
    flow_device = _device(cli.flow_device)
    rae_device = _device(cli.rae_device)

    sources = resolve_sources(cli)
    f1_map = parse_kv(cli.organ_f1_checkpoint, name="--organ_f1_checkpoint")
    m2_map = parse_kv(cli.organ_m2_checkpoint, name="--organ_m2_checkpoint")
    required_pairs: list[tuple[str, str]] = []
    for src in sources:
        f1_path = checkpoint_for_source(src, organ_map=f1_map, fallback=cli.f1_checkpoint)
        m2_path = m2_checkpoint_for_source(src, organ_map=m2_map, fallback=cli.m2_checkpoint)
        pair = (f1_path, m2_path)
        if pair not in required_pairs:
            required_pairs.append(pair)
        if not Path(f1_path).is_file():
            raise FileNotFoundError(f"F1 checkpoint not found: {f1_path}")
        if not Path(m2_path).is_file():
            raise FileNotFoundError(f"downstream M2 checkpoint not found: {m2_path}")

    # Important: F1-v5 head is loaded first inside load_bundle(), then M2 is reloaded.
    # Key by (F1,M2), because one generic F1 may be evaluated with different organ M2s.
    bundles = {
        pair: load_bundle(pair[0], m2_path=pair[1], cli=cli, metric_device=metric_device)
        for pair in required_pairs
    }

    # Heavy frozen RAE/Flow are shared.  DiT/Flow is intentionally created AFTER
    # every F1+M2 bundle below, so the explicit CLI DiT checkpoint has final precedence.
    first_args = bundles[required_pairs[0]]["args"]
    for pair, bundle in bundles.items():
        a = bundle["args"]
        common = (
            a.project_root == first_args.project_root
            and a.dit_checkpoint == first_args.dit_checkpoint
            and a.image_size == first_args.image_size
            and a.dit_variant == first_args.dit_variant
            and a.num_steps == first_args.num_steps
            and a.solver == first_args.solver
        )
        if not common:
            raise RuntimeError(
                "All organ F1 checkpoints must share RAE/Flow runtime settings in one batch eval. "
                f"Mismatch at F1/M2 pair={pair}"
            )

    transform = f1.build_transform(first_args.image_size, augment=False)

    # LOAD ORDER 3/4: RAE after F1 head + downstream M2 reload.
    torch.cuda.set_device(rae_device)
    rae = f1.build_rae(first_args.project_root, rae_device)
    rae.eval()
    for p in rae.parameters():
        p.requires_grad_(False)

    # LOAD ORDER 4/4: explicit DiT/Flow reload LAST.
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

    print("=" * 118)
    print("[F1-v5 PRETRAIN CHECKPOINT EVAL | M2-ALIGNED TEST-TEST CLASS FOLDERS]")
    print(f"batch_root={cli.batch_root or cli.data_root}")
    print(f"organ F1 checkpoints={f1_map or {'fallback': cli.f1_checkpoint}}")
    print(f"organ M2 checkpoints={m2_map or {'fallback': cli.m2_checkpoint}}")
    print(f"DiT/Flow RELOADED after F1+M2={first_args.dit_checkpoint}")
    print(f"devices metric/head={metric_device}, flow={flow_device}, rae={rae_device}")
    print(f"episodes={cli.episodes}/dataset | n_shot={cli.n_shot} | n_query={cli.n_query} | way=all actual classes unless --n_way is set")
    print(f"episode_batch_size={cli.episode_batch_size} | pair_batch_size={cli.pair_batch_size} | rae_chunk_size={cli.rae_chunk_size}")
    print("F1 inference = deterministic gamma; relative exploration = OFF")
    print("Test-test rule = all leakage-safe query classes; query excluded; undersized support uses replacement")
    print("=" * 118)

    output = Path(cli.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir = output.parent / f"{output.stem}_datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for src in sources:
        ckpt_path = checkpoint_for_source(src, organ_map=f1_map, fallback=cli.f1_checkpoint)
        m2_path = m2_checkpoint_for_source(src, organ_map=m2_map, fallback=cli.m2_checkpoint)
        bundle = bundles[(ckpt_path, m2_path)]
        args = bundle["args"]
        pool, labels = eligible_test_test_pool(src["path"], cli.n_shot)
        n_way = len(labels) if cli.n_way is None else min(int(cli.n_way), len(labels))
        if n_way < 2:
            raise RuntimeError(f"invalid n_way={n_way} for {src['path']}")
        dataset_seed = task_seed(cli.seed, src.get("hospital"), src.get("organ"))
        plans = generate_episodes(
            pool,
            labels,
            n_way=n_way,
            n_shot=cli.n_shot,
            n_query=cli.n_query,
            episodes=cli.episodes,
            seed=dataset_seed,
        )

        tag = (
            f"{src['hospital']}-{src['organ']}"
            if src.get("hospital") is not None
            else Path(src["path"]).name
        )
        print("\n" + "=" * 118)
        print(f"[DATASET] {tag}")
        print(f"source={src['path']}")
        print(f"F1 checkpoint={ckpt_path}")
        print(f"M2 RELOADED checkpoint={m2_path}")
        print(f"DIT/Flow RELOADED checkpoint={args.dit_checkpoint}")
        print(f"protocol={n_way}-way {cli.n_shot}-shot {cli.n_query}-query, episodes={cli.episodes}, seed={dataset_seed}")
        print(f"trajectory_score_mode={args.trajectory_score_mode}")

        metrics, details = evaluate_dataset(
            plans=plans,
            bundle=bundle,
            rae=rae,
            flow=flow,
            transform=transform,
            metric_device=metric_device,
            rae_device=rae_device,
            cli=cli,
            dataset_seed=dataset_seed,
        )
        ap = metrics["accuracy_percent"]
        print(
            f"RESULT {tag} | F1={ap['adaptive']:.2f}% ± {metrics['adaptive_accuracy_ci95_percent']:.2f} "
            f"| Endpoint={ap['endpoint']:.2f}% "
            f"| Uniform={ap['uniform']:.2f}% | Fixed({cli.fixed_decay:g})={ap['fixed_decay']:.2f}% "
        )
        print("  gamma_t mean   = " + str([round(x, 4) for x in metrics["gamma_t_mean"]]))
        print("  state w_t mean = " + str([round(x, 4) for x in metrics["state_weight_t_mean"]]))

        one: dict[str, Any] = {
            "stage": "f1-v5-pretrain-checkpoint-eval-m2-aligned-test-test-nodirect",
            "hospital": src.get("hospital"),
            "organ": src.get("organ"),
            "source_path": src["path"],
            "f1_checkpoint": ckpt_path,
            "m2_checkpoint": m2_path,
            "dit_checkpoint_reloaded": args.dit_checkpoint,
            "f1_checkpoint_epoch": int(bundle["ckpt"].get("epoch", -1)),
            "trajectory_score_mode": args.trajectory_score_mode,
            "seed": dataset_seed,
            "n_way_eval": n_way,
            "eligible_classes": len(labels),
            "candidate_rule": "all leakage-safe query classes; no K-shot count filtering",
            "support_sampling": "query excluded; replacement only when remaining supports < n_shot",
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
        safe = tag.replace("/", "_").replace(os.sep, "_")
        (dataset_dir / f"{safe}.json").write_text(
            json.dumps(one, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "stage": "f1-v5-pretrain-checkpoint-eval-m2-aligned-test-test-nodirect",
        "batch_root": cli.batch_root,
        "single_data_root": cli.data_root,
        "dataset_count": len(results),
        "organ_f1_checkpoints": f1_map,
        "fallback_f1_checkpoint": cli.f1_checkpoint,
        "organ_m2_checkpoints": m2_map,
        "fallback_m2_checkpoint": cli.m2_checkpoint,
        "dit_checkpoint_reloaded": first_args.dit_checkpoint,
        "protocol": {
            "test_test": True,
            "query_support_same_pool": True,
            "query_excluded_from_support": True,
            "way_rule": "all_leakage_safe_query_classes_unless_explicit_n_way",
            "support_sampling": "query_excluded_no_replace_if_enough_else_with_replacement",
            "n_shot": cli.n_shot,
            "n_query": cli.n_query,
            "episodes_per_dataset": cli.episodes,
            "fixed_decay_diagnostic": cli.fixed_decay,
        },
        "macro": macro_summary(results),
        "datasets": results,
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = write_csv(output, results)
    print("\n" + "=" * 118)
    print(f"Saved JSON -> {output}")
    print(f"Saved CSV  -> {csv_path}")
    print("=" * 118)


if __name__ == "__main__":
    main()
