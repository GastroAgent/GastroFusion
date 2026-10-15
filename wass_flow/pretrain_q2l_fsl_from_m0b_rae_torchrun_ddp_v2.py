#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q2L-M0B-RAE: pure Query-to-Last FSL metric pretraining.

Purpose
-------
Reuse the RAE M0-B MetricEncoder checkpoint and continue optimizing the SAME
MetricEncoder with a pure Query-to-Last objective.

This stage intentionally has:
  * frozen single-image RAE,
  * frozen support-conditioned DiT Flow + frozen EndoViT,
  * trainable M0-B MetricEncoder,
  * NO Query-to-Support Wasserstein term,
  * NO F1/router/gate/trajectory-weight head,
  * NO direct episodic branch.

For every anchor query q in a class-balanced P x K batch, choose:
  s+ : another image from the same class,
  s- : an image from a different class.

The frozen Flow is NOT retrained.  It is only probed under two candidate
conditions:

    z_q --Flow(condition=s+)--> z_T+
    z_q --Flow(condition=s-)--> z_T-

Then the M0-B MetricEncoder is applied symmetrically:

    F_q  = M(z_q)
    F_T+ = M(z_T+)
    F_T- = M(z_T-)

and the only classification/metric evidence is Query-to-Last SK-Wasserstein:

    D_same  = W_SK(F_q, F_T+)
    D_cross = W_SK(F_q, F_T-)

Training objective:

    L_rank = relu(margin + D_same - D_cross)

Optionally VISReg can be enabled as a representation regularizer, but its
weight is 0 by default so the default stage is strictly Q2L ranking.

Why this matches the intended hypothesis
----------------------------------------
The Flow remains a same-class-trained generator.  Correct/same-class support
conditions are therefore in-distribution conditions, while cross-class support
conditions are used only as frozen negative probes.  The trainable FSL metric
is optimized so the resulting Query-to-Last transport cost is discriminative.

Memory
------
Only the final generated state is retained.  Intermediate Flow states are not
stored.  Flow pairs are generated in chunks.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from functools import partial
from typing import Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from data.class_pool import (
    ImageRecord,
    RecordDataset,
    build_rae_transform,
    load_records,
)
from losss.visreg import VISReg
from models.metric_encoder import MetricEncoder, pool_square_tokens
from models.sk_sinkhorn import aligned_sk_sinkhorn_distance
from utils_.checkpoint import save_checkpoint
from utils_.distributed import (
    cleanup_distributed,
    seed_everything,
    setup_distributed,
)
from utils_.rae_runtime import build_rae, encode_rae_chunked

# -----------------------------------------------------------------------------
# Self-contained frozen RAE DiT/EndoViT runtime.
#
# IMPORTANT:
# Do NOT import train_f1_flow_episode.py here.  That file was only a reference
# implementation and may not exist in the runtime directory on the cluster.
# The Q2L trainer only needs the frozen Flow loader / support-condition encoder,
# so the required runtime is embedded here directly.
# -----------------------------------------------------------------------------
class FrozenFlowRuntime:
    """Frozen RAE DiT + frozen EndoViT support-condition encoder."""

    DATASET_MEAN = (0.3464, 0.2280, 0.2228)
    DATASET_STD = (0.2520, 0.2128, 0.2093)

    def __init__(
        self,
        *,
        project_root: str,
        checkpoint: str,
        device: torch.device,
        dit_variant: str,
        num_classes: int,
        image_size: int,
        use_ema: bool,
        hint_signal_ratio: Optional[float],
        allow_condition_mismatch: bool,
        amp: bool,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError("Q2L DiT generation currently requires CUDA")

        self.project_root = str(Path(project_root).resolve())
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)
        self.device = device
        self.amp = bool(amp and device.type == "cuda")
        self.image_size = int(image_size)

        from flow_models.dit_rae import create_dit_rae
        from timm.models.vision_transformer import VisionTransformer

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError("DiT checkpoint must be a dict checkpoint")

        self.condition_config = ckpt.get("condition_config") or {}
        saved_mode = self.condition_config.get("condition_mode")
        saved_size = self.condition_config.get("image_size")
        mismatch = (
            saved_mode != "vision_hint_v1"
            or (saved_size is not None and int(saved_size) != self.image_size)
        )
        if mismatch and not allow_condition_mismatch:
            raise RuntimeError(
                "DiT checkpoint condition_config is not V1-compatible.\n"
                f"saved={self.condition_config}\n"
                f"requested image_size={self.image_size}\n"
                "Use --allow_condition_mismatch only for an intentional ablation."
            )

        self.hint_signal_ratio = float(
            self.condition_config.get("hint_signal_ratio", 0.10)
            if hint_signal_ratio is None
            else hint_signal_ratio
        )
        if not 0.0 <= self.hint_signal_ratio <= 1.0:
            raise ValueError("hint_signal_ratio must be in [0,1]")

        dit = create_dit_rae(
            variant=dit_variant,
            rae_latent_dim=1024,
            rae_spatial_size=16,
            cross_attn_dim=768,
            num_classes=num_classes,
            use_cross_attention=True,
        )
        if use_ema and isinstance(ckpt.get("ema_model"), dict):
            dit_state = ckpt["ema_model"]
            self.loaded_dit_key = "ema_model"
        elif isinstance(ckpt.get("dit_model"), dict):
            dit_state = ckpt["dit_model"]
            self.loaded_dit_key = "dit_model"
        else:
            dit_state = ckpt
            self.loaded_dit_key = "checkpoint_root"

        missing, unexpected = dit.load_state_dict(dit_state, strict=False)
        if missing:
            print(f"[Q2L][WARN] DiT missing keys: {missing[:12]}")
        if unexpected:
            print(f"[Q2L][WARN] DiT unexpected keys: {unexpected[:12]}")
        dit = dit.to(device).eval()
        dit.device = device
        for p in dit.parameters():
            p.requires_grad_(False)
        self.dit = dit

        vision = VisionTransformer(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=False,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ).eval()
        endovit_path = os.path.join(
            self.project_root,
            "flow_matcher_otcfm/EndoViT/pytorch_model.bin",
        )
        vision.load_state_dict(
            torch.load(endovit_path, map_location="cpu", weights_only=False),
            strict=False,
        )
        if isinstance(ckpt.get("vision_model"), dict):
            vision.load_state_dict(ckpt["vision_model"], strict=False)
        vision = vision.to(device).eval()
        vision.device = device
        for p in vision.parameters():
            p.requires_grad_(False)
        self.vision = vision
        del ckpt

    @torch.no_grad()
    def encode_target_vision(self, support_images_01: torch.Tensor) -> torch.Tensor:
        """[0,1] support images -> EndoViT CLS + patch tokens."""
        x = F.interpolate(
            support_images_01.float(),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(
            self.DATASET_MEAN, device=x.device, dtype=x.dtype
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.DATASET_STD, device=x.device, dtype=x.dtype
        ).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.amp,
        ):
            out = self.vision.forward_features(x)
        return out.float()

    @torch.no_grad()
    def build_hint(
        self,
        support_images_01: torch.Tensor,
        *,
        stochastic: bool,
        pure_noise: bool,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """V1 weak grayscale hint, identical to the canonical RAE Flow runtime."""
        x = support_images_01.float()
        gray = (
            0.2989 * x[:, 0:1]
            + 0.5870 * x[:, 1:2]
            + 0.1140 * x[:, 2:3]
        ).repeat(1, 3, 1, 1)
        gray = gray * 2.0 - 1.0

        if seed is None:
            noise = (
                torch.randn_like(gray)
                if stochastic or pure_noise
                else torch.zeros_like(gray)
            )
        else:
            # Deterministic validation noise without mutating global CUDA RNG.
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            noise = torch.randn(
                gray.shape,
                generator=g,
                dtype=torch.float32,
            ).to(gray.device)
            if not (stochastic or pure_noise):
                noise.zero_()

        if pure_noise:
            return noise
        alpha = self.hint_signal_ratio
        return alpha * gray + math.sqrt(max(1.0 - alpha * alpha, 0.0)) * noise


STAGE = "Q2L-M0B-RAE-v1-pure-query-to-last-pretrain"


class Q2LMetricModel(nn.Module):
    """Thin DDP-safe wrapper around the pretrained M0-B MetricEncoder."""

    def __init__(self, metric: MetricEncoder) -> None:
        super().__init__()
        self.metric = metric

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.metric(latent)
        return features.raw, features.cloud


class RankAwarePKBatchSampler(Sampler[list[int]]):
    """Class-balanced P x K sampler, one deterministic stream per rank."""

    def __init__(
        self,
        targets: Iterable[int],
        classes_per_batch: int,
        samples_per_class: int,
        batches_per_epoch: int,
        *,
        seed: int,
        rank: int,
    ) -> None:
        super().__init__(None)
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, target in enumerate(targets):
            self.class_to_indices[int(target)].append(index)

        self.p = int(classes_per_batch)
        self.k = int(samples_per_class)
        self.batches_per_epoch = int(batches_per_epoch)
        self.seed = int(seed)
        self.rank = int(rank)
        self.epoch = 0

        if self.p < 2:
            raise ValueError("classes_per_batch must be >= 2")
        if self.k < 2:
            raise ValueError("samples_per_class must be >= 2")
        if self.batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be > 0")

        self.classes = sorted(
            cls for cls, indices in self.class_to_indices.items()
            if len(indices) >= self.k
        )
        if len(self.classes) < self.p:
            raise ValueError(
                f"Need P={self.p} classes with at least K={self.k} images; "
                f"only {len(self.classes)} classes qualify."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = random.Random(
            self.seed + self.epoch * 1_000_003 + self.rank * 100_003
        )
        for _ in range(self.batches_per_epoch):
            chosen_classes = rng.sample(self.classes, self.p)
            batch: list[int] = []
            for cls in chosen_classes:
                batch.extend(rng.sample(self.class_to_indices[cls], self.k))
            rng.shuffle(batch)
            yield batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Q2L-M0B-RAE: pure Query-to-Last SK-Wasserstein FSL pretraining"
    )

    train_src = p.add_mutually_exclusive_group(required=True)
    train_src.add_argument("--train_root", type=str)
    train_src.add_argument("--train_json_glob", type=str)

    val_src = p.add_mutually_exclusive_group(required=False)
    val_src.add_argument("--val_root", type=str, default=None)
    val_src.add_argument("--val_json_glob", type=str, default=None)
    p.add_argument("--val_fraction", type=float, default=0.05)

    ckpt = p.add_mutually_exclusive_group(required=True)
    ckpt.add_argument(
        "--init_m0b_checkpoint",
        type=str,
        help="RAE M0-B metric_imagenet_ot.pt; metric_encoder is reused directly.",
    )
    ckpt.add_argument(
        "--resume_checkpoint",
        type=str,
        help="Resume a checkpoint produced by this Q2L trainer.",
    )

    p.add_argument("--dit_checkpoint", type=str, required=True)
    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--rae_chunk_size", type=int, default=8)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)

    # Class-balanced pretraining batch. Values are per GPU under DDP.
    p.add_argument("--classes_per_batch", type=int, default=8)
    p.add_argument("--samples_per_class", type=int, default=4)
    p.add_argument("--batches_per_epoch", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=8)

    # Flow.  Defaults mirror the canonical RAE F1 runtime.
    p.add_argument("--dit_variant", type=str, default="L", choices=["S", "B", "L", "L12"])
    p.add_argument("--num_classes", type=int, default=1122)
    p.add_argument("--num_steps", type=int, default=8)
    p.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    p.add_argument(
        "--pair_chunk_size",
        type=int,
        default=2,
        help="Number of q/support-conditioned Flow pairs generated at once.",
    )
    p.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hint_signal_ratio", type=float, default=None)
    p.add_argument("--stochastic_hint", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pure_noise_hint", action="store_true")
    p.add_argument("--allow_condition_mismatch", action="store_true")

    # Metric architecture fallback only.  Normally inherited from M0-B checkpoint args.
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--metric_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)

    # Q2L Dynamic-SK Sinkhorn.
    p.add_argument("--ot_pool_factor", type=int, default=2)
    p.add_argument("--sinkhorn_reg", type=float, default=0.1)
    p.add_argument("--sinkhorn_iter", type=int, default=30)
    p.add_argument("--sk_weight_eps", type=float, default=1e-3)
    p.add_argument("--triplet_margin", type=float, default=0.2)

    # Pair-condition selection.  There is deliberately no direct-Metric hard mining:
    # the stage stays pure Query-to-Last rather than reintroducing q-to-support evidence.
    p.add_argument("--support_selection", choices=["random", "fixed"], default="random")
    p.add_argument("--val_support_selection", choices=["random", "fixed"], default="fixed")

    # Optimization.  Default objective is rank only.
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--rank_weight", type=float, default=1.0)
    p.add_argument(
        "--visreg_weight",
        type=float,
        default=0.0,
        help="0 keeps the default stage strictly Q2L ranking.",
    )
    p.add_argument("--visreg_num_proj", type=int, default=256)
    p.add_argument("--visreg_projection_chunk", type=int, default=128)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def _stratified_split(
    records: list[ImageRecord],
    val_fraction: float,
    min_per_split: int,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0,1)")

    by_label: dict[str, list[ImageRecord]] = defaultdict(list)
    for rec in records:
        by_label[rec.label].append(rec)

    rng = random.Random(seed)
    train_records: list[ImageRecord] = []
    val_records: list[ImageRecord] = []
    too_small: list[tuple[str, int]] = []

    for label, items in sorted(by_label.items()):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        if n < 2 * min_per_split:
            too_small.append((label, n))
            continue
        n_val = max(min_per_split, int(round(n * val_fraction)))
        n_val = min(n_val, n - min_per_split)
        val_records.extend(items[:n_val])
        train_records.extend(items[n_val:])

    if too_small:
        preview = ", ".join(f"{label}:{n}" for label, n in too_small[:10])
        raise ValueError(
            "Stratified Q2L split requires at least "
            f"{2 * min_per_split} samples per class. Too-small classes: {preview}"
        )
    return train_records, val_records


def _checkpoint_args(checkpoint: dict) -> dict:
    value = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if hasattr(value, "__dict__"):
        value = vars(value)
    return dict(value) if isinstance(value, dict) else {}


def _extract_metric_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dict")
    for key in ("metric_encoder", "metric", "metric_state_dict", "fsl_model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            state = value
            break
    else:
        state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise KeyError(
                "No MetricEncoder state found; expected metric_encoder/metric/metric_state_dict/fsl_model"
            )

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("module.", "metric.", "metric_encoder."):
            while name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def _build_metric_from_checkpoint(
    checkpoint: dict,
    args: argparse.Namespace,
) -> MetricEncoder:
    saved = _checkpoint_args(checkpoint)
    hidden_dim = int(saved.get("hidden_dim", args.hidden_dim))
    metric_dim = int(saved.get("metric_dim", args.metric_dim))
    dropout = float(saved.get("dropout", args.dropout))

    metric = MetricEncoder(
        1024,
        hidden_dim,
        metric_dim,
        dropout=dropout,
    )
    metric.load_state_dict(_extract_metric_state(checkpoint), strict=True)

    # Record the exact reused M0-B architecture in current args/checkpoints.
    args.hidden_dim = hidden_dim
    args.metric_dim = metric_dim
    args.dropout = dropout
    return metric


def _select_support_indices(
    labels: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For every anchor choose one same-class support and one cross-class support."""
    b = int(labels.numel())
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(b, dtype=torch.bool, device=labels.device)
    pos_mask = same & ~eye
    neg_mask = ~same

    if not bool(pos_mask.any(dim=1).all()):
        raise RuntimeError("Every query anchor needs another same-class support")
    if not bool(neg_mask.any(dim=1).all()):
        raise RuntimeError("Every query anchor needs a cross-class support")

    if mode == "random":
        pos_idx = torch.multinomial(pos_mask.float(), 1).squeeze(1)
        neg_idx = torch.multinomial(neg_mask.float(), 1).squeeze(1)
        return pos_idx, neg_idx
    if mode == "fixed":
        pos_idx = pos_mask.to(torch.int64).argmax(dim=1)
        neg_idx = neg_mask.to(torch.int64).argmax(dim=1)
        return pos_idx, neg_idx
    raise ValueError(mode)


@torch.no_grad()
def _integrate_last_state(
    flow: FrozenFlowRuntime,
    query_z: torch.Tensor,
    target_vision: torch.Tensor,
    hint: torch.Tensor,
    *,
    num_steps: int,
    solver: str,
) -> torch.Tensor:
    """Endpoint-only clone of FrozenFlowRuntime.generate_post_states().

    It preserves the exact Euler/Heun update but never stores intermediate states.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >=1")
    if solver not in {"euler", "heun"}:
        raise ValueError(f"unknown solver={solver}")

    x = query_z.to(flow.device, non_blocking=True).float()
    target_vision = target_vision.to(flow.device, non_blocking=True)
    hint = hint.to(flow.device, non_blocking=True)
    patch = target_vision[:, 1:, :]
    cls = target_vision[:, 0, :]

    t_span = torch.linspace(0.0, 1.0, int(num_steps) + 1, device=flow.device)
    dt = float(1.0 / int(num_steps))

    for i in range(int(num_steps)):
        t = torch.full((x.shape[0],), float(t_span[i]), device=flow.device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=flow.amp,
        ):
            dx1 = flow.dit(
                xt=x,
                timestep=t,
                class_labels=None,
                encoder_hidden_states=patch,
                hint=hint,
                text_embeds=None,
                image_embeds=cls,
            )["sample"]

            if solver == "euler":
                x = x + dx1.float() * dt
            else:
                x_euler = x + dx1.float() * dt
                t_next = torch.full(
                    (x.shape[0],),
                    float(t_span[i + 1]),
                    device=flow.device,
                )
                dx2 = flow.dit(
                    xt=x_euler,
                    timestep=t_next,
                    class_labels=None,
                    encoder_hidden_states=patch,
                    hint=hint,
                    text_embeds=None,
                    image_embeds=cls,
                )["sample"]
                x = x + 0.5 * (dx1.float() + dx2.float()) * dt

    return x.detach()


@torch.no_grad()
def _generate_last_for_pairs(
    flow: FrozenFlowRuntime,
    query_z: torch.Tensor,
    support_images_01: torch.Tensor,
    args: argparse.Namespace,
    *,
    hint_seed: Optional[int],
) -> torch.Tensor:
    """Generate one candidate-conditioned Last per aligned query/support pair."""
    if query_z.shape[0] != support_images_01.shape[0]:
        raise ValueError(
            f"pair batch mismatch query_z={query_z.shape[0]} support={support_images_01.shape[0]}"
        )

    outs: list[torch.Tensor] = []
    chunk = max(int(args.pair_chunk_size), 1)
    for start in range(0, query_z.shape[0], chunk):
        end = min(start + chunk, query_z.shape[0])
        q = query_z[start:end].to(flow.device, non_blocking=True)
        s = support_images_01[start:end].to(flow.device, non_blocking=True)

        target_vision = flow.encode_target_vision(s)
        seed = None if hint_seed is None else int(hint_seed + start * 1009)
        hint = flow.build_hint(
            s,
            stochastic=args.stochastic_hint,
            pure_noise=args.pure_noise_hint,
            seed=seed,
        )
        last = _integrate_last_state(
            flow,
            q,
            target_vision,
            hint,
            num_steps=args.num_steps,
            solver=args.solver,
        )
        outs.append(last)
        del target_vision, hint, last

    return torch.cat(outs, dim=0)


def _q2l_distances(
    query_raw: torch.Tensor,
    same_last_raw: torch.Tensor,
    cross_last_raw: torch.Tensor,
    *,
    pool_factor: int,
    reg: float,
    n_iter: int,
    weight_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = pool_square_tokens(query_raw, pool_factor)
    same = pool_square_tokens(same_last_raw, pool_factor)
    cross = pool_square_tokens(cross_last_raw, pool_factor)

    qn = F.normalize(q.float(), p=2, dim=-1)
    sn = F.normalize(same.float(), p=2, dim=-1)
    cn = F.normalize(cross.float(), p=2, dim=-1)

    b = q.shape[0]
    out = aligned_sk_sinkhorn_distance(
        torch.cat([qn, qn], dim=0),
        torch.cat([sn, cn], dim=0),
        query_raw=torch.cat([q, q], dim=0),
        support_raw=torch.cat([same, cross], dim=0),
        reg=reg,
        n_iter=n_iter,
        weight_eps=weight_eps,
        return_transport=False,
    ).distance.float()
    return out[:b], out[b:]


def _empty_sums(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.zeros((), device=device, dtype=torch.float32)
        for key in ("total", "rank", "vis", "pair_acc", "same_d", "cross_d", "gap")
    }


def _all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    out = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def _finalize_metrics(
    sums: dict[str, torch.Tensor],
    batches: int,
) -> dict[str, float]:
    count = torch.tensor(float(batches), device=next(iter(sums.values())).device)
    count = _all_reduce_sum(count).clamp_min(1.0)
    return {
        key: float((_all_reduce_sum(value) / count).cpu())
        for key, value in sums.items()
    }


def _compute_batch(
    model: Q2LMetricModel | DDP,
    rae,
    flow: FrozenFlowRuntime,
    visreg: VISReg,
    images: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    *,
    support_selection: str,
    hint_seed: Optional[int],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Input must remain RAE/Flow-ready [0,1].  This catches an accidental reuse of
    # ImageNet-normalized tensors before they silently corrupt Flow conditioning.
    with torch.no_grad():
        lo = float(images.detach().amin().item())
        hi = float(images.detach().amax().item())
    if lo < -1e-4 or hi > 1.0001:
        raise RuntimeError(
            f"Q2L RAE/Flow input must be [0,1], got min={lo:.4f}, max={hi:.4f}"
        )

    pos_idx, neg_idx = _select_support_indices(labels, support_selection)

    # Query/source RAE latent.  Real supports do NOT enter the metric distance;
    # they are used only as frozen Flow conditions.
    query_z = encode_rae_chunked(rae, images, args.rae_chunk_size).float()

    with torch.no_grad():
        same_support = images.index_select(0, pos_idx)
        cross_support = images.index_select(0, neg_idx)

        # Generate same-class and cross-class candidate-conditioned endpoints.
        # Flow itself is frozen in both cases.
        same_last = _generate_last_for_pairs(
            flow,
            query_z,
            same_support,
            args,
            hint_seed=hint_seed,
        )
        cross_last = _generate_last_for_pairs(
            flow,
            query_z,
            cross_support,
            args,
            hint_seed=None if hint_seed is None else hint_seed + 10_000_019,
        )

    # One DDP forward over q / same-last / cross-last keeps the shared MetricEncoder
    # symmetric and avoids three DDP bookkeeping passes.
    all_z = torch.cat([query_z, same_last, cross_last], dim=0)
    with torch.autocast(
        device_type=images.device.type,
        dtype=torch.float16,
        enabled=args.amp and images.device.type == "cuda",
    ):
        all_raw, all_cloud = model(all_z)

    b = images.shape[0]
    query_raw = all_raw[:b]
    same_raw = all_raw[b:2 * b]
    cross_raw = all_raw[2 * b:]

    same_d, cross_d = _q2l_distances(
        query_raw,
        same_raw,
        cross_raw,
        pool_factor=args.ot_pool_factor,
        reg=args.sinkhorn_reg,
        n_iter=args.sinkhorn_iter,
        weight_eps=args.sk_weight_eps,
    )

    rank_loss = F.relu(args.triplet_margin + same_d - cross_d).mean()

    if args.visreg_weight > 0:
        vis_loss = visreg(all_cloud.unsqueeze(0).float())
    else:
        vis_loss = all_cloud.sum() * 0.0

    total = args.rank_weight * rank_loss + args.visreg_weight * vis_loss

    metrics = {
        "total": total.detach().float(),
        "rank": rank_loss.detach().float(),
        "vis": vis_loss.detach().float(),
        "pair_acc": (same_d.detach() < cross_d.detach()).float().mean(),
        "same_d": same_d.detach().mean(),
        "cross_d": cross_d.detach().mean(),
        "gap": (cross_d.detach() - same_d.detach()).mean(),
    }
    return total, metrics


@torch.no_grad()
def evaluate(
    model: Q2LMetricModel | DDP,
    rae,
    flow: FrozenFlowRuntime,
    visreg: VISReg,
    loader: DataLoader,
    sampler: RankAwarePKBatchSampler,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    sampler.set_epoch(0)
    sums = _empty_sums(device)
    batches = 0

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # Deterministic validation hint stream even when stochastic_hint=True.
        hint_seed = int(args.seed + 8_000_003 + batch_idx * 97_003)
        _, metrics = _compute_batch(
            model,
            rae,
            flow,
            visreg,
            images,
            labels,
            args,
            support_selection=args.val_support_selection,
            hint_seed=hint_seed,
        )
        for key, value in metrics.items():
            sums[key] += value
        batches += 1

    model.train()
    return _finalize_metrics(sums, batches)


def main() -> None:
    args = parse_args()
    device, rank, world_size, is_main = setup_distributed()
    seed_everything(args.seed, rank)
    os.makedirs(args.output_dir, exist_ok=True)

    if device.type != "cuda":
        raise RuntimeError("Q2L DiT Flow pretraining currently requires CUDA")
    if args.ot_pool_factor <= 0:
        raise ValueError("ot_pool_factor must be >=1")
    if args.samples_per_class < 2 or args.classes_per_batch < 2:
        raise ValueError("Q2L PK pretraining needs P>=2 and K>=2")
    if args.pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be >=1")

    # ------------------------------------------------------------------
    # Data: same RAE class-pool pipeline as M0-B.
    # ------------------------------------------------------------------
    all_train_records = load_records(args.train_root, args.train_json_glob)
    explicit_val = args.val_root is not None or args.val_json_glob is not None

    if explicit_val:
        train_records = all_train_records
        val_records = load_records(args.val_root, args.val_json_glob)
    else:
        train_records, val_records = _stratified_split(
            all_train_records,
            val_fraction=args.val_fraction,
            min_per_split=args.samples_per_class,
            seed=args.seed,
        )

    train_dataset = RecordDataset(
        train_records,
        build_rae_transform(args.image_size, augment=args.augment),
    )
    val_dataset = RecordDataset(
        val_records,
        build_rae_transform(args.image_size, augment=False),
        label_to_index=train_dataset.label_to_index,
    )

    train_sampler = RankAwarePKBatchSampler(
        train_dataset.targets,
        args.classes_per_batch,
        args.samples_per_class,
        args.batches_per_epoch,
        seed=args.seed,
        rank=rank,
    )
    val_sampler = RankAwarePKBatchSampler(
        val_dataset.targets,
        args.classes_per_batch,
        args.samples_per_class,
        args.val_batches,
        seed=args.seed + 777_777,
        rank=rank,
    )

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        **loader_kwargs,
    )

    # ------------------------------------------------------------------
    # Frozen RAE.
    # ------------------------------------------------------------------
    rae = build_rae(args.project_root, device, resolution=args.image_size)
    if hasattr(rae, "eval"):
        rae.eval()
    if hasattr(rae, "parameters"):
        for p in rae.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Frozen support-conditioned DiT Flow / EndoViT.
    # ------------------------------------------------------------------
    flow = FrozenFlowRuntime(
        project_root=args.project_root,
        checkpoint=args.dit_checkpoint,
        device=device,
        dit_variant=args.dit_variant,
        num_classes=args.num_classes,
        image_size=args.image_size,
        use_ema=args.use_ema,
        hint_signal_ratio=args.hint_signal_ratio,
        allow_condition_mismatch=args.allow_condition_mismatch,
        amp=args.amp,
    )
    args.hint_signal_ratio = flow.hint_signal_ratio

    # ------------------------------------------------------------------
    # Trainable M0-B MetricEncoder initialization / Q2L resume.
    # ------------------------------------------------------------------
    init_path = args.resume_checkpoint or args.init_m0b_checkpoint
    assert init_path is not None
    checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
    metric = _build_metric_from_checkpoint(checkpoint, args)
    raw_model = Q2LMetricModel(metric).to(device)

    if world_size > 1:
        model: Q2LMetricModel | DDP = DDP(
            raw_model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    else:
        model = raw_model

    visreg = VISReg(
        num_projections=args.visreg_num_proj,
        projection_chunk_size=args.visreg_projection_chunk,
    ).to(device)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp,
    )

    start_epoch = 0
    best_pair_acc = -math.inf
    best_gap = -math.inf
    if args.resume_checkpoint:
        start_epoch = int(checkpoint.get("epoch", 0))
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])
        best_pair_acc = float(checkpoint.get("best_pair_acc", -math.inf))
        best_gap = float(checkpoint.get("best_gap", -math.inf))

    if is_main:
        n_params = sum(p.numel() for p in raw_model.metric.parameters())
        batch_per_gpu = args.classes_per_batch * args.samples_per_class
        print("=" * 108)
        print("[Q2L-M0B-RAE] pure Query-to-Last FSL metric pretraining")
        print(f"stage                  = {STAGE}")
        print(f"M0-B/Q2L init          = {init_path}")
        print(f"DiT/Flow checkpoint    = {args.dit_checkpoint}")
        print(f"MetricEncoder params   = {n_params / 1e6:.3f} M")
        print(f"Metric architecture    = 1024 -> hidden={args.hidden_dim} -> out={args.metric_dim}")
        print(f"world_size             = {world_size}")
        print(f"PK / GPU               = {args.classes_per_batch} x {args.samples_per_class} = {batch_per_gpu}")
        print(f"Flow pairs / PK batch  = {2 * batch_per_gpu} (same + cross), chunk={args.pair_chunk_size}")
        print(f"Flow solver/steps      = {args.solver}/{args.num_steps}")
        print(f"hint alpha/stochastic  = {args.hint_signal_ratio}/{args.stochastic_hint}")
        print(f"OT pool/reg/iter       = {args.ot_pool_factor}/{args.sinkhorn_reg}/{args.sinkhorn_iter}")
        print(f"loss                   = {args.rank_weight}*Q2L_rank + {args.visreg_weight}*VISReg")
        print("distance               = Dynamic-SK Sinkhorn(Query Metric, generated Last Metric)")
        print("real support Metric    = NEVER used; support is Flow condition only")
        print("intermediate trajectory= NOT stored")
        print("target condition        = same class for positive / cross class for frozen negative probe")
        print("=" * 108)

    # ------------------------------------------------------------------
    # Training.
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        sums = _empty_sums(device)
        batches = 0

        for step, (images, labels, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            total, metrics = _compute_batch(
                model,
                rae,
                flow,
                visreg,
                images,
                labels,
                args,
                support_selection=args.support_selection,
                hint_seed=None,
            )

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            for key, value in metrics.items():
                sums[key] += value
            batches += 1

            if is_main and args.log_every > 0 and step % args.log_every == 0:
                local = {k: float(v / batches) for k, v in sums.items()}
                print(
                    f"[Q2L][E{epoch+1:03d}][{step:04d}/{len(train_loader):04d}] "
                    f"loss={local['total']:.4f} rank={local['rank']:.4f} "
                    f"pair={local['pair_acc']:.4f} "
                    f"same={local['same_d']:.4f} cross={local['cross_d']:.4f} "
                    f"gap={local['gap']:.4f} vis={local['vis']:.4f}"
                )

        train_metrics = _finalize_metrics(sums, batches)
        scheduler.step()

        # Save LAST before validation, following the robust M0-B convention.
        if is_main:
            last_state = dict(
                stage=STAGE,
                checkpoint_kind="last_after_train_before_val",
                epoch=epoch + 1,
                metric_encoder=raw_model.metric.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict() if scaler.is_enabled() else None,
                args=vars(args),
                classes=train_dataset.classes,
                train_metrics=train_metrics,
                best_pair_acc=best_pair_acc,
                best_gap=best_gap,
                selection_metric="val_q2l_pair_acc_then_gap",
                validation_completed=False,
                init_m0b_checkpoint=args.init_m0b_checkpoint,
                dit_checkpoint=args.dit_checkpoint,
                flow_condition_config=getattr(flow, "condition_config", {}),
                loaded_dit_key=getattr(flow, "loaded_dit_key", None),
            )
            save_checkpoint(
                os.path.join(args.output_dir, "metric_q2l_m0b_rae_last.pt"),
                **last_state,
            )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        val_metrics = evaluate(
            model,
            rae,
            flow,
            visreg,
            val_loader,
            val_sampler,
            device,
            args,
        )

        if is_main:
            print(
                f"[Q2L][{epoch+1:03d}/{args.epochs}] "
                f"TRAIN pair={train_metrics['pair_acc']:.4f} "
                f"same={train_metrics['same_d']:.4f} cross={train_metrics['cross_d']:.4f} "
                f"gap={train_metrics['gap']:.4f} | "
                f"VAL pair={val_metrics['pair_acc']:.4f} "
                f"same={val_metrics['same_d']:.4f} cross={val_metrics['cross_d']:.4f} "
                f"gap={val_metrics['gap']:.4f}"
            )

            improved = (
                val_metrics["pair_acc"] > best_pair_acc
                or (
                    math.isclose(val_metrics["pair_acc"], best_pair_acc)
                    and val_metrics["gap"] > best_gap
                )
            )
            if improved:
                best_pair_acc = val_metrics["pair_acc"]
                best_gap = val_metrics["gap"]
                best_state = dict(
                    stage=STAGE,
                    checkpoint_kind="best_after_validation",
                    epoch=epoch + 1,
                    metric_encoder=raw_model.metric.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    scaler=scaler.state_dict() if scaler.is_enabled() else None,
                    args=vars(args),
                    classes=train_dataset.classes,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    best_pair_acc=best_pair_acc,
                    best_gap=best_gap,
                    selection_metric="val_q2l_pair_acc_then_gap",
                    validation_completed=True,
                    init_m0b_checkpoint=args.init_m0b_checkpoint,
                    dit_checkpoint=args.dit_checkpoint,
                    flow_condition_config=getattr(flow, "condition_config", {}),
                    loaded_dit_key=getattr(flow, "loaded_dit_key", None),
                )
                save_checkpoint(
                    os.path.join(args.output_dir, "metric_q2l_m0b_rae.pt"),
                    **best_state,
                )
                print(
                    f"[Q2L] new best: pair={best_pair_acc:.4f}, gap={best_gap:.4f} "
                    "-> metric_q2l_m0b_rae.pt"
                )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
