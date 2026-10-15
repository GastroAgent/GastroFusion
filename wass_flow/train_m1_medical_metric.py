#!/usr/bin/env python3
"""M1: non-episodic medical-domain SK-Wasserstein metric adaptation.

Purpose
-------
Adapt the M0-B ImageNet/general metric encoder to medical/endoscopy images
before episodic FSL calibration.  The stage is deliberately *not* episodic.
It uses class-balanced P x K medical batches and optimizes local distribution
separation with dynamic-SK cosine Sinkhorn distances:

    L = lambda_ot * relu(margin + D_pos - D_neg)
        + lambda_cls * CE_medical
        + lambda_vis * VISReg

Key M1 differences from M0-B
----------------------------
1. Initialize only the MetricEncoder from ``metric_imagenet_ot.pt``.
   The ImageNet classifier is NOT reused; a fresh medical classifier is built.
2. Default OT uses the full 16x16 = 256 metric nodes (pool_factor=1), matching
   the downstream M2/F1 geometry more closely.
3. Use a smaller learning rate and weaker CE semantic anchor by default to
   adapt rather than overwrite the M0-B metric space.
4. Validation uses deterministic positive/negative selection by default and
   best checkpoint selection is based on medical ``val_pair_acc`` (gap breaks
   ties).

The RAE is always frozen.  Supports single-GPU and torchrun DDP.
"""

from __future__ import annotations

# import debugpy
# try:
#     print("Waiting for debugger attach")
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9504))
#     # debugpy.listen(("172.16.0.108", 9504)) 
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import argparse
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

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


class MedicalMetricModel(nn.Module):
    """Shared metric encoder plus a disposable medical classification head."""

    def __init__(self, metric: MetricEncoder, num_classes: int) -> None:
        super().__init__()
        self.metric = metric
        self.classifier = nn.Linear(metric.out_dim, int(num_classes))
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, latent: torch.Tensor):
        features = self.metric(latent)
        logits = self.classifier(features.cloud)
        return features, logits


class RankAwarePKBatchSampler(Sampler[list[int]]):
    """P-class x K-image sampler with deterministic rank-specific streams."""

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
            counts = sorted(
                ((cls, len(indices)) for cls, indices in self.class_to_indices.items()),
                key=lambda x: x[1],
            )
            preview = ", ".join(f"{c}:{n}" for c, n in counts[:10])
            raise ValueError(
                f"Need P={self.p} classes with >=K={self.k} images; only "
                f"{len(self.classes)} classes qualify. Smallest class counts: {preview}"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1_000_003 + self.rank * 100_003)
        for _ in range(self.batches_per_epoch):
            selected = rng.sample(self.classes, self.p)
            batch: list[int] = []
            for cls in selected:
                batch.extend(rng.sample(self.class_to_indices[cls], self.k))
            rng.shuffle(batch)
            yield batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "M1: medical-domain dynamic-SK cosine-Sinkhorn metric adaptation"
    )

    train_src = p.add_mutually_exclusive_group(required=True)
    train_src.add_argument(
        "--train_root", type=str,
        help="Medical ImageFolder root: root/class_name/images",
    )
    train_src.add_argument(
        "--train_json_glob", type=str,
        help="Glob for medical class-pool JSON files",
    )

    val_src = p.add_mutually_exclusive_group(required=False)
    val_src.add_argument("--val_root", type=str, default=None)
    val_src.add_argument("--val_json_glob", type=str, default=None)
    p.add_argument(
        "--val_fraction", type=float, default=0.10,
        help="Per-class split fraction when no explicit validation source is supplied",
    )

    ckpt = p.add_mutually_exclusive_group(required=True)
    ckpt.add_argument(
        "--init_checkpoint", type=str,
        help="M0-B metric_imagenet_ot.pt. Only MetricEncoder weights are inherited.",
    )
    ckpt.add_argument(
        "--resume_checkpoint", type=str,
        help="Resume an M1 checkpoint including classifier/optimizer/scheduler.",
    )

    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--rae_chunk_size", type=int, default=8)

    # PK values are per GPU in DDP. Medical domains commonly have fewer classes
    # than ImageNet, so P=8 is a safer default than M0-B's P=16.
    p.add_argument("--classes_per_batch", type=int, default=8)
    p.add_argument("--samples_per_class", type=int, default=4)
    p.add_argument(
        "--val_classes_per_batch", type=int, default=0,
        help="0 means use classes_per_batch",
    )
    p.add_argument("--val_samples_per_class", type=int, default=2)
    p.add_argument("--batches_per_epoch", type=int, default=300)
    p.add_argument("--val_batches", type=int, default=100)

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=4)

    # MetricEncoder compatibility args. Current V1 implementation is fixed at
    # 512 hidden / 3 Transformer blocks / ~10.25M parameters.
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--metric_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)

    # Full 256 nodes by default in M1; use 2 only if memory/runtime requires it.
    p.add_argument("--ot_pool_factor", type=int, default=1)
    p.add_argument("--sinkhorn_reg", type=float, default=0.1)
    p.add_argument("--sinkhorn_iter", type=int, default=30)
    p.add_argument("--sk_weight_eps", type=float, default=1e-3)
    p.add_argument("--triplet_margin", type=float, default=0.2)
    p.add_argument(
        "--mining", choices=["cloud_hard", "random"], default="cloud_hard",
        help="Pair selection only; optimization distance is always dynamic-SK Sinkhorn",
    )
    p.add_argument(
        "--val_mining", choices=["fixed", "cloud_hard"], default="fixed",
        help="Fixed validation pairs make epoch-to-epoch comparison stable",
    )
    p.add_argument(
        "--hard_mining_start_epoch", type=int, default=0,
        help="If >0, use random pairs before this 0-based epoch, then switch to --mining",
    )

    # M1 is adaptation: OT dominates, CE is a weaker medical semantic anchor.
    p.add_argument("--ot_weight", type=float, default=1.0)
    p.add_argument("--cls_weight_start", type=float, default=0.20)
    p.add_argument("--cls_weight_end", type=float, default=0.05)
    p.add_argument("--visreg_weight", type=float, default=1e-2)
    p.add_argument("--visreg_num_proj", type=int, default=512)
    p.add_argument("--visreg_projection_chunk", type=int, default=128)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log_every", type=int, default=25)
    return p.parse_args()


def _stratified_medical_split(
    records: list[ImageRecord],
    *,
    val_fraction: float,
    min_train: int,
    min_val: int,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Per-class split while guaranteeing enough samples for train/val PK batches."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0,1)")
    if min_train < 2 or min_val < 2:
        raise ValueError("M1 train/val PK sampling requires at least 2 samples/class")

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
        if n < min_train + min_val:
            too_small.append((label, n))
            continue
        n_val = max(min_val, int(round(n * val_fraction)))
        n_val = min(n_val, n - min_train)
        val_records.extend(items[:n_val])
        train_records.extend(items[n_val:])

    if too_small:
        preview = ", ".join(f"{label}:{n}" for label, n in too_small[:20])
        raise ValueError(
            "Automatic M1 split cannot preserve PK sampling for every class. "
            f"Need at least train_K({min_train}) + val_K({min_val}) images/class. "
            f"Too-small classes: {preview}. Provide an explicit val source or lower K."
        )
    return train_records, val_records


def _load_initial_state(
    args: argparse.Namespace,
    metric: MetricEncoder,
    classifier: nn.Linear,
    medical_classes: list[str],
) -> tuple[dict, int]:
    """Load M0-B metric for init, or full M1 state for resume."""
    path = args.resume_checkpoint or args.init_checkpoint
    assert path is not None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    state = checkpoint.get("metric_encoder", checkpoint.get("model", checkpoint))
    metric.load_state_dict(state, strict=True)

    if args.init_checkpoint:
        # Intentional: M0-B classifier lives in ImageNet/general class space.
        # M1's medical classifier must start fresh.
        start_epoch = 0
        return checkpoint, start_epoch

    # Resume path must be a genuine M1 checkpoint with the same medical labels.
    saved_stage = str(checkpoint.get("stage", ""))
    if saved_stage and saved_stage != "M1":
        raise RuntimeError(
            f"--resume_checkpoint expects stage='M1', got stage={saved_stage!r}. "
            "Use --init_checkpoint for M0-B checkpoints."
        )
    saved_classes = checkpoint.get("classes")
    if saved_classes is not None and list(saved_classes) != list(medical_classes):
        raise RuntimeError(
            "Medical class mapping changed since the M1 checkpoint. "
            f"checkpoint={list(saved_classes)}, current={list(medical_classes)}"
        )
    cls_state = checkpoint.get("classifier")
    if cls_state is None:
        raise RuntimeError("M1 resume checkpoint has no 'classifier' state")
    classifier.load_state_dict(cls_state, strict=True)
    return checkpoint, int(checkpoint.get("epoch", 0))


def _mine_indices(
    cloud: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    b = labels.numel()
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(b, dtype=torch.bool, device=labels.device)
    pos_mask = same & ~eye
    neg_mask = ~same

    if not bool(pos_mask.any(dim=1).all()):
        raise RuntimeError("Every anchor needs >=1 positive inside the PK batch")
    if not bool(neg_mask.any(dim=1).all()):
        raise RuntimeError("Every anchor needs >=1 negative inside the PK batch")

    if mode == "cloud_hard":
        z = F.normalize(cloud.detach().float(), p=2, dim=-1)
        d = 1.0 - z @ z.t()
        pos_idx = d.masked_fill(~pos_mask, -torch.inf).argmax(dim=1)
        neg_idx = d.masked_fill(~neg_mask, torch.inf).argmin(dim=1)
        return pos_idx, neg_idx

    if mode == "random":
        pos_idx = torch.multinomial(pos_mask.float(), 1).squeeze(1)
        neg_idx = torch.multinomial(neg_mask.float(), 1).squeeze(1)
        return pos_idx, neg_idx

    if mode == "fixed":
        pos_idx = pos_mask.to(torch.int64).argmax(dim=1)
        neg_idx = neg_mask.to(torch.int64).argmax(dim=1)
        return pos_idx, neg_idx

    raise ValueError(mode)


def _pair_ot_distances(
    raw_tokens: torch.Tensor,
    pos_idx: torch.Tensor,
    neg_idx: torch.Tensor,
    *,
    pool_factor: int,
    reg: float,
    n_iter: int,
    weight_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = pool_square_tokens(raw_tokens, pool_factor)
    normalized = F.normalize(raw, p=2, dim=-1)

    # 2B aligned pairs in one Sinkhorn call: [anchor-positive, anchor-negative].
    q_norm = torch.cat([normalized, normalized], dim=0)
    s_norm = torch.cat([normalized[pos_idx], normalized[neg_idx]], dim=0)
    q_raw = torch.cat([raw, raw], dim=0)
    s_raw = torch.cat([raw[pos_idx], raw[neg_idx]], dim=0)

    distances = aligned_sk_sinkhorn_distance(
        q_norm,
        s_norm,
        query_raw=q_raw,
        support_raw=s_raw,
        reg=reg,
        n_iter=n_iter,
        weight_eps=weight_eps,
        return_transport=False,
    ).distance.float()
    b = raw.shape[0]
    return distances[:b], distances[b:]


def _linear_weight(start: float, end: float, epoch: int, epochs: int) -> float:
    if epochs <= 1:
        return float(end)
    p = epoch / float(epochs - 1)
    return float(start + p * (end - start))


def _all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    out = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def _empty_sums(device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "total", "rank", "cls", "vis", "cls_acc",
        "pair_acc", "pos_d", "neg_d", "gap", "active_triplet",
    )
    return {k: torch.zeros((), device=device, dtype=torch.float32) for k in keys}


def _finalize_metrics(
    sums: dict[str, torch.Tensor], batches: int
) -> dict[str, float]:
    count = torch.tensor(float(batches), device=next(iter(sums.values())).device)
    count = _all_reduce_sum(count).clamp_min(1.0)
    return {
        key: float((_all_reduce_sum(value) / count).cpu())
        for key, value in sums.items()
    }


def _compute_batch(
    model: MedicalMetricModel | DDP,
    visreg: VISReg,
    latent: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    *,
    cls_weight: float,
    mining_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    features, logits = model(latent)
    pos_idx, neg_idx = _mine_indices(features.cloud, labels, mining_mode)
    pos_d, neg_d = _pair_ot_distances(
        features.raw,
        pos_idx,
        neg_idx,
        pool_factor=args.ot_pool_factor,
        reg=args.sinkhorn_reg,
        n_iter=args.sinkhorn_iter,
        weight_eps=args.sk_weight_eps,
    )

    triplet = args.triplet_margin + pos_d - neg_d
    rank_loss = F.relu(triplet).mean()
    cls_loss = F.cross_entropy(logits.float(), labels)
    vis_loss = visreg(features.cloud.unsqueeze(0).float())
    total = (
        args.ot_weight * rank_loss
        + cls_weight * cls_loss
        + args.visreg_weight * vis_loss
    )

    metrics = {
        "total": total.detach().float(),
        "rank": rank_loss.detach().float(),
        "cls": cls_loss.detach().float(),
        "vis": vis_loss.detach().float(),
        "cls_acc": (logits.detach().argmax(dim=-1) == labels).float().mean(),
        "pair_acc": (pos_d.detach() < neg_d.detach()).float().mean(),
        "pos_d": pos_d.detach().mean(),
        "neg_d": neg_d.detach().mean(),
        "gap": (neg_d.detach() - pos_d.detach()).mean(),
        "active_triplet": (triplet.detach() > 0).float().mean(),
    }
    return total, metrics


@torch.no_grad()
def evaluate(
    model: MedicalMetricModel | DDP,
    rae,
    visreg: VISReg,
    loader: DataLoader,
    sampler: RankAwarePKBatchSampler,
    device: torch.device,
    args: argparse.Namespace,
    cls_weight: float,
) -> dict[str, float]:
    model.eval()
    sampler.set_epoch(0)  # fixed stream across epochs
    sums = _empty_sums(device)
    batches = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        latent = encode_rae_chunked(rae, images, args.rae_chunk_size)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            _, metrics = _compute_batch(
                model,
                visreg,
                latent,
                labels,
                args,
                cls_weight=cls_weight,
                mining_mode=args.val_mining,
            )
        for key, value in metrics.items():
            sums[key] += value
        batches += 1

    return _finalize_metrics(sums, batches)


def main() -> None:
    args = parse_args()
    device, rank, world_size, is_main = setup_distributed()
    seed_everything(args.seed, rank)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.ot_pool_factor <= 0:
        raise ValueError("ot_pool_factor must be >=1")
    if args.samples_per_class < 2:
        raise ValueError("samples_per_class must be >=2")
    if args.val_samples_per_class < 2:
        raise ValueError("val_samples_per_class must be >=2")
    if args.classes_per_batch < 2:
        raise ValueError("classes_per_batch must be >=2")

    val_p = args.val_classes_per_batch or args.classes_per_batch
    if val_p < 2:
        raise ValueError("val_classes_per_batch must be >=2")

    # ------------------------------------------------------------------
    # Medical data
    # ------------------------------------------------------------------
    all_train_records = load_records(args.train_root, args.train_json_glob)
    explicit_val = args.val_root is not None or args.val_json_glob is not None

    if explicit_val:
        train_records = all_train_records
        val_records = load_records(args.val_root, args.val_json_glob)
    else:
        train_records, val_records = _stratified_medical_split(
            all_train_records,
            val_fraction=args.val_fraction,
            min_train=args.samples_per_class,
            min_val=args.val_samples_per_class,
            seed=args.seed,
        )

    train_dataset = RecordDataset(
        train_records,
        build_rae_transform(args.image_size, augment=True),
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
        val_p,
        args.val_samples_per_class,
        args.val_batches,
        seed=args.seed + 777_777,
        rank=rank,
    )

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **loader_kwargs)

    # ------------------------------------------------------------------
    # Frozen RAE + M0-B initialized metric + fresh medical classifier
    # ------------------------------------------------------------------
    rae = build_rae(args.project_root, device, resolution=args.image_size)
    metric = MetricEncoder(
        1024,
        args.hidden_dim,
        args.metric_dim,
        dropout=args.dropout,
    )
    classifier = nn.Linear(metric.out_dim, len(train_dataset.classes))
    nn.init.trunc_normal_(classifier.weight, std=0.02)
    nn.init.zeros_(classifier.bias)

    checkpoint, start_epoch = _load_initial_state(
        args,
        metric,
        classifier,
        train_dataset.classes,
    )

    model = MedicalMetricModel(metric, len(train_dataset.classes))
    model.classifier = classifier
    model = model.to(device)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )
    raw_model = model.module if isinstance(model, DDP) else model

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
        enabled=args.amp and device.type == "cuda",
    )

    best_pair_acc = -math.inf
    best_gap = -math.inf
    if args.resume_checkpoint:
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
        init_stage = checkpoint.get("stage", "unknown")
        source_classes = checkpoint.get("classes")
        print("=" * 100)
        print("[M1] Medical-domain SK-Wasserstein metric adaptation")
        print(f"init checkpoint stage   = {init_stage}")
        print(f"source class count      = {len(source_classes) if source_classes is not None else 'unknown'}")
        print(f"medical class count     = {len(train_dataset.classes)}")
        print(f"medical classifier      = fresh init ({len(train_dataset.classes)} classes)")
        print(f"metric params           = {n_params / 1e6:.3f} M")
        print(f"world_size              = {world_size}")
        print(
            f"train PK / GPU          = {args.classes_per_batch} x "
            f"{args.samples_per_class} = {args.classes_per_batch * args.samples_per_class}"
        )
        print(
            f"val PK / GPU            = {val_p} x {args.val_samples_per_class} = "
            f"{val_p * args.val_samples_per_class}"
        )
        print(
            f"OT nodes                = {256 // (args.ot_pool_factor ** 2)} "
            f"(pool_factor={args.ot_pool_factor})"
        )
        print(f"train/val mining        = {args.mining} / {args.val_mining}")
        print(f"train images/classes    = {len(train_dataset)} / {len(train_dataset.classes)}")
        print(f"val images              = {len(val_dataset)}")
        print(f"LR                      = {args.lr:g}")
        print(
            f"loss weights            = OT {args.ot_weight:g}, "
            f"CE {args.cls_weight_start:g}->{args.cls_weight_end:g}, "
            f"VIS {args.visreg_weight:g}"
        )
        print(f"start epoch             = {start_epoch}")
        print("=" * 100)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        cls_weight = _linear_weight(
            args.cls_weight_start,
            args.cls_weight_end,
            epoch,
            args.epochs,
        )
        train_mining = (
            "random" if epoch < args.hard_mining_start_epoch else args.mining
        )

        sums = _empty_sums(device)
        batches = 0

        for step, (images, labels, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            latent = encode_rae_chunked(rae, images, args.rae_chunk_size)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                total, metrics = _compute_batch(
                    model,
                    visreg,
                    latent,
                    labels,
                    args,
                    cls_weight=cls_weight,
                    mining_mode=train_mining,
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
                    f"[M1][E{epoch+1:03d}][{step:04d}/{len(train_loader):04d}] "
                    f"loss={local['total']:.4f} rank={local['rank']:.4f} "
                    f"pair={local['pair_acc']:.4f} gap={local['gap']:.4f} "
                    f"D+={local['pos_d']:.4f} D-={local['neg_d']:.4f} "
                    f"active={local['active_triplet']:.3f} "
                    f"cls={local['cls']:.4f}/{local['cls_acc']:.4f} "
                    f"vis={local['vis']:.4f} cls_w={cls_weight:.3f} "
                    f"mining={train_mining}"
                )

        train_metrics = _finalize_metrics(sums, batches)
        val_metrics = evaluate(
            model,
            rae,
            visreg,
            val_loader,
            val_sampler,
            device,
            args,
            cls_weight,
        )
        scheduler.step()

        if is_main:
            print(
                f"[M1][{epoch+1:03d}/{args.epochs}] "
                f"TRAIN pair={train_metrics['pair_acc']:.4f} "
                f"D+={train_metrics['pos_d']:.4f} D-={train_metrics['neg_d']:.4f} "
                f"gap={train_metrics['gap']:.4f} cls_acc={train_metrics['cls_acc']:.4f} | "
                f"VAL pair={val_metrics['pair_acc']:.4f} "
                f"D+={val_metrics['pos_d']:.4f} D-={val_metrics['neg_d']:.4f} "
                f"gap={val_metrics['gap']:.4f} cls_acc={val_metrics['cls_acc']:.4f}"
            )

            pair = val_metrics["pair_acc"]
            gap = val_metrics["gap"]
            improved = (
                pair > best_pair_acc + 1e-8
                or (abs(pair - best_pair_acc) <= 1e-8 and gap > best_gap)
            )
            if improved:
                best_pair_acc = pair
                best_gap = gap

            state = dict(
                stage="M1",
                epoch=epoch + 1,
                source_checkpoint=(args.init_checkpoint or args.resume_checkpoint),
                source_stage=checkpoint.get("stage", "unknown"),
                metric_encoder=raw_model.metric.state_dict(),
                classifier=raw_model.classifier.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict() if scaler.is_enabled() else None,
                args=vars(args),
                classes=train_dataset.classes,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                best_pair_acc=best_pair_acc,
                best_gap=best_gap,
                selection_metric="val_pair_acc_then_gap",
            )
            save_checkpoint(
                os.path.join(args.output_dir, "metric_medical_adapt_last.pt"),
                **state,
            )
            if improved:
                save_checkpoint(
                    os.path.join(args.output_dir, "metric_medical_adapt.pt"),
                    **state,
                )
                print(
                    f"[M1] new best: val_pair_acc={best_pair_acc:.4f}, "
                    f"gap={best_gap:.4f} -> metric_medical_adapt.pt"
                )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
