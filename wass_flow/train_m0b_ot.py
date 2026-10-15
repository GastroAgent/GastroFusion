#!/usr/bin/env python3
"""M0-B: ImageNet/general SK-Wasserstein metric pretraining.

Stage objective
---------------
Start from the M0-A classification checkpoint and adapt the shared MetricEncoder
from a purely semantic classification representation to a local distribution
metric suitable for the downstream SK-Wasserstein FSL pipeline.

The training batch is class-balanced P x K.  For every anchor we mine one
positive and one negative inside the current batch, then optimize

    L = lambda_ot * relu(margin + D_pos - D_neg)
        + lambda_cls * CE
        + lambda_vis * VISReg

where D_pos / D_neg are cosine-ground-cost Sinkhorn distances using dynamic
pair-conditioned SK node masses.  The MetricEncoder is trainable; the RAE is
always frozen.  The M0-A classifier is kept only as a semantic anchor.

Default V1 choices
------------------
* P=16 classes, K=4 images / class / GPU
* 16x16 metric tokens are pooled to 8x8 only for OT (factor=2)
* hard positive / hard negative mining uses pooled cloud cosine distance
* lambda_cls linearly decays 0.5 -> 0.1
* lambda_vis = 1e-3
* best checkpoint is selected by validation pair-order accuracy: D_pos < D_neg

This script supports both single-GPU and torchrun DDP.  In DDP each rank draws
an independent PK batch stream; gradients are synchronized normally by DDP.
"""

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


class PairMetricModel(nn.Module):
    """Metric encoder plus the disposable M0-A classification head."""

    def __init__(self, metric: MetricEncoder, num_classes: int = 1122) -> None:
        super().__init__()
        self.metric = metric
        self.classifier = nn.Linear(metric.out_dim, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, latent: torch.Tensor):
        features = self.metric(latent)
        logits = self.classifier(features.cloud)
        return features, logits


class RankAwarePKBatchSampler(Sampler[list[int]]):
    """Class-balanced P x K sampler with an independent deterministic stream per rank."""

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
        # Large coprime-ish rank offset avoids identical PK streams across GPUs.
        rng = random.Random(self.seed + self.epoch * 1_000_003 + self.rank * 100_003)
        for _ in range(self.batches_per_epoch):
            chosen_classes = rng.sample(self.classes, self.p)
            batch: list[int] = []
            for cls in chosen_classes:
                batch.extend(rng.sample(self.class_to_indices[cls], self.k))
            rng.shuffle(batch)
            yield batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "M0-B: ImageNet/general dynamic-SK cosine-Sinkhorn metric pretraining"
    )

    train_src = p.add_mutually_exclusive_group(required=True)
    train_src.add_argument("--train_root", type=str, help="ImageFolder root: root/class/images")
    train_src.add_argument("--train_json_glob", type=str, help="Glob for class-pool JSON files")

    val_src = p.add_mutually_exclusive_group(required=False)
    val_src.add_argument("--val_root", type=str, default=None)
    val_src.add_argument("--val_json_glob", type=str, default=None)
    p.add_argument(
        "--val_fraction", type=float, default=0.05,
        help="Per-class validation fraction when no explicit validation source is supplied",
    )

    ckpt = p.add_mutually_exclusive_group(required=True)
    ckpt.add_argument(
        "--init_checkpoint", type=str,
        help="M0-A metric_cls_pretrain.pt used as initialization",
    )
    ckpt.add_argument(
        "--resume_checkpoint", type=str,
        help="Resume a previously saved M0-B checkpoint including optimizer/scheduler",
    )

    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--rae_chunk_size", type=int, default=8)

    # PK batch. Values are per GPU in DDP.
    p.add_argument("--classes_per_batch", type=int, default=16)
    p.add_argument("--samples_per_class", type=int, default=4)
    p.add_argument("--batches_per_epoch", type=int, default=500)
    p.add_argument("--val_batches", type=int, default=100)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=8)

    # Kept compatible with the current MetricEncoder constructor. The V1
    # transformer implementation internally fixes its width at 512 / depth 3.
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--metric_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)

    # OT / mining.
    p.add_argument(
        "--ot_pool_factor", type=int, default=2,
        help="1: use 16x16=256 nodes; 2: pool to 8x8=64 nodes for M0-B OT",
    )
    p.add_argument("--sinkhorn_reg", type=float, default=0.1)
    p.add_argument("--sinkhorn_iter", type=int, default=30)
    p.add_argument("--sk_weight_eps", type=float, default=1e-3)
    p.add_argument("--triplet_margin", type=float, default=0.2)
    p.add_argument(
        "--mining", choices=["cloud_hard", "random"], default="cloud_hard",
        help="Training pair mining; actual optimization distance is always SK-OT",
    )
    p.add_argument(
        "--val_mining", choices=["fixed", "cloud_hard"], default="fixed",
        help="Validation uses fixed pairs by default so best-checkpoint comparison is stable",
    )

    # Loss weights.
    p.add_argument("--ot_weight", type=float, default=1.0)
    p.add_argument("--cls_weight_start", type=float, default=0.4)
    p.add_argument("--cls_weight_end", type=float, default=0.1)
    p.add_argument("--visreg_weight", type=float, default=1e-2)
    p.add_argument("--visreg_num_proj", type=int, default=256)
    p.add_argument("--visreg_projection_chunk", type=int, default=128)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--strict_class_match", action=argparse.BooleanOptionalAction, default=False,
        help="Require current training classes to exactly match the M0-A checkpoint classes",
    )
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def _stratified_split(
    records: list[ImageRecord],
    val_fraction: float,
    min_per_split: int,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split each class independently so PK validation remains valid."""
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
            "Stratified M0-B split requires at least "
            f"{2 * min_per_split} samples per class. Too-small classes: {preview}"
        )
    return train_records, val_records


def _checkpoint_classes_ok(checkpoint: dict, dataset: RecordDataset, strict: bool) -> None:
    classes = checkpoint.get("classes")
    if classes is None:
        if strict:
            raise RuntimeError(
                "Checkpoint has no 'classes' field. Use a checkpoint saved by train_m0a_cls.py "
                "or pass --no-strict_class_match if you intentionally accept this."
            )
        return
    old = list(classes)
    new = list(dataset.classes)
    if old != new:
        msg = (
            f"Class mapping mismatch: checkpoint has {len(old)} classes, "
            f"current data has {len(new)} classes."
        )
        if strict:
            raise RuntimeError(msg)
        if len(old) != len(new):
            # Classifier shape cannot be loaded if class counts differ.
            print(f"[WARN] {msg} Classifier will be re-initialized.")


def _load_initial_state(
    args: argparse.Namespace,
    metric: MetricEncoder,
    classifier: nn.Linear,
) -> tuple[dict, int]:
    path = args.resume_checkpoint or args.init_checkpoint
    assert path is not None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metric_state = checkpoint.get("metric_encoder", checkpoint.get("model", checkpoint))
    metric.load_state_dict(metric_state, strict=True)

    cls_state = checkpoint.get("classifier")
    if cls_state is None:
        if args.init_checkpoint:
            raise RuntimeError(
                "M0-B expects the M0-A classification head in init_checkpoint, but no "
                "'classifier' state was found."
            )
    else:
        try:
            classifier.load_state_dict(cls_state, strict=True)
        except RuntimeError:
            if args.strict_class_match:
                raise
            print("[WARN] Classifier shape/mapping mismatch; classifier re-initialized.")

    start_epoch = int(checkpoint.get("epoch", 0)) if args.resume_checkpoint else 0
    return checkpoint, start_epoch


def _mine_indices(
    cloud: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one positive and one negative sample index for every anchor."""
    b = labels.numel()
    if b < 4:
        raise ValueError("PK batch is too small for pair mining")

    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(b, dtype=torch.bool, device=labels.device)
    pos_mask = same & ~eye
    neg_mask = ~same

    if not bool(pos_mask.any(dim=1).all()):
        raise RuntimeError("Every anchor needs at least one positive in the PK batch")
    if not bool(neg_mask.any(dim=1).all()):
        raise RuntimeError("Every anchor needs at least one negative in the PK batch")

    if mode == "cloud_hard":
        # Mining is a discrete selection step. Detach so no gradient path is
        # implied through argmin/argmax. The actual optimization uses SK-OT.
        z = F.normalize(cloud.detach().float(), p=2, dim=-1)
        d = 1.0 - z @ z.t()
        pos_d = d.masked_fill(~pos_mask, -torch.inf)
        neg_d = d.masked_fill(~neg_mask, torch.inf)
        pos_idx = pos_d.argmax(dim=1)   # farthest same-class sample
        neg_idx = neg_d.argmin(dim=1)   # closest different-class sample
        return pos_idx, neg_idx

    if mode == "random":
        # torch.multinomial keeps sampling on-device and honors the process seed.
        pos_prob = pos_mask.float()
        neg_prob = neg_mask.float()
        pos_idx = torch.multinomial(pos_prob, 1).squeeze(1)
        neg_idx = torch.multinomial(neg_prob, 1).squeeze(1)
        return pos_idx, neg_idx

    if mode == "fixed":
        # Deterministic pair selection for comparable validation across epochs.
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
    """Compute aligned positive and negative dynamic-SK Sinkhorn distances."""
    raw = pool_square_tokens(raw_tokens, pool_factor)
    normalized = F.normalize(raw, p=2, dim=-1)

    # Concatenate 2B aligned pairs so Sinkhorn is called only once.
    query_norm = torch.cat([normalized, normalized], dim=0)
    support_norm = torch.cat([normalized[pos_idx], normalized[neg_idx]], dim=0)
    query_raw = torch.cat([raw, raw], dim=0)
    support_raw = torch.cat([raw[pos_idx], raw[neg_idx]], dim=0)

    distances = aligned_sk_sinkhorn_distance(
        query_norm,
        support_norm,
        query_raw=query_raw,
        support_raw=support_raw,
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
    progress = epoch / float(epochs - 1)
    return float(start + progress * (end - start))


def _all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    out = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def _finalize_metrics(sums: dict[str, torch.Tensor], batches: int) -> dict[str, float]:
    count = torch.tensor(float(batches), device=next(iter(sums.values())).device)
    count = _all_reduce_sum(count).clamp_min(1.0)
    result: dict[str, float] = {}
    for key, value in sums.items():
        result[key] = float((_all_reduce_sum(value) / count).cpu())
    return result


def _empty_sums(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: torch.zeros((), device=device, dtype=torch.float32)
        for key in (
            "total", "rank", "cls", "vis", "cls_acc",
            "pair_acc", "pos_d", "neg_d", "gap",
        )
    }


def _compute_batch(
    model: PairMetricModel | DDP,
    visreg: VISReg,
    latent: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    cls_weight: float,
    mining_mode: str | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # Always call the wrapped model itself so DDP performs its normal forward
    # bookkeeping before backward/all-reduce.
    features, logits = model(latent)
    pos_idx, neg_idx = _mine_indices(
        features.cloud, labels, mining_mode or args.mining
    )
    pos_d, neg_d = _pair_ot_distances(
        features.raw,
        pos_idx,
        neg_idx,
        pool_factor=args.ot_pool_factor,
        reg=args.sinkhorn_reg,
        n_iter=args.sinkhorn_iter,
        weight_eps=args.sk_weight_eps,
    )

    rank_loss = F.relu(args.triplet_margin + pos_d - neg_d).mean()
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
    }
    return total, metrics


@torch.no_grad()
def evaluate(
    model: PairMetricModel | DDP,
    rae,
    visreg: VISReg,
    loader: DataLoader,
    sampler: RankAwarePKBatchSampler,
    device: torch.device,
    args: argparse.Namespace,
    cls_weight: float,
) -> dict[str, float]:
    model.eval()
    sampler.set_epoch(0)  # identical validation stream across training epochs
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
                model, visreg, latent, labels, args, cls_weight,
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
        raise ValueError("ot_pool_factor must be >= 1")
    if args.samples_per_class < 2 or args.classes_per_batch < 2:
        raise ValueError("M0-B needs at least P=2 and K=2")

    # ------------------------------------------------------------------
    # Data
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

    train_transform = build_rae_transform(args.image_size, augment=True)
    val_transform = build_rae_transform(args.image_size, augment=False)
    train_dataset = RecordDataset(train_records, train_transform)
    val_dataset = RecordDataset(
        val_records,
        val_transform,
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
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **loader_kwargs)

    # ------------------------------------------------------------------
    # Frozen RAE + M0-A initialized metric encoder / classifier
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

    checkpoint, start_epoch = _load_initial_state(args, metric, classifier)
    _checkpoint_classes_ok(checkpoint, train_dataset, args.strict_class_match)

    model = PairMetricModel(metric, len(train_dataset.classes))
    # PairMetricModel created a new classifier; replace it with the initialized one.
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
    if args.resume_checkpoint:
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])
        best_pair_acc = float(checkpoint.get("best_pair_acc", -math.inf))

    if is_main:
        n_params = sum(p.numel() for p in raw_model.metric.parameters())
        batch_per_gpu = args.classes_per_batch * args.samples_per_class
        print("=" * 100)
        print("[M0-B] ImageNet/general SK-Wasserstein metric pretraining")
        print(f"metric params          = {n_params / 1e6:.3f} M")
        print(f"world_size             = {world_size}")
        print(f"PK / GPU               = {args.classes_per_batch} x {args.samples_per_class} = {batch_per_gpu}")
        print(f"effective images/batch = {batch_per_gpu * world_size}")
        print(f"OT nodes               = {256 // (args.ot_pool_factor ** 2)} (pool_factor={args.ot_pool_factor})")
        print(f"train/val mining       = {args.mining} / {args.val_mining}")
        print(f"train images/classes   = {len(train_dataset)} / {len(train_dataset.classes)}")
        print(f"val images             = {len(val_dataset)}")
        print(f"start epoch            = {start_epoch}")
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
                    model, visreg, latent, labels, args, cls_weight
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
                    f"[M0-B][E{epoch+1:03d}][{step:04d}/{len(train_loader):04d}] "
                    f"loss={local['total']:.4f} rank={local['rank']:.4f} "
                    f"pair={local['pair_acc']:.4f} gap={local['gap']:.4f} "
                    f"cls={local['cls']:.4f}/{local['cls_acc']:.4f} "
                    f"vis={local['vis']:.4f} cls_w={cls_weight:.3f}"
                )
        # --------------------------------------------------------------
        # 1) TRAIN epoch completed -> save LAST immediately, BEFORE validation.
        #
        # This is intentional: if validation or best-checkpoint saving fails,
        # the just-finished training epoch is still recoverable from LAST.
        # Step the scheduler first so LAST is resume-ready for the next epoch.
        # --------------------------------------------------------------
        train_metrics = _finalize_metrics(sums, batches)
        scheduler.step()

        if is_main:
            last_state = dict(
                stage="M0-B",
                checkpoint_kind="last_after_train_before_val",
                epoch=epoch + 1,
                metric_encoder=raw_model.metric.state_dict(),
                classifier=raw_model.classifier.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict() if scaler.is_enabled() else None,
                args=vars(args),
                classes=train_dataset.classes,
                train_metrics=train_metrics,
                # This is the best score from PREVIOUS completed validations.
                # The current epoch has deliberately not been validated yet.
                best_pair_acc=best_pair_acc,
                selection_metric="val_pair_acc",
                validation_completed=False,
            )
            save_checkpoint(
                os.path.join(args.output_dir, "metric_imagenet_ot_last.pt"),
                **last_state,
            )
            print(
                f"[M0-B][{epoch+1:03d}/{args.epochs}] "
                "saved LAST immediately after training -> metric_imagenet_ot_last.pt"
            )

        # In DDP, non-main ranks must not enter validation while rank0 is still
        # serializing LAST; DDP forward may perform collectives/buffer sync.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # --------------------------------------------------------------
        # 2) Validation.  BEST is considered/saved only after this succeeds.
        # --------------------------------------------------------------
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

        if is_main:
            print(
                f"[M0-B][{epoch+1:03d}/{args.epochs}] "
                f"TRAIN loss={train_metrics['total']:.4f} "
                f"pair={train_metrics['pair_acc']:.4f} "
                f"D+={train_metrics['pos_d']:.4f} D-={train_metrics['neg_d']:.4f} "
                f"gap={train_metrics['gap']:.4f} cls_acc={train_metrics['cls_acc']:.4f} | "
                f"VAL loss={val_metrics['total']:.4f} "
                f"pair={val_metrics['pair_acc']:.4f} "
                f"D+={val_metrics['pos_d']:.4f} D-={val_metrics['neg_d']:.4f} "
                f"gap={val_metrics['gap']:.4f} cls_acc={val_metrics['cls_acc']:.4f}"
            )

            improved = val_metrics["pair_acc"] > best_pair_acc
            if improved:
                best_pair_acc = val_metrics["pair_acc"]

                # Construct BEST payload only after updating best_pair_acc, so
                # the checkpoint metadata is never stale (e.g. -inf on epoch 1).
                best_state = dict(
                    stage="M0-B",
                    checkpoint_kind="best_after_validation",
                    epoch=epoch + 1,
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
                    selection_metric="val_pair_acc",
                    validation_completed=True,
                )
                save_checkpoint(
                    os.path.join(args.output_dir, "metric_imagenet_ot.pt"),
                    **best_state,
                )
                print(
                    f"[M0-B] new best: val_pair_acc={best_pair_acc:.4f} -> "
                    "metric_imagenet_ot.pt"
                )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()