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
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split

from data.class_pool import (
    RecordDataset,
    build_rae_transform,
    load_records,
)
from models.metric_encoder import MetricEncoder
from utils_.checkpoint import save_checkpoint
from utils_.distributed import (
    cleanup_distributed,
    reduce_mean,
    seed_everything,
    setup_distributed,
)
from utils_.rae_runtime import build_rae, encode_rae, encode_rae_chunked


class ClassificationWarmup(nn.Module):
    def __init__(self, metric: MetricEncoder, num_classes: int) -> None:
        super().__init__()
        self.metric = metric
        self.classifier = nn.Linear(metric.out_dim, num_classes)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.metric(latent).cloud)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("M0-A: ImageNet/general classification warm-up for MetricEncoder")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--train_root", type=str, help="ImageFolder root: root/class/images")
    source.add_argument("--train_json_glob", type=str)
    p.add_argument("--val_root", type=str, default=None)
    p.add_argument("--val_json_glob", type=str, default=None)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--rae_chunk_size", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--log_interval", type=int, default=10,
                   help="Print progress every N batches using local-rank stats only (no all-reduce)")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--metric_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == labels).float().mean()


@torch.no_grad()
def evaluate(model, rae, loader, device, amp_enabled: bool, rae_chunk_size: int) -> tuple[float, float]:
    model.eval()
    loss_sum = torch.zeros((), device=device)
    acc_sum = torch.zeros((), device=device)
    count = 0
    for images, labels, _ in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        latent = encode_rae_chunked(rae, images, rae_chunk_size)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(latent)
            loss = F.cross_entropy(logits.float(), labels)
        loss_sum += loss
        acc_sum += accuracy(logits, labels)
        count += 1
    denom = max(count, 1)
    # reduce_mean 是集合操作，所有 rank 都会进入 evaluate，因此在这里是安全的。
    return float(reduce_mean(loss_sum / denom)), float(reduce_mean(acc_sum / denom))


def main() -> None:
    args = parse_args()
    device, rank, world_size, is_main = setup_distributed()
    seed_everything(args.seed, rank)
    os.makedirs(args.output_dir, exist_ok=True)

    train_records = load_records(args.train_root, args.train_json_glob)
    transform_train = build_rae_transform(args.image_size, augment=True)
    transform_val = build_rae_transform(args.image_size, augment=False)
    full_dataset = RecordDataset(train_records, transform_train)

    explicit_val = args.val_root or args.val_json_glob
    if explicit_val:
        train_dataset = full_dataset
        val_records = load_records(args.val_root, args.val_json_glob)
        val_dataset = RecordDataset(val_records, transform_val, full_dataset.label_to_index)
    else:
        n_val = max(1, int(len(full_dataset) * args.val_fraction))
        n_train = len(full_dataset) - n_val
        generator = torch.Generator().manual_seed(args.seed)
        train_subset, val_subset = random_split(full_dataset, [n_train, n_val], generator=generator)
        train_dataset = train_subset
        # random_split shares transform; this is acceptable for a warm-up, but disable
        # stochastic augmentation on the validation subset by wrapping records.
        val_records = [train_records[i] for i in val_subset.indices]
        val_dataset = RecordDataset(val_records, transform_val, full_dataset.label_to_index)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    # 多开 worker + persistent_workers + 更高 prefetch，缓解网络盘随机小文件读取的瓶颈。
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0, prefetch_factor=args.prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0, prefetch_factor=args.prefetch_factor,
    )

    rae = build_rae(args.project_root, device)
    metric = MetricEncoder(1024, args.hidden_dim, args.metric_dim, dropout=args.dropout)
    model = ClassificationWarmup(metric, len(full_dataset.classes)).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
    raw_model = model.module if isinstance(model, DDP) else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_acc = -1.0

    # 预热 RAE 与 MetricEncoder 的 CUDA kernel，避免首个 batch 被 kernel 编译/显存分配拖累。
    if is_main:
        print("Warming up RAE and model kernels...")
    warmup = torch.rand(2, 3, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        warm_latent = encode_rae_chunked(rae, warmup, args.rae_chunk_size)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=args.amp and device.type == "cuda"):
            _ = model(warm_latent)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print("Starting M0-A training...")
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        loss_sum = torch.zeros((), device=device)
        acc_sum = torch.zeros((), device=device)
        count = 0
        epoch_t0 = time.time()
        for images, labels, _ in train_loader:
            batch_t0 = time.time()
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            t_load = time.time()
            latent = encode_rae_chunked(rae, images, args.rae_chunk_size)
            t_encode = time.time()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=args.amp and device.type == "cuda"):
                logits = model(latent)
                loss = F.cross_entropy(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            t_end = time.time()
            with torch.no_grad():
                loss_sum += loss.detach()
                acc_sum += accuracy(logits.detach(), labels)
            count += 1
            # 训练循环内只用本地 rank 的 loss/acc 打印进度（零集合通信）。
            # 注意：此处绝不能调用 reduce_mean / all_reduce —— 集合操作要求所有 rank
            # 同时参与，若只由 rank0 发起会与 DDP backward 的梯度 AllReduce 互相等待而死锁。
            if is_main and (count == 1 or count % args.log_interval == 0):
                train_loss = float(loss_sum / max(count, 1))
                train_acc = float(acc_sum / max(count, 1))
                print(
                    f"[M0-A][{count}] train=loss:{train_loss:.4f}/acc:{train_acc:.4f} norm:{norm:.4f} "
                    f"load:{t_load - batch_t0:.2f}s enc:{t_encode - t_load:.2f}s "
                    f"fwd_bwd:{t_end - t_encode:.2f}s step:{t_end - batch_t0:.2f}s"
                )
        scheduler.step()
        # epoch 结束:所有 rank 一起参与 reduce（集合操作必须所有 rank 同步调用）。
        train_loss = float(reduce_mean(loss_sum / max(count, 1)))
        train_acc = float(reduce_mean(acc_sum / max(count, 1)))
        val_loss, val_acc = evaluate(model, rae, val_loader, device,
                                     args.amp and device.type == "cuda", args.rae_chunk_size)
        if is_main:
            print(f"[M0-A][{epoch+1:03d}/{args.epochs}] train=loss:{train_loss:.4f}/acc:{train_acc:.4f} "
                  f"val=loss:{val_loss:.4f}/acc{val_acc:.4f} epoch_time:{(time.time() - epoch_t0):.1f}s")
            state = dict(
                stage="M0-A",
                epoch=epoch + 1,
                metric_encoder=raw_model.metric.state_dict(),
                classifier=raw_model.classifier.state_dict(),
                optimizer=optimizer.state_dict(),
                args=vars(args),
                classes=full_dataset.classes,
                val_accuracy=val_acc,
            )
            save_checkpoint(os.path.join(args.output_dir, "metric_cls_last.pt"), **state)
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(os.path.join(args.output_dir, "metric_cls_pretrain.pt"), **state)
    cleanup_distributed()


if __name__ == "__main__":
    main()