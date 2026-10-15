#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3: Encoder--Decoder causal masked trajectory world model for RAE-Flow latents.

Task
----
For a same-class source/target image pair:

    source image -> frozen RAE -> x0
                 -> frozen support-conditioned Flow -> [x0, x1, ..., x8]
    target image -> frozen RAE -> s

The nine latent slots t=0..8 are split into two regions:

    Encoder context:  t=0..3   -- ALWAYS fully observed
    Decoder future:   t=4..8   -- t=4..7 randomly masked, t=8 ALWAYS masked

Targets are defined ONLY at masked decoder positions:

    masked t=4..7 : corresponding frozen Flow latent x_t
    masked t=8    : TRUE target/support frozen-RAE latent s (not Flow x8)

Unmasked decoder states are context only: they are never reconstructed and never
contribute to prediction loss or VISReg.  The prediction head is evaluated only
on masked decoder states.

Architecture
------------
Native latent shape per frame: (C=1024,H=16,W=16), i.e. N=256 tokens, D=1024.

Encoder (default 8 blocks):
    - receives x0..x3 only;
    - spatial self-attention is bidirectional within each frame;
    - temporal Delta-Attention across the four known frames is bidirectional;
    - all four encoder frames are fully known.

Decoder (default 8 blocks; 16 total blocks by default):
    - receives t=4..8 with masked frames replaced by a learned MASK frame;
    - spatial self-attention is bidirectional within each frame;
    - temporal Delta-Attention is CAUSAL across decoder time;
    - cross-attention reads the fully observed encoder context;
    - prediction head runs ONLY on masked decoder states.

Delta-Attention uses
    Q = W_q_state X + W_q_delta Delta(X)
    K = W_k X
    V = W_v X
where Delta(X_t)=X_t-X_{t-1} is exposed only when the current and previous
native trajectory frames are both observed.  Delta is computed before MASK
replacement and before learned 3D position encoding.

Objective
---------
No classification loss and no Wasserstein loss are used in M3 pretraining.
Only masked decoder positions contribute:

    L_pred   = prediction loss over masked t in {4,5,6,7,8}
    L_VISReg = mean VISReg over the same masked predictions
    L_total  = L_pred + visreg_weight * L_VISReg

Prediction loss is a single Smooth-L1 objective averaged uniformly over ALL
masked decoder positions.  No separate trajectory/endpoint weighting and no
cosine prediction loss are used.  VISReg is the only auxiliary regularizer.
RAE and Flow remain frozen; only the Encoder--Decoder world model is optimized.

Distributed training
--------------------
Single-node multi-GPU training uses torchrun + DistributedDataParallel (DDP).
Each rank owns one complete local pipeline on exactly one GPU:

    local GPU = frozen RAE + frozen Flow/EndoViT + one DDP replica of world model

Thus --batch_size is PER GPU. With 2 GPUs and --batch_size 8, global batch size is 16.
DDP improves throughput/global batch but does not shard model/optimizer memory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from PIL import Image
from torchvision import transforms

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

try:
    from data.class_pool import load_records
except ImportError:
    from data.class_pool import load_records

try:
    from losss.visreg import VISReg
except ImportError:
    from losss.visreg import VISReg

# Optional debugger. Disabled by default so batch/Slurm jobs never block.
# Enable explicitly with: M3_DEBUGPY=1 python ...
if os.environ.get("M3_DEBUGPY", "0") == "1":
    try:
        import debugpy
        print("[M3] Waiting for debugger attach on localhost:9504")
        debugpy.listen(("localhost", 9504))
        debugpy.wait_for_client()
    except Exception as e:
        print(f"[M3][WARN] debugpy unavailable/failed: {e}")



# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "M3: causal Delta-Attention latent world model for frozen RAE-Flow trajectories"
    )

    train = p.add_mutually_exclusive_group(required=True)
    train.add_argument("--train_root", type=str, default=None,
                       help="Class-folder root: root/class_name/images")
    train.add_argument("--train_json_glob", type=str, default=None,
                       help="Folder/glob of the project's class-pool JSONs")

    val = p.add_mutually_exclusive_group(required=False)
    val.add_argument("--val_root", type=str, default=None)
    val.add_argument("--val_json_glob", type=str, default=None)
    p.add_argument("--val_fraction", type=float, default=0.10,
                   help="Per-class holdout if no explicit validation source is supplied")

    p.add_argument("--project_root", type=str, required=True)
    p.add_argument("--flow_checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--resume_checkpoint", type=str, default=None)

    # Runtime placement. Under torchrun/DDP, LOCAL_RANK determines the GPU and
    # RAE + Flow/EndoViT + world model all live on that same local GPU.
    # --device is kept only for single-process fallback. --flow_device is accepted
    # for backward-compatible shell scripts but ignored in DDP/local-pipeline mode.
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--flow_device", type=str, default="same",
                   help="Deprecated compatibility option; Flow is colocated with the local rank GPU")
    p.add_argument("--local_rank", "--local-rank", type=int, default=-1,
                   help="Usually supplied by torchrun via LOCAL_RANK")

    # Frozen RAE / Flow.
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--rae_chunk_size", type=int, default=8)
    p.add_argument("--flow_num_steps", type=int, default=8,
                   help="Must be 8 for the canonical t=0..8 world-model sequence")
    p.add_argument("--flow_solver", choices=["euler", "heun"], default="heun")
    p.add_argument("--flow_dit_variant", type=str, default="L")
    p.add_argument("--flow_endovit_path", type=str, default=None)
    p.add_argument("--flow_hint_signal_ratio", type=float, default=None)
    p.add_argument("--flow_hint_dropout", type=float, default=0.0)
    p.add_argument("--flow_stochastic_hint", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--flow_allow_condition_mismatch", action="store_true")

    # Pair sampling.
    p.add_argument("--batch_size", type=int, default=2,
                   help="PER-GPU pair batch size; global batch = batch_size * WORLD_SIZE under DDP")
    p.add_argument("--steps_per_epoch", type=int, default=1000)
    p.add_argument("--val_steps", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=0,
                   help="Reserved for future Dataset/DataLoader version; current sampler is synchronous")

    # Masking: endpoint t=8 is mandatory; t=4..7 are random.
    p.add_argument("--tail_start", type=int, default=4)
    p.add_argument("--tail_end", type=int, default=7)
    p.add_argument("--tail_mask_prob", type=float, default=0.50)
    p.add_argument("--ensure_extra_tail_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="Ensure at least one of t=4..7 is masked in addition to mandatory t=8")

    # Encoder--Decoder world model.  --depth is TOTAL block count so the existing
    # depth=16 setting remains meaningful.  By default: 8 Encoder + 8 Decoder.
    # D=1024 is deliberately identical to the frozen RAE latent dimension.
    p.add_argument("--model_dim", type=int, default=1024)
    p.add_argument("--depth", type=int, default=16,
                   help="Total Encoder + Decoder block count")
    p.add_argument("--encoder_depth", type=int, default=8,
                   help="Number of fully-observed context Encoder blocks; decoder_depth=depth-encoder_depth")
    p.add_argument("--encoder_time", type=int, default=4,
                   help="Number of always-observed prefix frames; canonical task uses t=0..3")
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--ffn_hidden_dim", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--attn_dropout", type=float, default=0.0)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)

    # Prediction objective: ONLY masked decoder positions are predicted/lossed.
    # Every masked frame has equal weight; endpoint t=8 is NOT separately reweighted.
    p.add_argument("--smooth_l1_beta", type=float, default=1.0)

    # VISReg is applied to EVERY masked prediction (t=4..8 when masked), never to
    # unmasked/context states.  RAE stays frozen; gradients update only the world model.
    p.add_argument("--visreg_weight", type=float, default=1e-3)
    p.add_argument("--visreg_num_proj", type=int, default=512)
    p.add_argument("--visreg_projection_chunk", type=int, default=128)

    # Optimization.
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--save_every", type=int, default=1)

    return p.parse_args()


# -----------------------------------------------------------------------------
# Data pools: same folder/JSON style as the existing M2 code
# -----------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _scan_image_dir(path: str) -> list[str]:
    root = Path(path)
    if not root.exists():
        return []
    return sorted(
        str(p.resolve())
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def load_class_folders(root: str) -> dict[str, list[str]]:
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"class-folder root does not exist: {root}")
    out: dict[str, list[str]] = {}
    for child in sorted(root_p.iterdir()):
        if child.is_dir():
            paths = _scan_image_dir(str(child))
            if paths:
                out[child.name] = paths
    if not out:
        raise RuntimeError(f"no class images found under: {root}")
    return out


def _record_path(record) -> str:
    for attr in ("path", "image_path", "x", "img_path", "file", "filepath"):
        value = getattr(record, attr, None)
        if isinstance(value, (str, os.PathLike)) and str(value).strip():
            return os.fspath(value)
    payload = getattr(record, "__dict__", None)
    if isinstance(payload, dict):
        for key in ("path", "image_path", "x", "img_path", "file", "filepath"):
            value = payload.get(key)
            if isinstance(value, (str, os.PathLike)) and str(value).strip():
                return os.fspath(value)
    raise AttributeError(f"cannot find image path on record type={type(record).__name__}")


def load_class_json_glob(pattern: str) -> dict[str, list[str]]:
    """Use the project's canonical JSON loader, same as the newer M2 script."""
    records = load_records(None, pattern)
    if not records:
        raise RuntimeError(f"data.class_pool.load_records returned no records: {pattern}")

    merged: dict[str, list[str]] = {}
    bad_label = 0
    bad_path = 0
    for record in records:
        label = getattr(record, "label", None)
        if not isinstance(label, str) or not label.strip():
            bad_label += 1
            continue
        try:
            path = _record_path(record)
        except (AttributeError, TypeError, ValueError):
            bad_path += 1
            continue
        merged.setdefault(label.strip(), []).append(str(path))

    merged = {
        label: list(dict.fromkeys(paths))
        for label, paths in merged.items()
        if paths
    }
    if not merged:
        raise RuntimeError(
            "canonical class-pool loader returned records but no usable label/path pairs: "
            f"pattern={pattern}, records={len(records)}, bad_label={bad_label}, bad_path={bad_path}"
        )
    print(
        f"[class-pool] records={len(records)} classes={len(merged)} "
        f"images={sum(len(v) for v in merged.values())} "
        f"bad_label={bad_label} bad_path={bad_path}"
    )
    return merged


def load_class_pool(root: Optional[str], json_glob: Optional[str]) -> dict[str, list[str]]:
    if bool(root) == bool(json_glob):
        raise ValueError("exactly one of root/json_glob must be supplied")
    pool = load_class_folders(root) if root else load_class_json_glob(json_glob)
    # World-model source/target pair must be two distinct images.
    pool = {k: v for k, v in pool.items() if len(v) >= 2}
    if not pool:
        raise RuntimeError("no class contains >=2 images for distinct source/target pairing")
    return pool


def split_pool_per_class(
    pool: dict[str, list[str]], val_fraction: float, seed: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must lie in (0,1)")
    rng = random.Random(seed)
    train: dict[str, list[str]] = {}
    val: dict[str, list[str]] = {}
    for label, paths0 in sorted(pool.items()):
        paths = list(paths0)
        rng.shuffle(paths)
        # Need >=2 distinct images in BOTH partitions; small classes remain train-only.
        if len(paths) < 4:
            train[label] = paths
            continue
        n_val = max(2, int(round(len(paths) * val_fraction)))
        n_val = min(n_val, len(paths) - 2)
        val_paths = paths[:n_val]
        train_paths = paths[n_val:]
        if len(train_paths) >= 2:
            train[label] = train_paths
        if len(val_paths) >= 2:
            val[label] = val_paths
    if not train:
        raise RuntimeError("empty train pool after split")
    if not val:
        raise RuntimeError(
            "empty validation pool after split; provide --val_root/--val_json_glob "
            "or use classes with at least 4 images"
        )
    return train, val


@dataclass
class PairBatch:
    source_paths: list[str]
    target_paths: list[str]
    labels: list[str]


class SameClassPairSampler:
    def __init__(self, pool: dict[str, list[str]], seed: int) -> None:
        self.pool = {k: list(v) for k, v in pool.items() if len(v) >= 2}
        self.labels = sorted(self.pool)
        if not self.labels:
            raise RuntimeError("pair sampler has no class with >=2 images")
        self.rng = random.Random(seed)

    def sample(self, batch_size: int) -> PairBatch:
        src, tgt, labels = [], [], []
        for _ in range(batch_size):
            label = self.rng.choice(self.labels)
            a, b = self.rng.sample(self.pool[label], 2)
            src.append(a)
            tgt.append(b)
            labels.append(label)
        return PairBatch(src, tgt, labels)


# -----------------------------------------------------------------------------
# Frozen RAE / Flow
# -----------------------------------------------------------------------------


def build_transform(image_size: int):
    # Keep world-model targets deterministic.  Augmentation can be added later as
    # a dedicated ablation; it is deliberately absent from the first version.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def load_images(paths: list[str], transform) -> torch.Tensor:
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(transform(im.convert("RGB")))
    return torch.stack(images, dim=0)


def build_rae(project_root: str, device: torch.device):
    project_root = str(Path(project_root).resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from rae.rae import RAE

    model = RAE(
        encoder_name="dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]",
        resolution=256,
        decoder_config_path=os.path.join(project_root, "rae/configs/decoder/ViTXL/config.json"),
        pretrained_decoder_path=os.path.join(project_root, "flow_matcher_otcfm/rae_v2/decoder.pt"),
        noise_tau=0.0,
        normalization_stat_path=os.path.join(project_root, "flow_matcher_otcfm/rae_v2/stats.pt"),
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_rae_chunked(rae, images: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if chunk_size <= 0:
        chunk_size = images.shape[0]
    chunks = []
    for start in range(0, images.shape[0], chunk_size):
        chunks.append(rae.encode(images[start:start + chunk_size]))
    latent = torch.cat(chunks, dim=0)
    if latent.ndim != 4 or latent.shape[1:] != (1024, 16, 16):
        raise RuntimeError(f"RAE output must be (B,1024,16,16), got {tuple(latent.shape)}")
    return latent


class FrozenFlowRuntime:
    def __init__(self, args, local_device: torch.device) -> None:
        # DDP/local-pipeline design: each rank owns its own frozen Flow on the SAME
        # GPU as its RAE and trainable world-model replica. This avoids cross-GPU
        # copies and gives symmetric per-rank pipelines.
        self.device = local_device
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Flow requested CUDA but CUDA is unavailable")
            if self.device.index is not None and self.device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"local Flow device={self.device}, visible CUDA count={torch.cuda.device_count()}"
                )

        try:
            from flow_adapter import FrozenVisionHintDiTFlow
        except ImportError:
            from flow_adapter import FrozenVisionHintDiTFlow

        self.num_steps = int(args.flow_num_steps)
        self.stochastic_hint = bool(args.flow_stochastic_hint)
        self.flow = FrozenVisionHintDiTFlow(
            args.project_root,
            args.flow_checkpoint,
            self.device,
            dit_variant=args.flow_dit_variant,
            endovit_path=args.flow_endovit_path,
            num_steps=self.num_steps,
            solver=args.flow_solver,
            hint_signal_ratio=args.flow_hint_signal_ratio,
            hint_dropout=args.flow_hint_dropout,
            amp=args.amp,
            allow_condition_mismatch=args.flow_allow_condition_mismatch,
        ).to(self.device).eval()
        for p in self.flow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def generate(
        self,
        source_z: torch.Tensor,
        target_images: torch.Tensor,
        *,
        stochastic_hint: Optional[bool] = None,
    ) -> torch.Tensor:
        if stochastic_hint is None:
            stochastic_hint = self.stochastic_hint
        traj = self.flow.generate(
            source_z.to(self.device, non_blocking=True),
            target_images.to(self.device, non_blocking=True),
            stochastic_hint=bool(stochastic_hint),
        )
        expected_t = self.num_steps + 1
        if traj.ndim != 5 or traj.shape[1] != expected_t:
            raise RuntimeError(
                f"Flow trajectory must be (B,{expected_t},C,H,W), got {tuple(traj.shape)}"
            )
        if traj.shape[2:] != (1024, 16, 16):
            raise RuntimeError(
                f"Flow latent frame must be (1024,16,16), got {tuple(traj.shape[2:])}"
            )
        return traj.detach()


# -----------------------------------------------------------------------------
# Encoder--Decoder 3D Delta-Attention world model
# -----------------------------------------------------------------------------


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate(x)) * self.value(x)
        return self.dropout(self.out(x))


class TemporalDeltaAttention(nn.Module):
    """Temporal Delta-Attention at each fixed 16x16 spatial location.

    x:          (B,T,N,D)
    delta_hint: (B,T,N,D), computed from native RAE/Flow latents
    causal:     when True, time t cannot attend to any future time > t

    The Encoder uses causal=False because t=0..3 are fully observed context.
    The Decoder uses causal=True because t=4..8 are future/prediction slots.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = float(attn_dropout)
        self.causal = bool(causal)

        self.q_state = nn.Linear(dim, dim, bias=False)
        self.q_delta = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _heads(self, z: torch.Tensor) -> torch.Tensor:
        # (B,T,N,D) -> (B,N,H,T,dh)
        b, t, n, _ = z.shape
        return (
            z.view(b, t, n, self.num_heads, self.head_dim)
             .permute(0, 2, 3, 1, 4)
             .contiguous()
        )

    def forward(self, x: torch.Tensor, delta_hint: torch.Tensor) -> torch.Tensor:
        b, t, n, d = x.shape
        if delta_hint.shape != x.shape:
            raise ValueError(
                f"delta_hint shape={tuple(delta_hint.shape)} != state shape={tuple(x.shape)}"
            )

        delta_norm = F.layer_norm(delta_hint, (d,))
        q = self._heads(self.q_state(x) + self.q_delta(delta_norm))
        k = self._heads(self.k_proj(x))
        v = self._heads(self.v_proj(x))

        # (B,N,H,T,T); T is only 4 for Encoder or 5 for Decoder.
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.causal:
            causal = torch.ones((t, t), dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(
                ~causal.view(1, 1, 1, t, t),
                torch.finfo(scores.dtype).min,
            )

        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        if self.attn_dropout > 0.0:
            attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
        y = torch.matmul(attn, v)
        y = (
            y.permute(0, 3, 1, 2, 4)
             .contiguous()
             .view(b, t, n, d)
        )
        return self.out_proj(y)


class EncoderDeltaBlock(nn.Module):
    """Fully-observed context block for t=0..3."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        self.norm_spatial = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm_temporal = nn.LayerNorm(dim)
        self.temporal_delta_attn = TemporalDeltaAttention(
            dim, num_heads, attn_dropout, causal=False
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden_dim, dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, delta_hint: torch.Tensor) -> torch.Tensor:
        b, t, n, d = x.shape

        # Full spatial attention within every known context frame.
        z = self.norm_spatial(x).reshape(b * t, n, d)
        z, _ = self.spatial_attn(z, z, z, need_weights=False)
        x = x + self.resid_dropout(z.reshape(b, t, n, d))

        # The four context frames are all known, so temporal context is bidirectional.
        z = self.temporal_delta_attn(self.norm_temporal(x), delta_hint)
        x = x + self.resid_dropout(z)

        x = x + self.resid_dropout(self.ffn(self.norm_ffn(x)))
        return x


class DecoderDeltaBlock(nn.Module):
    """Causal future block for t=4..8 with cross-attention to Encoder context."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        self.norm_spatial = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm_temporal = nn.LayerNorm(dim)
        self.temporal_delta_attn = TemporalDeltaAttention(
            dim, num_heads, attn_dropout, causal=True
        )

        # Cross-attention is performed at each fixed spatial site: every decoder
        # site may read all four fully-observed Encoder times at that site.  Encoder
        # spatial attention has already mixed information within each known frame.
        self.norm_cross_q = nn.LayerNorm(dim)
        self.norm_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_dropout, batch_first=True
        )

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = SwiGLU(dim, ffn_hidden_dim, dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        delta_hint: torch.Tensor,
        encoder_context: torch.Tensor,
    ) -> torch.Tensor:
        b, td, n, d = x.shape
        if encoder_context.ndim != 4:
            raise ValueError("encoder_context must be (B,Tenc,N,D)")
        be, te, ne, de = encoder_context.shape
        if (be, ne, de) != (b, n, d):
            raise ValueError(
                f"encoder context shape mismatch: decoder={(b,td,n,d)}, "
                f"encoder={tuple(encoder_context.shape)}"
            )

        # Spatially bidirectional within each decoder frame (including MASK frames).
        z = self.norm_spatial(x).reshape(b * td, n, d)
        z, _ = self.spatial_attn(z, z, z, need_weights=False)
        x = x + self.resid_dropout(z.reshape(b, td, n, d))

        # Temporally causal within decoder time: prediction at global t cannot read >t.
        z = self.temporal_delta_attn(self.norm_temporal(x), delta_hint)
        x = x + self.resid_dropout(z)

        # Cross-attention to the fully known prefix.  Shape by spatial site:
        # q=(B*N,Tdec,D), kv=(B*N,Tenc,D).
        q = (
            self.norm_cross_q(x)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(b * n, td, d)
        )
        kv = (
            self.norm_cross_kv(encoder_context)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(b * n, te, d)
        )
        z, _ = self.cross_attn(q, kv, kv, need_weights=False)
        z = z.view(b, n, td, d).permute(0, 2, 1, 3).contiguous()
        x = x + self.resid_dropout(z)

        x = x + self.resid_dropout(self.ffn(self.norm_ffn(x)))
        return x


class CausalDeltaWorldModel(nn.Module):
    """Encoder--Decoder causal masked trajectory world model.

    The model returns predictions ONLY for masked decoder frames.  All context and
    unmasked future states still produce hidden representations because they are
    needed as conditioning information, but they never pass through pred_head.
    """

    def __init__(
        self,
        *,
        dim: int = 1024,
        depth: int = 16,
        encoder_depth: int = 8,
        encoder_time: int = 4,
        num_heads: int = 16,
        ffn_hidden_dim: int = 2048,
        max_time: int = 9,
        height: int = 16,
        width: int = 16,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if dim != 1024:
            raise ValueError(
                "This implementation keeps native frozen-RAE D=1024; "
                f"got model_dim={dim}."
            )
        if depth < 2:
            raise ValueError("depth must be >=2")
        if not 1 <= encoder_depth < depth:
            raise ValueError(
                f"need 1 <= encoder_depth < depth, got encoder_depth={encoder_depth}, depth={depth}"
            )
        if not 1 <= encoder_time < max_time:
            raise ValueError(
                f"need 1 <= encoder_time < max_time, got {encoder_time}, {max_time}"
            )

        self.dim = int(dim)
        self.depth = int(depth)
        self.encoder_depth = int(encoder_depth)
        self.decoder_depth = int(depth) - int(encoder_depth)
        self.encoder_time = int(encoder_time)
        self.max_time = int(max_time)
        self.height = int(height)
        self.width = int(width)
        self.n_tokens = self.height * self.width
        self.gradient_checkpointing = bool(gradient_checkpointing)

        # One learned D-vector is broadcast across a whole masked 16x16 frame; row/
        # column embeddings make the 256 masked spatial sites distinguishable.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, dim))

        # Learned 3D positional encoding: global time + row + column.
        self.time_pos = nn.Parameter(torch.zeros(1, max_time, 1, dim))
        self.row_pos = nn.Parameter(torch.zeros(1, 1, height, dim))
        self.col_pos = nn.Parameter(torch.zeros(1, 1, width, dim))

        self.encoder_blocks = nn.ModuleList([
            EncoderDeltaBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(self.encoder_depth)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderDeltaBlock(
                dim=dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(self.decoder_depth)
        ])

        self.encoder_norm = nn.LayerNorm(dim)
        self.decoder_norm = nn.LayerNorm(dim)
        self.pred_head = nn.Linear(dim, dim, bias=True)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.time_pos, std=0.02)
        nn.init.normal_(self.row_pos, std=0.02)
        nn.init.normal_(self.col_pos, std=0.02)
        nn.init.normal_(self.pred_head.weight, std=0.02)
        nn.init.zeros_(self.pred_head.bias)

    def _pos3d(self, start_t: int, end_t: int) -> torch.Tensor:
        # Return (1,Tslice,N,D) using GLOBAL time indices.
        if not (0 <= start_t < end_t <= self.max_time):
            raise ValueError(f"invalid position slice {start_t}:{end_t}")
        spatial = self.row_pos[:, :, :, None, :] + self.col_pos[:, :, None, :, :]
        spatial = spatial.reshape(1, 1, self.n_tokens, self.dim)
        return self.time_pos[:, start_t:end_t] + spatial

    @staticmethod
    def _native_tokens(trajectory: torch.Tensor) -> torch.Tensor:
        # (B,T,C,H,W) -> (B,T,N,D)
        b, t, c, h, w = trajectory.shape
        return trajectory.permute(0, 1, 3, 4, 2).contiguous().view(b, t, h * w, c)

    def _delta_hints(
        self,
        raw: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        """Native latent deltas, exposed only for consecutive observed frames."""
        b, t, n, d = raw.shape
        delta = torch.zeros_like(raw)
        if t > 1:
            valid = (observed[:, 1:] & observed[:, :-1]).to(raw.dtype)
            delta[:, 1:] = (raw[:, 1:] - raw[:, :-1]) * valid[:, :, None, None]
        return delta

    def forward(
        self,
        trajectory: torch.Tensor,
        observed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run Encoder--Decoder and predict ONLY masked decoder frames.

        Args:
            trajectory: (B,9,1024,16,16).  Its t=8 Flow state is never exposed
                        because t=8 must be masked; the true support RAE target is
                        supplied only to the loss outside this module.
            observed:   (B,9) bool.  t=0..3 must all be True; t=8 must be False.

        Returns:
            pred_masked:       (M,256,1024), only masked decoder frames
            masked_indices:    (M,2), [batch_index, GLOBAL_time_index]
            masked_decoder:    (B,5) bool for diagnostics
        """
        if trajectory.ndim != 5:
            raise ValueError(f"trajectory must be 5D, got {tuple(trajectory.shape)}")
        b, t, c, h, w = trajectory.shape
        if t != self.max_time:
            raise ValueError(f"expected T={self.max_time}, got T={t}")
        if (c, h, w) != (self.dim, self.height, self.width):
            raise ValueError(
                f"trajectory frame expected ({self.dim},{self.height},{self.width}), "
                f"got {(c,h,w)}"
            )
        if observed.shape != (b, t):
            raise ValueError(f"observed shape={tuple(observed.shape)} != {(b,t)}")
        if not bool(observed[:, :self.encoder_time].all().item()):
            raise ValueError("Encoder prefix t=0..encoder_time-1 must be fully observed")
        if bool(observed[:, -1].any().item()):
            raise ValueError("endpoint t=8 must always be masked")

        raw = self._native_tokens(trajectory)
        delta_full = self._delta_hints(raw, observed)

        # ------------------------------------------------------------------
        # Encoder: t=0..3, fully observed and temporally bidirectional.
        # ------------------------------------------------------------------
        te = self.encoder_time
        enc = raw[:, :te]
        enc = enc + self._pos3d(0, te).to(dtype=enc.dtype, device=enc.device)
        enc_delta = delta_full[:, :te]

        for block in self.encoder_blocks:
            if self.gradient_checkpointing and self.training:
                enc = checkpoint(block, enc, enc_delta, use_reentrant=False)
            else:
                enc = block(enc, enc_delta)
        enc = self.encoder_norm(enc)

        # ------------------------------------------------------------------
        # Decoder: t=4..8.  Randomly masked frames are replaced by MASK; causal
        # self-attention prevents any future state from leaking backward in time.
        # ------------------------------------------------------------------
        dec_raw = raw[:, te:]
        dec_observed = observed[:, te:]
        dec_masked = ~dec_observed
        dec = torch.where(
            dec_masked[:, :, None, None],
            self.mask_token.to(dtype=dec_raw.dtype, device=dec_raw.device),
            dec_raw,
        )
        dec = dec + self._pos3d(te, t).to(dtype=dec.dtype, device=dec.device)
        dec_delta = delta_full[:, te:]

        for block in self.decoder_blocks:
            if self.gradient_checkpointing and self.training:
                dec = checkpoint(block, dec, dec_delta, enc, use_reentrant=False)
            else:
                dec = block(dec, dec_delta, enc)
        dec = self.decoder_norm(dec)

        # BERT-style principle: prediction head and loss exist ONLY on selected MASK
        # positions.  Unmasked decoder frames are context only.
        masked_local = dec_masked.nonzero(as_tuple=False)  # (M,2): [b, decoder_t]
        if masked_local.numel() == 0:
            raise RuntimeError("no masked decoder positions; t=8 should make this impossible")
        hidden_masked = dec[dec_masked]  # (M,N,D)
        pred_masked = self.pred_head(hidden_masked)

        masked_indices = masked_local.clone()
        masked_indices[:, 1] += te  # convert decoder-local t=0..4 to global t=4..8
        return {
            "pred_masked": pred_masked,
            "masked_indices": masked_indices,
            "masked_decoder": dec_masked,
        }


# -----------------------------------------------------------------------------
# Masking and losses
# -----------------------------------------------------------------------------


def sample_observed_mask(
    batch_size: int,
    *,
    total_time: int,
    tail_start: int,
    tail_end: int,
    tail_mask_prob: float,
    ensure_extra_tail_mask: bool,
    rng: random.Random,
    device: torch.device,
) -> torch.Tensor:
    """Return observed=True / masked=False for each pair/time.

    t=total_time-1 is always masked.  tail_start..tail_end are independently
    random-masked.  Earlier states remain observed.
    """
    if total_time != 9:
        raise ValueError(f"canonical M3 expects total_time=9, got {total_time}")
    if not (0 <= tail_start <= tail_end < total_time - 1):
        raise ValueError(
            f"need 0 <= tail_start <= tail_end < endpoint={total_time-1}, "
            f"got {tail_start}..{tail_end}"
        )
    if not (0.0 <= tail_mask_prob <= 1.0):
        raise ValueError("tail_mask_prob must be in [0,1]")

    observed = torch.ones((batch_size, total_time), dtype=torch.bool, device=device)
    observed[:, total_time - 1] = False  # endpoint mandatory mask

    tail_ids = list(range(tail_start, tail_end + 1))
    for b in range(batch_size):
        masked_tail = []
        for t in tail_ids:
            if rng.random() < tail_mask_prob:
                observed[b, t] = False
                masked_tail.append(t)
        if ensure_extra_tail_mask and not masked_tail:
            observed[b, rng.choice(tail_ids)] = False
    return observed


def latent_prediction_loss(
    pred_masked: torch.Tensor,
    target_masked: torch.Tensor,
    masked_time_idx: torch.Tensor,
    *,
    endpoint_t: int,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Single Smooth-L1 prediction loss over masked decoder frames only.

    pred_masked / target_masked: (M,N,D)
    masked_time_idx:             (M,), GLOBAL time index for each predicted frame

    Every masked frame contributes equally.  Endpoint-specific loss is returned only
    as a monitoring metric; it is NOT separately weighted in the optimization target.
    """
    if pred_masked.shape != target_masked.shape:
        raise ValueError(
            f"pred_masked {tuple(pred_masked.shape)} != target_masked {tuple(target_masked.shape)}"
        )
    if pred_masked.ndim != 3:
        raise ValueError(f"masked predictions must be (M,N,D), got {tuple(pred_masked.shape)}")
    if masked_time_idx.ndim != 1 or masked_time_idx.shape[0] != pred_masked.shape[0]:
        raise ValueError("masked_time_idx shape mismatch")

    per_frame = F.smooth_l1_loss(
        pred_masked.float(), target_masked.float(), beta=beta, reduction="none"
    ).mean(dim=(-1, -2))  # (M,)

    pred_loss = per_frame.mean()

    # Monitoring only: endpoint t=8 is always masked, but receives no extra weight.
    is_endpoint = masked_time_idx.eq(int(endpoint_t))
    if not is_endpoint.any():
        raise RuntimeError("mandatory endpoint t=8 is missing from masked predictions")
    endpoint_pred = per_frame[is_endpoint].mean()

    return pred_loss, {
        "endpoint_pred": endpoint_pred,
        "masked_total": torch.tensor(
            float(pred_masked.shape[0]), device=per_frame.device, dtype=per_frame.dtype
        ),
    }


def masked_visreg_loss(pred_masked: torch.Tensor, visreg: VISReg) -> torch.Tensor:
    """Average VISReg over EVERY masked predicted frame, and no unmasked frame.

    Each predicted frame is an independent N x D latent cloud.  We deliberately
    regularize per-frame and then average, rather than concatenating different
    trajectory times into one artificial cloud.
    """
    if pred_masked.ndim != 3:
        raise ValueError(f"pred_masked must be (M,N,D), got {tuple(pred_masked.shape)}")
    if pred_masked.shape[0] == 0:
        return pred_masked.new_zeros(())
    losses = [visreg(pred_masked[i].unsqueeze(0).float()) for i in range(pred_masked.shape[0])]
    return torch.stack(losses).mean()


# -----------------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------------


def init_distributed(args) -> tuple[bool, int, int, int, torch.device]:
    """Initialize torchrun/DDP if WORLD_SIZE>1; otherwise use single-process mode."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    local_rank = env_local_rank if env_local_rank >= 0 else int(args.local_rank)
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL DDP requires CUDA")
        if local_rank < 0:
            raise RuntimeError("WORLD_SIZE>1 but LOCAL_RANK is missing; launch with torchrun")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        local_rank = device.index if device.type == "cuda" and device.index is not None else 0

    return distributed, rank, local_rank, world_size, device


def is_main_process(rank: int) -> bool:
    return int(rank) == 0


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def reduce_epoch_sums(
    sums: dict[str, float],
    *,
    steps: int,
    device: torch.device,
    distributed: bool,
    world_size: int,
) -> dict[str, float]:
    keys = list(sums.keys())
    values = torch.tensor([sums[k] for k in keys], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    denom = float(max(steps, 1) * max(world_size, 1))
    return {k: float(values[i].item() / denom) for i, k in enumerate(keys)}


# -----------------------------------------------------------------------------
# Optimization / checkpointing
# -----------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def lr_lambda(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine


def save_checkpoint(
    path: str,
    model: CausalDeltaWorldModel,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_val: float,
    args,
) -> None:
    state = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val": float(best_val),
        "world_model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
        "task": {
            "name": "causal_masked_future_latent_prediction",
            "time_slots": 9,
            "encoder_context": [0, 1, 2, 3],
            "decoder_future": [4, 5, 6, 7, 8],
            "encoder_depth": int(args.encoder_depth),
            "decoder_depth": int(args.depth) - int(args.encoder_depth),
            "endpoint_t": 8,
            "endpoint_target": "frozen_RAE(target/support_image)",
            "intermediate_target": "frozen_Flow_latent_at_same_t_if_masked",
            "mandatory_mask": [8],
            "random_mask_range": [int(args.tail_start), int(args.tail_end)],
            "prediction_head_scope": "masked_decoder_frames_only",
            "loss_scope": "masked_decoder_frames_only",
            "visreg_scope": "all_masked_predictions_only",
            "encoder_temporal_attention": "bidirectional",
            "decoder_temporal_attention": "causal",
            "spatial_attention": "bidirectional_within_frame",
            "support_RAE_is_model_input": False,
            "prediction_loss": "SmoothL1_on_masked_decoder_frames_only",
            "endpoint_extra_weight": False,
            "cosine_prediction_loss": False,
            "classification_loss": False,
            "wasserstein_loss": False,
        },
    }
    torch.save(state, path)


def resume_checkpoint(path: str, model, optimizer, scheduler, scaler) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    unwrap_model(model).load_state_dict(ckpt["world_model"], strict=True)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return (
        int(ckpt.get("epoch", 0)),
        int(ckpt.get("global_step", 0)),
        float(ckpt.get("best_val", float("inf"))),
    )


# -----------------------------------------------------------------------------
# One train/validation pass
# -----------------------------------------------------------------------------


def run_epoch(
    *,
    training: bool,
    model: CausalDeltaWorldModel,
    rae,
    flow: FrozenFlowRuntime,
    sampler: SameClassPairSampler,
    transform,
    visreg: VISReg,
    optimizer,
    scheduler,
    scaler,
    args,
    device: torch.device,
    epoch: int,
    global_step: int,
    rank: int,
    world_size: int,
    distributed: bool,
) -> tuple[dict[str, float], int]:
    model.train(training)
    steps = int(args.steps_per_epoch if training else args.val_steps)
    # Validation mask/pair stream is deterministic per epoch-independent seed.
    mask_rng = random.Random(
        args.seed
        + (1000003 if training else 2000003)
        + (epoch if training else 0)
        + int(rank) * 10007
    )

    sums = {
        "loss": 0.0,
        "pred": 0.0,
        "endpoint_pred": 0.0,  # monitoring only; not separately weighted
        "vis": 0.0,
        "masked_total": 0.0,
    }

    amp_enabled = bool(args.amp and device.type == "cuda")

    for step in range(steps):
        pair = sampler.sample(int(args.batch_size))
        source_img = load_images(pair.source_paths, transform).to(device, non_blocking=True)
        target_img_primary = load_images(pair.target_paths, transform).to(device, non_blocking=True)

        # Frozen RAE target space: source x0 and true target/support endpoint S.
        with torch.no_grad():
            both = torch.cat([source_img, target_img_primary], dim=0)
            z = encode_rae_chunked(rae, both, int(args.rae_chunk_size))
            bsz = source_img.shape[0]
            source_z = z[:bsz]
            support_z = z[bsz:]

        # Frozen Flow is colocated with RAE + world model on this rank's GPU.
        # No inter-GPU transfer is needed inside a rank.
        with torch.no_grad():
            traj = flow.generate(
                source_z,
                target_img_primary,
                stochastic_hint=(bool(args.flow_stochastic_hint) if training else False),
            )

        if traj.shape[1] != 9:
            raise RuntimeError(
                f"M3 task is defined on t=0..8 (9 slots), but Flow returned T={traj.shape[1]}. "
                "Use --flow_num_steps 8."
            )

        # Build the target bank, but ONLY masked decoder slots are later gathered.
        # t=4..7 target their corresponding Flow states when masked; t=8 targets the
        # TRUE support RAE latent.  Context/unmasked states never enter the loss.
        target_frames = traj.clone()
        target_frames[:, 8] = support_z
        target_tokens = (
            target_frames.permute(0, 1, 3, 4, 2)
                         .contiguous()
                         .view(bsz, 9, 256, 1024)
        )

        observed = sample_observed_mask(
            bsz,
            total_time=9,
            tail_start=int(args.tail_start),
            tail_end=int(args.tail_end),
            tail_mask_prob=float(args.tail_mask_prob),
            ensure_extra_tail_mask=bool(args.ensure_extra_tail_mask),
            rng=mask_rng,
            device=device,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        grad_ctx = torch.enable_grad() if training else torch.no_grad()
        with grad_ctx:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(traj, observed)
                pred_masked = output["pred_masked"]              # (M,256,1024)
                masked_indices = output["masked_indices"]        # (M,2), global t

                # IMPORTANT: gather targets ONLY at the same masked positions.
                # Unmasked t=0..7 never pass through the prediction objective.
                masked_global = ~observed
                target_masked = target_tokens[masked_global]      # (M,256,1024)
                if target_masked.shape != pred_masked.shape:
                    raise RuntimeError(
                        f"masked target/pred mismatch: {tuple(target_masked.shape)} vs "
                        f"{tuple(pred_masked.shape)}"
                    )

                pred_loss, parts = latent_prediction_loss(
                    pred_masked,
                    target_masked,
                    masked_indices[:, 1],
                    endpoint_t=8,
                    beta=float(args.smooth_l1_beta),
                )

                if float(args.visreg_weight) > 0.0:
                    # VISReg follows the exact same supervision set as prediction:
                    # EVERY masked predicted frame, and NO unmasked/context frame.
                    vis_loss = masked_visreg_loss(pred_masked, visreg)
                else:
                    vis_loss = pred_loss.new_zeros(())

                loss = pred_loss + float(args.visreg_weight) * vis_loss

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

        sums["loss"] += float(loss.detach().item())
        sums["pred"] += float(pred_loss.detach().item())
        sums["endpoint_pred"] += float(parts["endpoint_pred"].detach().item())
        sums["vis"] += float(vis_loss.detach().item())
        sums["masked_total"] += float(parts["masked_total"].detach().item())

        if (
            training
            and is_main_process(rank)
            and args.log_every > 0
            and (step + 1) % int(args.log_every) == 0
        ):
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [train {step+1:04d}/{steps}] "
                f"loss={loss.item():.5f} pred={pred_loss.item():.5f} "
                f"end_pred={parts['endpoint_pred'].item():.5f} "
                f"vis={vis_loss.item():.5f} "
                f"masked_total={int(parts['masked_total'].item())} lr={lr:.3e}"
            )

    out = reduce_epoch_sums(
        sums,
        steps=steps,
        device=device,
        distributed=distributed,
        world_size=world_size,
    )
    return out, global_step


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if int(args.flow_num_steps) != 8:
        raise ValueError(
            "Current task definition requires --flow_num_steps 8, giving x0..x8. "
            "t=8 is then masked and supervised by true support RAE latent."
        )
    if int(args.encoder_time) != 4:
        raise ValueError(
            f"Current canonical Encoder--Decoder task requires --encoder_time 4, got {args.encoder_time}"
        )
    if not (1 <= int(args.encoder_depth) < int(args.depth)):
        raise ValueError(
            f"need 1 <= encoder_depth < depth, got {args.encoder_depth}, {args.depth}"
        )

    distributed, rank, local_rank, world_size, device = init_distributed(args)
    main_process = is_main_process(rank)

    # Different stochastic streams per rank, while class-pool splitting remains shared.
    process_seed = int(args.seed) + rank * 100003
    random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)

    if int(args.tail_start) != 4 or int(args.tail_end) != 7:
        if main_process:
            print(
                f"[WARN] canonical random-mask range is t=4..7; current={args.tail_start}..{args.tail_end}"
            )

    if distributed and str(args.flow_device).strip() not in {"", "same", "primary"} and main_process:
        print(
            f"[WARN] --flow_device={args.flow_device} is ignored under DDP. "
            "Each rank colocates RAE + Flow/EndoViT + world model on its LOCAL_RANK GPU."
        )

    output_dir = Path(args.output_dir)
    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    # Data metadata/split is identical on every rank. The actual pair streams use
    # rank-specific sampler seeds so DDP replicas see different examples.
    train_pool = load_class_pool(args.train_root, args.train_json_glob)
    if args.val_root or args.val_json_glob:
        val_pool = load_class_pool(args.val_root, args.val_json_glob)
    else:
        train_pool, val_pool = split_pool_per_class(train_pool, args.val_fraction, args.seed)

    train_sampler = SameClassPairSampler(
        train_pool, seed=args.seed + 11 + rank * 10007
    )
    transform = build_transform(args.image_size)

    # One COMPLETE frozen conditioning pipeline per rank/GPU.
    if main_process:
        print("[M3] loading one frozen RAE + Flow/EndoViT copy per local rank ...")
    rae = build_rae(args.project_root, device)
    flow = FrozenFlowRuntime(args, device)

    # Trainable world-model replica for this rank.
    base_model = CausalDeltaWorldModel(
        dim=args.model_dim,
        depth=args.depth,
        encoder_depth=args.encoder_depth,
        encoder_time=args.encoder_time,
        num_heads=args.num_heads,
        ffn_hidden_dim=args.ffn_hidden_dim,
        max_time=9,
        height=16,
        width=16,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    visreg = VISReg(
        num_projections=args.visreg_num_proj,
        projection_chunk_size=args.visreg_projection_chunk,
    ).to(device)

    total_params, trainable_params = count_parameters(base_model)

    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )
    total_optim_steps = int(args.epochs) * int(args.steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: lr_lambda(
            s,
            total_steps=total_optim_steps,
            warmup_steps=int(args.warmup_steps),
            min_lr_ratio=float(args.min_lr_ratio),
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    if args.resume_checkpoint:
        start_epoch, global_step, best_val = resume_checkpoint(
            args.resume_checkpoint, base_model, optimizer, scheduler, scaler
        )
        if main_process:
            print(
                f"[M3] resumed {args.resume_checkpoint}: epoch={start_epoch}, "
                f"global_step={global_step}, best_val={best_val:.6f}"
            )

    # Wrap only the trainable world model. Frozen RAE/Flow are local inference modules.
    if distributed:
        model: nn.Module = DDP(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    else:
        model = base_model

    if main_process:
        print("=" * 100)
        print("[M3 ENCODER--DECODER CAUSAL MASKED WORLD MODEL | DDP READY]")
        print(f"distributed           = {distributed}")
        print(f"world size            = {world_size}")
        print(f"per-GPU batch         = {args.batch_size}")
        print(f"global batch          = {int(args.batch_size) * int(world_size)}")
        print(f"steps/epoch           = {args.steps_per_epoch} per rank")
        print(f"train classes/images  = {len(train_pool)} / {sum(len(v) for v in train_pool.values())}")
        print(f"val   classes/images  = {len(val_pool)} / {sum(len(v) for v in val_pool.values())}")
        print(f"local pipeline        = RAE + Flow/EndoViT + DDP world-model replica on each GPU")
        print(f"world-model params    = {trainable_params/1e6:.2f}M trainable / {total_params/1e6:.2f}M total per rank")
        print(f"blocks                = {args.encoder_depth} Encoder + {int(args.depth)-int(args.encoder_depth)} Decoder")
        print("Flow hint             = deterministic when --no-flow_stochastic_hint; use ratio=0.1, dropout=0")
        print("prediction head       = masked decoder states only")
        print("loss                  = masked SmoothL1 + visreg_weight * masked VISReg")
        print("DDP note              = model/optimizer memory is replicated, not sharded")
        print("=" * 100)

    history = []
    try:
        for epoch in range(start_epoch, int(args.epochs)):
            if main_process:
                print(f"\n[M3] epoch {epoch+1}/{args.epochs}")

            train_stats, global_step = run_epoch(
                training=True,
                model=model,
                rae=rae,
                flow=flow,
                sampler=train_sampler,
                transform=transform,
                visreg=visreg,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                device=device,
                epoch=epoch,
                global_step=global_step,
                rank=rank,
                world_size=world_size,
                distributed=distributed,
            )

            # Stable but distinct validation stream on each rank; reduced metrics are
            # therefore averages over world_size * val_steps local batches.
            val_sampler = SameClassPairSampler(
                val_pool, seed=args.seed + 97 + rank * 10007
            )
            val_stats, _ = run_epoch(
                training=False,
                model=model,
                rae=rae,
                flow=flow,
                sampler=val_sampler,
                transform=transform,
                visreg=visreg,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                device=device,
                epoch=0,
                global_step=global_step,
                rank=rank,
                world_size=world_size,
                distributed=distributed,
            )

            # All ranks receive identical reduced val_stats, so best_val stays synced.
            improved = val_stats["loss"] < best_val
            if improved:
                best_val = val_stats["loss"]

            if main_process:
                row = {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "world_size": world_size,
                    "per_gpu_batch": int(args.batch_size),
                    "global_batch": int(args.batch_size) * int(world_size),
                    "train": train_stats,
                    "val": val_stats,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                history.append(row)
                with (output_dir / "history.json").open("w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)

                print(
                    f"[M3][epoch {epoch+1}] "
                    f"train loss={train_stats['loss']:.6f} pred={train_stats['pred']:.6f} "
                    f"end_pred={train_stats['endpoint_pred']:.6f} | "
                    f"val loss={val_stats['loss']:.6f} pred={val_stats['pred']:.6f} "
                    f"end_pred={val_stats['endpoint_pred']:.6f}"
                )

                save_checkpoint(
                    str(output_dir / "world_model_latest.pt"),
                    model, optimizer, scheduler, scaler,
                    epoch + 1, global_step, best_val, args,
                )

                if improved:
                    save_checkpoint(
                        str(output_dir / "world_model_best.pt"),
                        model, optimizer, scheduler, scaler,
                        epoch + 1, global_step, best_val, args,
                    )
                    print(f"[M3] new best validation total loss = {best_val:.6f}")

                if int(args.save_every) > 0 and (epoch + 1) % int(args.save_every) == 0:
                    save_checkpoint(
                        str(output_dir / f"world_model_epoch_{epoch+1:03d}.pt"),
                        model, optimizer, scheduler, scaler,
                        epoch + 1, global_step, best_val, args,
                    )

            if distributed:
                dist.barrier()

        if main_process:
            print(f"[M3] done. best validation total loss={best_val:.6f}")
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
