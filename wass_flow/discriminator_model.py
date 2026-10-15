"""
DINO-based Discriminator model for GAN training (extracted from RAEv2 Stage-1).

This file consolidates all components related to the discriminator used in
train_stage1.py's GAN training, including:
  - DinoDiscriminator (DINO ViT backbone + multi-scale discriminator heads)
  - DiffAug (Differentiable Augmentation for data-efficient GAN training)
  - GAN losses (hinge, vanilla discriminator/generator losses, adaptive weight)
  - LPIPS (Learned Perceptual Image Patch Similarity)
  - build_discriminator() factory function

Original source: /mnt/inaisfs/data/home/tansy_criait/jmf/GitHub/RAEv2-main/src/stage1/disc/
"""

from __future__ import annotations

import math
import os
import hashlib
import pathlib
from collections import namedtuple
from typing import List, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision import models as tv_models
from tqdm import tqdm


# ============================================================
# Utility: RandomWindowCrop (from disc/utils.py)
# ============================================================

def _linspace_indices(limit: int, count: int) -> List[int]:
    if count <= 1:
        return [0]
    return sorted({int(round(i * (limit / (count - 1)))) for i in range(count)})


def _gen_positions_1d(length: int, crop: int, slots: int) -> List[int]:
    limit = max(length - crop, 0)
    pos = _linspace_indices(limit, max(slots, 1))
    pos = [max(0, min(p, limit)) for p in pos]
    if slots > 1:
        pos[0] = 0
        pos[-1] = limit
    return pos


class RandomWindowCrop:
    """Random crop with a fixed catalog of windows (XLA-friendly variant)."""

    def __init__(
        self,
        input_size: int | Tuple[int, int],
        crop: int,
        num_windows: int,
        per_sample: bool = False,
    ):
        if isinstance(input_size, int):
            self.H = self.W = int(input_size)
        else:
            self.H, self.W = map(int, input_size)
        self.crop = int(crop)
        self.per_sample = bool(per_sample)

        if self.crop <= 0:
            raise ValueError("crop must be > 0")
        if self.crop > self.H or self.crop > self.W:
            raise ValueError(f"crop={self.crop} exceeds input {(self.H, self.W)}")
        if num_windows <= 0:
            raise ValueError("num_windows must be > 0")

        rows_min = math.ceil(self.H / self.crop)
        cols_min = math.ceil(self.W / self.crop)
        n_min = rows_min * cols_min
        if num_windows < n_min:
            raise ValueError(
                f"num_windows={num_windows} too small to cover {(self.H, self.W)} with crop {self.crop}"
            )

        t_rows = _gen_positions_1d(self.H, self.crop, rows_min)
        l_cols = _gen_positions_1d(self.W, self.crop, cols_min)
        base_offsets = [(t, l) for t in t_rows for l in l_cols]

        offsets = list(base_offsets)
        if num_windows > len(offsets):
            rows_t = max(rows_min, int(math.floor(math.sqrt(num_windows * self.H / self.W))))
            cols_t = max(cols_min, int(math.ceil(num_windows / rows_t)))
            while rows_t * cols_t < num_windows:
                cols_t += 1

            t_more = _gen_positions_1d(self.H, self.crop, rows_t)
            l_more = _gen_positions_1d(self.W, self.crop, cols_t)
            dense = [(t, l) for t in t_more for l in l_more]

            seen = set(offsets)
            for off in dense:
                if len(offsets) >= num_windows:
                    break
                if off not in seen:
                    offsets.append(off)
                    seen.add(off)

            idx = 0
            while len(offsets) < num_windows and idx < len(dense):
                offsets.append(dense[idx])
                idx += 1

        self.offsets: List[Tuple[int, int]] = offsets[:num_windows]
        self.num_windows = len(self.offsets)

    def __repr__(self) -> str:
        return (
            f"RandomWindowCrop(input={(self.H, self.W)}, crop={self.crop}, "
            f"windows={self.num_windows}, per_sample={self.per_sample})"
        )

    def _rand_idx(self) -> int:
        return torch.randint(0, self.num_windows, (1,)).item()

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        H, W = tensor.shape[-2], tensor.shape[-1]
        if (H, W) != (self.H, self.W):
            raise ValueError(f"Expected input {(self.H, self.W)}, got {(H, W)}")

        crop = self.crop
        if self.per_sample and tensor.dim() >= 4 and tensor.shape[0] > 1:
            outputs = []
            for i in range(tensor.shape[0]):
                top, left = self.offsets[self._rand_idx()]
                outputs.append(tensor[i, ..., top:top + crop, left:left + crop])
            return torch.stack(outputs, dim=0)

        top, left = self.offsets[self._rand_idx()]
        return tensor[..., top:top + crop, left:left + crop]


# ============================================================
# DINO Discriminator Core (from disc/dinodisc.py)
# ============================================================

fused_mlp_func = None
flash_attn_qkvpacked_func = None


def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
    attn = query.mul(scale) @ key.transpose(-2, -1)
    if attn_mask is not None:
        attn.add_(attn_mask)
    return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value


class MLPNoDrop(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight,
                bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training,
                return_residual=False, checkpoint_lvl=0, heuristic=0, process_group=None,
            )
        else:
            return self.fc2(self.act(self.fc1(x)))

    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class SelfAttentionNoDrop(nn.Module):
    def __init__(self, block_idx, embed_dim=768, num_heads=12, flash_if_available=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.using_flash_attn = False

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        if self.using_flash_attn and qkv.dtype != torch.float32:
            oup = flash_attn_qkvpacked_func(qkv, softmax_scale=self.scale).view(B, L, C)
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
        return self.proj(oup)

    def extra_repr(self) -> str:
        return f'using_flash_attn={self.using_flash_attn}'


class SABlockNoDrop(nn.Module):
    def __init__(self, block_idx, embed_dim, num_heads, mlp_ratio, norm_eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = SelfAttentionNoDrop(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, flash_if_available=True)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = MLPNoDrop(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), fused_if_available=True)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.ratio = 1 / np.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fn(x)
        if out.shape[-1] != x.shape[-1]:
            # Even kernel sizes with padding=ks//2 produce output that is 1 element
            # larger than the input.  Trim to match so the residual add works.
            min_spatial = min(out.shape[-1], x.shape[-1])
            out = out[..., :min_spatial]
            x = x[..., :min_spatial]
        return (out.add(x)).mul_(self.ratio)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 1, eps: float = 1e-6):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.float()
        G = np.ceil(x.size(0) / self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))
        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]
        return x.view(shape)


# DINO ViT recipes (predefined configurations)
recipes = {
    "S_16": {
        "depth": 12, "key_depths": (2, 5, 8, 11), "norm_eps": 1e-6,
        "patch_size": 16, "in_chans": 3, "embed_dim": 384, "num_heads": 6, "mlp_ratio": 4.0,
    },
    "S_8": {
        "depth": 12, "key_depths": (2, 5, 8, 11), "norm_eps": 1e-6,
        "patch_size": 8, "in_chans": 3, "embed_dim": 384, "num_heads": 6, "mlp_ratio": 4.0,
    },
    "B_16": {
        "depth": 12, "key_depths": (2, 5, 8, 11), "norm_eps": 1e-6,
        "patch_size": 16, "in_chans": 3, "embed_dim": 768, "num_heads": 12, "mlp_ratio": 4.0,
    },
}


def make_block(channels: int, kernel_size: int, norm_type: str, norm_eps: float, using_spec_norm: bool) -> nn.Module:
    if norm_type == "bn":
        norm = BatchNormLocal(channels, eps=norm_eps)
    elif norm_type == "gn":
        norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=norm_eps, affine=True)
    else:
        raise NotImplementedError(f"Unknown norm_type '{norm_type}'")
    conv = SpectralConv1d if using_spec_norm else nn.Conv1d
    return nn.Sequential(
        conv(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode="circular"),
        norm,
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class FrozenDINONoDrop(nn.Module):
    """Frozen DINO ViT backbone (no dropout). Extracts multi-scale patch features."""

    def __init__(
        self,
        depth=12,
        key_depths=(2, 5, 8, 11),
        norm_eps=1e-6,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4.0,
        crop_prob: float = -0.5,
        no_resize: bool = False,
        original_input_size: int | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.img_size = 224
        self.original_input_size = original_input_size if original_input_size is not None else self.img_size
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_size = patch_size
        self.patch_nums = self.img_size // patch_size

        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        self.register_buffer("x_scale", (0.5 / std).reshape(1, 3, 1, 1))
        self.register_buffer("x_shift", ((0.5 - mean) / std).reshape(1, 3, 1, 1))
        self.crop = RandomWindowCrop(self.original_input_size, self.img_size, 9, False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_nums * self.patch_nums + 1, embed_dim))

        self.key_depths = set(d for d in key_depths if d < depth)
        self.blocks = nn.Sequential(
            *[
                SABlockNoDrop(block_idx=i, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_eps=norm_eps)
                for i in range(max(depth, 1 + max(self.key_depths, default=0)))
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.crop_prob = crop_prob
        self.no_resize = no_resize
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def inter_pos_embed(self, patch_nums=(14, 14)):
        if patch_nums[0] == self.patch_nums and patch_nums[1] == self.patch_nums:
            return self.pos_embed
        pe_cls, pe_grid = self.pos_embed[:, :1], self.pos_embed[0, 1:]
        pe_grid = pe_grid.reshape(1, self.patch_nums, self.patch_nums, -1).permute(0, 3, 1, 2)
        pe_grid = F.interpolate(pe_grid, size=patch_nums, mode="bilinear", align_corners=False)
        pe_grid = pe_grid.permute(0, 2, 3, 1).reshape(1, patch_nums[0] * patch_nums[1], -1)
        return torch.cat([pe_cls, pe_grid], dim=1)

    def forward(self, x, grad_ckpt=False):
        if not self.no_resize:
            x = F.interpolate(x, size=(self.original_input_size, self.original_input_size), mode="bilinear", align_corners=False)
            if self.crop_prob > 0 and torch.rand(()) < self.crop_prob:
                x = self.crop(x)
        else:
            if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
                x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = x * self.x_scale + self.x_shift
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        if x.shape[1] != self.pos_embed.shape[1]:
            h = w = int(math.sqrt(x.shape[1] - 1))
            pos_embed = self.inter_pos_embed((h, w))
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed

        activations = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.key_depths:
                activations.append(x[:, 1:, :].transpose(1, 2))
        activations.insert(0, x[:, 1:, :].transpose(1, 2))
        return activations


class DinoDisc(nn.Module):
    """
    DINO-based multi-scale discriminator.

    Uses a frozen DINO ViT backbone to extract patch-level features at multiple depths,
    then applies separate discriminator heads on each scale's features.
    """

    def __init__(
        self,
        device: torch.device,
        dino_ckpt_path: str,
        ks: int,
        key_depths=(2, 5, 8, 11),
        norm_type="bn",
        using_spec_norm=True,
        norm_eps=1e-6,
        recipe: str = "S_16",
    ):
        super().__init__()
        state = torch.load(dino_ckpt_path, map_location="cpu")
        for key in sorted(state.keys()):
            if ".attn.qkv.bias" in key:
                bias = state[key]
                C = bias.numel() // 3
                bias[C: 2 * C].zero_()

        recipe_cfg = dict(recipes[recipe])
        key_depths = tuple(d for d in key_depths if d < recipe_cfg["depth"])
        recipe_cfg.update({"key_depths": key_depths, "norm_eps": norm_eps})
        dino = FrozenDINONoDrop(**recipe_cfg)
        missing, unexpected = dino.load_state_dict(state, strict=False)
        missing = [m for m in missing if all(x not in m for x in {"x_scale", "x_shift"})]
        if missing:
            raise RuntimeError(f"DINO checkpoint missing keys: {missing}")
        if unexpected:
            raise RuntimeError(f"DINO checkpoint has unexpected keys: {unexpected}")
        dino.eval()
        self.dino_proxy: Tuple[FrozenDINONoDrop, ...] = (dino.to(device=device),)
        dino_C = self.dino_proxy[0].embed_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    make_block(dino_C, kernel_size=1, norm_type=norm_type, norm_eps=norm_eps, using_spec_norm=using_spec_norm),
                    ResidualBlock(
                        make_block(dino_C, kernel_size=ks, norm_type=norm_type, norm_eps=norm_eps, using_spec_norm=using_spec_norm)
                    ),
                    (SpectralConv1d if using_spec_norm else nn.Conv1d)(dino_C, 1, kernel_size=1, padding=0),
                )
                for _ in range(len(key_depths) + 1)
            ]
        )
        for p in self.heads:
            p.requires_grad_(True)
        self.dino_proxy[0].requires_grad_(False)

    def forward(self, x_in_pm1: torch.Tensor, grad_ckpt: bool = False) -> torch.Tensor:
        if grad_ckpt and x_in_pm1.requires_grad:
            raise RuntimeError("DINO discriminator does not support grad checkpointing.")
        activations: List[torch.Tensor] = self.dino_proxy[0](x_in_pm1, grad_ckpt=False)
        batch = x_in_pm1.shape[0]
        outputs = []
        for head, act in zip(self.heads, activations):
            out = head(act).view(batch, -1)
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class DinoDiscriminator(DinoDisc):
    """Thin wrapper aligning with legacy API expecting (fake, real) outputs."""

    def __init__(self, device: torch.device, **kwargs):
        super().__init__(device=device, **kwargs)

    def classify(self, img: torch.Tensor) -> torch.Tensor:
        return super().forward(img)

    def forward(self, fake: torch.Tensor, real: torch.Tensor | None = None):
        logits_fake = self.classify(fake)
        logits_real = self.classify(real) if real is not None else None
        return logits_fake, logits_real


# ============================================================
# DiffAug: Differentiable Augmentation (from disc/diffaug.py)
# ============================================================

class DiffAug(object):
    """
    Differentiable Augmentation for Data-Efficient GAN Training.
    Adapted from https://github.com/autonomousvision/stylegan-t
    """

    def __init__(self, prob=1.0, cutout=0.2):
        self.grids = {}
        self.prob = abs(prob)
        self.using_cutout = prob > 0
        self.cutout = cutout
        self.img_channels = -1
        self.last_blur_radius = -1
        self.last_blur_kernel_h = self.last_blur_kernel_w = None

    def __str__(self):
        return f'DiffAug(p={self.prob}, cutout={self.cutout if self.using_cutout else 0})'

    def get_grids(self, B, x, y, dev):
        if (B, x, y) in self.grids:
            return self.grids[(B, x, y)]
        self.grids[(B, x, y)] = ret = torch.meshgrid(
            torch.arange(B, dtype=torch.long, device=dev),
            torch.arange(x, dtype=torch.long, device=dev),
            torch.arange(y, dtype=torch.long, device=dev),
            indexing='ij'
        )
        return ret

    def aug(self, BCHW: torch.Tensor, warmup_blur_schedule: float = 0) -> torch.Tensor:
        if BCHW.dtype != torch.float32:
            BCHW = BCHW.float()
        if warmup_blur_schedule > 0:
            self.img_channels = BCHW.shape[1]
            sigma0 = (BCHW.shape[-2] * 0.5) ** 0.5
            sigma = sigma0 * warmup_blur_schedule
            blur_radius = math.floor(sigma * 3)
            if blur_radius >= 1:
                if self.last_blur_radius != blur_radius:
                    self.last_blur_radius = blur_radius
                    gaussian = torch.arange(-blur_radius, blur_radius + 1, dtype=torch.float32, device=BCHW.device)
                    gaussian = gaussian.mul_(1 / sigma).square_().neg_().exp2_()
                    gaussian.div_(gaussian.sum())
                    self.last_blur_kernel_h = gaussian.view(1, 1, 2 * blur_radius + 1, 1).repeat(self.img_channels, 1, 1, 1).contiguous()
                    self.last_blur_kernel_w = gaussian.view(1, 1, 1, 2 * blur_radius + 1).repeat(self.img_channels, 1, 1, 1).contiguous()
                BCHW = F.pad(BCHW, [blur_radius, blur_radius, blur_radius, blur_radius], mode='reflect')
                BCHW = F.conv2d(input=BCHW, weight=self.last_blur_kernel_h, bias=None, groups=self.img_channels)
                BCHW = F.conv2d(input=BCHW, weight=self.last_blur_kernel_w, bias=None, groups=self.img_channels)

        if self.prob < 1e-6:
            return BCHW
        trans, color, cut = torch.rand(3) <= self.prob
        trans, color, cut = trans.item(), color.item(), cut.item()
        B, dev = BCHW.shape[0], BCHW.device
        rand01 = torch.rand(7, B, 1, 1, device=dev) if (trans or color or cut) else None

        raw_h, raw_w = BCHW.shape[-2:]
        if trans:
            ratio = 0.125
            delta_h = round(raw_h * ratio)
            delta_w = round(raw_w * ratio)
            translation_h = rand01[0].mul(delta_h + delta_h + 1).floor().long() - delta_h
            translation_w = rand01[1].mul(delta_w + delta_w + 1).floor().long() - delta_w
            grid_B, grid_h, grid_w = self.get_grids(B, raw_h, raw_w, dev)
            grid_h = (grid_h + translation_h).add_(1).clamp_(0, raw_h + 1)
            grid_w = (grid_w + translation_w).add_(1).clamp_(0, raw_w + 1)
            bchw_pad = F.pad(BCHW, [1, 1, 1, 1, 0, 0, 0, 0])
            BCHW = bchw_pad.permute(0, 2, 3, 1).contiguous()[grid_B, grid_h, grid_w].permute(0, 3, 1, 2).contiguous()

        if color:
            BCHW = BCHW.add(rand01[2].unsqueeze(-1).sub(0.5))
            bchw_mean = BCHW.mean(dim=1, keepdim=True)
            BCHW = BCHW.sub(bchw_mean).mul(rand01[3].unsqueeze(-1).mul(2)).add_(bchw_mean)
            bchw_mean = BCHW.mean(dim=(1, 2, 3), keepdim=True)
            BCHW = BCHW.sub(bchw_mean).mul(rand01[4].unsqueeze(-1).add(0.5)).add_(bchw_mean)

        if self.using_cutout and cut:
            ratio = self.cutout
            cutout_h = round(raw_h * ratio)
            cutout_w = round(raw_w * ratio)
            offset_h = rand01[5].mul(raw_h + (1 - cutout_h % 2)).floor().long()
            offset_w = rand01[6].mul(raw_w + (1 - cutout_w % 2)).floor().long()
            grid_B, grid_h, grid_w = self.get_grids(B, cutout_h, cutout_w, dev)
            grid_h = (grid_h + offset_h).sub_(cutout_h // 2).clamp(min=0, max=raw_h - 1)
            grid_w = (grid_w + offset_w).sub_(cutout_w // 2).clamp(min=0, max=raw_w - 1)
            mask = torch.ones(B, raw_h, raw_w, dtype=BCHW.dtype, device=dev)
            mask[grid_B, grid_h, grid_w] = 0
            BCHW = BCHW.mul(mask.unsqueeze(1))

        return BCHW


# ============================================================
# GAN Losses (from disc/gan_loss.py)
# ============================================================

def hinge_d_loss(logits_real, logits_fake, reduction: str = "mean") -> torch.Tensor:
    """Hinge discriminator loss used by VQGAN."""
    reduce = torch.mean if reduction == "mean" else torch.sum
    loss_real = reduce(F.relu(1.0 - logits_real))
    loss_fake = reduce(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real, logits_fake, reduction: str = "mean") -> torch.Tensor:
    """Original GAN discriminator loss."""
    reduce = torch.mean if reduction == "mean" else torch.sum
    return 0.5 * (
        reduce(F.softplus(-logits_real)) + reduce(F.softplus(logits_fake))
    )


def vanilla_g_loss(logits_fake, reduction: str = "mean") -> torch.Tensor:
    """Original GAN generator loss."""
    if reduction == "mean":
        return - torch.mean(logits_fake)
    if reduction == "sum":
        return - torch.sum(logits_fake)
    raise ValueError(f"Unsupported reduction '{reduction}'")


def select_gan_losses(disc_kind: str, gen_kind: str):
    """Select discriminator and generator loss functions by name."""
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")
    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, gen_loss_fn


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    """Calculate adaptive weight for GAN loss based on gradient norms."""
    torch._functorch.config.donated_buffer = False
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()


# ============================================================
# LPIPS (from disc/lpips.py + disc/lpips_utils.py)
# ============================================================

def _lpips_download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def _md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


_LPIPS_URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}
_LPIPS_CKPT_MAP = {"vgg_lpips": "vgg.pth"}
_LPIPS_MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


def _get_lpips_ckpt_path(name, root=None, check=False):
    assert name in _LPIPS_URL_MAP
    if root is None:
        root = pathlib.Path(__file__).parent.absolute()
        root = os.path.join(root, '.caches')
    path = os.path.join(root, _LPIPS_CKPT_MAP[name])
    if not os.path.exists(path) or (check and not _md5_hash(path) == _LPIPS_MD5_MAP[name]):
        print(f"Downloading {name} model from {_LPIPS_URL_MAP[name]} to {path}")
        _lpips_download(_LPIPS_URL_MAP[name], path)
        md5 = _md5_hash(path)
        assert md5 == _LPIPS_MD5_MAP[name], md5
    return path


class ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers.append(nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super().__init__()
        features = tv_models.vgg16(pretrained=pretrained).features
        self.slice1 = nn.Sequential(*[features[x] for x in range(4)])
        self.slice2 = nn.Sequential(*[features[x] for x in range(4, 9)])
        self.slice3 = nn.Sequential(*[features[x] for x in range(9, 16)])
        self.slice4 = nn.Sequential(*[features[x] for x in range(16, 23)])
        self.slice5 = nn.Sequential(*[features[x] for x in range(23, 30)])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, tensor: torch.Tensor):
        h = self.slice1(tensor)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)


def _normalize(tensor: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(tensor ** 2, dim=1, keepdim=True))
    return tensor / (norm_factor + eps)


def _spatial_average(tensor: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return tensor.mean([2, 3], keepdim=keepdim)


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS) metric."""

    def __init__(self, use_dropout: bool = True) -> None:
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]
        self.net = VGG16FeatureExtractor(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self._load_pretrained_weights()
        for param in self.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self, name: str = "vgg_lpips") -> None:
        ckpt = _get_lpips_ckpt_path(name)
        state = torch.load(ckpt, map_location=torch.device("cpu"))
        self.load_state_dict(state, strict=False)
        print(f"[LPIPS] Loaded pretrained weights from {ckpt}")

    def forward(self, input: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        input_scaled, target_scaled = self.scaling_layer(input), self.scaling_layer(target)
        feats_input = self.net(input_scaled)
        feats_target = self.net(target_scaled)
        diffs = []
        lin_layers = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for idx, (feat_in, feat_tgt) in enumerate(zip(feats_input, feats_target)):
            feat_in = _normalize(feat_in)
            feat_tgt = _normalize(feat_tgt)
            diff = (feat_in - feat_tgt) ** 2
            diffs.append(_spatial_average(lin_layers[idx].model(diff), keepdim=True))
        value = diffs[0]
        for diff in diffs[1:]:
            value = value + diff
        if reduction == "none":
            return value
        if reduction == "sum":
            return torch.sum(value)
        if reduction == "mean":
            return torch.mean(value)
        raise ValueError(f"Unsupported reduction '{reduction}'")


# ============================================================
# Factory function: build_discriminator
# ============================================================

def build_discriminator(
    dino_ckpt_path: str,
    device: torch.device,
    ks: int = 4,
    key_depths: tuple = (2, 5, 8, 11),
    norm_type: str = "bn",
    using_spec_norm: bool = True,
    norm_eps: float = 1e-6,
    recipe: str = "S_16",
    aug_prob: float = 1.0,
    aug_cutout: float = 0.2,
) -> tuple[DinoDiscriminator, DiffAug]:
    """
    Instantiate DINO-based discriminator and its augmentation policy.

    Args:
        dino_ckpt_path: Path to pretrained DINO ViT checkpoint (e.g., dino_vit_small_patch8_224_gan.pth).
        device: Target device (torch.device).
        ks: Kernel size for discriminator head convolution layers.
        key_depths: Transformer layer indices to extract multi-scale features from.
        norm_type: Normalization type ('bn' for BatchNormLocal, 'gn' for GroupNorm).
        using_spec_norm: Whether to use spectral normalization in discriminator heads.
        norm_eps: Epsilon for normalization layers.
        recipe: DINO ViT recipe name ('S_16', 'S_8', 'B_16').
        aug_prob: Probability of applying DiffAug augmentation.
        aug_cutout: Cutout ratio for DiffAug.

    Returns:
        Tuple of (DinoDiscriminator, DiffAug)
    """
    disc = DinoDiscriminator(
        device=device,
        dino_ckpt_path=dino_ckpt_path,
        ks=ks,
        key_depths=key_depths,
        norm_type=norm_type,
        using_spec_norm=using_spec_norm,
        norm_eps=norm_eps,
        recipe=recipe,
    ).to(device)

    augment = DiffAug(prob=aug_prob, cutout=aug_cutout)

    return disc, augment


# ============================================================
# Usage Example
# ============================================================

if __name__ == "__main__":
    """
    Usage Example:
    ==============

    This example demonstrates how to:
    1. Build a DINO-based discriminator with DiffAug augmentation
    2. Use the discriminator for GAN training (compute logits for fake/real images)
    3. Compute GAN losses (hinge discriminator loss, vanilla generator loss)
    4. Use LPIPS perceptual loss
    5. Calculate adaptive weight for balancing reconstruction and GAN losses

    Prerequisites:
    - A pretrained DINO ViT checkpoint (e.g., dino_vit_small_patch8_224_gan.pth)
      Download from: https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth

    In train_stage1.py, the discriminator is used as:
        discriminator, disc_aug = build_discriminator(config.gan.arch, device, config.gan.augment)
        ddp_disc = DDP(discriminator, device_ids=[device.index], broadcast_buffers=False)
        discriminator.train()
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Build the discriminator
    # ------------------------------------------------------------------
    # Try known checkpoint locations in this project
    _possible_ckpt_paths = [
        # Project-local RAEv2 checkpoint (patch_size=8, recipe="S_8")
        os.path.join(os.path.dirname(__file__), "flow_matcher_otcfm", "rae_v2", "dino_vit_small_patch8_224_gan.pth"),
        # Alternative common name
        os.path.join(os.path.dirname(__file__), "flow_matcher_otcfm", "rae_v2", "dino_deitsmall8_300ep_pretrain.pth"),
    ]
    dino_ckpt_path = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/flow_matcher_otcfm/rae_v2/dino_vit_small_patch8_224_gan.pth"
    for p in _possible_ckpt_paths:
        if os.path.exists(p):
            dino_ckpt_path = p
            break

    if dino_ckpt_path is None:
        print("\n[WARNING] DINO checkpoint not found!")
        print("Searched paths:")
        for p in _possible_ckpt_paths:
            print(f"  - {p}")
        print("\nTo download a DINO ViT checkpoint (S_8 recipe):")
        print("  wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth")
        print("\n--- Skipping live demo, showing API usage instead ---\n")
    else:
        print(f"Found DINO checkpoint: {dino_ckpt_path}")

        # Determine recipe based on checkpoint filename
        recipe = "S_8" if "patch8" in dino_ckpt_path or "small8" in dino_ckpt_path else "S_16"

        discriminator, disc_aug = build_discriminator(
            dino_ckpt_path=dino_ckpt_path,
            device=device,
            ks=4,
            recipe=recipe,
            aug_prob=1.0,
            aug_cutout=0.2,
        )
        discriminator.train()
        print(f"Discriminator built successfully (recipe={recipe})")

        # ------------------------------------------------------------------
        # 2. Simulate fake and real images (in [-1, 1] range, 224x224)
        # ------------------------------------------------------------------
        batch_size = 4
        fake_images = torch.randn(batch_size, 3, 224, 224, device=device).tanh()  # [-1, 1]
        real_images = torch.randn(batch_size, 3, 224, 224, device=device).tanh()  # [-1, 1]

        # ------------------------------------------------------------------
        # 3. Apply DiffAug augmentation to both fake and real images
        # ------------------------------------------------------------------
        fake_augmented = disc_aug.aug(fake_images)
        real_augmented = disc_aug.aug(real_images)

        # ------------------------------------------------------------------
        # 4. Get discriminator logits
        # ------------------------------------------------------------------
        logits_fake, logits_real = discriminator(fake_augmented, real_augmented)
        print(f"\nlogits_fake shape: {logits_fake.shape}")  # [B, num_heads * num_patches]
        print(f"logits_real shape: {logits_real.shape}")  # [B, num_heads * num_patches]

        # ------------------------------------------------------------------
        # 5. Compute GAN losses
        # ------------------------------------------------------------------
        d_loss = hinge_d_loss(logits_real, logits_fake)
        print(f"Discriminator hinge loss: {d_loss.item():.4f}")

        g_loss = vanilla_g_loss(logits_fake)
        print(f"Generator vanilla loss: {g_loss.item():.4f}")

        disc_loss_fn, gen_loss_fn = select_gan_losses("hinge", "vanilla")
        print(f"Selected disc_loss_fn: {disc_loss_fn.__name__}, gen_loss_fn: {gen_loss_fn.__name__}")

        # ------------------------------------------------------------------
        # 6. LPIPS perceptual loss (between fake and real)
        # ------------------------------------------------------------------
        print("\n--- LPIPS (will download VGG weights on first use, ~50MB) ---")
        # Uncomment to run LPIPS:
        # lpips_model = LPIPS().to(device).eval()
        # perceptual_loss = lpips_model(fake_images, real_images)
        # print(f"LPIPS perceptual loss: {perceptual_loss.item():.4f}")

        # ------------------------------------------------------------------
        # 7. Adaptive weight calculation (for balancing recon vs GAN loss)
        # ------------------------------------------------------------------
        # Uncomment to compute adaptive weight:
        # recon_loss = F.mse_loss(fake_images, real_images)
        # generator_layer = discriminator.heads[0][0]  # any trainable layer
        # adaptive_w = calculate_adaptive_weight(recon_loss, g_loss, generator_layer)
        # print(f"Adaptive weight: {adaptive_w.item():.4f}")

    # ------------------------------------------------------------------
    # 8. Integration into training loop (pseudocode)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)