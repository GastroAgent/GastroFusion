from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MetricFeatures:
    """Raw and normalized local metric tokens.

    Attributes:
        raw: (..., N, D), before L2 normalization. Used by SK, VISReg and
            trajectory geometry.
        normalized: (..., N, D), L2-normalized. Used by cosine OT cost.
        cloud: (..., D), mean pooled raw representation.
    """

    raw: torch.Tensor
    normalized: torch.Tensor
    cloud: torch.Tensor


class MetricTransformerBlock(nn.Module):
    """Pre-norm Transformer block for local RAE metric tokens."""

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        mlp_hidden = int(round(dim * mlp_ratio))

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(h)
        x = x + self.mlp(self.norm2(x))
        return x


class MetricEncoder(nn.Module):
    """Shared ~10M Transformer metric encoder for RAE / Flow tokens.

    Recommended V1 architecture:
        RAE tokens (..., 256, 1024)
          -> Linear(1024, 512)
          -> learnable 2-D positional embedding
          -> 3 x TransformerBlock(dim=512, heads=8, MLP ratio=4)
          -> LayerNorm
          -> Linear(512, out_dim=256)

    The same weights are used for query, support and every Flow trajectory
    state.  The public constructor and output interface intentionally remain
    compatible with the earlier lightweight projector so the rest of the V1
    pipeline does not need to change.

    Compatibility note:
        ``hidden_dim`` is retained in the signature because existing training
        scripts pass it positionally.  V1 fixes the Transformer model width to
        512 so that simply replacing this file gives the intended ~10M model
        in M0-A/M0-B/M1/M2/F1, even when an older script still passes
        ``--hidden_dim 256``.  ``hidden_dim`` is therefore treated as a legacy
        compatibility argument and stored as ``requested_hidden_dim`` only.
    """

    # Fixed V1 capacity. Keeping these here makes every stage instantiate the
    # exact same architecture without requiring edits to all training scripts.
    MODEL_DIM = 512
    DEPTH = 4
    NUM_HEADS = 8
    MLP_RATIO = 4.0
    BASE_GRID_SIZE = 16

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 256,
        out_dim: int = 256,
        dropout: float = 0.0,
        residual: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.eps = float(eps)
        self.requested_hidden_dim = int(hidden_dim)

        # Kept only for API compatibility with the old projector. The new
        # encoder changes channel width, so the old direct input-output
        # residual is deliberately disabled.
        self.residual = False
        if residual:
            # Do not silently create a second shortcut around the Transformer.
            # Transformer blocks already contain residual paths internally.
            self.residual = False

        model_dim = self.MODEL_DIM
        self.model_dim = model_dim
        self.depth = self.DEPTH
        self.num_heads = self.NUM_HEADS
        self.mlp_ratio = self.MLP_RATIO

        self.input_proj = nn.Linear(self.in_dim, model_dim)

        # Stored as a 2-D grid so position information remains spatially
        # meaningful and can be interpolated if a square token grid other than
        # 16x16 is ever supplied.
        self.pos_embed_2d = nn.Parameter(
            torch.zeros(1, model_dim, self.BASE_GRID_SIZE, self.BASE_GRID_SIZE)
        )

        self.blocks = nn.ModuleList(
            [
                MetricTransformerBlock(
                    dim=model_dim,
                    num_heads=self.NUM_HEADS,
                    mlp_ratio=self.MLP_RATIO,
                    dropout=dropout,
                )
                for _ in range(self.DEPTH)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, self.out_dim)

        self.reset_parameters()

    @property
    def net(self):
        """Legacy read-only compatibility view.

        Older smoke tests accessed ``metric.net[0].weight`` from the previous
        Sequential projector.  Returning a tuple keeps that diagnostic working
        without registering duplicate module names in the checkpoint.
        """
        return (self.input_proj, self.blocks, self.final_norm, self.output_proj)

    def reset_parameters(self) -> None:
        # ViT-style initialization. MultiheadAttention exposes its packed QKV
        # matrix separately, so handle both regular Linear layers and QKV.
        nn.init.trunc_normal_(self.pos_embed_2d, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if module.in_proj_weight is not None:
                    nn.init.trunc_normal_(module.in_proj_weight, std=0.02)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)

    @staticmethod
    def to_tokens(x: torch.Tensor) -> torch.Tensor:
        """Convert (..., C, H, W) to (..., H*W, C); keep token input as-is."""
        if x.ndim < 3:
            raise ValueError(f"MetricEncoder input needs >=3 dims, got {tuple(x.shape)}")
        if x.ndim >= 4:
            # RAE tensors are channel-first and normally C=1024. Distinguish
            # them from (..., N, C) by the final two square spatial dimensions.
            if x.shape[-2] == x.shape[-1] and x.shape[-3] > x.shape[-2]:
                leading = x.shape[:-3]
                c, h, w = x.shape[-3:]
                return x.reshape(*leading, c, h * w).transpose(-1, -2).contiguous()
        return x

    def _position_tokens(
        self,
        n_tokens: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return (1, N, model_dim) learnable 2-D positional tokens."""
        side = int(round(math.sqrt(n_tokens)))
        if side * side != n_tokens:
            raise ValueError(
                f"MetricEncoder expects a square spatial token grid; got N={n_tokens}."
            )

        pos = self.pos_embed_2d
        if side != self.BASE_GRID_SIZE:
            pos = F.interpolate(
                pos.float(),
                size=(side, side),
                mode="bicubic",
                align_corners=False,
            ).to(dtype=pos.dtype)

        pos = pos.flatten(2).transpose(1, 2).contiguous()  # (1, N, D)
        return pos.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> MetricFeatures:
        tokens = self.to_tokens(x)
        if tokens.shape[-1] != self.in_dim:
            raise ValueError(
                f"MetricEncoder expected token dim {self.in_dim}, got {tokens.shape[-1]} "
                f"from shape {tuple(x.shape)}"
            )

        # The V1 pipeline sometimes has many leading axes, e.g.
        # (B, Q, way, shot, T, N, C). MultiheadAttention consumes 3-D tensors,
        # so collapse only those leading axes and restore them afterwards.
        leading = tokens.shape[:-2]
        n_tokens = tokens.shape[-2]
        flat = tokens.reshape(-1, n_tokens, self.in_dim)

        h = self.input_proj(flat)
        h = h + self._position_tokens(n_tokens, device=h.device, dtype=h.dtype)
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        raw_flat = self.output_proj(h)

        raw = raw_flat.reshape(*leading, n_tokens, self.out_dim)
        normalized = F.normalize(raw, p=2, dim=-1, eps=self.eps)
        cloud = raw.mean(dim=-2)
        return MetricFeatures(raw=raw, normalized=normalized, cloud=cloud)


def pool_square_tokens(tokens: torch.Tensor, factor: int) -> torch.Tensor:
    """Spatially average-pool flattened square token grids.

    Args:
        tokens: (..., N, D), N must be a perfect square.
        factor: 1 keeps tokens; 2 maps 16x16 to 8x8.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor must be >=1")
    if factor == 1:
        return tokens
    n, d = tokens.shape[-2:]
    side = int(round(math.sqrt(n)))
    if side * side != n or side % factor != 0:
        raise ValueError(f"Cannot pool N={n} square tokens by factor={factor}")
    leading = tokens.shape[:-2]
    x = tokens.reshape(-1, side, side, d).permute(0, 3, 1, 2)
    x = F.avg_pool2d(x, kernel_size=factor, stride=factor)
    new_side = side // factor
    return x.permute(0, 2, 3, 1).reshape(*leading, new_side * new_side, d)