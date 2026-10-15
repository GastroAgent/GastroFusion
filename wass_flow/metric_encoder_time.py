"""Time-conditioned MetricEncoder for trajectory-aware Q2L / Q2Step pretraining.

Design (agreed with user):
  * The base MetricEncoder is entirely *unconditional* (no class embedding).
    Here time ``t in [0,1]`` is the ONLY conditioning signal.
  * Kept the stronger AdaLN-Zero-style in-block conditioning, but in a
    re-parameterized form so that all old ``MetricEncoder`` state dict keys
    (``input_proj / pos_embed_2d / blocks.*.norm1 / blocks.*.attn / ...``)
    keep their exact names and can be loaded with ``strict=False``.
  * Conditioning semantics per block::

        attn_branch:  x = x + gate1 * attn( norm1( x*(1+scale1)+shift1 ) )
        mlp_branch:   x = x + gate2 * mlp( norm2( x*(1+scale2)+shift2 ) )

  * Initialization preserves the previously trained Q2L metric at step one:
        scale1 == shift1 == scale2 == shift2 == 0   -> norms unchanged
        gate1 == gate2 == 1.0                        -> residual equals original
    so ``TimestepMetricEncoder(init=Q2L)`` is identical to the Q2L model
    before training, and time modulation only *extends* it afterwards.

Time stamp convention used by the training scripts:
  * Real latents (query images, support images, z0) are encoded with t=0.
  * Flow trajectory landing points z_k (k=1..T) are encoded with t=k/T,
    therefore E(z_T, t=1) is the Q2L endpoint and Q2StepMean scans k=1..T.

The public output interface (MetricFeatures) is preserved so downstream
Sinkhorn-Knopp, VISReg and evaluation code does not change.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from metric_encoder import MetricEncoder, MetricFeatures


# ---------------------------------------------------------------------------
# Timestep conditioning helpers
# ---------------------------------------------------------------------------
def _timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal embedding of a continuous timestep (DiT / DDPM style).

    Args:
        t: (...,) scalar timesteps in any range (training uses 0..1).
        dim: output feature dimension (must be even).
    Returns:
        (..., dim) tensor of cos/sin Fourier features.
    """
    if dim % 2 != 0:
        raise ValueError(f"_timestep_embedding needs even dim, got {dim}")
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.to(freqs.dtype).unsqueeze(-1) * freqs  # (..., half)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _AdaLNModulation(nn.Module):
    """Predicts (scale1, shift1, gate1, scale2, shift2, gate2) from time.

    ``gate1 / gate2`` residuals are initialised to ``gate_bias`` (default 1.0)
    so that at step zero the time-conditioned block equals the original
    unconditional block, while gains/shifts stay at 0 (AdaLN-Zero spirit but
    with continuation of the pre-trained Q2L encoder).
    """

    CHUNK_ORDER = ("scale1", "shift1", "gate1", "scale2", "shift2", "gate2")

    def __init__(self, cond_dim: int, dim: int, gate_bias: float = 1.0) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.dim = int(dim)
        self.gate_bias = float(gate_bias)
        self.mod = nn.Linear(self.cond_dim, 6 * self.dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # AdaLN-Zero style init plus continuation gates:
        #   weights -> 0 (outputs are pure biases)
        #   scale/shift slots -> 0         (norm unchanged)
        #   gate1/gate2 bias   -> gate_bias (residual equals original block)
        gate_idx1 = 2
        gate_idx2 = 5
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        with torch.no_grad():
            dim = self.dim
            self.mod.bias[gate_idx1 * dim : (gate_idx1 + 1) * dim].fill_(self.gate_bias)
            self.mod.bias[gate_idx2 * dim : (gate_idx2 + 1) * dim].fill_(self.gate_bias)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """Returns (..., 6, dim) modulations in CHUNK_ORDER."""
        out = self.mod(cond)  # (..., 6*dim)
        *lead, _ = out.shape
        return out.reshape(*lead, 6, self.dim)


def _modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Per-channel affine modulation: x*(1+scale)+shift. Broadcast over tokens."""
    return x * (1.0 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)


class TimestepMetricTransformerBlock(nn.Module):
    """Pre-norm Transformer block with timestep AdaLN-Zero conditioning.

    Module names (norm1 / attn / drop1 / norm2 / mlp / drop2) intentionally
    mirror :class:`MetricTransformerBlock` (metric_encoder.py) so old
    checkpoints load 1:1.  The new conditioning weights live in
    ``adaln_mod`` which only exists in the time-conditioned model.
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_dim: int = 512,
        gate_bias: float = 1.0,
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

        self.adaln_mod = _AdaLNModulation(cond_dim=time_dim, dim=dim, gate_bias=gate_bias)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond is None:
            # Plain unconditional path, identical to base MetricTransformerBlock.
            h = self.norm1(x)
            h, _ = self.attn(h, h, h, need_weights=False)
            x = x + self.drop1(h)
            x = x + self.mlp(self.norm2(x))
            return x

        # (..., 6, dim) -> per-channel vectors.
        scale1, shift1, gate1, scale2, shift2, gate2 = cond.unbind(dim=-2)

        # Attention sub-block.
        h = _modulate(self.norm1(x), scale1, shift1)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate1.unsqueeze(-2) * self.drop1(h)

        # MLP sub-block.
        h = _modulate(self.norm2(x), scale2, shift2)
        x = x + gate2.unsqueeze(-2) * self.mlp(h)
        return x


class TimestepMetricEncoder(MetricEncoder):
    """Time-conditioned MetricEncoder with Q2L state-dict compatibility.

    Re-uses every inherited public helper (``to_tokens``, ``_position_tokens``,
    ``MetricFeatures`` container).  The Transformer blocks are time-conditioned
    clones of the original blocks and share all original key names.
    """

    def __init__(
        self,
        in_dim: int = 1024,
        hidden_dim: int = 256,
        out_dim: int = 256,
        dropout: float = 0.0,
        residual: bool = False,
        eps: float = 1e-8,
        time_dim: int = 512,
        gate_bias: float = 1.0,
    ) -> None:
        # Deliberately do NOT call MetricEncoder.__init__: we rebuild the same
        # architecture with time-conditioned blocks while keeping identical
        # parameter key names. All inherited utilities remain available.
        nn.Module.__init__(self)

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.eps = float(eps)
        self.requested_hidden_dim = int(hidden_dim)
        # Kept for API compatibility with the old projector. The Transformer
        # blocks already contain residual paths internally.
        self.residual = True if residual else False

        self.model_dim = self.MODEL_DIM  # fixed V1 capacity, 512
        self.depth = self.DEPTH
        self.num_heads = self.NUM_HEADS
        self.mlp_ratio = self.MLP_RATIO
        self.time_dim = int(time_dim)
        self.gate_bias = float(gate_bias)

        self.input_proj = nn.Linear(self.in_dim, self.model_dim)
        # Fixed V1 capacity: inherited constant name retained as-is.
        self.pos_embed_2d = nn.Parameter(
            torch.zeros(1, self.model_dim, self.BASE_GRID_SIZE, self.BASE_GRID_SIZE)
        )

        # Shared timestep Fourier -> MLP conditioner.
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.blocks = nn.ModuleList(
            [
                TimestepMetricTransformerBlock(
                    dim=self.model_dim,
                    num_heads=self.NUM_HEADS,
                    mlp_ratio=self.MLP_RATIO,
                    dropout=dropout,
                    time_dim=self.time_dim,
                    gate_bias=self.gate_bias,
                )
                for _ in range(self.DEPTH)
            ]
        )

        self.final_norm = nn.LayerNorm(self.model_dim)
        self.output_proj = nn.Linear(self.model_dim, self.out_dim)

        self.reset_parameters()

    # -- initialization ----------------------------------------------------
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed_2d, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and not isinstance(module, _AdaLNModulation):
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
        # Re-apply the continuation init for every AdaLN modulator (the loop
        # above skips their inner Linear so the gate bias survives).
        for module in self.modules():
            if isinstance(module, _AdaLNModulation):
                module.reset_parameters()

    # -- time embedding ----------------------------------------------------
    def embed_time(self, t: torch.Tensor) -> torch.Tensor:
        """FFN embedding of timestep scalars (...,) -> (..., time_dim)."""
        feats = _timestep_embedding(t, dim=self.time_dim)  # (..., time_dim)
        return self.time_embed(feats)

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> MetricFeatures:
        """Encode latent tokens.  Pass ``t`` (...,) to time-condition.

        When ``t is None`` this returns exactly the base MetricEncoder
        computation (useful for eval / weight-diff comparisons).
        """
        tokens = self.to_tokens(x)
        if tokens.shape[-1] != self.in_dim:
            raise ValueError(
                f"MetricEncoder expected token dim {self.in_dim}, got {tokens.shape[-1]} "
                f"from shape {tuple(x.shape)}"
            )

        leading = tokens.shape[:-2]
        n_tokens = tokens.shape[-2]
        flat = tokens.reshape(-1, n_tokens, self.in_dim)
        n_sets = flat.shape[0]

        h = self.input_proj(flat)
        h = h + self._position_tokens(n_tokens, device=h.device, dtype=h.dtype)

        if t is None:
            for block in self.blocks:
                h = block(h)
        else:
            # Flatten any leading layout to one scalar per token set.
            t_flat = t.reshape(n_sets).to(device=h.device, dtype=h.dtype)
            if (t_flat < 0.0).any() or (t_flat > 1.0).any():
                raise ValueError(
                    f"Timestep values must be inside [0,1], got range "
                    f"[{t_flat.min().item():.3f}, {t_flat.max().item():.3f}]"
                )
            cond_all = self.embed_time(t_flat)  # (M, time_dim)
            for block in self.blocks:
                cond = block.adaln_mod(cond_all)  # (M, 6, dim)
                h = block(h, cond)

        h = self.final_norm(h)
        raw_flat = self.output_proj(h)

        raw = raw_flat.reshape(*leading, n_tokens, self.out_dim)
        normalized = F.normalize(raw, p=2, dim=-1, eps=self.eps)
        cloud = raw.mean(dim=-2)
        return MetricFeatures(raw=raw, normalized=normalized, cloud=cloud)


def build_timestep_metric_from_checkpoint(
    state_dict: dict[str, torch.Tensor],
    *,
    in_dim: int = 1024,
    out_dim: int = 256,
    dropout: float = 0.0,
    time_dim: int = 512,
    gate_bias: float = 1.0,
    verbose: bool = True,
) -> TimestepMetricEncoder:
    """Create a TimestepMetricEncoder and load an old MetricEncoder state dict.

    All original keys are loaded exactly; only the new ``time_embed`` and
    ``blocks.*.adaln_mod`` parameters remain at their continuation init, so a
    freshly built model produces exactly the same outputs as the Q2L encoder
    it was initialized from.
    """
    model = TimestepMetricEncoder(
        in_dim=in_dim,
        out_dim=out_dim,
        dropout=dropout,
        time_dim=time_dim,
        gate_bias=gate_bias,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # All missing keys must belong to the *new* time-conditioning path.
        def _is_new_key(k: str) -> bool:
            return k.startswith("time_embed.") or ".adaln_mod." in k

        bad = [k for k in missing if not _is_new_key(k)]
        if bad:
            raise ValueError(
                f"Old MetricEncoder state dict misses original keys "
                f"(architecture mismatch): {sorted(bad)[:5]} ..."
            )
        if verbose:
            print(
                f"[TimestepMetricEncoder] initialized from Q2L checkpoint: "
                f"{len(missing)} new time-conditioning keys at continuation init."
            )
    if unexpected:
        raise ValueError(
            f"State dict contains keys not part of TimestepMetricEncoder: "
            f"{sorted(unexpected)[:10]} ..."
        )
    return model
