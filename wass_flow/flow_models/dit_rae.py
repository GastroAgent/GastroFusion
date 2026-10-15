"""
DiT for RAE Flow Matching.

直接在 RAE 潜变量空间 (B, 1024, 16, 16) 上运行 Flow Matching,
无需 UNet 和 RAE Mapper.

设计要点:
  - DiT-L: hidden_dim=1024, depth=24, num_heads=16, 完美匹配 RAE 输出维度
  - 3 层条件注入:
    1. AdaLN-Zero: timestep + class + text_cls + image_cls → 全局调制
    2. Cross-Attention: encoder_hidden_states (B, seq, 768) → 序列条件
    3. Spatial Addition: hint 图 → 空间对齐的结构条件
  - 输入输出都是 (B, 1024, 16, 16), 端到端在 RAE 空间

参考:
  - Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)
  - Flow Matching (Lipman et al., 2023)
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
from einops import rearrange


# ──────────────────────────────────────────────
# 基础组件
# ──────────────────────────────────────────────

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + MLP."""

    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.freq_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return self.mlp(emb)


class AdaLNZero(nn.Module):
    """Adaptive Layer Norm with zero-initialized gate.

    条件向量 c → MLP → (γ₁, β₁, α₁, γ₂, β₂, α₂, ...)
    每个 (γ, β, α) 三元组调制一个子层 (Self-Attn / Cross-Attn / MLP).
    α 初始化为 0, 保证训练初期为恒等映射.
    """

    def __init__(self, cond_dim: int, hidden_dim: int, num_modulations: int = 9):
        """
        Args:
            cond_dim: 条件向量维度
            hidden_dim: token 维度
            num_modulations: 调制参数数量 (3 per sub-layer: γ, β, α)
                - 3 sub-layers (Self-Attn, Cross-Attn, MLP) → 9
                - 2 sub-layers (Self-Attn, MLP) → 6
        """
        super().__init__()
        self.num_modulations = num_modulations
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        # 条件 → 调制参数
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_modulations * hidden_dim),
        )
        # zero-init the last linear
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Args:
            x: (B, N, D) token 序列
            c: (B, D) 条件向量

        Returns:
            list of (modulated_x, gate) tuples, 每个 sub-layer 一个
        """
        B, N, D = x.shape
        params = self.proj(c)  # (B, num_modulations * D)
        params = params.unsqueeze(1)  # (B, 1, num_modulations * D)

        # 切分为 (γ, β, α) 三元组
        chunks = params.chunk(self.num_modulations, dim=-1)  # each (B, 1, D)
        # 每 3 个一组: (γ, β, α)
        results = []
        for i in range(0, self.num_modulations, 3):
            gamma = chunks[i]       # (B, 1, D) scale
            beta = chunks[i + 1]    # (B, 1, D) shift
            alpha = chunks[i + 2]   # (B, 1, D) gate

            # 先对 x 做一次 LN, 然后用同一组 (γ, β) 调制
            x_norm = self.norm(x)
            x_mod = x_norm * (1 + gamma) + beta
            results.append((x_mod, alpha))

        return results


class SelfAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # RoPE freqs
        self._rope_init()

    def _rope_init(self):
        # Pre-compute RoPE frequencies (缓冲区, 不参与梯度)
        dim = self.head_dim
        freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('rope_freqs', freqs, persistent=False)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple:
        """Apply Rotary Position Embedding to q and k."""
        # q, k: (B, H, N, D_head)
        seq_len = q.shape[-2]
        t = torch.arange(seq_len, device=q.device, dtype=self.rope_freqs.dtype)
        freqs = torch.outer(t, self.rope_freqs)  # (N, D_head//2)
        cos = freqs.cos()  # (N, D_head//2)
        sin = freqs.sin()  # (N, D_head//2)
        # 重复以匹配完整的 head_dim: (N, D_head)
        cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, D_head)
        sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # RoPE
        q, k = self._apply_rope(q, k)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class CrossAttention(nn.Module):
    """Cross-Attention: Q from x, KV from encoder_hidden_states."""

    def __init__(self, hidden_dim: int, cond_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Args:
            hidden_dim: Q 维度 (DiT hidden)
            cond_dim: KV 维度 (encoder_hidden_states dim, 768)
            num_heads: 注意力头数
            dropout: dropout 率
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.k_proj = nn.Linear(cond_dim, hidden_dim, bias=True)
        self.v_proj = nn.Linear(cond_dim, hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_dim) token 序列 (Q)
            encoder_hidden_states: (B, M, cond_dim) 条件序列 (K, V)

        Returns:
            (B, N, hidden_dim)
        """
        B, N, _ = x.shape
        M = encoder_hidden_states.shape[1]

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_hidden_states).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_hidden_states).reshape(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    """Feed-Forward Network."""

    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
# DiT Block
# ──────────────────────────────────────────────

class DiTBlock(nn.Module):
    """DiT Block with AdaLN-Zero + Self-Attention + Cross-Attention + MLP.

    条件注入:
      - Self-Attention: AdaLN 调制 (γ₁, β₁) + zero-gate α₁
      - Cross-Attention: AdaLN 调制 (γ₂, β₂) + zero-gate α₂
      - MLP: AdaLN 调制 (γ₃, β₃) + zero-gate α₃

    每个子层的 α 初始化为 0, 保证训练初期为恒等映射.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        cond_dim: int = 1024,
        cross_attn_dim: int = 768,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention

        # AdaLN: 9 个调制参数 (3 sub-layers × 3 params)
        num_mod = 9 if use_cross_attention else 6
        self.adaln = AdaLNZero(cond_dim, hidden_dim, num_modulations=num_mod)

        # Self-Attention
        self.self_attn = SelfAttention(hidden_dim, num_heads, dropout)

        # Cross-Attention (可选)
        if use_cross_attention:
            self.cross_attn = CrossAttention(
                hidden_dim, cross_attn_dim, num_heads, dropout
            )

        # MLP
        self.mlp = MLP(hidden_dim, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) token 序列
            c: (B, D) 条件向量 (AdaLN)
            encoder_hidden_states: (B, M, cross_dim) cross-attention 条件

        Returns:
            (B, N, D)
        """
        mods = self.adaln(x, c)

        # Self-Attention
        x_mod, gate = mods[0]
        x = x + gate * self.self_attn(x_mod)

        # Cross-Attention
        if self.use_cross_attention and encoder_hidden_states is not None:
            x_mod, gate = mods[1]
            x = x + gate * self.cross_attn(x_mod, encoder_hidden_states)

        # MLP
        mlp_idx = 2 if self.use_cross_attention else 1
        x_mod, gate = mods[mlp_idx]
        x = x + gate * self.mlp(x_mod)

        return x


# ──────────────────────────────────────────────
# Hint Encoder
# ──────────────────────────────────────────────

class HintEncoder(nn.Module):
    """将 hint 图像编码为与 RAE 潜变量对齐的空间特征.

    (B, 3, H, W) → CNN → (B, 1024, 16, 16) → flatten → (B, 256, 1024)

    使用轻量 CNN, 自适应池化到目标空间尺寸.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 1024,
        target_spatial: int = 16,
    ):
        super().__init__()
        self.target_spatial = target_spatial
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 256, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Conv2d(512, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((target_spatial, target_spatial))

        # 初始化为接近零, 训练初期 hint 不影响主路径
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hint: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hint: (B, 3, H, W)

        Returns:
            (B, 256, hidden_dim) — 256 = 16×16
        """
        h = self.encoder(hint)
        h = self.pool(h)  # (B, hidden_dim, 16, 16)
        h = rearrange(h, 'b c h w -> b (h w) c')  # (B, 256, hidden_dim)
        return h


# ──────────────────────────────────────────────
# Final Layer
# ──────────────────────────────────────────────

class FinalLayer(nn.Module):
    """最终层: AdaLN 调制 + Linear.

    参考 DiT 的 final layer, 将 token 映射回目标维度.
    Linear 层 zero-initialized.
    """

    def __init__(self, hidden_dim: int, cond_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_dim),
        )
        self.linear = nn.Linear(hidden_dim, out_dim)

        # zero-init
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, hidden_dim)
            c: (B, cond_dim)

        Returns:
            (B, N, out_dim)
        """
        params = self.adaLN_mod(c).unsqueeze(1)  # (B, 1, 2*hidden_dim)
        gamma, beta = params.chunk(2, dim=-1)
        x = self.norm(x) * (1 + gamma) + beta
        return self.linear(x)


# ──────────────────────────────────────────────
# DiTForRAEFlow 主模型
# ──────────────────────────────────────────────

class DiTForRAEFlow(nn.Module):
    """DiT 直接在 RAE 潜变量空间做 Flow Matching.

    输入/输出都是 (B, 1024, 16, 16), 无需 RAE Mapper.

    条件注入 (3 层):
      1. AdaLN-Zero: timestep + class + text_cls + image_cls + hint_global
      2. Cross-Attention: encoder_hidden_states (文本+视觉 patch tokens)
      3. Spatial Addition: hint 图 (空间对齐的结构条件, 加到 x tokens)

    用法:
        model = DiTForRAEFlow(hidden_dim=1024, depth=24, num_heads=16)

        # 训练
        vt_pred = model(
            xt=xt,                          # (B, 1024, 16, 16)
            timestep=t,                     # (B,)
            class_labels=y,                 # (B,) int
            encoder_hidden_states=enc_hs,   # (B, seq, 768)
            hint=hint,                      # (B, 3, H, W)
            text_embeds=text_cls,           # (B, 768)
            image_embeds=image_cls,         # (B, 768)
        )
        # vt_pred: (B, 1024, 16, 16)
    """

    def __init__(
        self,
        # ── DiT 架构参数 ──
        hidden_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        # ── RAE 空间参数 ──
        rae_latent_dim: int = 1024,
        rae_spatial_size: int = 16,
        # ── 条件参数 ──
        cross_attn_dim: int = 768,          # encoder_hidden_states 维度
        num_classes: int = 1000,            # class embedding 数量
        hint_in_channels: int = 3,
        use_cross_attention: bool = True,
        # ── 学习参数 ──
        learn_sigma: bool = False,          # 是否学习方差 (暂未实现)
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.rae_latent_dim = rae_latent_dim
        self.rae_spatial_size = rae_spatial_size
        self.num_tokens = rae_spatial_size ** 2  # 256
        self.use_cross_attention = use_cross_attention

        # ═══ 输入处理 ═══
        # RAE 潜变量: (B, rae_latent_dim, 16, 16) → flatten → (B, 256, rae_latent_dim)
        # 如果 hidden_dim == rae_latent_dim (DiT-L: 1024==1024), 不需要投影
        if hidden_dim == rae_latent_dim:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Linear(rae_latent_dim, hidden_dim)

        # 位置编码 (可学习)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_tokens, hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ═══ Hint Encoder ═══
        self.hint_encoder = HintEncoder(
            in_channels=hint_in_channels,
            hidden_dim=hidden_dim,
            target_spatial=rae_spatial_size,
        )

        # ═══ 条件向量构建 (用于 AdaLN) ═══
        # c = time_embed + class_embed + text_proj + image_proj
        self.time_embed = TimestepEmbedding(hidden_dim, freq_dim=256)
        self.class_embed = nn.Embedding(num_classes, hidden_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(cross_attn_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(cross_attn_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ═══ Transformer Blocks ═══
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                cond_dim=hidden_dim,
                cross_attn_dim=cross_attn_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_cross_attention=use_cross_attention,
            )
            for _ in range(depth)
        ])

        # ═══ 输出层 ═══
        self.final_layer = FinalLayer(hidden_dim, hidden_dim, rae_latent_dim)

        # ═══ 输出 reshape: 如果 hidden_dim != rae_latent_dim, 需要额外投影 ═══
        # FinalLayer 已经输出 rae_latent_dim, 直接 reshape 即可

        self.initialize_weights()

    def initialize_weights(self):
        """初始化权重."""
        # 时间/类别/条件投影
        def _init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.time_embed.apply(_init_linear)
        self.text_proj.apply(_init_linear)
        self.image_proj.apply(_init_linear)
        nn.init.normal_(self.class_embed.weight, std=0.02)

        # Transformer blocks 中的 Self-Attn / Cross-Attn / MLP
        for block in self.blocks:
            # Self-Attention
            nn.init.xavier_uniform_(block.self_attn.qkv.weight)
            nn.init.zeros_(block.self_attn.qkv.bias)
            nn.init.xavier_uniform_(block.self_attn.proj.weight)
            nn.init.zeros_(block.self_attn.proj.bias)
            # Cross-Attention
            if block.use_cross_attention:
                nn.init.xavier_uniform_(block.cross_attn.q_proj.weight)
                nn.init.zeros_(block.cross_attn.q_proj.bias)
                nn.init.xavier_uniform_(block.cross_attn.k_proj.weight)
                nn.init.zeros_(block.cross_attn.k_proj.bias)
                nn.init.xavier_uniform_(block.cross_attn.v_proj.weight)
                nn.init.zeros_(block.cross_attn.v_proj.bias)
                nn.init.xavier_uniform_(block.cross_attn.out_proj.weight)
                nn.init.zeros_(block.cross_attn.out_proj.bias)
            # MLP
            nn.init.xavier_uniform_(block.mlp.net[0].weight)
            nn.init.zeros_(block.mlp.net[0].bias)
            nn.init.xavier_uniform_(block.mlp.net[3].weight)
            nn.init.zeros_(block.mlp.net[3].bias)
            # AdaLN (已 zero-init)

    def _build_condition_vector(
        self,
        timestep: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """构建 AdaLN 条件向量 c.

        c = time_embed(t) + class_embed(y) + text_proj(text_cls) + image_proj(image_cls)

        Args:
            timestep: (B,) 时间步
            class_labels: (B,) int 类别 ID
            text_embeds: (B, cross_attn_dim) 文本 CLS embedding
            image_embeds: (B, cross_attn_dim) 视觉 CLS embedding

        Returns:
            c: (B, hidden_dim)
        """
        B = timestep.shape[0]
        c = self.time_embed(timestep)  # (B, hidden_dim)

        if class_labels is not None:
            c = c + self.class_embed(class_labels)  # (B, hidden_dim)

        if text_embeds is not None:
            c = c + self.text_proj(text_embeds)  # (B, hidden_dim)

        if image_embeds is not None:
            c = c + self.image_proj(image_embeds)  # (B, hidden_dim)

        return c

    def forward(
        self,
        xt: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        hint: Optional[torch.Tensor] = None,
        text_embeds: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        **kwargs,  # 兼容 UNet 接口的额外参数 (如 added_cond_kwargs, image_hint_model)
    ) -> dict:
        """前向传播: 预测速度场 vt.

        Args:
            xt: (B, 1024, 16, 16) 噪声潜变量 (RAE 空间)
            timestep: (B,) 时间步
            class_labels: (B,) int 类别 ID
            encoder_hidden_states: (B, seq, 768) 文本+视觉 token 序列
            hint: (B, 3, H, W) hint 图像
            text_embeds: (B, 768) 文本 CLS embedding
            image_embeds: (B, 768) 视觉 CLS embedding
            **kwargs: 兼容性参数, 忽略

        Returns:
            dict with 'sample': (B, 1024, 16, 16) 预测速度场
        """
        B = xt.shape[0]

        # ① Input: flatten + project + positional encoding
        x = rearrange(xt, 'b c h w -> b (h w) c')  # (B, 256, rae_latent_dim)
        x = self.input_proj(x)                       # (B, 256, hidden_dim)
        x = x + self.pos_embed                       # 位置编码

        # ② Hint: 空间加法 (与 x 对齐)
        if hint is not None:
            hint_tokens = self.hint_encoder(hint)  # (B, 256, hidden_dim)
            x = x + hint_tokens

        # ③ 条件向量 c (用于 AdaLN)
        c = self._build_condition_vector(
            timestep, class_labels, text_embeds, image_embeds
        )  # (B, hidden_dim)

        # ④ Transformer Blocks
        for block in self.blocks:
            x = block(x, c, encoder_hidden_states)

        # ⑤ Final Layer + reshape
        x = self.final_layer(x, c)  # (B, 256, rae_latent_dim)
        vt = rearrange(
            x, 'b (h w) c -> b c h w',
            h=self.rae_spatial_size,
            w=self.rae_spatial_size,
        )

        return {'sample': vt}

    def get_model_info(self) -> dict:
        """返回模型信息."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_dim': self.hidden_dim,
            'num_tokens': self.num_tokens,
            'depth': len(self.blocks),
            'num_heads': self.blocks[0].self_attn.num_heads,
        }


# ──────────────────────────────────────────────
# 工厂函数 & 预设配置
# ──────────────────────────────────────────────

def create_dit_rae(
    variant: str = 'L',
    rae_latent_dim: int = 1024,
    rae_spatial_size: int = 16,
    cross_attn_dim: int = 768,
    num_classes: int = 1000,
    use_cross_attention: bool = True,
    dropout: float = 0.0,
) -> DiTForRAEFlow:
    """创建 DiT for RAE Flow Matching 模型.

    Args:
        variant: 模型变体
            'S'  - DiT-S:  hidden=384,  depth=12, heads=6   (~33M, 需要投影)
            'B'  - DiT-B:  hidden=768,  depth=12, heads=12  (~86M, 需要投影)
            'L'  - DiT-L:  hidden=1024, depth=24, heads=16  (~400M, 完美匹配RAE)
            'L12'- DiT-L12: hidden=1024, depth=12, heads=16 (~200M, 匹配RAE, 减半层数)
        rae_latent_dim: RAE 潜变量维度 (默认 1024)
        rae_spatial_size: RAE 空间尺寸 (默认 16)
        cross_attn_dim: encoder_hidden_states 维度 (默认 768)
        num_classes: 类别数量
        use_cross_attention: 是否使用 cross-attention
        dropout: dropout 率

    Returns:
        DiTForRAEFlow 实例
    """
    configs = {
        'S': dict(hidden_dim=384, depth=12, num_heads=6, mlp_ratio=4.0),
        'B': dict(hidden_dim=768, depth=12, num_heads=12, mlp_ratio=4.0),
        'L': dict(hidden_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0),
        'L12': dict(hidden_dim=1024, depth=12, num_heads=16, mlp_ratio=4.0),
    }

    if variant not in configs:
        raise ValueError(
            f"Unknown variant: {variant}, must be one of {list(configs.keys())}"
        )

    cfg = configs[variant]
    return DiTForRAEFlow(
        hidden_dim=cfg['hidden_dim'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        dropout=dropout,
        rae_latent_dim=rae_latent_dim,
        rae_spatial_size=rae_spatial_size,
        cross_attn_dim=cross_attn_dim,
        num_classes=num_classes,
        use_cross_attention=use_cross_attention,
    )


# ──────────────────────────────────────────────
# 验证脚本
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 60)
    print('DiT for RAE Flow Matching — 维度验证')
    print('=' * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 2

    for variant in ['S', 'B', 'L', 'L12']:
        print(f'\n{"─" * 50}')
        print(f'Variant: {variant}')
        print(f'{"─" * 50}')

        model = create_dit_rae(variant=variant).to(device)
        info = model.get_model_info()
        print(f'  hidden_dim={info["hidden_dim"]}, depth={info["depth"]}, '
              f'heads={info["num_heads"]}, params={info["total_parameters"]:,}')

        # 前向传播
        xt = torch.randn(B, 1024, 16, 16, device=device)
        t = torch.rand(B, device=device)
        y = torch.randint(0, 1000, (B,), device=device)
        enc_hs = torch.randn(B, 50, 768, device=device)
        hint = torch.randn(B, 3, 224, 224, device=device)
        text_cls = torch.randn(B, 768, device=device)
        img_cls = torch.randn(B, 768, device=device)

        output = model(
            xt=xt,
            timestep=t,
            class_labels=y,
            encoder_hidden_states=enc_hs,
            hint=hint,
            text_embeds=text_cls,
            image_embeds=img_cls,
        )

        vt = output['sample']
        print(f'  Input:  {list(xt.shape)}')
        print(f'  Output: {list(vt.shape)}')
        assert vt.shape == xt.shape, f'Shape mismatch: {vt.shape} != {xt.shape}'

        # 梯度检查
        xt_grad = torch.randn(B, 1024, 16, 16, device=device, requires_grad=True)
        out = model(xt=xt_grad, timestep=t, class_labels=y,
                    encoder_hidden_states=enc_hs, hint=hint,
                    text_embeds=text_cls, image_embeds=img_cls)
        out['sample'].sum().backward()
        assert xt_grad.grad is not None, 'Gradient broken!'
        print(f'  Gradient: OK (norm={xt_grad.grad.norm().item():.4f})')

    print(f'\n{"=" * 60}')
    print('所有维度验证通过! ✓')
    print(f'{"=" * 60}')