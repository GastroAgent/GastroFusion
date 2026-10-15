"""
RAE ↔ UNet 潜变量空间映射模块.

RAE 输出维度: (B, 1024, 16, 16)
UNet 输入维度: (B, in_channels, sample_size, sample_size)  默认 (B, 8, 64, 64)
UNet 输出维度: (B, out_channels, sample_size, sample_size)  默认 (B, 4, 64, 64)

映射策略:
  - 空间: PixelShuffle 进行 4× 上采样 (16×16 → 64×64)
  - 通道: 先用 1×1 卷积将通道调整到 PixelShuffle 所需的输入通道数,
          再通过 PixelShuffle 同时完成空间上采样和通道降维.

逆映射策略:
  - 空间: PixelUnshuffle 进行 4× 下采样 (64×64 → 16×16)
  - 通道: 先用 PixelUnshuffle 将空间压缩到通道维度,
          再用 1×1 卷积将通道从 out_channels * r^2 映射回 rae_latent_dim.
"""

import torch
import torch.nn as nn


class RAEmapper(nn.Module):
    """将 RAE 潜变量映射到 UNet 输入空间 (正向), 以及从 UNet 输出空间映射回 RAE 空间 (逆向).

    正向映射:  (B, rae_dim, 16, 16) → (B, unet_in_channels, 64, 64)
    逆向映射:  (B, unet_out_channels, 64, 64) → (B, rae_dim, 16, 16)

    其中空间缩放因子 r = sample_size // rae_spatial_size (默认 64 // 16 = 4).

    PixelShuffle 工作原理 (以 r=4 为例):
      输入:  (B, C_in, H, W)
      输出:  (B, C_in / r², H*r, W*r)
      要求:  C_in 必须能被 r² = 16 整除.

    因此正向映射中:
      1×1 卷积: rae_dim → unet_in_channels * r² = 8 * 16 = 128
      PixelShuffle: (B, 128, 16, 16) → (B, 8, 64, 64)  ✓

    逆向映射中:
      PixelUnshuffle: (B, 4, 64, 64) → (B, 4*16=64, 16, 16)
      1×1 卷积: 64 → rae_dim = 1024
    """

    def __init__(
        self,
        rae_latent_dim: int = 1024,
        rae_spatial_size: int = 16,
        unet_in_channels: int = 8,
        unet_out_channels: int = 4,
        sample_size: int = 64,
    ):
        super().__init__()

        self.rae_latent_dim = rae_latent_dim
        self.rae_spatial_size = rae_spatial_size
        self.unet_in_channels = unet_in_channels
        self.unet_out_channels = unet_out_channels
        self.sample_size = sample_size

        # 空间缩放因子
        self.r = sample_size // rae_spatial_size  # 64 // 16 = 4
        assert sample_size % rae_spatial_size == 0, (
            f"sample_size ({sample_size}) 必须能被 rae_spatial_size ({rae_spatial_size}) 整除"
        )
        r2 = self.r * self.r  # 16

        # ── 正向映射: RAE → UNet ──
        # PixelShuffle 要求输入通道数 = out_channels * r²
        forward_mid_channels = unet_in_channels * r2  # 8 * 16 = 128

        # 检查 rae_latent_dim 是否足够大以支撑该映射
        if rae_latent_dim < forward_mid_channels:
            # 当 rae_latent_dim 较小时, 先上采样通道再 PixelShuffle
            self.forward_proj = nn.Sequential(
                nn.Conv2d(rae_latent_dim, forward_mid_channels, 1, bias=False),
                nn.GroupNorm(8, forward_mid_channels),
                nn.SiLU(),
            )
        else:
            self.forward_proj = nn.Sequential(
                nn.GroupNorm(32, rae_latent_dim),
                nn.SiLU(),
                nn.Conv2d(rae_latent_dim, forward_mid_channels, 1, bias=False),
            )

        self.forward_shuffle = nn.PixelShuffle(self.r)

        # ── 逆向映射: UNet → RAE ──
        # PixelUnshuffle: (B, unet_out_channels, sample_size, sample_size)
        #              → (B, unet_out_channels * r², rae_spatial_size, rae_spatial_size)
        backward_mid_channels = unet_out_channels * r2  # 4 * 16 = 64

        self.backward_unshuffle = nn.PixelUnshuffle(self.r)

        self.backward_proj = nn.Sequential(
            nn.GroupNorm(8, backward_mid_channels),
            nn.SiLU(),
            nn.Conv2d(backward_mid_channels, rae_latent_dim, 1, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重, 使得初始映射接近恒等/零映射, 保证训练初期稳定."""
        for module in [self.forward_proj, self.backward_proj]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    # 使用很小的随机初始化, 让映射从近似零开始学习
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, z_rae: torch.Tensor) -> torch.Tensor:
        """正向映射: RAE 潜变量 → UNet 输入空间.

        Args:
            z_rae: RAE 编码器输出, shape (B, rae_latent_dim, 16, 16)

        Returns:
            z_unet: UNet 输入空间的潜变量, shape (B, unet_in_channels, 64, 64)
        """
        # (B, 1024, 16, 16) → conv → (B, 128, 16, 16) → shuffle → (B, 8, 64, 64)
        h = self.forward_proj(z_rae)
        h = self.forward_shuffle(h)
        return h

    def inverse(self, z_unet: torch.Tensor) -> torch.Tensor:
        """逆向映射: UNet 输出 → RAE 潜变量空间.

        Args:
            z_unet: UNet 输出, shape (B, unet_out_channels, 64, 64)

        Returns:
            z_rae: RAE 空间的潜变量, shape (B, rae_latent_dim, 16, 16)
        """
        # (B, 4, 64, 64) → unshuffle → (B, 64, 16, 16) → conv → (B, 1024, 16, 16)
        h = self.backward_unshuffle(z_unet)
        h = self.backward_proj(h)
        return h

    def encode_pair(self, z0_rae: torch.Tensor, z1_rae: torch.Tensor):
        """将一对 RAE 潜变量同时映射到 UNet 空间 (用于 Flow Matching 训练).

        Args:
            z0_rae: 源图像 RAE 潜变量, shape (B, rae_latent_dim, 16, 16)
            z1_rae: 目标图像 RAE 潜变量, shape (B, rae_latent_dim, 16, 16)

        Returns:
            (z0_unet, z1_unet): 映射后的 UNet 空间潜变量对
        """
        return self.forward(z0_rae), self.forward(z1_rae)


class ResBlock(nn.Module):
    """残差块: Conv3×3 + GN + SiLU + Conv3×3 + GN + shortcut."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class RAEmapperDeep(nn.Module):
    """深层 RAE↔UNet 映射器 (约 3M 参数).

    使用多层 ResBlock + 3×3 卷积 + PixelShuffle/PixelUnshuffle,
    相比 RAEmapper(RAEmapperResidual) 的单层 1×1 卷积, 具有更强的
    空间建模能力和更大的参数量, 能更好地完成 RAE↔UNet 之间的维度转换.

    Forward:  (B, rae_dim, 16, 16) → ResBlocks → PixelShuffle → (B, unet_in, 64, 64)
    Inverse:  (B, unet_out, 64, 64) → PixelUnshuffle → ResBlocks → (B, rae_dim, 16, 16)
    """

    def __init__(
        self,
        rae_latent_dim: int = 1024,
        rae_spatial_size: int = 16,
        unet_in_channels: int = 4,
        unet_out_channels: int = 4,
        sample_size: int = 64,
        mid_channels: int = 128,
        num_res_blocks: int = 5,
    ):
        super().__init__()

        self.rae_latent_dim = rae_latent_dim
        self.rae_spatial_size = rae_spatial_size
        self.unet_in_channels = unet_in_channels
        self.unet_out_channels = unet_out_channels
        self.sample_size = sample_size

        # 空间缩放因子
        self.r = sample_size // rae_spatial_size  # 64 // 16 = 4
        assert sample_size % rae_spatial_size == 0
        r2 = self.r * self.r  # 16

        # ── Forward: RAE (1024, 16, 16) → UNet (4, 64, 64) ──
        # PixelShuffle(4) 需要 in_channels = out_channels × r² = 4 × 16 = 64
        forward_ps_in = unet_in_channels * r2  # 64

        self.forward_in = nn.Sequential(
            nn.Conv2d(rae_latent_dim, mid_channels, 1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU(),
        )
        self.forward_res = nn.Sequential(
            *[ResBlock(mid_channels) for _ in range(num_res_blocks)]
        )
        self.forward_out = nn.Sequential(
            nn.Conv2d(mid_channels, forward_ps_in, 1, bias=False),
            nn.GroupNorm(4, forward_ps_in),
            nn.SiLU(),
        )
        self.forward_shuffle = nn.PixelShuffle(self.r)

        # ── Inverse: UNet (4, 64, 64) → RAE (1024, 16, 16) ──
        # PixelUnshuffle(4): (B, 4, 64, 64) → (B, 64, 16, 16)
        backward_us_out = unet_out_channels * r2  # 64

        self.backward_unshuffle = nn.PixelUnshuffle(self.r)
        self.backward_in = nn.Sequential(
            nn.Conv2d(backward_us_out, mid_channels, 1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU(),
        )
        self.backward_res = nn.Sequential(
            *[ResBlock(mid_channels) for _ in range(num_res_blocks)]
        )
        self.backward_out = nn.Sequential(
            nn.Conv2d(mid_channels, rae_latent_dim, 1, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重."""
        for module in [self.forward_in, self.forward_out,
                       self.backward_in, self.backward_out]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        # ResBlock 使用默认初始化
        for res_block in self.forward_res:
            for m in res_block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        for res_block in self.backward_res:
            for m in res_block.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, z_rae: torch.Tensor) -> torch.Tensor:
        """正向映射: RAE → UNet (纯主路径, 无残差旁路)."""
        h = self.forward_in(z_rae)
        h = self.forward_res(h)
        h = self.forward_out(h)
        h = self.forward_shuffle(h)
        return h

    def inverse(self, z_unet: torch.Tensor) -> torch.Tensor:
        """逆向映射: UNet → RAE (纯主路径, 无残差旁路)."""
        h = self.backward_unshuffle(z_unet)
        h = self.backward_in(h)
        h = self.backward_res(h)
        h = self.backward_out(h)
        return h

    def encode_pair(self, z0_rae: torch.Tensor, z1_rae: torch.Tensor):
        return self.forward(z0_rae), self.forward(z1_rae)


class RAEmapperResidual(RAEmapperDeep):
    """Deep + 残差旁路的 RAE 映射器.

    继承 RAEmapperDeep 的多层 ResBlock + PixelShuffle 主路径,
    额外添加双线性插值 + 1×1 卷积的残差旁路, 确保即使主路径
    学习不充分, 信息也能通过直连路径传递.

    Forward:  main_path(z_rae) + bypass(z_rae) * scale
    Inverse:  main_path(z_unet) + bypass(z_unet) * scale
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ── 残差旁路: 双线性上/下采样 + 1×1 卷积通道调整 ──
        self.residual_upsample = nn.Upsample(
            scale_factor=self.r, mode='bilinear', align_corners=False
        )
        self.residual_proj = nn.Conv2d(
            self.rae_latent_dim, self.unet_in_channels, 1, bias=True
        )
        self.residual_downsample = nn.Upsample(
            scale_factor=1.0 / self.r, mode='bilinear', align_corners=False
        )
        self.residual_inv_proj = nn.Conv2d(
            self.unet_out_channels, self.rae_latent_dim, 1, bias=True
        )

        # 残差缩放因子 (初始较小, 让主路径逐渐学习)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # 初始化残差旁路为近似零
        nn.init.zeros_(self.residual_proj.weight)
        nn.init.zeros_(self.residual_proj.bias)
        nn.init.zeros_(self.residual_inv_proj.weight)
        nn.init.zeros_(self.residual_inv_proj.bias)

    def forward(self, z_rae: torch.Tensor) -> torch.Tensor:
        """正向映射: RAE → UNet (主路径 + 残差旁路)."""
        # 主路径 (from RAEmapperDeep)
        h = self.forward_in(z_rae)
        h = self.forward_res(h)
        h = self.forward_out(h)
        h = self.forward_shuffle(h)

        # 残差旁路
        h_res = self.residual_upsample(z_rae)
        h_res = self.residual_proj(h_res)

        return h + h_res * self.residual_scale

    def inverse(self, z_unet: torch.Tensor) -> torch.Tensor:
        """逆向映射: UNet → RAE (主路径 + 残差旁路)."""
        # 主路径 (from RAEmapperDeep)
        h = self.backward_unshuffle(z_unet)
        h = self.backward_in(h)
        h = self.backward_res(h)
        h = self.backward_out(h)

        # 残差旁路
        h_res = self.residual_downsample(z_unet)
        h_res = self.residual_inv_proj(h_res)

        return h + h_res * self.residual_scale


# ──────────────────────────────────────────────
# 纯 reshape 映射器 (零参数, 零损失, 严格可逆)
# ──────────────────────────────────────────────

class RAEmapperReshape(nn.Module):
    """纯 reshape 映射器: 零参数, 零损失, 严格可逆.

    使用 PixelShuffle / PixelUnshuffle 在通道与空间维度之间做无损重排,
    不引入任何可学习参数, 数学上 forward ∘ inverse 为恒等映射.

    Forward:  (B, rae_latent_dim, rae_spatial, rae_spatial)
              → PixelShuffle(r) →
              (B, unet_channels, sample_size, sample_size)

    Inverse:  (B, unet_channels, sample_size, sample_size)
              → PixelUnshuffle(r) →
              (B, rae_latent_dim, rae_spatial, rae_spatial)

    约束: rae_latent_dim == unet_channels * r², 其中 r = sample_size // rae_spatial_size.
    默认配置: 1024 == 64 * (4² = 16) ✓

    与卷积类映射器 (RAEmapper / RAEmapperDeep / RAEmapperResidual) 相比:
      - 优势: 无信息瓶颈, roundtrip 严格无损 (PSNR = ∞)
      - 代价: UNet 输入通道从 4/8 增加到 64, 但仅影响 UNet 第一层卷积,
              参数量增加可忽略 (< 0.1%), 信息量提升 16 倍.
    """

    def __init__(
        self,
        rae_latent_dim: int = 1024,
        rae_spatial_size: int = 16,
        unet_in_channels: int = 64,
        unet_out_channels: int = 64,
        sample_size: int = 64,
    ):
        super().__init__()

        self.rae_latent_dim = rae_latent_dim
        self.rae_spatial_size = rae_spatial_size
        self.unet_in_channels = unet_in_channels
        self.unet_out_channels = unet_out_channels
        self.sample_size = sample_size

        # 空间缩放因子
        self.r = sample_size // rae_spatial_size  # 64 // 16 = 4
        assert sample_size % rae_spatial_size == 0, (
            f"sample_size ({sample_size}) 必须能被 rae_spatial_size "
            f"({rae_spatial_size}) 整除"
        )
        r2 = self.r * self.r  # 16

        # 无损约束: 正向 rae_latent_dim 必须恰好等于 unet_in_channels * r²
        assert rae_latent_dim == unet_in_channels * r2, (
            f"无损约束失败: rae_latent_dim ({rae_latent_dim}) 必须等于 "
            f"unet_in_channels ({unet_in_channels}) * r² ({r2}) = "
            f"{unet_in_channels * r2}. 当前不匹配, 无法实现纯 reshape."
        )
        # 逆向同理: unet_out_channels * r² 必须等于 rae_latent_dim
        assert unet_out_channels * r2 == rae_latent_dim, (
            f"无损约束失败: unet_out_channels ({unet_out_channels}) * r² ({r2}) = "
            f"{unet_out_channels * r2} 必须等于 rae_latent_dim ({rae_latent_dim})."
        )

        # 纯 reshape 模块 (无可学习参数)
        self.forward_shuffle = nn.PixelShuffle(self.r)
        self.backward_unshuffle = nn.PixelUnshuffle(self.r)

    def forward(self, z_rae: torch.Tensor) -> torch.Tensor:
        """正向映射: RAE 潜变量 → UNet 输入空间 (纯 reshape).

        Args:
            z_rae: RAE 编码器输出, shape (B, rae_latent_dim, rae_spatial, rae_spatial)

        Returns:
            z_unet: UNet 输入空间的潜变量,
                    shape (B, unet_in_channels, sample_size, sample_size)
        """
        # (B, 1024, 16, 16) → (B, 64, 64, 64)
        return self.forward_shuffle(z_rae)

    def inverse(self, z_unet: torch.Tensor) -> torch.Tensor:
        """逆向映射: UNet 输出 → RAE 潜变量空间 (纯 reshape, 严格可逆).

        Args:
            z_unet: UNet 输出, shape (B, unet_out_channels, sample_size, sample_size)

        Returns:
            z_rae: RAE 空间的潜变量,
                   shape (B, rae_latent_dim, rae_spatial, rae_spatial)
        """
        # (B, 64, 64, 64) → (B, 1024, 16, 16)
        return self.backward_unshuffle(z_unet)

    def encode_pair(self, z0_rae: torch.Tensor, z1_rae: torch.Tensor):
        """将一对 RAE 潜变量同时映射到 UNet 空间 (用于 Flow Matching 训练).

        Args:
            z0_rae: 源图像 RAE 潜变量, shape (B, rae_latent_dim, rae_spatial, rae_spatial)
            z1_rae: 目标图像 RAE 潜变量, shape (B, rae_latent_dim, rae_spatial, rae_spatial)

        Returns:
            (z0_unet, z1_unet): 映射后的 UNet 空间潜变量对
        """
        return self.forward(z0_rae), self.forward(z1_rae)


# ──────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────

def create_rae_mapper(
    variant: str = 'residual',
    rae_latent_dim: int = 1024,
    rae_spatial_size: int = 16,
    unet_in_channels: int = 8,
    unet_out_channels: int = 4,
    sample_size: int = 64,
    mid_channels: int = 192,
    num_res_blocks: int = 5,
):
    """创建 RAE 映射器的工厂函数.

    Args:
        variant: 映射器变体, 'base' / 'residual' / 'deep' / 'reshape'
        rae_latent_dim: RAE 潜变量通道维度 (默认 1024)
        rae_spatial_size: RAE 潜变量空间尺寸 (默认 16)
        unet_in_channels: UNet 输入通道数 (默认 8; reshape 变体忽略, 自动推导)
        unet_out_channels: UNet 输出通道数 (默认 4; reshape 变体忽略, 自动推导)
        sample_size: UNet 空间尺寸 (默认 64)
        mid_channels: deep 变体的中间通道数 (默认 192)
        num_res_blocks: deep 变体的 ResBlock 数量 (默认 5)

    Returns:
        mapper 实例
    """
    if variant == 'reshape':
        # 纯 reshape 变体: 自动推导 UNet 通道数, 保证 rae_latent_dim == C * r²
        r = sample_size // rae_spatial_size
        r2 = r * r
        required_channels = rae_latent_dim // r2
        assert rae_latent_dim % r2 == 0, (
            f"reshape 变体要求 rae_latent_dim ({rae_latent_dim}) 能被 r² ({r2}) 整除"
        )
        return RAEmapperReshape(
            rae_latent_dim=rae_latent_dim,
            rae_spatial_size=rae_spatial_size,
            unet_in_channels=required_channels,
            unet_out_channels=required_channels,
            sample_size=sample_size,
        )
    if variant == 'base':
        return RAEmapper(
            rae_latent_dim=rae_latent_dim,
            rae_spatial_size=rae_spatial_size,
            unet_in_channels=unet_in_channels,
            unet_out_channels=unet_out_channels,
            sample_size=sample_size,
        )
    elif variant == 'residual':
        return RAEmapperResidual(
            rae_latent_dim=rae_latent_dim,
            rae_spatial_size=rae_spatial_size,
            unet_in_channels=unet_in_channels,
            unet_out_channels=unet_out_channels,
            sample_size=sample_size,
            mid_channels=mid_channels,
            num_res_blocks=num_res_blocks,
        )
    elif variant == 'deep':
        return RAEmapperDeep(
            rae_latent_dim=rae_latent_dim,
            rae_spatial_size=rae_spatial_size,
            unet_in_channels=unet_in_channels,
            unet_out_channels=unet_out_channels,
            sample_size=sample_size,
            mid_channels=mid_channels,
            num_res_blocks=num_res_blocks,
        )
    else:
        raise ValueError(
            f"Unknown mapper variant: {variant}, "
            f"must be 'base', 'residual', 'deep', or 'reshape'"
        )


# ──────────────────────────────────────────────
# 验证脚本
# ──────────────────────────────────────────────

if __name__ == '__main__':
    """验证映射器的维度正确性."""
    print("=" * 60)
    print("RAE ↔ UNet 潜变量映射器 维度验证")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 各变体的 UNet 输入/输出通道配置 (与 create_rae_mapper 默认值一致)
    variant_configs = {
        'base':     dict(unet_in=8, unet_out=4),
        'residual': dict(unet_in=8, unet_out=4),  # 工厂默认 unet_in=8
        'reshape':  dict(unet_in=64, unet_out=64),  # 自动推导: 1024 // 16 = 64
    }

    for variant, cfg in variant_configs.items():
        print(f"\n{'─' * 40}")
        print(f"变体: {variant}")
        print(f"{'─' * 40}")

        mapper = create_rae_mapper(variant=variant).to(device)

        # 正向映射: RAE → UNet
        z_rae = torch.randn(2, 1024, 16, 16, device=device)
        z_unet = mapper(z_rae)
        print(f"  正向: {list(z_rae.shape)} → {list(z_unet.shape)}")
        assert z_unet.shape == (2, cfg['unet_in'], 64, 64), (
            f"正向映射维度错误: {z_unet.shape}, 期望 {(2, cfg['unet_in'], 64, 64)}"
        )

        # 逆向映射: UNet → RAE
        z_unet_out = torch.randn(2, cfg['unet_out'], 64, 64, device=device)
        z_rae_rec = mapper.inverse(z_unet_out)
        print(f"  逆向: {list(z_unet_out.shape)} → {list(z_rae_rec.shape)}")
        assert z_rae_rec.shape == (2, 1024, 16, 16), (
            f"逆向映射维度错误: {z_rae_rec.shape}"
        )

        # 梯度流检查
        z_rae_grad = torch.randn(2, 1024, 16, 16, device=device, requires_grad=True)
        z_unet_grad = mapper(z_rae_grad)
        loss = z_unet_grad.sum()
        loss.backward()
        assert z_rae_grad.grad is not None, "正向映射梯度断裂!"
        print(f"  正向梯度: OK (grad norm = {z_rae_grad.grad.norm().item():.4f})")

        z_unet_grad2 = torch.randn(
            2, cfg['unet_out'], 64, 64, device=device, requires_grad=True
        )
        z_rae_grad2 = mapper.inverse(z_unet_grad2)
        loss2 = z_rae_grad2.sum()
        loss2.backward()
        assert z_unet_grad2.grad is not None, "逆向映射梯度断裂!"
        print(f"  逆向梯度: OK (grad norm = {z_unet_grad2.grad.norm().item():.4f})")

        # 参数量
        num_params = sum(p.numel() for p in mapper.parameters())
        print(f"  参数量: {num_params:,}")

        # ── reshape 变体专属: 验证严格无损可逆 ──
        if variant == 'reshape':
            z_test = torch.randn(2, 1024, 16, 16, device=device)
            z_roundtrip = mapper.inverse(mapper(z_test))
            is_lossless = torch.equal(z_test, z_roundtrip)
            max_diff = (z_test - z_roundtrip).abs().max().item()
            print(f"  Roundtrip 无损检查: {'✓ 严格无损' if is_lossless else '✗ 有损'}")
            print(f"    max |z - inverse(forward(z))| = {max_diff:.2e}")
            assert is_lossless, f"reshape 变体 roundtrip 不无损! max_diff = {max_diff}"
            assert num_params == 0, f"reshape 变体应有 0 参数, 实际 {num_params}"
            print(f"  ✓ 零参数确认")

    print(f"\n{'=' * 60}")
    print("所有维度验证通过! OK")
    print(f"{'=' * 60}")
