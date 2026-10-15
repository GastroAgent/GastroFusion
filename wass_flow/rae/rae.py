import os
import sys
sys.path.append("/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/rae")
from math import sqrt
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoConfig
from encoders.vision_encoder import create_encoder
from decoders import GeneralDecoder

def _load_decoder(config_path, hidden_size, patch_size, num_patches, pretrained_path=None):
    config = AutoConfig.from_pretrained(config_path)
    config.hidden_size = hidden_size
    config.patch_size = patch_size
    config.image_size = int(patch_size * sqrt(num_patches))
    decoder = GeneralDecoder(config, num_patches=num_patches)
    if pretrained_path is not None:
        print(f"Loading pretrained decoder from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        keys = decoder.load_state_dict(state_dict, strict=False)
        if keys.missing_keys:
            print(f"Missing keys: {keys.missing_keys}")
    return decoder

def _load_normalization_stats(path):
    if path is None:
        return None, None, False
    stats = torch.load(path, map_location='cpu', weights_only=False)
    print(f"Loaded normalization stats from {path}")
    return stats.get('mean', None), stats.get('var', None), True


class RAE(nn.Module):
    def __init__(self,
        encoder_name: str = 'dinov3mls',
        resolution: int = 256,
        decoder_config_path: str = 'vit_mae-base',
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        noise_tau: float = 0.1,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.encoder = create_encoder(encoder_name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), resolution=resolution)
        self.resolution = resolution
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        self.base_patches = (resolution // 16) ** 2
        self.eps = eps
        self.noise_tau = noise_tau

        self.decoder = _load_decoder(
            decoder_config_path, self.latent_dim, decoder_patch_size,
            self.base_patches, pretrained_decoder_path,
        )
        self.latent_mean, self.latent_var, self.do_normalization = _load_normalization_stats(normalization_stat_path)
        print(f"RAE: encoder={encoder_name}, resolution={resolution}, "
              f"patch_size={self.encoder_patch_size}, hidden_size={self.latent_dim}")

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        return x + noise_sigma * torch.randn_like(x)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.max() <= 1.0:
            x = x * 255.0
        _, _, h, w = x.shape
        if h != self.resolution or w != self.resolution:
            x = nn.functional.interpolate(x, size=(self.resolution, self.resolution), mode='bicubic', align_corners=False)

        z = self.encoder(x)

        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        # Reshape [B, N, C] -> [B, C, H, W]
        b, n, c = z.shape
        h = w = int(sqrt(n))
        z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean
        # Reshape [B, C, H, W] -> [B, N, C]
        b, c, h, w = z.shape
        z = z.view(b, c, h * w).transpose(1, 2)
        # forward 必须经由 self.decoder 本身: 若被 DDP 包装, 才会触发梯度同步.
        # 但 unpatchify 是普通方法, DDP.__getattr__ 透传不到, 需经 .module 取原始 module.
        output = self.decoder(z, drop_cls_token=False).logits
        decoder_module = getattr(self.decoder, 'module', self.decoder)
        return decoder_module.unpatchify(output)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x)
        x_rec = self.decode(z)
        if return_latent:
            return x_rec, z
        return x_rec


if __name__ == '__main__':
    """
    RAE 模型加载与推理测试示例。

    用法:
        cd /mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow
        python -m rae.rae

    encoder_name 格式: "encoder_type-architecture-model_config"
        - encoder_type:  dinov2 | dinov2mls | dinov3 | dinov3mls | siglip2 |
                         mae | webssl | eupe | eupemls | tipsv2 | clip | ...
        - architecture:  任意标识符（当前未使用，可填 'base' 等）
        - model_config:  模型规模标识，如 b16, l16, b 等
    示例:
        dinov3-base-b16      DINOv3 ViT-B/16
        dinov3mls-base-b16   DINOv3 ViT-B/16 多层平均
        dinov2-base-b        DINOv2 ViT-B/14
        mae-base-b           MAE ViT-Base
    """
    import sys
    import os
    os.environ["DINOV3_CKPT_DIR"] = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/flow_matcher_otcfm/rae_v2"
    # 确保项目根目录在 sys.path 中
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")

    # ============================
    # 1. 构建模型
    # ============================
    # 可根据实际环境修改 encoder_name / decoder_config_path / pretrained_decoder_path
    model = RAE(
        encoder_name='dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]',           # encoder_type-architecture-model_config
        resolution=256,                            # 输入图像分辨率
        decoder_config_path='/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/rae/configs/decoder/ViTXL/config.json',  # HuggingFace decoder config
        pretrained_decoder_path='/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/flow_matcher_otcfm/rae_v2/decoder.pt',              # 可选: 'flow_matcher_otcfm/rae_v2/decoder.pt'
        noise_tau=0.0,                             # 测试时关闭噪声
        normalization_stat_path='/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/flow_matcher_otcfm/rae_v2/stats.pt',              # 可选: latent 归一化统计文件路径
    ).to(device)
    model.eval()
    print(f"[INFO] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ============================
    # 2. 构造随机输入
    # ============================
    # RAE.encode 期望输入范围 [0, 255]（若 max <= 1.0 会自动 × 255）
    batch_size = 2
    x = torch.rand(batch_size, 3, 256, 256, device=device) * 255.0
    print(f"[INFO] 输入 shape: {x.shape}, 范围: [{x.min():.1f}, {x.max():.1f}]")

    # ============================
    # 3. 测试编码
    # ============================
    with torch.no_grad():
        z = model.encode(x)
    print(f"[INFO] 编码输出 (latent) shape: {z.shape}")
    # 期望: [B, latent_dim, H, W]，其中 H = W = resolution // patch_size = 256 // 16 = 16

    # ============================
    # 4. 测试解码
    # ============================
    with torch.no_grad():
        x_rec = model.decode(z)
    print(f"[INFO] 解码输出 (重建) shape: {x_rec.shape}")
    # 期望: [B, 3, 256, 256]
    # ============================
    # 计算重建误差
    # ============================
    recon_error = (x_rec - x).abs().mean().item()
    print(f"[INFO] 编码-解码 重建绝对误差 (MAE): {recon_error:.4f}")
    print("[INFO] RAE 加载与推理测试完成!")

    # ============================
    # 5. 测试完整前向传播
    # ============================
    with torch.no_grad():
        x_rec2, z2 = model(x, return_latent=True)
    print(f"[INFO] 前向传播 - 重建 shape: {x_rec2.shape}, 潜变量 shape: {z2.shape}")

    # ============================
    # 6. 计算重建误差
    # ============================
    recon_error = (x_rec - x).abs().mean().item()
    print(f"[INFO] 重建绝对误差 (MAE): {recon_error:.4f}")
    print("[INFO] RAE 加载与推理测试完成!")
