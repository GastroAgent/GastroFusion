from __future__ import annotations

import os
import sys

import torch


DEFAULT_ENCODER_NAME = "dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]"


def build_rae(
    project_root: str,
    device: torch.device,
    encoder_name: str = DEFAULT_ENCODER_NAME,
    resolution: int = 256,
    decoder_config: str | None = None,
    decoder_weight: str | None = None,
    normalization_stats: str | None = None,
):
    project_root = os.path.abspath(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from rae.rae import RAE

    decoder_config = decoder_config or os.path.join(
        project_root, "rae/configs/decoder/ViTXL/config.json"
    )
    decoder_weight = decoder_weight or os.path.join(
        project_root, "flow_matcher_otcfm/rae_v2/decoder.pt"
    )
    normalization_stats = normalization_stats or os.path.join(
        project_root, "flow_matcher_otcfm/rae_v2/stats.pt"
    )
    model = RAE(
        encoder_name=encoder_name,
        resolution=resolution,
        decoder_config_path=decoder_config,
        pretrained_decoder_path=decoder_weight,
        noise_tau=0.0,
        normalization_stat_path=normalization_stats,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def encode_rae(rae_model, images: torch.Tensor) -> torch.Tensor:
    """Encode [0,1] images and enforce the native (B,1024,16,16) interface."""
    if images.min().item() < -1e-4 or images.max().item() > 1.0001:
        raise ValueError(
            f"RAE images must be [0,1], range=({images.min().item():.3f}, {images.max().item():.3f})"
        )
    latent = rae_model.encode(images)
    if latent.ndim != 4:
        raise RuntimeError(f"RAE returned unexpected shape {tuple(latent.shape)}")
    return latent


@torch.no_grad()
def encode_rae_chunked(rae_model, images: torch.Tensor, chunk_size: int = 8) -> torch.Tensor:
    """Memory-safe frozen RAE encoding while keeping concatenated latents on device."""
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or images.shape[0] <= chunk_size:
        return encode_rae(rae_model, images)
    outputs = []
    for start in range(0, images.shape[0], chunk_size):
        outputs.append(encode_rae(rae_model, images[start:start + chunk_size]))
    return torch.cat(outputs, dim=0)
