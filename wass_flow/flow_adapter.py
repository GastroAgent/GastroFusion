from __future__ import annotations

import math
import os
import sys
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenVisionHintDiTFlow(nn.Module):
    """Frozen V1 DiT inference adapter.

    V1 conditions:
      - timestep
      - target/support image CLS -> AdaLN
      - target/support patch tokens -> cross attention
      - weak noisy grayscale target hint -> spatial addition
      - no caption and no class ID
    """

    ENDOVIT_MEAN = (0.3464, 0.2280, 0.2228)
    ENDOVIT_STD = (0.2520, 0.2128, 0.2093)

    def __init__(
        self,
        project_root: str,
        checkpoint: str,
        device: torch.device,
        dit_variant: str = "L",
        endovit_path: str | None = None,
        use_ema: bool = True,
        num_steps: int = 8,
        solver: str = "heun",
        hint_signal_ratio: float | None = None,
        hint_dropout: float = 0.20,
        amp: bool = True,
        allow_condition_mismatch: bool = False,
    ) -> None:
        super().__init__()
        project_root = os.path.abspath(project_root)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from flow_models.dit_rae import create_dit_rae

        self.device_obj = device
        self.num_steps = int(num_steps)
        self.solver = solver
        self.hint_dropout = float(hint_dropout)
        self.amp = bool(amp and device.type == "cuda")
        if not (0.0 <= self.hint_dropout <= 1.0):
            raise ValueError("hint_dropout must be in [0,1]")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be euler or heun")

        checkpoint_obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
        condition_config = checkpoint_obj.get("condition_config")
        mode = condition_config.get("condition_mode") if condition_config else None
        if mode != "vision_hint_v1" and not allow_condition_mismatch:
            raise RuntimeError(
                "Flow checkpoint is not marked as vision_hint_v1. "
                f"condition_config={condition_config}. Retrain with train_dit_flow_v1.py "
                "or explicitly allow a condition mismatch for an ablation."
            )
        if mode != "vision_hint_v1":
            print(f"[WARN] using condition-mismatched Flow checkpoint: {condition_config}")
        rae_range = condition_config.get("rae_input_range") if condition_config else None
        if rae_range not in (None, "[0,1]"):
            print(f"[WARN] checkpoint rae_input_range={rae_range!r}; V1 metric pipeline uses [0,1]")
        saved_alpha = condition_config.get("hint_signal_ratio") if condition_config else None
        if hint_signal_ratio is None:
            hint_signal_ratio = 0.10 if saved_alpha is None else float(saved_alpha)
        elif saved_alpha is not None and abs(float(hint_signal_ratio) - float(saved_alpha)) > 1e-8:
            print(
                f"[WARN] overriding Flow-training hint alpha {saved_alpha} "
                f"with downstream alpha {hint_signal_ratio}"
            )
        self.hint_signal_ratio = float(hint_signal_ratio)
        if not (0.0 <= self.hint_signal_ratio <= 1.0):
            raise ValueError("hint_signal_ratio must be in [0,1]")

        self.dit = create_dit_rae(
            variant=dit_variant,
            rae_latent_dim=1024,
            rae_spatial_size=16,
            cross_attn_dim=768,
            num_classes=int(checkpoint_obj.get("num_classes", 1000)),
            use_cross_attention=True,
        )
        if use_ema and checkpoint_obj.get("ema_model") is not None:
            dit_state = checkpoint_obj["ema_model"]
        elif checkpoint_obj.get("dit_model") is not None:
            dit_state = checkpoint_obj["dit_model"]
        else:
            dit_state = checkpoint_obj
        self.dit.load_state_dict(dit_state, strict=False)
        self.dit.to(device).eval()
        self.dit.device = device

        endovit_path = endovit_path or os.path.join(
            project_root, "flow_matcher_otcfm/EndoViT/pytorch_model.bin"
        )
        from timm.models.vision_transformer import VisionTransformer

        self.vision = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=False,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.vision.load_state_dict(torch.load(endovit_path, map_location="cpu", weights_only=False), strict=False)
        vision_state = checkpoint_obj.get("vision_model")
        if vision_state is not None:
            self.vision.load_state_dict(vision_state, strict=False)
        self.vision.to(device).eval()

        for module in (self.dit, self.vision):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        mean = torch.tensor(self.ENDOVIT_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(self.ENDOVIT_STD).view(1, 3, 1, 1)
        self.register_buffer("vision_mean", mean, persistent=False)
        self.register_buffer("vision_std", std, persistent=False)

    def train(self, mode: bool = True):
        # This adapter stays frozen/eval even when the outer classifier trains.
        super().train(False)
        self.dit.eval()
        self.vision.eval()
        return self

    def _vision_features(self, target_images: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(target_images, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.vision_mean) / self.vision_std
        return self.vision.forward_features(x)

    def _build_hint(self, target_images: torch.Tensor, stochastic: bool) -> torch.Tensor | None:
        # RGB [0,1] -> grayscale [-1,1], repeated to the 3-channel HintEncoder input.
        gray = (
            0.2989 * target_images[:, 0:1]
            + 0.5870 * target_images[:, 1:2]
            + 0.1140 * target_images[:, 2:3]
        )
        signal = (gray * 2.0 - 1.0).repeat(1, 3, 1, 1)
        alpha = self.hint_signal_ratio
        if alpha <= 0:
            hint = torch.randn_like(signal) if stochastic else torch.zeros_like(signal)
        else:
            noise = torch.randn_like(signal) if stochastic else torch.zeros_like(signal)
            hint = alpha * signal + math.sqrt(max(1.0 - alpha * alpha, 0.0)) * noise
        if stochastic and self.hint_dropout > 0:
            keep = (torch.rand(signal.shape[0], 1, 1, 1, device=signal.device) >= self.hint_dropout).to(signal.dtype)
            hint = hint * keep
        return hint

    @torch.no_grad()
    def generate(
        self,
        query_latent: torch.Tensor,
        target_images: torch.Tensor,
        stochastic_hint: bool = True,
    ) -> torch.Tensor:
        """Generate predicted states only: (B, T+1, 1024,16,16).

        The GT support endpoint is deliberately not appended.
        """
        query_latent = query_latent.to(self.device_obj)
        target_images = target_images.to(self.device_obj)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            vision = self._vision_features(target_images)
            image_cls = vision[:, 0]
            image_tokens = vision[:, 1:]
            hint = self._build_hint(target_images, stochastic=stochastic_hint)

            x = query_latent
            trajectory = [x.float()]
            t_span = torch.linspace(0.0, 1.0, self.num_steps + 1, device=x.device)
            for step in range(self.num_steps):
                t = t_span[step].expand(x.shape[0])
                t_next = t_span[step + 1].expand(x.shape[0])
                dt = t_span[step + 1] - t_span[step]
                dx1 = self.dit(
                    xt=x,
                    timestep=t,
                    class_labels=None,
                    encoder_hidden_states=image_tokens,
                    hint=hint,
                    text_embeds=None,
                    image_embeds=image_cls,
                )["sample"]
                if self.solver == "euler":
                    x = x + dx1 * dt
                else:
                    x_euler = x + dx1 * dt
                    dx2 = self.dit(
                        xt=x_euler,
                        timestep=t_next,
                        class_labels=None,
                        encoder_hidden_states=image_tokens,
                        hint=hint,
                        text_embeds=None,
                        image_embeds=image_cls,
                    )["sample"]
                    x = x + 0.5 * (dx1 + dx2) * dt
                trajectory.append(x.float())
        return torch.stack(trajectory, dim=1)

    @torch.no_grad()
    def generate_chunked(
        self,
        query_latent: torch.Tensor,
        target_images: torch.Tensor,
        chunk_size: int,
        stochastic_hint: bool = True,
    ) -> torch.Tensor:
        outputs = []
        for start in range(0, query_latent.shape[0], chunk_size):
            outputs.append(self.generate(
                query_latent[start:start + chunk_size],
                target_images[start:start + chunk_size],
                stochastic_hint=stochastic_hint,
            ).cpu())
        return torch.cat(outputs, dim=0)
