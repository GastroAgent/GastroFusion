"""
gen_trajectory_dit.py
=====================
DiT V1 轨迹生成脚本（目标视觉 + 弱噪声 hint；caption/class ID 关闭）。

基于 gen_trajectory_rae.py (UNet 版) 改造，核心差异:
  - 模型: DiTForRAEFlow (flow_models/dit_rae.py) 替代 UNet2DConditionModel
  - 潜变量空间: 直接在 RAE 原生空间 (B, 1024, 16, 16) 运行
  - 无需 RAE Mapper: 编码→RAE.encode, 解码→RAE.decode (省去 mapper.forward/inverse)
  - 前向签名: dit_model(xt, timestep, class_labels, encoder_hidden_states,
                        hint, text_embeds, image_embeds)['sample']

用法:
    python gen_trajectory_dit.py \
        --data_path <json> \
        --checkpoint <pt> \
        --output_dir <dir> \
        --dit_variant L \
        --num_steps 8 --solver heun
"""

# import debugpy

# try:
#     print("Waiting for debugger attach")
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9505))
#     # debugpy.listen(("172.16.0.108", 9504))
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import sys
import json
import math
import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image, make_grid
from timm.models.vision_transformer import VisionTransformer
from safetensors.torch import load_file

# ---- 本地模块 ----
_PROJECT_ROOT = '/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow'
sys.path.append(_PROJECT_ROOT)

from rae.rae import RAE
from flow_models.dit_rae import create_dit_rae
from transformers import (
    AutoTokenizer, ChineseCLIPTextModel, ChineseCLIPTextConfig,
)
from data_loader import MedicalJsonDataset


# ============================================================
# 常量 / 默认参数
# ============================================================
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

DEFAULT_DATASET_MEAN = [0.3464, 0.2280, 0.2228]
DEFAULT_DATASET_STD = [0.2520, 0.2128, 0.2093]

# ── RAE 配置 ──
_RAE_ENCODER_NAME = 'dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]'
_RAE_RESOLUTION = 256
_RAE_DECODER_CONFIG = os.path.join(
    _PROJECT_ROOT, 'rae/configs/decoder/ViTXL/config.json'
)
_RAE_DECODER_WEIGHT = os.path.join(
    _PROJECT_ROOT, 'flow_matcher_otcfm/rae_v2/decoder.pt'
)
_RAE_NORMALIZATION_STATS = os.path.join(
    _PROJECT_ROOT, 'flow_matcher_otcfm/rae_v2/stats.pt'
)

# ── 视觉/文本编码器路径 ──
_ENDOVIT_PATH = os.path.join(
    _PROJECT_ROOT, 'flow_matcher_otcfm/EndoViT/pytorch_model.bin'
)
_TEXT_ENCODER_PATH = os.path.join(
    _PROJECT_ROOT, 'flow_matcher_otcfm/text_encoder'
)


# ============================================================
# 通用工具函数
# ============================================================

def process_single_image(
    image_path: str,
    input_size: int = 224,
    dataset_mean: list = DEFAULT_DATASET_MEAN,
    dataset_std: list = DEFAULT_DATASET_STD,
) -> torch.Tensor:
    """读取图片并应用标准化变换。"""
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=dataset_mean, std=dataset_std),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image)


def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def save_json(obj: Any, path: str):
    """原子写入 JSON 文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


# ============================================================
# TrajectoryGenerator (DiT 版)
# ============================================================

class TrajectoryGenerator:
    """DiT 版 Flow-Matching 轨迹生成器（固定步数，无距离计算）。

    与 UNet 版的关键差异:
      - 直接在 RAE 空间 (B, 1024, 16, 16) 运行
      - 无需 RAE Mapper, 编码/解码路径更短
    """

    def __init__(self, args):
        self.args = args
        self.use_gt_vt = bool(args.use_gt_vt)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dit_model, rae_model, (text_model, text_tokenizer), (vision_model, proc_fn) = \
            self._load_model(args.checkpoint, args.use_ema)

        self.dit_model = dit_model
        self.rae_model = rae_model
        if self.args.hint_signal_ratio is None:
            self.args.hint_signal_ratio = float(
                self.condition_config.get('hint_signal_ratio', 0.10)
            )
        self.text_model = text_model
        self.text_tokenizer = text_tokenizer
        self.vision_model = vision_model
        self.process_single_image = proc_fn

        self._setup_directories()
        os.makedirs(self.args.temp, exist_ok=True)

    # ----------------------------------------------------------
    # 目录设置
    # ----------------------------------------------------------
    def _setup_directories(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        if self.args.save_latents:
            self.latents_dir = os.path.join(self.args.output_dir, 'latents')
            os.makedirs(self.latents_dir, exist_ok=True)
        if self.args.save_images:
            self.images_dir = os.path.join(self.args.output_dir, 'images')
            os.makedirs(self.images_dir, exist_ok=True)

    # ----------------------------------------------------------
    # 模型加载 (DiT 版: 移除 UNet / RAE Mapper)
    # ----------------------------------------------------------
    def _load_model(self, checkpoint: str, use_ema: bool = False):
        """加载所有模型（DiT, RAE, Text, Vision）。

        与 UNet 版差异:
          - DiT 替代 UNet (create_dit_rae)
          - 无 RAE Mapper
        """
        print(f"[DiT Weights] from {checkpoint}")

        # ---- DiT 模型 ----
        dit_model = create_dit_rae(
            variant=self.args.dit_variant,
            rae_latent_dim=1024,
            rae_spatial_size=16,
            cross_attn_dim=768,
            num_classes=self.args.num_classes,
            use_cross_attention=True,
        )
        state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
        condition_config = state_dict.get('condition_config')
        self.condition_config = condition_config or {}
        saved_mode = self.condition_config.get('condition_mode')
        saved_size = self.condition_config.get('image_size')
        requested_size = int(self.args.image_size[0])
        mismatch = (
            saved_mode != 'vision_hint_v1'
            or (saved_size is not None and int(saved_size) != requested_size)
        )
        if mismatch and not self.args.allow_condition_mismatch:
            raise RuntimeError(
                "checkpoint 条件配置与 V1 生成配置不一致。"
                f" saved={self.condition_config}, image_size={self.args.image_size}. "
                "确需消融时使用 --allow_condition_mismatch。"
            )
        if mismatch:
            print(f"[WARN] condition mismatch: saved={self.condition_config}")
        # 优先使用 EMA 权重 (推理通常用 EMA)
        if use_ema and 'ema_model' in state_dict:
            dit_model.load_state_dict(state_dict['ema_model'], strict=False)
            print(f"[INFO] 已加载 EMA 权重")
        elif 'dit_model' in state_dict:
            dit_model.load_state_dict(state_dict['dit_model'], strict=False)
            print(f"[INFO] 已加载 dit_model 权重")
        else:
            # 兼容: 直接尝试整体加载
            dit_model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] 已直接加载 checkpoint 权重 (strict=False)")

        # ---- RAE ----
        rae_model = RAE(
            encoder_name=_RAE_ENCODER_NAME,
            resolution=_RAE_RESOLUTION,
            decoder_config_path=_RAE_DECODER_CONFIG,
            pretrained_decoder_path=_RAE_DECODER_WEIGHT,
            noise_tau=0.0,
            normalization_stat_path=_RAE_NORMALIZATION_STATS,
        ).to(device='cuda').eval()
        print(f"[INFO] RAE 模型已加载, latent_dim={rae_model.latent_dim}")

        # ---- Text Encoder ----
        # V1 does not use caption conditioning. Keep None to save memory.
        text_model, text_tokenizer = None, None

        # ---- Vision Encoder (EndoViT) ----
        vision_model = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=False,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ).eval()
        vision_model.load_state_dict(
            torch.load(_ENDOVIT_PATH, weights_only=False), strict=False
        )
        if state_dict.get('vision_model') is not None:
            vision_model.load_state_dict(state_dict['vision_model'], strict=False)
            print(f"[INFO] Vision 权重已从检查点恢复")

        # ---- 全部放到 CUDA ----
        dit_model = dit_model.to(device='cuda')
        rae_model = rae_model.to('cuda')
        if text_model is not None:
            text_model = text_model.to('cuda')
        if vision_model is not None:
            vision_model = vision_model.to('cuda')
            vision_model.device = 'cuda'

        # 设置 DiT 的 device 属性 (前向时多处使用 dit_model.device)
        dit_model.device = torch.device('cuda')

        dit_info = dit_model.get_model_info()
        print(f"[INFO] DiT 模型: hidden_dim={dit_info['hidden_dim']}, "
              f"depth={dit_info['depth']}, heads={dit_info['num_heads']}, "
              f"params={dit_info['total_parameters']:,}")

        return dit_model, rae_model, (text_model, text_tokenizer), (vision_model, process_single_image)

    @torch.no_grad()
    def _decode_latent_to_image(self, z_rae: torch.Tensor) -> torch.Tensor:
        """将 RAE 潜变量解码为图像。

        DiT 版: 直接 rae.decode, 无需 mapper.inverse()
        (B, 1024, 16, 16) → RAE decode → (B, 3, H, W)
        """
        img = self.rae_model.decode(z_rae)
        return img

    @torch.no_grad()
    def _encode_image_to_latent(self, images: torch.Tensor) -> torch.Tensor:
        """将图像编码为 RAE 潜变量。

        DiT 版: 直接 rae.encode, 无需 mapper.forward()
        RAE.encode 期望输入范围为 [0, 1]（内部会自动 ×255）。
        """
        # [-1, 1] → [0, 1]（存在负值说明经过 Normalize）
        if images.min() < 0:
            images = (images + 1.0) / 2.0
        # (B, 3, H, W) → RAE encode → (B, 1024, 16, 16)
        z_rae = self.rae_model.encode(images)
        return z_rae

    # ----------------------------------------------------------
    # 归一化 / 图像保存
    # ----------------------------------------------------------
    @staticmethod
    def normalize_samples(x: torch.Tensor) -> torch.Tensor:
        """将 RAE.decode 输出（约 [0, 255]）转换为 [0, 1] 用于图像保存。"""
        return (x / 255.0).clamp(0.0, 1.0)

    def save_trajectory_images(
        self, trajectory: List[torch.Tensor], pair_name: str, append_x1: bool = True
    ):
        """将整条轨迹解码并保存为逐帧 png + 一张 grid 总览。"""
        if not self.args.save_images:
            return

        pair_img_dir = os.path.join(self.images_dir, pair_name)
        os.makedirs(pair_img_dir, exist_ok=True)

        frames = []
        for step_idx, z in enumerate(trajectory[:-1] if append_x1 else trajectory):
            img = self._decode_latent_to_image(z)
            frames.append(img)
            save_image(img, os.path.join(pair_img_dir, f'step_{step_idx:02d}.png'))

        # 最后一步（生成结果 / GT 终点）单独标注
        last_idx = len(trajectory) - 1
        last_label = 'x1_gt' if append_x1 else f'step_{last_idx:02d}'
        z_last = trajectory[-1]
        img_last = self._decode_latent_to_image(z_last)
        save_image(img_last, os.path.join(pair_img_dir, f'{last_label}.png'))
        frames.append(img_last)

        # grid 总览
        if self.args.save_grid:
            grid = make_grid(torch.cat(frames, dim=0), nrow=min(8, len(frames)))
            save_image(grid, os.path.join(pair_img_dir, f'grid_{pair_name}.png'))

    # ----------------------------------------------------------
    # ODE Solvers（DiT 版前向签名）
    # ----------------------------------------------------------
    def euler_solver_fixed(self, x0, x1, t_span, batch) -> List[torch.Tensor]:
        """Euler 求解器（固定步数），返回轨迹 [x0, x_{1}, ..., x_{N}]。

        DiT 前向: dit_model(xt, timestep, class_labels, encoder_hidden_states,
                            hint, text_embeds, image_embeds)['sample']
        """
        x = x0.to(self.dit_model.device)
        caption_hidden_states = batch['caption_hidden_states']
        hint = batch['hint']
        image_embeds = batch['image_embeds']
        use_gt_vt = self.use_gt_vt

        trajectory = [x0.detach().clone()]
        dt = (t_span[1] - t_span[0]).item()

        for t_idx in range(len(t_span) - 1):
            t = t_span[t_idx] * torch.ones(x0.shape[0], device=self.device)
            if use_gt_vt:
                dx = batch['ut']
            else:
                dx = self.dit_model(
                    xt=x,
                    timestep=t.reshape(-1),
                    class_labels=None,
                    encoder_hidden_states=caption_hidden_states,
                    hint=hint,
                    text_embeds=None,
                    image_embeds=image_embeds,
                )['sample']
            x = x + dx * dt
            trajectory.append(x.detach().clone())

        return trajectory

    def heun_solver_fixed(self, x0, x1, t_span, batch) -> List[torch.Tensor]:
        """Heun 求解器（固定步数），返回轨迹 [x0, x_{1}, ..., x_{N}]。"""
        x = x0.to(self.dit_model.device)
        caption_hidden_states = batch['caption_hidden_states']
        hint = batch['hint']
        image_embeds = batch['image_embeds']

        trajectory = [x0.detach().clone()]
        dt = (t_span[1] - t_span[0]).item()

        for t_idx in range(len(t_span) - 1):
            t = t_span[t_idx] * torch.ones(x0.shape[0], device=self.device)
            t_next = t_span[t_idx + 1] * torch.ones(x0.shape[0], device=self.device)

            dx1 = self.dit_model(
                xt=x,
                timestep=t,
                class_labels=None,
                encoder_hidden_states=caption_hidden_states,
                hint=hint,
                text_embeds=None,
                image_embeds=image_embeds,
            )['sample']
            x_euler = x + dx1 * dt
            dx2 = self.dit_model(
                xt=x_euler,
                timestep=t_next,
                class_labels=None,
                encoder_hidden_states=caption_hidden_states,
                hint=hint,
                text_embeds=None,
                image_embeds=image_embeds,
            )['sample']
            x = x + (dx1 + dx2) * dt / 2
            trajectory.append(x.detach().clone())

        return trajectory

    # ----------------------------------------------------------
    # 单 pair 轨迹生成 + 保存
    # ----------------------------------------------------------
    @torch.no_grad()
    def generate_one_pair(self, idx: int, batch: dict, info: dict) -> dict:
        """生成单个 pair 的轨迹并保存。返回元信息字典。"""
        x0 = batch['x0']
        x1 = batch['x1']
        t_span = torch.linspace(0, 1, self.args.num_steps + 1, device=self.device)

        # 求解
        if self.args.solver == 'euler':
            trajectory = self.euler_solver_fixed(x0, x1, t_span, batch)
        else:
            trajectory = self.heun_solver_fixed(x0, x1, t_span, batch)

        # 追加 GT 终点 x1 作为参考（与原脚本一致）
        trajectory.append(x1.detach().clone())

        pair_name = f'pair_{idx:06d}'

        # 保存 latent
        if self.args.save_latents:
            latent_path = os.path.join(self.latents_dir, f'{pair_name}.pt')
            torch.save({
                'pair_idx': idx,
                'x0_latent': x0.cpu(),
                'x1_latent': x1.cpu(),
                'trajectory': [t.cpu() for t in trajectory],
                't_span': t_span.cpu(),
                'meta': {
                    **info,
                    'solver': self.args.solver,
                    'num_steps': self.args.num_steps,
                    'use_gt_vt': self.use_gt_vt,
                    'dit_variant': self.args.dit_variant,
                    'append_x1_gt': True,
                    'predicted_traj_len': self.args.num_steps + 1,
                    'classification_use_indices': list(range(self.args.num_steps + 1)),
                    'traj_len': len(trajectory),
                },
            }, latent_path)

        # 保存解码图像
        self.save_trajectory_images(trajectory, pair_name, append_x1=True)

        meta = {
            'pair_name': pair_name,
            'pair_idx': idx,
            'traj_len': len(trajectory),
            'predicted_traj_len': self.args.num_steps + 1,
            'gt_endpoint_index': len(trajectory) - 1,
            **info,
        }
        return meta

    # ----------------------------------------------------------
    # 主生成循环
    # ----------------------------------------------------------
    def generate(self):
        """主生成循环：遍历数据集，生成轨迹并保存。"""
        transform = T.Compose([
            T.Resize(tuple(self.args.image_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        transform_grey = T.Compose([
            T.Resize(tuple(self.args.image_size)),
            T.ToTensor(),
            T.RandomGrayscale(p=1),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = MedicalJsonDataset(
            path=self.args.data_path,
            transform=transform,
            transform_A=transform,
            transform_B=transform,
            hint_transform=transform_grey,
        )

        all_meta = []

        from conditional_flow_matcher import OptimalTransportConditionalFlowMatcher
        FM = OptimalTransportConditionalFlowMatcher(sigma=0.0, ot_method='exact')

        n_total = len(dataset)
        start_idx = max(self.args.start_idx, 0)
        end_idx = min(self.args.end_idx, n_total) if self.args.end_idx > 0 else n_total
        print(f'[INFO] 数据集共 {n_total} pair, 生成范围 [{start_idx}, {end_idx})')

        pbar = tqdm(range(start_idx, end_idx), desc='Generating trajectories (DiT)')
        for idx in pbar:
            batch = deepcopy(dataset[idx])

            if idx % 100 == 0:
                print(f"{batch.get('label_A', '?')} --> {batch.get('label_B', '?')}")
                print(f'---------------------------------{idx}/{n_total}----------------------------------')

            x0 = batch['x0'].to("cuda").unsqueeze(0)
            x1 = batch['x1'].to("cuda").unsqueeze(0)
            caption = [batch['caption']]  # only saved as metadata; not fed to DiT
            y = batch['class_id'].to("cuda").unsqueeze(0)  # metadata / optional GT-FM only
            x1_path = [batch['x1_path']]
            hint_base = batch['hint']
            alpha = float(self.args.hint_signal_ratio)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"hint_signal_ratio must be in [0,1], got {alpha}")
            if self.args.pure_noise_hint:
                hint = torch.randn_like(hint_base)
            else:
                noise = (
                    torch.randn_like(hint_base)
                    if self.args.stochastic_hint
                    else torch.zeros_like(hint_base)
                )
                hint = alpha * hint_base + math.sqrt(max(1.0 - alpha * alpha, 0.0)) * noise
            hint = hint.to("cuda").unsqueeze(0)

            # 保存原始图像（编码前）作为参考对照
            # transform 含 Normalize → [-1,1]，需转回 [0,1] 再保存
            if self.args.save_images:
                pair_img_dir = os.path.join(self.images_dir, f'pair_{idx:06d}')
                os.makedirs(pair_img_dir, exist_ok=True)
                save_image(((x0[0] + 1.0) / 2.0).clamp(0, 1).cpu(),
                           os.path.join(pair_img_dir, 'x0_orig.png'))
                save_image(((x1[0] + 1.0) / 2.0).clamp(0, 1).cpu(),
                           os.path.join(pair_img_dir, 'x1_orig.png'))

            # RAE 编码 (直接在 RAE 空间, 无需 Mapper)
            images = torch.cat([x0, x1], dim=0)
            with torch.no_grad():
                z_rae = self._encode_image_to_latent(images)
                x0, x1 = z_rae.chunk(2, dim=0)  # (1, 1024, 16, 16) each

            # Target visual conditions: CLS -> AdaLN; patch tokens -> cross-attention.
            if self.vision_model is not None:
                with torch.no_grad():
                    x1_images = torch.stack([process_single_image(p) for p in x1_path])
                    vision_embeds = self.vision_model.forward_features(
                        x1_images.to(self.vision_model.device))
            else:
                raise RuntimeError("vision_model is required for V1 generation")

            caption_hidden_states = vision_embeds[:, 1:, ...].to(self.dit_model.device)

            conds = {
                'x0': x0,
                'x1': x1,
                'x0_path': [batch['x0_path']],
                'x1_path': [batch['x1_path']],
                'caption': caption,
                'caption_hidden_states': caption_hidden_states.to(self.device),
                'y': y.to(self.device),
                'hint': hint.to(self.device),
                'text_embeds': torch.zeros_like(vision_embeds[:, 0]).to(self.device),
                'image_embeds': vision_embeds[:, 0].to(self.device),
            }

            if self.use_gt_vt:
                _, _, ut = FM.get_sample_location_and_conditional_flow(
                    x0, x1, sample_plan=False, cond=conds, print_info=False
                )
                conds['ut'] = ut.to(self.device)

            # pair 元信息（用于 JSON 汇总）
            info = {
                'x0_path': batch['x0_path'],
                'x1_path': batch['x1_path'],
                'caption': batch['caption'],
                'label_A': batch.get('label_A', ''),
                'label_B': batch.get('label_B', ''),
                'label_A_id': batch.get('label_A_id', -1),
                'label_B_id': batch.get('label_B_id', -1),
                'class_id': int(batch['class_id'].item()),
            }

            meta = self.generate_one_pair(idx, conds, info)
            all_meta.append(meta)

            if idx % self.args.save_every == 0:
                save_json(all_meta, os.path.join(self.args.output_dir, 'trajectory_info.json'))

            pbar.set_postfix({'pairs': len(all_meta), 'traj_len': meta['traj_len']})

        # 最终保存
        save_json(all_meta, os.path.join(self.args.output_dir, 'trajectory_info.json'))
        print(f'[DONE] 共生成 {len(all_meta)} 条轨迹')
        print(f'[INFO] latents -> {self.latents_dir if self.args.save_latents else "(disabled)"}')
        print(f'[INFO] images  -> {self.images_dir if self.args.save_images else "(disabled)"}')
        print(f'[INFO] meta    -> {os.path.join(self.args.output_dir, "trajectory_info.json")}')
        return all_meta

    def _get_text_embeds(self, caption, text_embeds_map):
        """获取 text embedding（带缓存）。"""
        if self.text_model is not None and caption[0] not in text_embeds_map:
            with torch.no_grad():
                caption_input = self.text_tokenizer(
                    caption, return_tensors="pt", padding=True
                ).to(self.text_model.device)
                caption_outputs = self.text_model(**caption_input)
                text_embeds = caption_outputs['last_hidden_state'].to(self.text_model.device)
            if len(text_embeds_map) < 200:
                text_embeds_map[caption[0]] = text_embeds
            return text_embeds
        elif caption[0] in text_embeds_map:
            return text_embeds_map[caption[0]]
        else:
            return torch.zeros((1, 2, 1))


# ============================================================
# CLI 参数解析
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='DiT Flow-Matching trajectory generation (fixed steps, no wass)'
    )

    # 数据与输出
    parser.add_argument('--data_path', type=str,
                        default="/mnt/inaisfs/data/home/tansy_criait/wass_flow_match_胃/data_tsy/胃_doctor/final_doctor_exam_modify_path_flat.json",
                        help='数据路径')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(
                            _PROJECT_ROOT,
                            'weights_dit_flow_v1/vision_hint_inclass/otcfm/'
                            'otcfm_dit_L_step_500000.pt'
                        ),
                        help='DiT 模型 checkpoint 路径')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(
                            _PROJECT_ROOT, 'fsl_multi_vae/rae_flow_metric_v1/outputs_trajectory/gen_trajectory_dit'
                        ),
                        help='输出目录')
    parser.add_argument('--temp', type=str,
                        default='/mnt/inaisfs/data/home/tansy_criait/'
                                'wass_flow_match_tsy/temp/temp',
                        help='临时文件目录')

    # DiT 配置
    parser.add_argument('--dit_variant', type=str, default='L',
                        choices=['S', 'B', 'L', 'L12'],
                        help='DiT 变体: S(384d), B(768d), L(1024d), L12(1024d shallow)')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='class embedding 数量 (需与训练时一致)')

    # ODE Solver（固定步数）
    parser.add_argument('--solver', type=str, default='heun',
                        choices=['euler', 'heun'],
                        help='ODE solver 类型')
    parser.add_argument('--num_steps', type=int, default=8,
                        help='固定采样步数（轨迹点数 = num_steps + 1 + GT终点）')
    parser.add_argument('--use_gt_vt', action=argparse.BooleanOptionalAction, default=False,
                        help='强制使用 GT velocity target 引导生成')
    parser.add_argument('--hint_signal_ratio', type=float, default=None,
                        help='灰度 hint 信号系数 alpha')
    parser.add_argument('--stochastic_hint', action=argparse.BooleanOptionalAction, default=True,
                        help='是否在弱灰度 hint 中采样随机噪声')
    parser.add_argument('--pure_noise_hint', action='store_true',
                        help='纯随机 hint 消融')
    parser.add_argument('--allow_condition_mismatch', action='store_true',
                        help='允许使用条件配置不同/缺失的旧 checkpoint，仅用于消融')

    # 图像参数
    parser.add_argument('--image_size', type=int, nargs=2,
                        default=[256, 256], help='图像尺寸')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='使用 EMA 模型权重')

    # 保存选项
    parser.add_argument('--save_latents', action='store_true', default=False,
                        help='保存轨迹 latent tensors (.pt)')
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='保存解码后的轨迹图像 (.png)')
    parser.add_argument('--save_grid', action='store_true', default=True,
                        help='保存每条轨迹的 grid 总览图')
    parser.add_argument('--save_every', type=int, default=50,
                        help='每 N 个 pair 保存一次 JSON 汇总')

    # 范围控制
    parser.add_argument('--start_idx', type=int, default=0,
                        help='起始 pair 索引')
    parser.add_argument('--end_idx', type=int, default=0,
                        help='结束 pair 索引（0=到末尾）')

    return parser.parse_args()


# ============================================================
# 入口
# ============================================================

def _save_args_json(args):
    """将 args 参数保存为单独的 args.json 文件。"""
    args_dict = vars(args)
    serializable = {}
    for k, v in args_dict.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            serializable[k] = v
        elif isinstance(v, (list, tuple)):
            serializable[k] = list(v)
        else:
            serializable[k] = str(v)
    args_path = os.path.join(args.output_dir, 'args.json')
    save_json(serializable, args_path)
    print(f'[INFO] args 已保存至 {args_path}')


def main():
    args = parse_args()
    generator = TrajectoryGenerator(args)
    _save_args_json(args)
    print('=' * 80)
    print('[INFO] 开始 DiT Flow-Matching 轨迹生成（固定步数，无 wass 计算）')
    print(f'  dit_variant = {args.dit_variant}')
    print(f'  solver      = {args.solver}')
    print(f'  num_steps   = {args.num_steps}')
    print(f'  use_gt_vt   = {args.use_gt_vt}')
    print('  conditions  = target vision CLS + patch tokens + weak noisy hint')
    print('  caption/class ID = disabled')
    print(f'  hint alpha  = {args.hint_signal_ratio}')
    print(f'  save_latents = {args.save_latents}')
    print(f'  save_images  = {args.save_images}')
    print('=' * 80)
    generator.generate()


if __name__ == '__main__':
    main()