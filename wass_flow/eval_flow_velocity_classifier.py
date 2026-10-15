#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flow Velocity Classifier
========================

A few-shot, image-conditioned adaptation of the core idea from:
    "Your Diffusion Model is Secretly a Zero-Shot Classifier"

The original Diffusion Classifier ranks a class condition c by how accurately
that condition lets a diffusion model predict the known injected noise.

Here the current project has no text/class-ID condition at inference.  Instead,
each sampled support image is already the visual condition of the frozen Flow
model.  Therefore each query-support pair is scored by the Flow model's own
training objective:

    x0 = query RAE latent
    x1 = one candidate support RAE latent
    t  = shared trial time across all candidate supports
    e  = shared Gaussian perturbation across all candidate supports

    x_t = (1-t) * x0 + t * x1 + sigma * e
    u_t = x1 - x0

    v_theta = Flow(x_t, t, condition=support_image)
    pair_error = E_trial[ mean((v_theta - u_t)^2) ]

Then pair_error is reshaped to [Q, way, shot], and the shot dimension is
aggregated by mean / min / softmin.  The class with the lowest score wins.

IMPORTANT adaptation relative to the paper
------------------------------------------
The paper can keep x_t identical across class conditions because its condition c
is separate from the noised input image.  In the current Flow model, however,
the candidate support is BOTH:
  1) the endpoint x1 of the Flow-Matching path, and
  2) the visual conditioning input (EndoViT tokens + image hint).

So x_t and u_t must remain candidate-specific to stay aligned with the actual
Flow training objective.  What is shared across candidates is the Monte-Carlo
randomness (t and Gaussian perturbation), preserving the paired-testing spirit.

No MetricEncoder / Dynamic-SK / trajectory Wasserstein distance is used for the
new classifier score.  RAE and Flow/EndoViT are frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# =============================================================================
# USER DEFAULTS
# =============================================================================

PROJECT_ROOT = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow"

# Keep the two uploaded evaluators' default data roots.
TEST_TRAIN_DATA_BASE = os.path.join(
    PROJECT_ROOT, "fsl_multi_vae/data/data_tsy_827_doctor_test_split"
)
TEST_TEST_DATA_BASE = os.path.join(PROJECT_ROOT, "data")

DEFAULT_DIT_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "weights_dit_flow_v1/vision_hint/otcfm/otcfm_dit_L_step_400000.pt",
)

DEFAULT_RESULT_ROOT = os.path.join(
    PROJECT_ROOT,
    "fsl_multi_vae/rae_flow_metric_v1/results_flow_velocity_classifier",
)

ORGAN_N_WAY = {
    "食管": 8,
    "胃": 8,
    "十二指肠": 7,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class EpisodePlan:
    episode_id: int
    candidate_labels: List[str]
    query_label: str
    query_path: str
    support_paths: Dict[str, List[str]]
    seed: int


@dataclass
class ScoreAccumulator:
    correct: int = 0
    total: int = 0
    pos_sum: float = 0.0
    neg_sum: float = 0.0
    margin_sum: float = 0.0
    pair_pos_sum: float = 0.0
    pair_neg_sum: float = 0.0


# =============================================================================
# CLI / basic helpers
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diffusion-Classifier-style velocity-error classification with the current RAE Flow model"
    )

    p.add_argument("--project-root", default=PROJECT_ROOT)
    p.add_argument("--eval-setting", choices=["test_train", "test_test"], default="test_train")
    p.add_argument("--data-base", default=None,
                   help="Override data root. Default follows the corresponding uploaded evaluator.")
    p.add_argument("--result-dir", default=DEFAULT_RESULT_ROOT)
    p.add_argument("--dit-checkpoint", default=os.environ.get("DIT_CHECKPOINT", DEFAULT_DIT_CHECKPOINT))

    p.add_argument("--hospitals", nargs="+", default=None)
    p.add_argument("--parts", nargs="+", default=["十二指肠", "胃", "食管"])

    p.add_argument("--n-shot", type=int, default=5)
    p.add_argument("--n-query", type=int, default=1)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle-candidate-slots", action=argparse.BooleanOptionalAction, default=True)

    # Flow runtime.
    p.add_argument("--dit-variant", default="L", choices=["S", "B", "L", "L12"])
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-condition-mismatch", action="store_true")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda:0")

    # RAE.
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--rae-chunk-size", type=int, default=8)

    # Diffusion-Classifier-style Monte Carlo scoring.
    p.add_argument("--num-trials", type=int, default=16)
    p.add_argument("--time-sampling", choices=["even", "uniform"], default="even")
    p.add_argument("--flow-sigma", type=float, default=0.10,
                   help="Training path noise scale. Current DiT Flow trainer uses sigma=0.1.")
    p.add_argument("--error-type", choices=["l2", "l1"], default="l2")
    p.add_argument("--pair-chunk-size", type=int, default=8,
                   help="How many (trial, support) pairs are sent through DiT at once.")

    # Current Flow input condition = support EndoViT tokens + support hint.
    p.add_argument(
        "--hint-mode",
        choices=["shared_noise", "runtime_independent", "signal_only"],
        default="shared_noise",
        help=(
            "shared_noise: same Gaussian hint noise across supports, while keeping each support's grayscale signal; "
            "runtime_independent: FrozenFlowRuntime helper with a deterministic seed; "
            "signal_only: no Gaussian hint noise."
        ),
    )
    p.add_argument("--hint-seed-offset", type=int, default=81173)

    # We compute all three cheaply from the SAME pair errors by default.
    p.add_argument(
        "--shot-aggs", nargs="+", choices=["mean", "min", "softmin"],
        default=["mean", "min", "softmin"],
    )
    p.add_argument("--shot-tau", type=float, default=0.10)

    p.add_argument("--save-episode-details", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-trial-errors", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args()


def setup_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def task_seed(base_seed: int, hospital: str, part: str) -> int:
    return (
        int(base_seed)
        + sum(ord(c) for c in str(hospital)) * 1009
        + sum(ord(c) for c in str(part)) * 9176
    ) % (2**31 - 1)


def fast_path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(os.path.expanduser(path))))


def torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def resolve_device(text: str) -> torch.device:
    d = torch.device(text)
    if d.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA unavailable, requested {d}")
        idx = 0 if d.index is None else int(d.index)
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested {d}, but visible CUDA device count={torch.cuda.device_count()}"
            )
    return d


# =============================================================================
# Episode protocol -- aligned to the uploaded M2 evaluators
# =============================================================================

def scan_class_folders(root: str) -> Dict[str, List[str]]:
    root_p = Path(root)
    if not root_p.is_dir():
        raise FileNotFoundError(f"Class root not found: {root}")

    result: Dict[str, List[str]] = {}
    for child in sorted(root_p.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        paths = sorted(str(p) for p in child.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        seen = set()
        unique = []
        for p in paths:
            key = fast_path_key(p)
            if key not in seen:
                seen.add(key)
                unique.append(p)
        if unique:
            result[child.name] = unique

    if not result:
        raise RuntimeError(f"No class images found under: {root}")
    return result


def prepare_task_pools(
    *,
    data_base: str,
    hospital: str,
    part: str,
    eval_setting: str,
    n_way: int,
    n_shot: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str], str, str]:
    query_root = os.path.join(data_base, "indomain_support_76", hospital, part)
    support_root = (
        os.path.join(data_base, f"{part}_1query", "train")
        if eval_setting == "test_train"
        else query_root
    )

    raw_query_pool = scan_class_folders(query_root)
    raw_support_pool = (
        raw_query_pool if eval_setting == "test_test" else scan_class_folders(support_root)
    )

    if eval_setting == "test_train":
        # Same reduced/intersection semantics as eval_m2_episode_testtrain.py:
        # preserve every shared class with >=1 image on both sides; shortage later
        # uses replacement instead of dropping the class.
        candidate_labels = [
            c for c in sorted(set(raw_query_pool) & set(raw_support_pool))
            if len(raw_query_pool[c]) >= 1 and len(raw_support_pool[c]) >= 1
        ]
        query_pool = {c: raw_query_pool[c] for c in candidate_labels}
        support_pool = {c: raw_support_pool[c] for c in candidate_labels}
        if len(candidate_labels) < 2:
            raise RuntimeError(
                f"{hospital}/{part}: fewer than 2 shared query/support classes; "
                f"intersection={candidate_labels}"
            )
    else:
        # Test-test needs one query plus K distinct supports from the same class pool.
        candidate_labels = [
            c for c in sorted(raw_query_pool)
            if len(raw_query_pool[c]) >= int(n_shot) + 1
        ]
        query_pool = {c: raw_query_pool[c] for c in candidate_labels}
        support_pool = {c: raw_support_pool[c] for c in candidate_labels}
        if len(candidate_labels) < 2:
            raise RuntimeError(
                f"{hospital}/{part}: fewer than 2 test-test classes with >= {n_shot + 1} images"
            )

    way_eff = (
        len(candidate_labels)
        if eval_setting == "test_train"
        else min(int(n_way), len(candidate_labels))
    )
    if way_eff < 2:
        raise RuntimeError(f"{hospital}/{part}: effective way={way_eff} < 2")

    return query_pool, support_pool, candidate_labels, query_root, support_root


def generate_episode_plans(
    query_pool: Dict[str, List[str]],
    support_pool: Dict[str, List[str]],
    labels: Sequence[str],
    *,
    n_way: int,
    n_shot: int,
    n_episodes: int,
    seed: int,
    eval_setting: str,
    shuffle_candidate_slots: bool,
) -> List[EpisodePlan]:
    rng = random.Random(int(seed))
    labels = list(labels)
    query_labels = sorted(c for c in query_pool if c in set(labels))
    if not query_labels:
        raise RuntimeError("No eligible query class exists in candidate space")
    if n_way > len(labels):
        raise RuntimeError(f"n_way={n_way} > candidate labels={len(labels)}")

    plans: List[EpisodePlan] = []
    for episode_id in range(int(n_episodes)):
        query_label = rng.choice(query_labels)
        if n_way >= len(labels):
            selected = list(labels)
        else:
            distractors = [c for c in labels if c != query_label]
            selected = [query_label] + rng.sample(distractors, n_way - 1)

        query_path = rng.choice(query_pool[query_label])
        query_key = fast_path_key(query_path)

        support_paths: Dict[str, List[str]] = {}
        for label in selected:
            candidates = [
                p for p in support_pool[label]
                if fast_path_key(p) != query_key
            ]
            if not candidates:
                raise RuntimeError(
                    f"episode={episode_id} class={label!r}: no non-query support remains"
                )

            if len(candidates) >= n_shot:
                chosen = rng.sample(candidates, n_shot)
            elif eval_setting == "test_train":
                chosen = rng.choices(candidates, k=n_shot)
            else:
                raise RuntimeError(
                    f"episode={episode_id} support insufficient: class={label!r} "
                    f"available={len(candidates)} need={n_shot}"
                )
            support_paths[label] = chosen

        if shuffle_candidate_slots:
            rng.shuffle(selected)

        plans.append(
            EpisodePlan(
                episode_id=episode_id,
                candidate_labels=list(selected),
                query_label=query_label,
                query_path=query_path,
                support_paths={c: support_paths[c] for c in selected},
                seed=int(seed),
            )
        )
    return plans


# =============================================================================
# Frozen RAE
# =============================================================================

def build_rae(project_root: str, device: torch.device):
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


def build_transform(image_size: int):
    # Same evaluation preprocessing as the uploaded M2 evaluator: [0,1].
    return transforms.Compose([
        transforms.Resize((int(image_size), int(image_size))),
        transforms.ToTensor(),
    ])


def load_images(paths: Sequence[str], transform) -> torch.Tensor:
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(transform(im.convert("RGB")))
    return torch.stack(images, dim=0)


@torch.inference_mode()
def encode_rae_chunked(
    rae: nn.Module,
    images: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        chunk_size = int(images.shape[0])
    chunks = []
    for start in range(0, images.shape[0], chunk_size):
        chunks.append(rae.encode(images[start:start + chunk_size]))
    latent = torch.cat(chunks, dim=0)
    if latent.ndim != 4 or latent.shape[1:] != (1024, 16, 16):
        raise RuntimeError(
            f"RAE latent expected (B,1024,16,16), got {tuple(latent.shape)}"
        )
    return latent


# =============================================================================
# Frozen current Flow runtime
# =============================================================================

class FrozenFlowRuntime:
    """Frozen current DiT Flow + frozen EndoViT support-condition encoder."""

    DATASET_MEAN = (0.3464, 0.2280, 0.2228)
    DATASET_STD = (0.2520, 0.2128, 0.2093)

    def __init__(
        self,
        *,
        project_root: str,
        checkpoint: str,
        device: torch.device,
        dit_variant: str,
        num_classes: int,
        image_size: int,
        use_ema: bool,
        allow_condition_mismatch: bool,
        amp: bool,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError("The current DiT Flow runtime requires CUDA")
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"DiT/Flow checkpoint not found: {checkpoint}")

        self.project_root = str(Path(project_root).resolve())
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)
        self.device = device
        self.amp = bool(amp)
        self.image_size = int(image_size)
        self.checkpoint = str(checkpoint)

        from flow_models.dit_rae import create_dit_rae
        from timm.models.vision_transformer import VisionTransformer

        ckpt = torch_load(checkpoint)
        if not isinstance(ckpt, dict):
            raise ValueError("DiT checkpoint must be a dict checkpoint")

        self.condition_config = ckpt.get("condition_config") or {}
        saved_mode = self.condition_config.get("condition_mode")
        saved_size = self.condition_config.get("image_size")
        mismatch = (
            saved_mode != "vision_hint_v1"
            or (saved_size is not None and int(saved_size) != self.image_size)
            or bool(self.condition_config.get("use_class_id", False))
        )
        if mismatch and not allow_condition_mismatch:
            raise RuntimeError(
                "DiT checkpoint condition_config is not the expected visual-condition V1.\n"
                f"saved={self.condition_config}\n"
                f"requested image_size={self.image_size}\n"
                "Use --allow-condition-mismatch only for an intentional ablation."
            )

        self.hint_signal_ratio = float(self.condition_config.get("hint_signal_ratio", 0.10))
        if not 0.0 <= self.hint_signal_ratio <= 1.0:
            raise ValueError("hint_signal_ratio from checkpoint must be in [0,1]")

        # Prefer architecture metadata saved with the checkpoint when present.
        saved_args = ckpt.get("args") if isinstance(ckpt.get("args"), dict) else {}
        if saved_args.get("dit_variant") is not None:
            dit_variant = str(saved_args["dit_variant"])
        elif ckpt.get("dit_variant") is not None:
            dit_variant = str(ckpt["dit_variant"])
        if saved_args.get("num_classes") is not None:
            num_classes = int(saved_args["num_classes"])
        elif ckpt.get("num_classes") is not None:
            num_classes = int(ckpt["num_classes"])
        self.dit_variant = dit_variant
        self.num_classes = int(num_classes)

        dit = create_dit_rae(
            variant=self.dit_variant,
            rae_latent_dim=1024,
            rae_spatial_size=16,
            cross_attn_dim=768,
            num_classes=self.num_classes,
            use_cross_attention=True,
        )

        if use_ema and isinstance(ckpt.get("ema_model"), dict):
            dit_state = ckpt["ema_model"]
            self.loaded_dit_key = "ema_model"
        elif isinstance(ckpt.get("dit_model"), dict):
            dit_state = ckpt["dit_model"]
            self.loaded_dit_key = "dit_model"
        else:
            dit_state = ckpt
            self.loaded_dit_key = "checkpoint_root"

        incompatible = dit.load_state_dict(dit_state, strict=False)
        if incompatible.missing_keys:
            print(f"[Flow][WARN] DiT missing keys: {list(incompatible.missing_keys)[:12]}")
        if incompatible.unexpected_keys:
            print(f"[Flow][WARN] DiT unexpected keys: {list(incompatible.unexpected_keys)[:12]}")
        dit = dit.to(self.device).eval()
        dit.device = self.device
        for p in dit.parameters():
            p.requires_grad_(False)
        self.dit = dit

        vision = VisionTransformer(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=False,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ).eval()
        endovit_path = os.path.join(
            self.project_root, "flow_matcher_otcfm/EndoViT/pytorch_model.bin"
        )
        if not Path(endovit_path).is_file():
            raise FileNotFoundError(f"EndoViT checkpoint not found: {endovit_path}")
        vision.load_state_dict(
            torch.load(endovit_path, map_location="cpu", weights_only=False),
            strict=False,
        )
        # The Flow checkpoint may contain the jointly trained vision encoder.
        if isinstance(ckpt.get("vision_model"), dict):
            vision.load_state_dict(ckpt["vision_model"], strict=False)
            self.loaded_vision_key = "vision_model"
        else:
            self.loaded_vision_key = "base_EndoViT"
        vision = vision.to(self.device).eval()
        vision.device = self.device
        for p in vision.parameters():
            p.requires_grad_(False)
        self.vision = vision

        self.checkpoint_step = ckpt.get("step")
        del ckpt

    @torch.no_grad()
    def encode_target_vision(self, support_images_01: torch.Tensor) -> torch.Tensor:
        """Support image -> EndoViT [CLS + patch] tokens, matching current Flow input."""
        x = support_images_01.to(self.device, non_blocking=True).float()
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor(self.DATASET_MEAN, device=self.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.DATASET_STD, device=self.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
            out = self.vision.forward_features(x)
        return out.float()

    @torch.no_grad()
    def build_runtime_hint(
        self,
        support_images_01: torch.Tensor,
        *,
        stochastic: bool,
        seed: Optional[int],
    ) -> torch.Tensor:
        """Exact helper semantics used by the existing frozen Flow runtime."""
        x = support_images_01.to(self.device, non_blocking=True).float()
        gray = (
            0.2989 * x[:, 0:1]
            + 0.5870 * x[:, 1:2]
            + 0.1140 * x[:, 2:3]
        ).repeat(1, 3, 1, 1)
        gray = gray * 2.0 - 1.0

        if seed is None:
            noise = torch.randn_like(gray) if stochastic else torch.zeros_like(gray)
        else:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            noise = torch.randn(gray.shape, generator=g, dtype=torch.float32).to(self.device)
            if not stochastic:
                noise.zero_()

        alpha = self.hint_signal_ratio
        return alpha * gray + math.sqrt(max(1.0 - alpha * alpha, 0.0)) * noise

    @torch.no_grad()
    def build_shared_noise_hint(
        self,
        support_images_01: torch.Tensor,
        *,
        seed: int,
    ) -> torch.Tensor:
        """Same marginal hint distribution, but pair the random noise across candidates."""
        x = support_images_01.to(self.device, non_blocking=True).float()
        gray = (
            0.2989 * x[:, 0:1]
            + 0.5870 * x[:, 1:2]
            + 0.1140 * x[:, 2:3]
        ).repeat(1, 3, 1, 1)
        gray = gray * 2.0 - 1.0

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        shared = torch.randn(
            (1, gray.shape[1], gray.shape[2], gray.shape[3]),
            generator=g,
            dtype=torch.float32,
        ).to(self.device)
        shared = shared.expand(gray.shape[0], -1, -1, -1)
        alpha = self.hint_signal_ratio
        return alpha * gray + math.sqrt(max(1.0 - alpha * alpha, 0.0)) * shared

    @torch.no_grad()
    def predict_velocity(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        *,
        target_vision: torch.Tensor,
        hint: torch.Tensor,
    ) -> torch.Tensor:
        """Current Flow forward: (x_t,t,support visual condition) -> v_theta."""
        patch = target_vision[:, 1:, :]
        cls = target_vision[:, 0, :]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
            out = self.dit(
                xt=xt,
                timestep=t.reshape(-1),
                class_labels=None,
                encoder_hidden_states=patch,
                hint=hint,
                text_embeds=None,
                image_embeds=cls,
            )["sample"]
        return out.float()


# =============================================================================
# Velocity-error classifier
# =============================================================================

def sample_trial_times(
    num_trials: int,
    *,
    mode: str,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if num_trials < 1:
        raise ValueError("num_trials must be >=1")
    if mode == "even":
        # Midpoints avoid exact t=0/1 while covering the complete training interval.
        return (torch.arange(num_trials, device=device, dtype=torch.float32) + 0.5) / float(num_trials)
    if mode == "uniform":
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        return torch.rand(num_trials, generator=g, dtype=torch.float32).to(device)
    raise ValueError(f"Unknown time sampling: {mode}")


def aggregate_shots(pair_error: torch.Tensor, mode: str, tau: float) -> torch.Tensor:
    """pair_error [Q,way,shot] -> class_error [Q,way]."""
    if mode == "mean":
        return pair_error.mean(dim=-1)
    if mode == "min":
        return pair_error.min(dim=-1).values
    if mode == "softmin":
        if tau <= 0:
            raise ValueError("shot_tau must be >0 for softmin")
        return -float(tau) * (
            torch.logsumexp(-pair_error / float(tau), dim=-1)
            - math.log(float(pair_error.shape[-1]))
        )
    raise ValueError(f"Unknown shot aggregation: {mode}")


@torch.inference_mode()
def flow_velocity_classify(
    *,
    flow: FrozenFlowRuntime,
    support_z: torch.Tensor,
    query_z: torch.Tensor,
    support_images_cpu: torch.Tensor,
    n_way: int,
    n_shot: int,
    num_trials: int,
    time_sampling: str,
    flow_sigma: float,
    error_type: str,
    pair_chunk_size: int,
    hint_mode: str,
    trial_seed: int,
    hint_seed: int,
    shot_aggs: Sequence[str],
    shot_tau: float,
    save_trial_errors: bool,
) -> Dict[str, Any]:
    """Score every query-support pair by the frozen Flow training-target error.

    Shared paired-testing randomness:
      * one set of t_r for all candidates of a query;
      * one Gaussian epsilon_r for all candidates of a query.

    Candidate-specific quantities (required by this Flow's training semantics):
      * endpoint x1 = support latent;
      * visual condition = the same support image;
      * therefore x_t and u_t are candidate-specific.
    """
    n_way = int(n_way)
    n_shot = int(n_shot)
    n_support = int(support_z.shape[0])
    n_query = int(query_z.shape[0])
    if n_support != n_way * n_shot:
        raise ValueError(
            f"support count={n_support} != n_way*n_shot={n_way*n_shot}"
        )

    support_z_f = support_z.to(flow.device, non_blocking=True).float()
    query_z_f = query_z.to(flow.device, non_blocking=True).float()

    # Candidate condition is computed once and held fixed across MC trials.
    target_vision = flow.encode_target_vision(support_images_cpu)
    if hint_mode == "shared_noise":
        hint = flow.build_shared_noise_hint(support_images_cpu, seed=hint_seed)
    elif hint_mode == "runtime_independent":
        hint = flow.build_runtime_hint(support_images_cpu, stochastic=True, seed=hint_seed)
    elif hint_mode == "signal_only":
        hint = flow.build_runtime_hint(support_images_cpu, stochastic=False, seed=hint_seed)
    else:
        raise ValueError(f"Unknown hint_mode={hint_mode}")

    times = sample_trial_times(
        num_trials,
        mode=time_sampling,
        seed=trial_seed,
        device=flow.device,
    )

    pair_means: List[torch.Tensor] = []
    pair_stds: List[torch.Tensor] = []
    all_trial_errors: List[torch.Tensor] = []

    for q_idx in range(n_query):
        q = query_z_f[q_idx:q_idx + 1]  # [1,C,H,W]

        # Gaussian perturbation follows the current Flow-Matching path.  One
        # epsilon_r is broadcast to every candidate support for paired testing.
        g = torch.Generator(device="cpu")
        g.manual_seed(int(trial_seed + q_idx * 104729))
        eps = torch.randn(
            (num_trials, q.shape[1], q.shape[2], q.shape[3]),
            generator=g,
            dtype=torch.float32,
        ).to(flow.device)

        # Flatten [trial,support] into one stream so DiT forwards are batched.
        r, s = num_trials, n_support
        q_rs = q.view(1, 1, *q.shape[1:]).expand(r, s, *q.shape[1:])
        s_rs = support_z_f.view(1, s, *support_z_f.shape[1:]).expand(r, s, *support_z_f.shape[1:])
        e_rs = eps.view(r, 1, *eps.shape[1:]).expand(r, s, *eps.shape[1:])
        t_rs = times.view(r, 1).expand(r, s)

        t_x = t_rs.view(r, s, 1, 1, 1)
        xt_rs = (1.0 - t_x) * q_rs + t_x * s_rs + float(flow_sigma) * e_rs
        ut_rs = s_rs - q_rs

        xt_flat = xt_rs.reshape(r * s, *q.shape[1:])
        ut_flat = ut_rs.reshape(r * s, *q.shape[1:])
        t_flat = t_rs.reshape(r * s)
        vision_flat = target_vision.view(1, s, *target_vision.shape[1:]).expand(
            r, s, *target_vision.shape[1:]
        ).reshape(r * s, *target_vision.shape[1:])
        hint_flat = hint.view(1, s, *hint.shape[1:]).expand(
            r, s, *hint.shape[1:]
        ).reshape(r * s, *hint.shape[1:])

        error_chunks = []
        chunk = max(int(pair_chunk_size), 1)
        for start in range(0, r * s, chunk):
            end = min(start + chunk, r * s)
            pred = flow.predict_velocity(
                xt_flat[start:end],
                t_flat[start:end],
                target_vision=vision_flat[start:end],
                hint=hint_flat[start:end],
            )
            diff = pred - ut_flat[start:end]
            if error_type == "l2":
                err = diff.pow(2).flatten(1).mean(dim=1)
            elif error_type == "l1":
                err = diff.abs().flatten(1).mean(dim=1)
            else:
                raise ValueError(f"Unknown error_type={error_type}")
            error_chunks.append(err.float())

        trial_error = torch.cat(error_chunks, dim=0).reshape(r, s)  # [R,S]
        pair_means.append(trial_error.mean(dim=0))
        pair_stds.append(trial_error.std(dim=0, unbiased=False))
        if save_trial_errors:
            all_trial_errors.append(trial_error.detach().cpu())

    pair_error_flat = torch.stack(pair_means, dim=0)  # [Q,S]
    pair_std_flat = torch.stack(pair_stds, dim=0)
    pair_error = pair_error_flat.reshape(n_query, n_way, n_shot)
    pair_std = pair_std_flat.reshape(n_query, n_way, n_shot)

    class_error_by_agg: Dict[str, torch.Tensor] = {}
    pred_by_agg: Dict[str, torch.Tensor] = {}
    for agg in shot_aggs:
        class_error = aggregate_shots(pair_error, agg, shot_tau)
        class_error_by_agg[agg] = class_error
        pred_by_agg[agg] = class_error.argmin(dim=-1)

    out: Dict[str, Any] = {
        "pair_error": pair_error,
        "pair_trial_std": pair_std,
        "class_error_by_agg": class_error_by_agg,
        "pred_by_agg": pred_by_agg,
        "trial_times": times.detach().cpu(),
    }
    if save_trial_errors:
        out["trial_error"] = torch.stack(all_trial_errors, dim=0)  # [Q,R,S]
    return out


# =============================================================================
# Metrics / evaluation
# =============================================================================

def ci95_from_binary(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    p = correct / total
    return 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / total)


def finalize_accumulator(acc: ScoreAccumulator) -> Dict[str, float]:
    n = max(int(acc.total), 1)
    accuracy = acc.correct / n
    return {
        "correct": int(acc.correct),
        "total": int(acc.total),
        "accuracy": float(accuracy),
        "accuracy_percent": float(100.0 * accuracy),
        "ci95": float(ci95_from_binary(acc.correct, acc.total)),
        "ci95_percent": float(100.0 * ci95_from_binary(acc.correct, acc.total)),
        "pos_class_error": float(acc.pos_sum / n),
        "nearest_neg_class_error": float(acc.neg_sum / n),
        "class_margin": float(acc.margin_sum / n),
        "mean_positive_pair_error": float(acc.pair_pos_sum / n),
        "mean_nearest_negative_pair_error": float(acc.pair_neg_sum / n),
    }


@torch.inference_mode()
def evaluate_task(
    *,
    hospital: str,
    part: str,
    args: argparse.Namespace,
    data_base: str,
    rae: nn.Module,
    flow: FrozenFlowRuntime,
    device: torch.device,
    transform,
) -> Dict[str, Any]:
    way_requested = int(ORGAN_N_WAY.get(part, 8))
    query_pool, support_pool, labels, query_root, support_root = prepare_task_pools(
        data_base=data_base,
        hospital=hospital,
        part=part,
        eval_setting=args.eval_setting,
        n_way=way_requested,
        n_shot=args.n_shot,
    )
    way_eff = (
        len(labels)
        if args.eval_setting == "test_train"
        else min(way_requested, len(labels))
    )

    seed_task = task_seed(args.seed, hospital, part)
    plans = generate_episode_plans(
        query_pool,
        support_pool,
        labels,
        n_way=way_eff,
        n_shot=args.n_shot,
        n_episodes=args.episodes,
        seed=seed_task,
        eval_setting=args.eval_setting,
        shuffle_candidate_slots=args.shuffle_candidate_slots,
    )

    print("=" * 118)
    print(
        f"[Flow Velocity Classifier] {hospital}/{part} | "
        f"{way_eff}-way {args.n_shot}-shot {args.n_query}-query | episodes={len(plans)}"
    )
    print(f"query_root   = {query_root}")
    print(f"support_root = {support_root}")
    print(
        f"trials={args.num_trials}, time={args.time_sampling}, sigma={args.flow_sigma}, "
        f"error={args.error_type}, hint={args.hint_mode}"
    )
    print(f"shot_aggs={args.shot_aggs}, softmin_tau={args.shot_tau}")
    print("=" * 118)

    accumulators = {agg: ScoreAccumulator() for agg in args.shot_aggs}
    episode_records: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep_idx, ep in enumerate(plans):
        support_flat = [
            p
            for label in ep.candidate_labels
            for p in ep.support_paths[label]
        ]
        support_cpu = load_images(support_flat, transform)
        query_cpu = load_images([ep.query_path], transform)
        all_cpu = torch.cat([support_cpu, query_cpu], dim=0)
        latent = encode_rae_chunked(
            rae,
            all_cpu.to(device, non_blocking=True),
            args.rae_chunk_size,
        )
        support_z = latent[:len(support_flat)]
        query_z = latent[len(support_flat):]

        gt_index = ep.candidate_labels.index(ep.query_label)
        trial_seed = int(seed_task + ep.episode_id * 1_000_003 + 97) % (2**31 - 1)
        hint_seed = int(trial_seed + args.hint_seed_offset) % (2**31 - 1)

        out = flow_velocity_classify(
            flow=flow,
            support_z=support_z,
            query_z=query_z,
            support_images_cpu=support_cpu,
            n_way=way_eff,
            n_shot=args.n_shot,
            num_trials=args.num_trials,
            time_sampling=args.time_sampling,
            flow_sigma=args.flow_sigma,
            error_type=args.error_type,
            pair_chunk_size=args.pair_chunk_size,
            hint_mode=args.hint_mode,
            trial_seed=trial_seed,
            hint_seed=hint_seed,
            shot_aggs=args.shot_aggs,
            shot_tau=args.shot_tau,
            save_trial_errors=args.save_trial_errors,
        )

        pair_e = out["pair_error"][0]  # [way,shot]
        pair_std = out["pair_trial_std"][0]

        ep_record: Dict[str, Any] = {
            "episode_id": int(ep.episode_id),
            "query_label": ep.query_label,
            "query_path": ep.query_path,
            "candidate_labels": list(ep.candidate_labels),
            "trial_seed": trial_seed,
            "hint_seed": hint_seed,
            "trial_times": [float(v) for v in out["trial_times"].tolist()],
        }
        if args.save_episode_details:
            ep_record["support_paths_by_class"] = {
                c: list(ep.support_paths[c]) for c in ep.candidate_labels
            }
            ep_record["pair_errors_by_class"] = {
                c: [float(v) for v in pair_e[i].detach().cpu().tolist()]
                for i, c in enumerate(ep.candidate_labels)
            }
            ep_record["pair_trial_std_by_class"] = {
                c: [float(v) for v in pair_std[i].detach().cpu().tolist()]
                for i, c in enumerate(ep.candidate_labels)
            }
            if args.save_trial_errors:
                # [Q,R,S] -> [R,way,shot]
                tr = out["trial_error"][0].reshape(
                    args.num_trials, way_eff, args.n_shot
                )
                ep_record["trial_errors"] = [
                    [
                        [float(x) for x in tr[r, i].tolist()]
                        for i in range(way_eff)
                    ]
                    for r in range(args.num_trials)
                ]

        ep_record["shot_aggregations"] = {}
        for agg in args.shot_aggs:
            class_e = out["class_error_by_agg"][agg][0]
            pred_idx = int(out["pred_by_agg"][agg][0].item())
            pred_label = ep.candidate_labels[pred_idx]

            pos = float(class_e[gt_index].item())
            neg_mask = torch.ones(way_eff, dtype=torch.bool, device=class_e.device)
            neg_mask[gt_index] = False
            nearest_neg = float(class_e[neg_mask].min().item())
            margin = nearest_neg - pos  # >0 means GT is better than nearest negative.

            pair_pos = float(pair_e[gt_index].mean().item())
            pair_neg_mask = torch.ones_like(pair_e, dtype=torch.bool)
            pair_neg_mask[gt_index] = False
            pair_neg = float(pair_e.masked_select(pair_neg_mask).min().item())

            acc = accumulators[agg]
            acc.total += 1
            acc.correct += int(pred_label == ep.query_label)
            acc.pos_sum += pos
            acc.neg_sum += nearest_neg
            acc.margin_sum += margin
            acc.pair_pos_sum += pair_pos
            acc.pair_neg_sum += pair_neg

            ep_record["shot_aggregations"][agg] = {
                "pred_label": pred_label,
                "correct": bool(pred_label == ep.query_label),
                "class_errors": {
                    c: float(class_e[i].item())
                    for i, c in enumerate(ep.candidate_labels)
                },
                "pos_class_error": pos,
                "nearest_neg_class_error": nearest_neg,
                "margin": margin,
            }

        if args.save_episode_details:
            episode_records.append(ep_record)

        if args.log_every > 0 and (ep_idx + 1) % args.log_every == 0:
            msg = " | ".join(
                f"{agg}:{100.0 * a.correct / max(a.total, 1):.2f}%"
                for agg, a in accumulators.items()
            )
            print(f"  [{ep_idx + 1:04d}/{len(plans):04d}] {msg}")

        del support_cpu, query_cpu, all_cpu, latent, support_z, query_z, out

    elapsed = time.time() - t0
    metrics = {agg: finalize_accumulator(acc) for agg, acc in accumulators.items()}

    result: Dict[str, Any] = {
        "stage": "flow_velocity_classifier",
        "hospital": hospital,
        "part": part,
        "eval_setting": args.eval_setting,
        "query_root": query_root,
        "support_root": support_root,
        "valid_classes": labels,
        "protocol": {
            "n_way_requested": way_requested,
            "n_way": way_eff,
            "n_shot": int(args.n_shot),
            "n_query_total": int(args.n_query),
            "episodes": int(args.episodes),
            "seed": int(seed_task),
            "query_excluded_from_support": True,
            "candidate_space": (
                "query_support_intersection"
                if args.eval_setting == "test_train"
                else "test_test_eligible"
            ),
            "rare_support_class_policy": (
                "keep_if_at_least_1_then_sample_with_replacement"
                if args.eval_setting == "test_train"
                else "distinct_support_only"
            ),
        },
        "flow_classifier": {
            "principle": "Diffusion-Classifier-style conditional prediction error",
            "candidate_condition": "one support image at a time",
            "flow_source_x0": "query RAE latent",
            "flow_endpoint_x1": "candidate support RAE latent",
            "flow_condition": "same candidate support EndoViT CLS+patch tokens and image hint",
            "path": "x_t=(1-t)*x0+t*x1+sigma*epsilon",
            "target_velocity": "u_t=x1-x0",
            "pair_score": f"trial mean of elementwise {args.error_type} error between v_theta and u_t",
            "paired_randomness": "same t and same epsilon across candidate supports for each query/trial",
            "candidate_specific_xt_and_ut": True,
            "candidate_specific_xt_reason": (
                "support is both Flow endpoint x1 and visual condition in the current trained model"
            ),
            "num_trials": int(args.num_trials),
            "time_sampling": args.time_sampling,
            "flow_sigma": float(args.flow_sigma),
            "error_type": args.error_type,
            "hint_mode": args.hint_mode,
            "hint_signal_ratio": float(flow.hint_signal_ratio),
            "trial_aggregation": "mean",
            "shot_aggregations": list(args.shot_aggs),
            "shot_tau": float(args.shot_tau),
        },
        "flow_runtime": {
            "checkpoint": flow.checkpoint,
            "checkpoint_step": flow.checkpoint_step,
            "loaded_dit_key": flow.loaded_dit_key,
            "loaded_vision_key": flow.loaded_vision_key,
            "dit_variant": flow.dit_variant,
            "num_classes_structure_only": flow.num_classes,
            "condition_config": flow.condition_config,
            "class_id_condition_used": False,
            "text_condition_used": False,
        },
        "metrics": metrics,
        "elapsed_sec": float(elapsed),
    }
    if args.save_episode_details:
        result["episodes"] = episode_records
    return result


# =============================================================================
# Output
# =============================================================================

def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "hospital", "part", "eval_setting", "shot_agg", "n_way", "n_shot",
        "episodes", "num_trials", "time_sampling", "flow_sigma", "error_type",
        "hint_mode", "correct", "accuracy_percent", "ci95_percent",
        "pos_class_error", "nearest_neg_class_error", "class_margin",
        "mean_positive_pair_error", "mean_nearest_negative_pair_error",
        "flow_checkpoint",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    if args.n_query != 1:
        raise ValueError("This evaluator currently keeps the canonical 1-query episode protocol")
    if args.n_shot < 1 or args.episodes < 1 or args.num_trials < 1:
        raise ValueError("n_shot / episodes / num_trials must be >=1")
    if args.flow_sigma < 0:
        raise ValueError("flow_sigma must be >=0")
    if "softmin" in args.shot_aggs and args.shot_tau <= 0:
        raise ValueError("shot_tau must be >0 when softmin is requested")

    setup_seed(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.data_base is None:
        data_base = (
            TEST_TRAIN_DATA_BASE
            if args.eval_setting == "test_train"
            else TEST_TEST_DATA_BASE
        )
    else:
        data_base = args.data_base

    hospitals = args.hospitals
    if hospitals is None:
        hospitals = (
            ["华西", "自贡", "南充", "苍溪"]
            if args.eval_setting == "test_train"
            else ["自贡", "南充", "苍溪"]
        )

    result_dir = os.path.join(args.result_dir, args.eval_setting)
    os.makedirs(result_dir, exist_ok=True)

    print("=" * 118)
    print("[FLOW VELOCITY CLASSIFIER | DIFFUSION-CLASSIFIER-STYLE ADAPTATION]")
    print(f"device          = {device}")
    print(f"eval_setting    = {args.eval_setting}")
    print(f"data_base       = {data_base}")
    print(f"hospitals       = {hospitals}")
    print(f"parts           = {args.parts}")
    print(f"dit_checkpoint  = {args.dit_checkpoint}")
    print(f"trials          = {args.num_trials} ({args.time_sampling})")
    print(f"flow_sigma      = {args.flow_sigma}")
    print(f"error_type      = {args.error_type}")
    print(f"hint_mode       = {args.hint_mode}")
    print(f"shot_aggs       = {args.shot_aggs}")
    print(f"result_dir      = {result_dir}")
    print("=" * 118)

    # One frozen RAE and one frozen Flow runtime are reused for all tasks.
    rae = build_rae(args.project_root, device)
    flow = FrozenFlowRuntime(
        project_root=args.project_root,
        checkpoint=args.dit_checkpoint,
        device=device,
        dit_variant=args.dit_variant,
        num_classes=args.num_classes,
        image_size=args.image_size,
        use_ema=args.use_ema,
        allow_condition_mismatch=args.allow_condition_mismatch,
        amp=args.amp,
    )
    transform = build_transform(args.image_size)

    summary_rows: List[Dict[str, Any]] = []
    summary_nested: Dict[str, Any] = {}

    for part in args.parts:
        for hospital in hospitals:
            result = evaluate_task(
                hospital=hospital,
                part=part,
                args=args,
                data_base=data_base,
                rae=rae,
                flow=flow,
                device=device,
                transform=transform,
            )

            out_json = os.path.join(
                result_dir,
                part,
                f"{hospital}_{part}_{args.eval_setting}_flow_velocity_classifier.json",
            )
            save_json(out_json, result)
            summary_nested.setdefault(part, {})[hospital] = result["metrics"]

            print(f"\n[RESULT] {hospital}/{part}")
            for agg, metrics in result["metrics"].items():
                print(
                    f"  {agg:<7} acc={metrics['accuracy_percent']:.2f}% "
                    f"±{metrics['ci95_percent']:.2f}% "
                    f"margin={metrics['class_margin']:.6f}"
                )
                summary_rows.append({
                    "hospital": hospital,
                    "part": part,
                    "eval_setting": args.eval_setting,
                    "shot_agg": agg,
                    "n_way": result["protocol"]["n_way"],
                    "n_shot": args.n_shot,
                    "episodes": args.episodes,
                    "num_trials": args.num_trials,
                    "time_sampling": args.time_sampling,
                    "flow_sigma": args.flow_sigma,
                    "error_type": args.error_type,
                    "hint_mode": args.hint_mode,
                    "correct": metrics["correct"],
                    "accuracy_percent": metrics["accuracy_percent"],
                    "ci95_percent": metrics["ci95_percent"],
                    "pos_class_error": metrics["pos_class_error"],
                    "nearest_neg_class_error": metrics["nearest_neg_class_error"],
                    "class_margin": metrics["class_margin"],
                    "mean_positive_pair_error": metrics["mean_positive_pair_error"],
                    "mean_nearest_negative_pair_error": metrics["mean_nearest_negative_pair_error"],
                    "flow_checkpoint": args.dit_checkpoint,
                })

            if device.type == "cuda":
                torch.cuda.empty_cache()

    save_summary_csv(os.path.join(result_dir, "summary.csv"), summary_rows)
    save_json(
        os.path.join(result_dir, "summary.json"),
        {
            "config": {
                "method": "flow_velocity_classifier",
                "project_root": args.project_root,
                "data_base": data_base,
                "eval_setting": args.eval_setting,
                "hospitals": hospitals,
                "parts": args.parts,
                "organ_n_way": ORGAN_N_WAY,
                "n_shot": args.n_shot,
                "n_query": args.n_query,
                "episodes": args.episodes,
                "seed": args.seed,
                "dit_checkpoint": args.dit_checkpoint,
                "num_trials": args.num_trials,
                "time_sampling": args.time_sampling,
                "flow_sigma": args.flow_sigma,
                "error_type": args.error_type,
                "hint_mode": args.hint_mode,
                "shot_aggs": args.shot_aggs,
                "shot_tau": args.shot_tau,
                "pairing_rule": "shared t and epsilon across candidate supports",
                "pair_score": "mean Flow velocity prediction error over trials",
            },
            "results": summary_nested,
        },
    )

    print("\n" + "=" * 118)
    print("[DONE]")
    print(f"summary CSV  = {os.path.join(result_dir, 'summary.csv')}")
    print(f"summary JSON = {os.path.join(result_dir, 'summary.json')}")
    print("=" * 118)


if __name__ == "__main__":
    main()
