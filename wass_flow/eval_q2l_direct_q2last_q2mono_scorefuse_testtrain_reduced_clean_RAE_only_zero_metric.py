#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean RAE-only Direct / Q2L / Q2Mono episodic evaluation (reduced test-train).

This is the zero learned-metric-checkpoint ablation. No learned metric projection
is imported, instantiated, or loaded. All geometric distances are computed
directly on frozen RAE latent maps. Each [B,1024,16,16] latent is interpreted
as 256 spatial samples in R^1024 and fed directly to Dynamic-SK/Sinkhorn.

Direct, Q2L, and Q2Mono use the same RAE token geometry, frozen Flow trajectory,
and K-shot aggregation. Q2Mono applies strict running-maximum filtering to the
post-flow Query-to-Step distances z_1,...,z_T, then appends the Direct
Query-to-Support distance and averages the retained values; z_0 is excluded
from monotonic filtering.

Reduced Test-Train protocol:
    candidate classes = query-side classes ∩ support-side classes.
    Support-only classes are removed from the candidate space.
    Every shared class with at least one support image is retained; if a class
    has fewer than K support images, K-shot support is sampled with replacement.
"""
from __future__ import annotations

import csv
import json
import shutil
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# =============================================================================
# USER CONFIG
# =============================================================================
PROJECT_ROOT = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow"
DATA_BASE = os.path.join(PROJECT_ROOT, "fsl_multi_vae/data/data_tsy_827_doctor_test_split")

EVAL_SETTING = "test_train"
BASE_RESULT_DIR = os.path.join(PROJECT_ROOT, "fsl_multi_vae/rae_flow_metric_v1", "results_q2l_scorefusion_RAE_only_zero_metric")
HOSPITALS = ["华西", "自贡", "南充", "苍溪"]
# HOSPITALS = ["苍溪"]
# Reference widths only. In REDUCED Test-Train, all shared query∩support classes are evaluated.
ORGAN_N_WAY = {"食管": 8, "胃": 8, "十二指肠": 7}
PARTS = ["十二指肠", "胃", "食管"]
# PARTS = ["食管"]

# -----------------------------------------------------------------------------
# CLEAN RAE-ONLY / ZERO-METRIC-CHECKPOINT CONFIG
# -----------------------------------------------------------------------------
# No learned Metric-model checkpoint is loaded anywhere in this evaluator.
# Direct, Q2L, and Q2Mono operate directly on frozen RAE spatial-token clouds.
# The only learned trajectory checkpoint used below is the frozen DiT/Flow model.
DIT_CHECKPOINT = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_dit_flow_v1/vision_hint_inclass/otcfm/otcfm_dit_L_step_500000.pt"
DIT_VARIANT = "L"
FLOW_NUM_CLASSES = 1000
FLOW_NUM_STEPS = 8
FLOW_SOLVER = "heun"
FLOW_PAIR_CHUNK_SIZE = 16
FLOW_USE_EMA = True
FLOW_HINT_SIGNAL_RATIO = None  # None -> read the hint ratio from the DiT/Flow checkpoint.
FLOW_STOCHASTIC_HINT = True
FLOW_PURE_NOISE_HINT = False
FLOW_ALLOW_CONDITION_MISMATCH = False
FLOW_AMP = True

# Raw RAE Sinkhorn geometry. With factor=1, [B,1024,16,16] -> [B,256,1024].
RAE_OT_POOL_FACTOR = 1
SINKHORN_REG = 0.10
SINKHORN_ITER = 30
SK_WEIGHT_EPS = 1e-3

EXPERIMENT_NAME = "RAE-only-zero-Metric"

N_SHOT = 5
N_QUERY = 1
EPISODES_PER_TASK = 100
SEED = 42
SHUFFLE_CANDIDATE_SLOTS = True

IMAGE_SIZE = 256
RAE_CHUNK_SIZE = 8

# K-shot aggregation is evaluation-only and shared by all three RAE-only branches.
SHOT_AGG = "softmin"          # "softmin", "min", "mean"
SHOT_TAU = 0.10

# Flexible fusion configuration
# Available metrics: direct, flow(Query2Last), q2mono(Q2Mono)
FUSION_METRICS = ["flow", "q2mono"]  # ["direct", "flow", "q2mono"]
# Fusion rules remain unchanged:
#   dist_sum    : sum of selected class distances
#   soft_vote   : sum of softmax votes from selected metrics
#   global_min  : choose the best metric-class pair
FUSION_MODES = ["dist_sum", "soft_vote", "global_min"]


def get_fusion_tag():
    return "fusion_" + "_".join(FUSION_METRICS)


RESULT_DIR = os.path.join(BASE_RESULT_DIR, get_fusion_tag())
CASE_SAVE_ROOT = os.path.join(RESULT_DIR, 'direct_wrong_fusion_correct_cases')

SAVE_EPISODE_DETAILS = False
LOG_EVERY = 25

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.sk_sinkhorn import aligned_sk_sinkhorn_distance

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class EpisodePlan:
    episode_id: int
    candidate_labels: List[str]
    query_label: str
    query_path: str
    support_paths: Dict[str, List[str]]
    seed: int


@dataclass
class EvalAccumulator:
    correct: int = 0
    total: int = 0
    pos_sum: float = 0.0
    neg_sum: float = 0.0
    margin_sum: float = 0.0
    pair_pos_sum: float = 0.0
    pair_neg_sum: float = 0.0


@dataclass
class DualEvalAccumulator:
    # Existing matched branches.
    direct: EvalAccumulator = field(default_factory=EvalAccumulator)
    flow: EvalAccumulator = field(default_factory=EvalAccumulator)
    # Q2Mono branch: strict running-max Query-to-Step distances plus Direct.
    q2mono: EvalAccumulator = field(default_factory=EvalAccumulator)

    # Direct vs Query2Last statistics (kept for backward compatibility).
    both_correct: int = 0
    direct_only: int = 0
    flow_only: int = 0
    both_wrong: int = 0
    prediction_changed: int = 0
    oracle_correct: int = 0

    # Direct vs Q2Mono statistics.
    direct_q2mono_both_correct: int = 0
    direct_only_vs_q2mono: int = 0
    q2mono_only_vs_direct: int = 0
    direct_q2mono_both_wrong: int = 0
    q2mono_prediction_changed: int = 0
    oracle_direct_q2mono_correct: int = 0

    # Oracle across all matched scores.
    oracle_flow_q2mono_correct: int = 0
    oracle_all_correct: int = 0

    # Real-time A/B/C score-fusion hits.
    fuse_a_correct: int = 0
    fuse_b_correct: int = 0
    fuse_c_correct: int = 0
    fusion_correct: int = 0


@dataclass(frozen=True)
class RAEOnlyRuntimeConfig:
    dit_checkpoint: str
    dit_variant: str
    num_classes: int
    num_steps: int
    solver: str
    pair_chunk_size: int
    use_ema: bool
    hint_signal_ratio: Optional[float]
    stochastic_hint: bool
    pure_noise_hint: bool
    allow_condition_mismatch: bool
    amp: bool
    ot_pool_factor: int
    sinkhorn_reg: float
    sinkhorn_iter: int
    sk_weight_eps: float


def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fast_path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(os.path.expanduser(path))))



def scan_class_folders(root: str) -> Dict[str, List[str]]:
    """Read ``root/class_name/**/*image`` into a class pool."""
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
    hospital: str,
    part: str,
    *,
    eval_setting: str,
    n_way: int,
    n_shot: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str], List[str], str, str]:
    query_root = os.path.join(DATA_BASE, "indomain_support_76", hospital, part)
    if eval_setting == "test_train":
        support_root = os.path.join(DATA_BASE, f"{part}_1query", "train")
    elif eval_setting == "test_test":
        support_root = query_root
    else:
        raise ValueError(eval_setting)

    query_pool = scan_class_folders(query_root)
    support_pool = query_pool if eval_setting == "test_test" else scan_class_folders(support_root)

    if eval_setting == "test_test":
        # Need one query + K distinct supports from the same test class.
        candidate_labels = [
            c for c in sorted(query_pool)
            if len(query_pool[c]) >= n_shot + 1
        ]
        query_labels = list(candidate_labels)
        query_pool = {c: query_pool[c] for c in query_labels}
        support_pool = {c: support_pool[c] for c in candidate_labels}
    else:
        # REDUCED Test-Train:
        #   candidate space = query classes ∩ support classes.
        # Support-only distractor classes are removed.  We intentionally do NOT
        # require >= n_shot support images here: K-shot shortage is handled later
        # with sampling WITH replacement, matching the reduced M2 evaluator.
        candidate_labels = [
            c for c in sorted(set(query_pool) & set(support_pool))
            if len(query_pool[c]) >= 1 and len(support_pool[c]) >= 1
        ]
        query_labels = list(candidate_labels)
        query_pool = {c: query_pool[c] for c in candidate_labels}
        support_pool = {c: support_pool[c] for c in candidate_labels}

    if not query_labels:
        raise RuntimeError(f"{hospital}/{part}: no valid GT query class represented in support")
    if len(candidate_labels) < 2:
        raise RuntimeError(f"{hospital}/{part}: fewer than 2 candidate classes")
    if eval_setting != "test_train" and n_way > len(candidate_labels):
        print(
            f"[WARN] {hospital}/{part} requested {n_way}-way but only "
            f"{len(candidate_labels)} candidates; will use all available."
        )

    return query_pool, support_pool, candidate_labels, query_labels, query_root, support_root


def generate_episode_plans(
    query_pool: Dict[str, List[str]],
    support_pool: Dict[str, List[str]],
    candidate_labels: Sequence[str],
    query_labels: Sequence[str],
    *,
    n_way: int,
    n_shot: int,
    n_episodes: int,
    seed: int,
    eval_setting: str,
) -> List[EpisodePlan]:
    rng = random.Random(seed)
    candidate_labels = list(candidate_labels)
    query_labels = list(query_labels)
    n_way = min(int(n_way), len(candidate_labels))
    if n_way < 2:
        raise ValueError("effective n_way must be >=2")

    plans: List[EpisodePlan] = []
    for episode_id in range(n_episodes):
        query_label = rng.choice(query_labels)

        # Reduced Test-Train evaluates the complete shared class space.
        # Test-Test keeps the requested N-way behavior.
        if eval_setting == "test_train" or n_way >= len(candidate_labels):
            selected = list(candidate_labels)
        else:
            distractors = [c for c in candidate_labels if c != query_label]
            selected = [query_label] + rng.sample(distractors, n_way - 1)

        query_path = rng.choice(query_pool[query_label])
        query_key = fast_path_key(query_path)
        support_paths: Dict[str, List[str]] = {}
        for label in selected:
            # Always remove an identical path. This is mandatory in Test-Test and
            # harmless protection against accidental overlap in Test-Train.
            candidates = [p for p in support_pool[label] if fast_path_key(p) != query_key]
            if not candidates:
                raise RuntimeError(
                    f"episode={episode_id} class={label!r}: no non-query support remains"
                )
            if len(candidates) >= n_shot:
                support_paths[label] = rng.sample(candidates, n_shot)
            elif eval_setting == "test_train":
                support_paths[label] = rng.choices(candidates, k=n_shot)
            else:
                raise RuntimeError(
                    f"episode={episode_id} class={label!r} support={len(candidates)} < shot={n_shot}"
                )

        if SHUFFLE_CANDIDATE_SLOTS:
            rng.shuffle(selected)

        plans.append(EpisodePlan(
            episode_id=episode_id,
            candidate_labels=list(selected),
            query_label=query_label,
            query_path=query_path,
            support_paths={c: support_paths[c] for c in selected},
            seed=seed,
        ))
    return plans


# =============================================================================
# Frozen RAE / Flow runtime
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
    return transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])




# =============================================================================
# Rescue case visualization
# =============================================================================
SAVE_RESCUE_CASES = True
MAX_RESCUE_CASES = 3

def _safe_decode_save(rae, z, path):
    """Decode RAE latent trajectory state and save a valid visualization image.

    Handles common decoder output ranges: [-1,1], [0,1], [0,255].
    Also reports abnormal near-zero outputs to avoid silently saving black frames.
    """
    x = rae.decode(z).float().detach().cpu()
    xmin = float(x.min())
    xmax = float(x.max())

    if xmin < -0.05:
        # tanh decoder output
        x = (x + 1.0) / 2.0
    elif xmax > 2.0:
        # uint8-like decoder output
        x = x / 255.0
    elif xmin >= 0.0 and xmax <= 1.0:
        # already normalized
        pass
    else:
        # fallback normalization for unexpected decoder ranges
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)

    x = x.clamp(0, 1)
    if float(x.max()) < 1e-5:
        print(f"[WARN][RAE decode] almost-black trajectory frame: range=({xmin:.6g},{xmax:.6g}) path={path}")

    from torchvision.utils import save_image
    save_image(x, str(path))

def save_q2l_rescue_case(case_dir, rae, ep, query_z, post_states):
    import shutil
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ep.query_path, case_dir / "query.png")
    idx = 0
    for cls in ep.candidate_labels:
        for p in ep.support_paths[cls]:
            shutil.copy2(p, case_dir / f"support_{idx:02d}.png")
            idx += 1

    traj_dir = case_dir / "trajectory"
    traj_dir.mkdir(exist_ok=True)
    _safe_decode_save(rae, query_z[:1], traj_dir / "step00.png")
    for t in range(post_states.shape[1]):
        _safe_decode_save(rae, post_states[0, t:t+1], traj_dir / f"step{t+1:02d}.png")

    with open(case_dir / "case_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "episode_id": ep.episode_id,
            "query_label": ep.query_label,
            "query_path": ep.query_path,
            "support_paths": ep.support_paths
        }, f, ensure_ascii=False, indent=2)

def load_images(paths: Sequence[str], transform) -> torch.Tensor:
    out = []
    for path in paths:
        with Image.open(path) as im:
            out.append(transform(im.convert("RGB")))
    return torch.stack(out, dim=0)


@torch.inference_mode()
def encode_rae_chunked(rae: nn.Module, images: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunk_size = int(images.shape[0]) if chunk_size <= 0 else int(chunk_size)
    z = torch.cat([rae.encode(images[i:i+chunk_size]) for i in range(0, images.shape[0], chunk_size)], dim=0)
    if z.ndim != 4 or z.shape[1:] != (1024, 16, 16):
        raise RuntimeError(f"RAE latent expected (B,1024,16,16), got {tuple(z.shape)}")
    return z.float()


class FrozenFlowRuntime:
    DATASET_MEAN = (0.3464, 0.2280, 0.2228)
    DATASET_STD = (0.2520, 0.2128, 0.2093)

    def __init__(self, *, project_root: str, cfg: RAEOnlyRuntimeConfig, device: torch.device) -> None:
        if device.type != "cuda":
            raise RuntimeError("Q2L Flow evaluation requires CUDA")
        self.project_root = str(Path(project_root).resolve())
        self.device = device
        self.amp = bool(cfg.amp)
        self.image_size = IMAGE_SIZE

        from flow_models.dit_rae import create_dit_rae
        from timm.models.vision_transformer import VisionTransformer

        ckpt = torch.load(cfg.dit_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise TypeError("DiT checkpoint must be dict")
        self.condition_config = ckpt.get("condition_config") or {}
        saved_mode = self.condition_config.get("condition_mode")
        saved_size = self.condition_config.get("image_size")
        mismatch = saved_mode != "vision_hint_v1" or (saved_size is not None and int(saved_size) != IMAGE_SIZE)
        if mismatch and not cfg.allow_condition_mismatch:
            raise RuntimeError(
                f"DiT condition mismatch: saved={self.condition_config}, image_size={IMAGE_SIZE}"
            )

        self.hint_signal_ratio = float(
            self.condition_config.get("hint_signal_ratio", 0.10)
            if cfg.hint_signal_ratio is None else cfg.hint_signal_ratio
        )

        dit = create_dit_rae(
            variant=cfg.dit_variant,
            rae_latent_dim=1024,
            rae_spatial_size=16,
            cross_attn_dim=768,
            num_classes=cfg.num_classes,
            use_cross_attention=True,
        )
        if cfg.use_ema and isinstance(ckpt.get("ema_model"), dict):
            state = ckpt["ema_model"]
            self.loaded_dit_key = "ema_model"
        elif isinstance(ckpt.get("dit_model"), dict):
            state = ckpt["dit_model"]
            self.loaded_dit_key = "dit_model"
        else:
            state = ckpt
            self.loaded_dit_key = "checkpoint_root"
        missing, unexpected = dit.load_state_dict(state, strict=False)
        if missing:
            print(f"[Q2L-EVAL][WARN] DiT missing keys: {missing[:12]}")
        if unexpected:
            print(f"[Q2L-EVAL][WARN] DiT unexpected keys: {unexpected[:12]}")
        self.dit = dit.to(device).eval()
        self.dit.device = device
        for p in self.dit.parameters():
            p.requires_grad_(False)

        vision = VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=False, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        ).eval()
        endovit = os.path.join(self.project_root, "flow_matcher_otcfm/EndoViT/pytorch_model.bin")
        vision.load_state_dict(torch.load(endovit, map_location="cpu", weights_only=False), strict=False)
        if isinstance(ckpt.get("vision_model"), dict):
            vision.load_state_dict(ckpt["vision_model"], strict=False)
        self.vision = vision.to(device).eval()
        self.vision.device = device
        for p in self.vision.parameters():
            p.requires_grad_(False)
        del ckpt

    @torch.no_grad()
    def encode_target_vision(self, support_images_01: torch.Tensor) -> torch.Tensor:
        x = support_images_01.to(self.device, non_blocking=True).float()
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor(self.DATASET_MEAN, device=self.device, dtype=x.dtype).view(1,3,1,1)
        std = torch.tensor(self.DATASET_STD, device=self.device, dtype=x.dtype).view(1,3,1,1)
        x = (x - mean) / std
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
            return self.vision.forward_features(x).float()

    @torch.no_grad()
    def build_hint(self, support_images_01: torch.Tensor, *, stochastic: bool, pure_noise: bool, seed: Optional[int]) -> torch.Tensor:
        x = support_images_01.to(self.device, non_blocking=True).float()
        gray = (0.2989*x[:,0:1] + 0.5870*x[:,1:2] + 0.1140*x[:,2:3]).repeat(1,3,1,1)
        gray = gray * 2.0 - 1.0
        if seed is None:
            noise = torch.randn_like(gray) if stochastic or pure_noise else torch.zeros_like(gray)
        else:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            noise = torch.randn(gray.shape, generator=g, dtype=torch.float32).to(self.device)
            if not (stochastic or pure_noise):
                noise.zero_()
        if pure_noise:
            return noise
        a = self.hint_signal_ratio
        return a * gray + math.sqrt(max(1.0 - a*a, 0.0)) * noise


@torch.no_grad()
def integrate_post_states(
    flow: FrozenFlowRuntime,
    query_z: torch.Tensor,
    target_vision: torch.Tensor,
    hint: torch.Tensor,
    *,
    num_steps: int,
    solver: str,
) -> torch.Tensor:
    """Return all post-flow states z_1,...,z_T for each query/support pair.

    Shape: [B, T, 1024, 16, 16].  The initial state z_0 is the encoded query
    itself and is intentionally not duplicated here.
    """
    x = query_z.to(flow.device, non_blocking=True).float()
    target_vision = target_vision.to(flow.device, non_blocking=True)
    hint = hint.to(flow.device, non_blocking=True)
    patch, cls = target_vision[:, 1:, :], target_vision[:, 0, :]
    t_span = torch.linspace(0.0, 1.0, int(num_steps) + 1, device=flow.device)
    dt = 1.0 / float(num_steps)
    states = []
    for i in range(int(num_steps)):
        t = torch.full((x.shape[0],), float(t_span[i]), device=flow.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=flow.amp):
            dx1 = flow.dit(
                xt=x, timestep=t, class_labels=None, encoder_hidden_states=patch,
                hint=hint, text_embeds=None, image_embeds=cls,
            )["sample"]
            if solver == "euler":
                x = x + dx1.float() * dt
            else:
                xe = x + dx1.float() * dt
                tn = torch.full((x.shape[0],), float(t_span[i + 1]), device=flow.device)
                dx2 = flow.dit(
                    xt=xe, timestep=tn, class_labels=None, encoder_hidden_states=patch,
                    hint=hint, text_embeds=None, image_embeds=cls,
                )["sample"]
                x = x + 0.5 * (dx1.float() + dx2.float()) * dt
        states.append(x.detach())
    return torch.stack(states, dim=1)


@torch.no_grad()
def generate_post_states_for_pairs(
    flow: FrozenFlowRuntime,
    query_z_pairs: torch.Tensor,
    support_images: torch.Tensor,
    cfg: RAEOnlyRuntimeConfig,
    *,
    hint_seed: int,
) -> torch.Tensor:
    """Generate the same support-conditioned trajectory used by Query2Last.

    The endpoint is post_states[:, -1].  Q2Mono reuses every post state for
    distance evaluation and running-maximum filtering, so no second Flow pass is required.
    """
    outs = []
    chunk = max(int(cfg.pair_chunk_size), 1)
    for start in range(0, query_z_pairs.shape[0], chunk):
        end = min(start + chunk, query_z_pairs.shape[0])
        q = query_z_pairs[start:end]
        s = support_images[start:end]
        target_vision = flow.encode_target_vision(s)
        hint = flow.build_hint(
            s,
            stochastic=cfg.stochastic_hint,
            pure_noise=cfg.pure_noise_hint,
            seed=int(hint_seed + start * 1009),
        )
        outs.append(integrate_post_states(
            flow, q, target_vision, hint,
            num_steps=cfg.num_steps, solver=cfg.solver,
        ))
        del target_vision, hint
    return torch.cat(outs, dim=0)


# =============================================================================
# Explicit RAE-only runtime configuration (NO Metric checkpoint)
# =============================================================================

def build_rae_only_runtime_config() -> RAEOnlyRuntimeConfig:
    return RAEOnlyRuntimeConfig(
        dit_checkpoint=DIT_CHECKPOINT,
        dit_variant=DIT_VARIANT,
        num_classes=FLOW_NUM_CLASSES,
        num_steps=FLOW_NUM_STEPS,
        solver=FLOW_SOLVER,
        pair_chunk_size=FLOW_PAIR_CHUNK_SIZE,
        use_ema=FLOW_USE_EMA,
        hint_signal_ratio=FLOW_HINT_SIGNAL_RATIO,
        stochastic_hint=FLOW_STOCHASTIC_HINT,
        pure_noise_hint=FLOW_PURE_NOISE_HINT,
        allow_condition_mismatch=FLOW_ALLOW_CONDITION_MISMATCH,
        amp=FLOW_AMP,
        ot_pool_factor=RAE_OT_POOL_FACTOR,
        sinkhorn_reg=SINKHORN_REG,
        sinkhorn_iter=SINKHORN_ITER,
        sk_weight_eps=SK_WEIGHT_EPS,
    )


def build_experiment_metadata(cfg: RAEOnlyRuntimeConfig) -> Dict[str, Dict[str, Any]]:
    return {
        EXPERIMENT_NAME: {
            "metric_model": "disabled",
            "metric_checkpoint": None,
            "feature_space": "raw_RAE_spatial_tokens",
            "rae_token_geometry": "[B,1024,H,W] -> [B,H*W,1024]",
            "flow_checkpoint": cfg.dit_checkpoint,
        }
    }


def aggregate_kshot(pair_distance: torch.Tensor) -> torch.Tensor:
    # pair_distance: [way, shot]
    if SHOT_AGG == "mean":
        return pair_distance.mean(dim=-1)
    if SHOT_AGG == "min":
        return pair_distance.min(dim=-1).values
    if SHOT_AGG == "softmin":
        tau = float(SHOT_TAU)
        return -tau * (torch.logsumexp(-pair_distance / tau, dim=-1) - math.log(float(pair_distance.shape[-1])))
    raise ValueError(SHOT_AGG)


def score_fusion_predictions(
    score_dict: Dict[str, torch.Tensor],
    mode: str,
) -> int:
    """Flexible fusion over selected metric scores.

    The selected metrics are controlled by FUSION_METRICS.
    """
    names = [n for n in FUSION_METRICS if n in score_dict]
    if not names:
        raise RuntimeError("FUSION_METRICS selected no available scores")
    d_all = torch.stack([score_dict[n] for n in names], dim=0)

    if mode == "dist_sum":
        return int(d_all.sum(dim=0).argmin().item())
    if mode == "soft_vote":
        tau = float(SHOT_TAU)
        return int(torch.softmax(-d_all / tau, dim=-1).sum(dim=0).argmax().item())
    if mode == "global_min":
        return int(d_all.argmin().item() % d_all.shape[1])
    raise ValueError(f"Unknown fusion mode={mode}")


@torch.inference_mode()
def _rae_latent_to_tokens(z: torch.Tensor, cfg: RAEOnlyRuntimeConfig) -> torch.Tensor:
    """Convert frozen RAE maps to an empirical spatial-token distribution.

    RAE output [B,1024,H,W] is optionally spatially average-pooled and then
    rearranged to [B,N,1024], where N=H'*W'.  No learned metric projection is
    applied.  These raw tokens are the empirical samples used by Sinkhorn.
    """
    if z.ndim != 4:
        raise ValueError(f"RAE latent must be [B,C,H,W], got {tuple(z.shape)}")
    if z.shape[1] != 1024:
        raise ValueError(f"Expected RAE channel dimension 1024, got {z.shape[1]}")

    factor = int(cfg.ot_pool_factor)
    if factor < 1:
        raise ValueError(f"RAE_OT_POOL_FACTOR must be >=1, got {factor}")
    if factor > 1:
        if z.shape[-2] % factor != 0 or z.shape[-1] % factor != 0:
            raise ValueError(
                f"RAE spatial size {tuple(z.shape[-2:])} is not divisible by pool factor={factor}"
            )
        z = F.avg_pool2d(z.float(), kernel_size=factor, stride=factor)

    return z.float().flatten(2).transpose(1, 2).contiguous()


@torch.inference_mode()
def aligned_pair_distances(query_z: torch.Tensor, target_z: torch.Tensor, cfg: RAEOnlyRuntimeConfig) -> torch.Tensor:
    """Dynamic-SK directly on raw RAE spatial-token clouds; no learned metric projection."""
    q_raw = _rae_latent_to_tokens(query_z, cfg)
    target_raw = _rae_latent_to_tokens(target_z, cfg)

    if q_raw.shape[0] == 1 and target_raw.shape[0] > 1:
        q_raw = q_raw.expand(target_raw.shape[0], -1, -1).contiguous()
    elif q_raw.shape[0] != target_raw.shape[0]:
        raise ValueError(
            f"query/target batch mismatch: query={q_raw.shape[0]} target={target_raw.shape[0]}"
        )

    q = F.normalize(q_raw, p=2, dim=-1)
    target = F.normalize(target_raw, p=2, dim=-1)

    return aligned_sk_sinkhorn_distance(
        q, target,
        query_raw=q_raw,
        support_raw=target_raw,
        reg=cfg.sinkhorn_reg,
        n_iter=cfg.sinkhorn_iter,
        weight_eps=cfg.sk_weight_eps,
        return_transport=False,
    ).distance.float()


@torch.inference_mode()
def q2mono_pair_distances(
    query_z: torch.Tensor,
    post_states: torch.Tensor,
    direct_distance: torch.Tensor,
    cfg: RAEOnlyRuntimeConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Q2Mono scores and raw Query-to-Step distances for every pair.

    Q2Mono follows the reference rule used in the final ablation:
      1) compute Query-to-Step Dynamic-SK distances for z_1,...,z_T;
      2) scan from early to late and retain a step only when its distance is
         strictly larger than all earlier retained values (strict running max);
      3) append the real Query-to-Support Direct distance;
      4) average the retained trajectory distances together with Direct.

    The original query latent z_0 is not treated as a trajectory step and is
    not part of the monotonic filtering.  This is the RAE-only/no-Metric-Model
    version: every distance is computed directly in the frozen RAE latent space.

    Returns:
        q2mono_distance: [P], Q2Mono score for each query/support pair.
        per_step:        [P,T], raw Query-to-Step distances for diagnostics.
    """
    if post_states.ndim != 5:
        raise ValueError(f"post_states must be [P,T,C,H,W], got {tuple(post_states.shape)}")

    direct_distance = direct_distance.reshape(-1)
    if direct_distance.numel() != post_states.shape[0]:
        raise ValueError(
            f"direct_distance must contain one value per pair: "
            f"got {direct_distance.numel()} for P={post_states.shape[0]}"
        )

    step_distances = []
    for t in range(post_states.shape[1]):
        step_distances.append(aligned_pair_distances(query_z, post_states[:, t], cfg))
    per_step = torch.stack(step_distances, dim=-1)

    outputs = []
    for pair_idx in range(per_step.shape[0]):
        kept = []
        running_max = -float("inf")
        for d in per_step[pair_idx]:
            value = float(d.item())
            if value > running_max:
                kept.append(d)
                running_max = value

        # Direct participates in the final average, but not in the monotonic
        # filtering over post-flow states.
        kept.append(direct_distance[pair_idx])
        outputs.append(torch.stack(kept).mean())

    return torch.stack(outputs), per_step


def ci95_from_binary(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    p = correct / total
    return 1.96 * math.sqrt(max(p * (1.0-p), 0.0) / total)


def finalize_branch(acc: EvalAccumulator) -> Dict[str, float]:
    n = max(acc.total, 1)
    p = acc.correct / n
    return {
        "correct": int(acc.correct),
        "total": int(acc.total),
        "accuracy": float(p),
        "accuracy_percent": float(100*p),
        "ci95_percent": float(100*ci95_from_binary(acc.correct, acc.total)),
        "positive_class_distance": float(acc.pos_sum/n),
        "nearest_negative_class_distance": float(acc.neg_sum/n),
        "class_margin": float(acc.margin_sum/n),
        "mean_positive_pair_distance": float(acc.pair_pos_sum/n),
        "nearest_negative_pair_distance": float(acc.pair_neg_sum/n),
    }


def update_branch(acc: EvalAccumulator, pair_d: torch.Tensor, class_d: torch.Tensor, gt_idx: int, correct: bool) -> None:
    n_way = int(class_d.numel())
    neg_mask = torch.ones(n_way, dtype=torch.bool, device=class_d.device)
    neg_mask[gt_idx] = False
    pos = float(class_d[gt_idx].item())
    neg = float(class_d[neg_mask].min().item())
    pair_pos = float(pair_d[gt_idx].mean().item())
    pair_neg_mask = torch.ones_like(pair_d, dtype=torch.bool)
    pair_neg_mask[gt_idx] = False
    pair_neg = float(pair_d.masked_select(pair_neg_mask).min().item())
    acc.total += 1
    acc.correct += int(correct)
    acc.pos_sum += pos
    acc.neg_sum += neg
    acc.margin_sum += (neg - pos)
    acc.pair_pos_sum += pair_pos
    acc.pair_neg_sum += pair_neg


def finalize_dual(acc: DualEvalAccumulator) -> Dict[str, Any]:
    direct = finalize_branch(acc.direct)
    flow = finalize_branch(acc.flow)
    q2mono = finalize_branch(acc.q2mono)
    total = max(acc.direct.total, 1)
    oracle_acc = acc.oracle_correct / total
    oracle_q2mono_acc = acc.oracle_direct_q2mono_correct / total
    oracle_all_acc = acc.oracle_all_correct / total
    return {
        "direct_accuracy_percent": direct["accuracy_percent"],
        "direct_ci95_percent": direct["ci95_percent"],
        "flow_accuracy_percent": flow["accuracy_percent"],
        "flow_ci95_percent": flow["ci95_percent"],
        "flow_gain_vs_direct_pp": flow["accuracy_percent"] - direct["accuracy_percent"],
        "q2mono_accuracy_percent": q2mono["accuracy_percent"],
        "q2mono_ci95_percent": q2mono["ci95_percent"],
        "q2mono_gain_vs_direct_pp": q2mono["accuracy_percent"] - direct["accuracy_percent"],
        "q2mono_gain_vs_query2last_pp": q2mono["accuracy_percent"] - flow["accuracy_percent"],
        "oracle_direct_flow_accuracy_percent": 100.0 * oracle_acc,
        "oracle_gain_vs_direct_pp": 100.0 * oracle_acc - direct["accuracy_percent"],
        "oracle_direct_q2mono_accuracy_percent": 100.0 * oracle_q2mono_acc,
        "oracle_flow_q2mono_accuracy_percent": 100.0 * (acc.oracle_flow_q2mono_correct / total),
        "oracle_all_three_accuracy_percent": 100.0 * oracle_all_acc,
        "fuse_a_accuracy_percent": 100.0 * acc.fuse_a_correct / total,
        "fuse_b_accuracy_percent": 100.0 * acc.fuse_b_correct / total,
        "fuse_c_accuracy_percent": 100.0 * acc.fuse_c_correct / total,
        "fuse_a_gain_vs_direct_pp": 100.0 * acc.fuse_a_correct / total - direct["accuracy_percent"],
        "fuse_b_gain_vs_direct_pp": 100.0 * acc.fuse_b_correct / total - direct["accuracy_percent"],
        "fuse_c_gain_vs_direct_pp": 100.0 * acc.fuse_c_correct / total - direct["accuracy_percent"],
        "fuse_a_correct": int(acc.fuse_a_correct),
        "fuse_b_correct": int(acc.fuse_b_correct),
        "fuse_c_correct": int(acc.fuse_c_correct),
        "fusion_accuracy_percent": 100.0 * acc.fusion_correct / total,
        "fusion_correct": int(acc.fusion_correct),
        "fusion_metrics": list(FUSION_METRICS),
        "fusion_modes": list(FUSION_MODES),
        "direct_correct": int(acc.direct.correct),
        "flow_correct": int(acc.flow.correct),
        "q2mono_correct": int(acc.q2mono.correct),
        "total": int(acc.direct.total),
        "both_correct": int(acc.both_correct),
        "direct_only": int(acc.direct_only),
        "flow_only": int(acc.flow_only),
        "both_wrong": int(acc.both_wrong),
        "flow_rescue": int(acc.flow_only),
        "flow_degradation": int(acc.direct_only),
        "flow_net_rescue": int(acc.flow_only - acc.direct_only),
        "prediction_changed": int(acc.prediction_changed),
        "prediction_changed_rate_percent": 100.0 * acc.prediction_changed / total,
        "direct_q2mono_both_correct": int(acc.direct_q2mono_both_correct),
        "direct_only_vs_q2mono": int(acc.direct_only_vs_q2mono),
        "q2mono_only_vs_direct": int(acc.q2mono_only_vs_direct),
        "direct_q2mono_both_wrong": int(acc.direct_q2mono_both_wrong),
        "q2mono_rescue_vs_direct": int(acc.q2mono_only_vs_direct),
        "q2mono_degradation_vs_direct": int(acc.direct_only_vs_q2mono),
        "q2mono_net_rescue_vs_direct": int(acc.q2mono_only_vs_direct - acc.direct_only_vs_q2mono),
        "q2mono_prediction_changed": int(acc.q2mono_prediction_changed),
        "q2mono_prediction_changed_rate_percent": 100.0 * acc.q2mono_prediction_changed / total,
        "direct": direct,
        "flow_q2l": flow,
        "q2mono": q2mono,
    }


@torch.inference_mode()
def evaluate_task(
    *, hospital: str, part: str, rae: nn.Module, flow: FrozenFlowRuntime,
    metrics: Dict[str, Any], metadata: Dict[str, Dict[str, Any]],
    cfg: RAEOnlyRuntimeConfig, device: torch.device, transform,
) -> Dict[str, Any]:
    requested_way = int(ORGAN_N_WAY.get(part, 8))
    query_pool, support_pool, candidate_labels, query_labels, query_root, support_root = prepare_task_pools(
        hospital, part, eval_setting=EVAL_SETTING, n_way=requested_way, n_shot=N_SHOT,
    )
    episode_n_way = len(candidate_labels) if EVAL_SETTING == "test_train" else min(requested_way, len(candidate_labels))

    task_seed = SEED + sum(ord(c) for c in hospital)*1009 + sum(ord(c) for c in part)*9176
    rescue_saved = 0

    plans = generate_episode_plans(
        query_pool, support_pool, candidate_labels, query_labels,
        n_way=episode_n_way, n_shot=N_SHOT, n_episodes=EPISODES_PER_TASK,
        seed=task_seed, eval_setting=EVAL_SETTING,
    )

    print("="*126)
    print(f"[Q2L Direct + Query2Last + Q2Mono Eval] {hospital} / {part} | {episode_n_way}-way {N_SHOT}-shot 1-query | episodes={len(plans)}")
    print(f"setting                     = {EVAL_SETTING}")
    print(f"query_root                  = {query_root}")
    print(f"support_root                = {support_root}")
    print(f"candidate support classes   = {len(candidate_labels)}")
    print(f"valid GT query classes       = {len(query_labels)}")
    if EVAL_SETTING == "test_train":
        print("candidate-space policy       = REDUCED (query ∩ support shared classes)")
    else:
        print("query excluded from support  = STRICT")
    print("Direct evidence              = raw-RAE Query-to-Support Dynamic-SK")
    print("Query2Last evidence          = raw-RAE Query-to-Last Dynamic-SK after frozen Flow")
    print("Q2Mono evidence              = raw-RAE strict running-max Query-to-Step over z_1..z_T plus Direct")
    print(f"Flow runtime                 = {cfg.num_steps} steps / {cfg.solver} / chunk={cfg.pair_chunk_size}")
    print(f"raw-RAE SK                   = pool={cfg.ot_pool_factor} reg={cfg.sinkhorn_reg} iter={cfg.sinkhorn_iter} eps={cfg.sk_weight_eps}")
    print(f"K-shot aggregation           = {SHOT_AGG} tau={SHOT_TAU if SHOT_AGG=='softmin' else 'N/A'}")
    print("metric model                 = DISABLED (zero Metric checkpoint)")
    print(f"flow checkpoint              = {cfg.dit_checkpoint}")
    print("RAE token geometry           = [B,1024,H,W] -> [B,H*W,1024]")
    print("="*126)

    accs = {name: DualEvalAccumulator() for name in metrics}
    episode_records = []
    t0 = time.time()

    for ep_idx, ep in enumerate(plans):
        support_flat: List[str] = []
        for c in ep.candidate_labels:
            support_flat.extend(ep.support_paths[c])

        support_images = load_images(support_flat, transform).to(device, non_blocking=True)
        query_image = load_images([ep.query_path], transform).to(device, non_blocking=True)

        # Encode the REAL supports and query exactly once. Direct uses support_z;
        # Flow uses the same query_z but replaces each support target with z_T(q|support).
        all_images = torch.cat([support_images, query_image], dim=0)
        all_z = encode_rae_chunked(rae, all_images, RAE_CHUNK_SIZE)
        support_z = all_z[:-1]
        query_z = all_z[-1:]

        q_pairs = query_z.expand(len(support_flat), -1, -1, -1).contiguous()
        hint_seed = int(task_seed + 8_000_003 + ep_idx * 97_003)
        post_states = generate_post_states_for_pairs(flow, q_pairs, support_images, cfg, hint_seed=hint_seed)
        last_z = post_states[:, -1]

        gt_idx = ep.candidate_labels.index(ep.query_label)
        ep_rec = {
            "episode_id": ep.episode_id,
            "query_label": ep.query_label,
            "query_path": ep.query_path,
            "candidate_labels": list(ep.candidate_labels),
        }
        if SAVE_EPISODE_DETAILS:
            ep_rec["support_paths_by_class"] = {c: list(ep.support_paths[c]) for c in ep.candidate_labels}
            ep_rec["experiment_results"] = {}

        for name, _ in metrics.items():
            direct_pair = aligned_pair_distances(query_z, support_z, cfg).reshape(episode_n_way, N_SHOT)
            flow_pair = aligned_pair_distances(query_z, last_z, cfg).reshape(episode_n_way, N_SHOT)
            q2mono_pair_flat, q2mono_per_step_flat = q2mono_pair_distances(
                query_z, post_states, direct_pair.reshape(-1), cfg
            )
            q2mono_pair = q2mono_pair_flat.reshape(episode_n_way, N_SHOT)
            q2mono_per_step = q2mono_per_step_flat.reshape(episode_n_way, N_SHOT, cfg.num_steps)

            direct_class = aggregate_kshot(direct_pair)
            flow_class = aggregate_kshot(flow_pair)
            q2mono_class = aggregate_kshot(q2mono_pair)

            direct_idx = int(direct_class.argmin().item())
            flow_idx = int(flow_class.argmin().item())
            q2mono_idx = int(q2mono_class.argmin().item())
            direct_label = ep.candidate_labels[direct_idx]
            flow_label = ep.candidate_labels[flow_idx]
            q2mono_label = ep.candidate_labels[q2mono_idx]
            direct_ok = direct_label == ep.query_label
            flow_ok = flow_label == ep.query_label
            q2mono_ok = q2mono_label == ep.query_label

            a = accs[name]
            update_branch(a.direct, direct_pair, direct_class, gt_idx, direct_ok)
            update_branch(a.flow, flow_pair, flow_class, gt_idx, flow_ok)
            update_branch(a.q2mono, q2mono_pair, q2mono_class, gt_idx, q2mono_ok)

            # Existing Direct vs Query2Last bookkeeping.
            a.both_correct += int(direct_ok and flow_ok)
            a.direct_only += int(direct_ok and not flow_ok)
            a.flow_only += int((not direct_ok) and flow_ok)
            a.both_wrong += int((not direct_ok) and (not flow_ok))
            a.prediction_changed += int(direct_label != flow_label)
            a.oracle_correct += int(direct_ok or flow_ok)

            # Direct vs Q2Mono bookkeeping.
            a.direct_q2mono_both_correct += int(direct_ok and q2mono_ok)
            a.direct_only_vs_q2mono += int(direct_ok and not q2mono_ok)
            a.q2mono_only_vs_direct += int((not direct_ok) and q2mono_ok)
            a.direct_q2mono_both_wrong += int((not direct_ok) and (not q2mono_ok))
            a.q2mono_prediction_changed += int(direct_label != q2mono_label)
            a.oracle_direct_q2mono_correct += int(direct_ok or q2mono_ok)
            a.oracle_flow_q2mono_correct += int(flow_ok or q2mono_ok)
            a.oracle_all_correct += int(direct_ok or flow_ok or q2mono_ok)

            # ---- Three fusion rules over the selected fusion objects ----
            score_dict = {"direct": direct_class, "flow": flow_class, "q2mono": q2mono_class}
            fuse_preds = {}
            for fm in FUSION_MODES:
                idx = score_fusion_predictions(score_dict, fm)
                fuse_preds[fm] = idx
            a.fuse_a_correct += int(ep.candidate_labels[fuse_preds["dist_sum"]] == ep.query_label)
            a.fuse_b_correct += int(ep.candidate_labels[fuse_preds["soft_vote"]] == ep.query_label)
            a.fuse_c_correct += int(ep.candidate_labels[fuse_preds["global_min"]] == ep.query_label)
            a.fusion_correct += int(ep.candidate_labels[fuse_preds["dist_sum"]] == ep.query_label)
            fusion_idx = int(fuse_preds["dist_sum"])
            if SAVE_RESCUE_CASES and rescue_saved < MAX_RESCUE_CASES:
                fusion_pred = ep.candidate_labels[fuse_preds["dist_sum"]]
                if (not direct_ok) and (fusion_pred == ep.query_label):
                    save_q2l_rescue_case(
                        Path(CASE_SAVE_ROOT) / hospital / part / f"case_{rescue_saved:03d}",
                        rae, ep, query_z[:1], post_states
                    )
                    rescue_saved += 1
                    print(f"[SAVE RESCUE CASE] {hospital}/{part}: {rescue_saved}")

            if SAVE_EPISODE_DETAILS:
                ep_rec["experiment_results"][name] = {
                    "direct_pred_label": direct_label,
                    "flow_pred_label": flow_label,
                    "q2mono_pred_label": q2mono_label,
                    "direct_correct": bool(direct_ok),
                    "flow_correct": bool(flow_ok),
                    "q2mono_correct": bool(q2mono_ok),
                    "flow_rescue": bool((not direct_ok) and flow_ok),
                    "flow_degradation": bool(direct_ok and (not flow_ok)),
                    "q2mono_rescue_vs_direct": bool((not direct_ok) and q2mono_ok),
                    "q2mono_degradation_vs_direct": bool(direct_ok and (not q2mono_ok)),
                    "oracle_direct_flow_correct": bool(direct_ok or flow_ok),
                    "oracle_direct_q2mono_correct": bool(direct_ok or q2mono_ok),
                    "oracle_all_three_correct": bool(direct_ok or flow_ok or q2mono_ok),
                    "fusion_pred_label": ep.candidate_labels[fusion_idx],
                    "fusion_correct": bool(ep.candidate_labels[fusion_idx] == ep.query_label),
                    "prediction_changed": bool(direct_label != flow_label),
                    "q2mono_prediction_changed": bool(direct_label != q2mono_label),
                    "direct_class_distances": {c: float(direct_class[i].item()) for i,c in enumerate(ep.candidate_labels)},
                    "flow_q2l_class_distances": {c: float(flow_class[i].item()) for i,c in enumerate(ep.candidate_labels)},
                    "q2mono_class_distances": {c: float(q2mono_class[i].item()) for i,c in enumerate(ep.candidate_labels)},
                    "direct_pair_distances_by_class": {c: [float(v) for v in direct_pair[i].cpu().tolist()] for i,c in enumerate(ep.candidate_labels)},
                    "flow_q2l_pair_distances_by_class": {c: [float(v) for v in flow_pair[i].cpu().tolist()] for i,c in enumerate(ep.candidate_labels)},
                    "q2mono_pair_distances_by_class": {c: [float(v) for v in q2mono_pair[i].cpu().tolist()] for i,c in enumerate(ep.candidate_labels)},
                    "q2mono_per_step_pair_distances_by_class": {
                        c: [[float(v) for v in shot_steps] for shot_steps in q2mono_per_step[i].cpu().tolist()]
                        for i, c in enumerate(ep.candidate_labels)
                    },
                }

        if SAVE_EPISODE_DETAILS:
            episode_records.append(ep_rec)
        if LOG_EVERY > 0 and (ep_idx+1) % LOG_EVERY == 0:
            msg = " | ".join(
                f"{n}:D={100*a.direct.correct/max(a.direct.total,1):.2f}%/Q2L={100*a.flow.correct/max(a.flow.total,1):.2f}%/Q2Mono={100*a.q2mono.correct/max(a.q2mono.total,1):.2f}%"
                for n,a in accs.items()
            )
            print(f"  [{ep_idx+1:04d}/{len(plans):04d}] {msg}")

        del support_images, query_image, all_images, all_z, support_z, query_z, q_pairs, post_states, last_z

    results = {name: finalize_dual(a) for name,a in accs.items()}
    out = {
        "hospital": hospital,
        "part": part,
        "eval_setting": EVAL_SETTING,
        "query_root": query_root,
        "support_root": support_root,
        "candidate_support_classes": candidate_labels,
        "valid_gt_query_classes": query_labels,
        "protocol": {
            "n_way": episode_n_way,
            "n_shot": N_SHOT,
            "n_query": N_QUERY,
            "episodes": EPISODES_PER_TASK,
            "seed": task_seed,
            "candidate_space": "query_support_intersection_reduced" if EVAL_SETTING == "test_train" else "same_test_pool",
            "query_excluded_from_support": True,
            "support_sampling": "without_replacement_if_enough_else_with_replacement" if EVAL_SETTING == "test_train" else "without_replacement",
            "matched_episode_for_all_three_scores": True,
            "direct_evidence": "raw-RAE Query-to-Support Dynamic-SK",
            "flow_evidence": "raw-RAE Query-to-Last Dynamic-SK",
            "q2mono_evidence": "raw-RAE mean of strict running-maximum Query-to-Step distances over post states z_1..z_T plus Direct distance; z0 excluded",
        },
        "rae_only_runtime": {
            "dit_checkpoint": cfg.dit_checkpoint,
            "dit_variant": cfg.dit_variant,
            "num_classes": cfg.num_classes,
            "num_steps": cfg.num_steps,
            "solver": cfg.solver,
            "pair_chunk_size": cfg.pair_chunk_size,
            "use_ema": cfg.use_ema,
            "hint_signal_ratio": cfg.hint_signal_ratio,
            "stochastic_hint": cfg.stochastic_hint,
            "pure_noise_hint": cfg.pure_noise_hint,
            "ot_pool_factor": cfg.ot_pool_factor,
            "sinkhorn_reg": cfg.sinkhorn_reg,
            "sinkhorn_iter": cfg.sinkhorn_iter,
            "sk_weight_eps": cfg.sk_weight_eps,
            "kshot_aggregation": SHOT_AGG,
            "shot_tau": SHOT_TAU if SHOT_AGG == "softmin" else None,
        },
        "experiment_metadata": metadata,
        "metrics": results,
        "elapsed_sec": time.time() - t0,
    }
    if SAVE_EPISODE_DETAILS:
        out["episodes"] = episode_records
    return out


def save_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "experiment", "hospital", "part", "eval_setting", "n_way", "n_shot", "episodes",
        "direct_accuracy_percent", "direct_ci95_percent",
        "flow_accuracy_percent", "flow_ci95_percent", "flow_gain_vs_direct_pp",
        "q2mono_accuracy_percent", "q2mono_ci95_percent",
        "q2mono_gain_vs_direct_pp", "q2mono_gain_vs_query2last_pp",
        "oracle_direct_flow_accuracy_percent", "oracle_gain_vs_direct_pp",
        "oracle_direct_q2mono_accuracy_percent",
        "oracle_flow_q2mono_accuracy_percent",
        "oracle_all_three_accuracy_percent",
        "direct_correct", "flow_correct", "q2mono_correct", "total",
        "both_correct", "direct_only", "flow_only", "both_wrong",
        "flow_rescue", "flow_degradation", "flow_net_rescue",
        "prediction_changed", "prediction_changed_rate_percent",
        "q2mono_rescue_vs_direct", "q2mono_degradation_vs_direct",
        "q2mono_net_rescue_vs_direct", "q2mono_prediction_changed",
        "q2mono_prediction_changed_rate_percent",
        "fuse_a_accuracy_percent", "fuse_b_accuracy_percent", "fuse_c_accuracy_percent",
        "fuse_a_gain_vs_direct_pp", "fuse_b_gain_vs_direct_pp", "fuse_c_gain_vs_direct_pp",
        "fuse_a_correct", "fuse_b_correct", "fuse_c_correct",
        "flow_checkpoint_path",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def main() -> None:
    if N_QUERY != 1 or N_SHOT < 1 or EPISODES_PER_TASK < 1:
        raise ValueError("This evaluator requires 1-query, shot>=1, episodes>=1")
    setup_seed(SEED)
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Q2L Direct+Flow evaluation requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = build_rae_only_runtime_config()
    metrics = {EXPERIMENT_NAME: None}
    metadata = build_experiment_metadata(cfg)
    if cfg.ot_pool_factor < 1:
        raise ValueError("RAE_OT_POOL_FACTOR must be >=1")
    if not os.path.isfile(cfg.dit_checkpoint):
        raise FileNotFoundError(f"DiT checkpoint not found: {cfg.dit_checkpoint}")

    print("="*126)
    print("[Clean RAE-only Direct + Query2Last + Q2Mono Episodic Evaluation | ZERO Metric Model]")
    print(f"device      = {device}")
    print(f"hospitals   = {HOSPITALS}")
    print(f"parts       = {PARTS}")
    print(f"setting     = {EVAL_SETTING}")
    print(f"result_dir  = {RESULT_DIR}")
    print("="*126)

    rae = build_rae(PROJECT_ROOT, device)
    flow = FrozenFlowRuntime(project_root=PROJECT_ROOT, cfg=cfg, device=device)
    transform = build_transform(IMAGE_SIZE)

    summary_rows: List[Dict[str, Any]] = []
    summary_nested: Dict[str, Any] = {}
    for part in PARTS:
        for hospital in HOSPITALS:
            result = evaluate_task(
                hospital=hospital, part=part, rae=rae, flow=flow,
                metrics=metrics, metadata=metadata, cfg=cfg,
                device=device, transform=transform,
            )
            out_json = os.path.join(
                RESULT_DIR, part, f"{hospital}_{part}_{EVAL_SETTING}_q2l_direct_q2last_q2mono.json"
            )
            save_json(out_json, result)
            summary_nested.setdefault(part, {})[hospital] = result["metrics"]

            print(f"\n[RESULT] {hospital} / {part}")
            for name, m in result["metrics"].items():
                print(
                    f"  {name:<10} "
                    f"Direct={m['direct_accuracy_percent']:.2f}% "
                    f"Q2L={m['flow_accuracy_percent']:.2f}% "
                    f"Q2Mono={m['q2mono_accuracy_percent']:.2f}% "
                    f"Oracle3={m['oracle_all_three_accuracy_percent']:.2f}% "
                    f"R/D/Net={m['flow_rescue']}/{m['flow_degradation']}/{m['flow_net_rescue']}"
                )
                summary_rows.append({
                    "experiment": name,
                    "hospital": hospital,
                    "part": part,
                    "eval_setting": EVAL_SETTING,
                    "n_way": result["protocol"]["n_way"],
                    "n_shot": N_SHOT,
                    "episodes": EPISODES_PER_TASK,
                    **{k: v for k, v in m.items() if not isinstance(v, dict)},
                    "flow_checkpoint_path": cfg.dit_checkpoint,
                })

    save_summary_csv(os.path.join(RESULT_DIR, "summary.csv"), summary_rows)
    save_json(os.path.join(RESULT_DIR, "summary.json"), {
        "config": {
            "hospitals": HOSPITALS,
            "parts": PARTS,
            "eval_setting": EVAL_SETTING,
            "n_shot": N_SHOT,
            "n_query": N_QUERY,
            "episodes_per_task": EPISODES_PER_TASK,
            "metric_model": "disabled",
            "metric_checkpoint": None,
            "flow_checkpoint": cfg.dit_checkpoint,
            "rae_token_geometry": "[B,1024,H,W] -> [B,H*W,1024]",
            "direct_evidence": "raw-RAE Query-to-Support Dynamic-SK",
            "flow_evidence": "raw-RAE Query-to-Last Dynamic-SK",
            "q2mono_evidence": "raw-RAE mean of strict running-maximum Query-to-Step distances over post states z_1..z_T plus Direct distance; z0 excluded",
        },
        "results": summary_nested,
    })
    print("\n" + "="*126)
    print("[DONE]")
    print(f"summary CSV  = {os.path.join(RESULT_DIR, 'summary.csv')}")
    print(f"summary JSON = {os.path.join(RESULT_DIR, 'summary.json')}")
    print("="*126)


if __name__ == "__main__":
    main()
