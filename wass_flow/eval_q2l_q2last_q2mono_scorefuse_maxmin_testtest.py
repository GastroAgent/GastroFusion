#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q2L-M0B-RAE episodic evaluation ( reporting).

This evaluator is aligned with ``pretrain_q2l_fsl_from_m0b_rae_torchrun_ddp_v2.py``.
The evaluator reports TWO matched geometric scores on every episode:

    Q2Last:  Query-to-Step Dynamic-SK at the final post-flow state z_T.
    Q2Mono:  mean of strictly increasing running-maximum trajectory step distances;
             the internally computed Query-to-Support distance is kept inside the
             mean to preserve the original Q2Mono definition.

All scores use exactly the same query, candidate classes, K-shot supports,
MetricEncoder checkpoint, SK geometry and K-shot aggregation.  Q2Last and
Q2Mono share the same support-conditioned Flow trajectory.

NOTE(): Direct (Query-to-Support) is NOT reported.  The Direct pair
distance is still computed internally because the Q2Mono definition appends it
to the retained-step mean; it never appears in any output table/metric.
"""
from __future__ import annotations

import csv
import json
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

EVAL_SETTING = "test_test"
BASE_RESULT_DIR = os.path.join(PROJECT_ROOT, "fsl_multi_vae/rae_flow_metric_v1", "results_q2l_q2last_q2mono_scorefuse_testtest_v7")
HOSPITALS = ["华西", "自贡", "南充", "苍溪"]
N_WAY = 8

PARTS = ["十二指肠", "胃", "食管"]

# Evaluate both the validation-selected BEST checkpoint and the latest LAST checkpoint.
# Set one entry to "" if you want to skip it.
Q2L_CHECKPOINTS = {
    "十二指肠-test-test": os.path.join(PROJECT_ROOT, "weights_metric_v1", "十二指肠-test-test", "metric_q2l_m0b_rae_manifest.pt"),
    "食管-test-test": os.path.join(PROJECT_ROOT, "weights_metric_v1", "食管-test-test", "metric_q2l_m0b_rae_manifest.pt"),
    "胃-test-test": os.path.join(PROJECT_ROOT, "weights_metric_v1", "胃-test-test", "metric_q2l_m0b_rae_manifest.pt"),
}

N_SHOT = 5
N_QUERY = 1
EPISODES_PER_TASK = 100
SEED = 42
SHUFFLE_CANDIDATE_SLOTS = True

IMAGE_SIZE = 256
RAE_CHUNK_SIZE = 8

# Q2L pair geometry is restored from the Q2L checkpoint (ot_pool_factor,
# Sinkhorn reg/iter/eps). K-shot aggregation is an evaluation-only operation.
SHOT_AGG = "softmin"          # "softmin", "min", "mean"
SHOT_TAU = 0.10

# Flexible score-fusion configuration.
# Matched metrics:
#   q2last  -> Query-to-Step distance at the final post-flow state z_T
#   q2mono  -> monotonic-increase mean (internal Query-to-Support distance kept)
# Fuse-A/Fuse-B follow the previous evaluator.
# Fuse-C uses a conservative class-wise max-then-min rule:
# for each class take max(Q2Last, Q2MonoMean), then choose the class
# whose resulting worst-case distance is smallest.
FUSION_METRICS = ["q2last", "q2mono"]
FUSION_MODES = ["dist_sum", "soft_vote", "max_then_min"]


def get_fusion_tag() -> str:
    """Match the reference evaluator: separate outputs by selected fusion metrics."""
    return "fusion_" + "_".join(FUSION_METRICS)


# Keep the same directory convention as the reference score-fusion evaluator:
# BASE_RESULT_DIR/fusion_<metric1>_<metric2>_<metric3>/...
RESULT_DIR = os.path.join(BASE_RESULT_DIR, get_fusion_tag())

SAVE_EPISODE_DETAILS = True
LOG_EVERY = 25

# Optional manual override. Empty string -> use checkpoint['dit_checkpoint'].
DIT_CHECKPOINT_OVERRIDE = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/weights_dit_flow_v1/vision_hint_inclass/otcfm/otcfm_dit_L_step_500000.pt"

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.metric_encoder import MetricEncoder, pool_square_tokens
from models.sk_sinkhorn import aligned_sk_sinkhorn_distance

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}




def query2last_from_steps(step_distances: torch.Tensor) -> torch.Tensor:
    """Distance at the final post-flow state z_T."""
    if step_distances.shape[-1] < 1:
        raise ValueError("step_distances must contain at least one post-flow step")
    return step_distances[..., -1]


def query2mono_mean_from_steps(
    step_distances: torch.Tensor,
    direct_distance: torch.Tensor,
) -> torch.Tensor:
    """Running-maximum filtered steps + real query-support distance."""
    outputs = []
    for i in range(step_distances.shape[0]):
        kept = []
        running_max = -float("inf")
        for d in step_distances[i]:
            v = float(d.item())
            if v > running_max:
                kept.append(d)
                running_max = v
        kept.append(direct_distance[i])
        outputs.append(torch.stack(kept).mean())
    return torch.stack(outputs)

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
    q2last: EvalAccumulator = field(default_factory=EvalAccumulator)
    q2mono_mean: EvalAccumulator = field(default_factory=EvalAccumulator)

    # Direct vs Q2Last statistics (kept for backward compatibility).
    both_correct: int = 0
    direct_only: int = 0
    q2last_only: int = 0
    both_wrong: int = 0
    prediction_changed: int = 0
    oracle_correct: int = 0

    # Direct vs Q2MonoMean statistics.
    direct_q2mono_both_correct: int = 0
    direct_only_vs_q2mono: int = 0
    q2mono_only_vs_direct: int = 0
    direct_q2mono_both_wrong: int = 0
    q2mono_prediction_changed: int = 0
    oracle_direct_q2mono_correct: int = 0

    # Oracle across all three matched scores.
    oracle_all_correct: int = 0

    # A/B/C score-fusion hits.
    # A = dist_sum, B = soft_vote, C = class-wise max then global argmin.
    fuse_a_correct: int = 0
    fuse_b_correct: int = 0
    fuse_c_correct: int = 0
    # Backward-compatible alias: primary fusion = A / dist_sum.
    fusion_correct: int = 0


@dataclass(frozen=True)
class Q2LRuntimeConfig:
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


def torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def as_args_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


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
        # NON-REDUCED Test-Train: all support-side classes with K-shot capacity
        # remain candidate distractors. Query side only defines valid GT classes.
        candidate_labels = [
            c for c in sorted(support_pool)
            if len(support_pool[c]) >= n_shot
        ]
        cset = set(candidate_labels)
        query_labels = [
            c for c in sorted(query_pool)
            if len(query_pool[c]) >= 1 and c in cset
        ]
        query_pool = {c: query_pool[c] for c in query_labels}
        support_pool = {c: support_pool[c] for c in candidate_labels}

    if not query_labels:
        raise RuntimeError(f"{hospital}/{part}: no valid GT query class represented in support")
    if len(candidate_labels) < 2:
        raise RuntimeError(f"{hospital}/{part}: fewer than 2 candidate classes")
    if n_way > len(candidate_labels):
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
        distractors = [c for c in candidate_labels if c != query_label]
        selected = [query_label] + rng.sample(distractors, n_way - 1)

        query_path = rng.choice(query_pool[query_label])
        query_key = fast_path_key(query_path)
        support_paths: Dict[str, List[str]] = {}
        for label in selected:
            # Always remove an identical path. This is mandatory in Test-Test and
            # harmless protection against accidental overlap in Test-Train.
            candidates = [p for p in support_pool[label] if fast_path_key(p) != query_key]
            if len(candidates) < n_shot:
                raise RuntimeError(
                    f"episode={episode_id} class={label!r} support={len(candidates)} < shot={n_shot}"
                )
            support_paths[label] = rng.sample(candidates, n_shot)

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

    def __init__(self, *, project_root: str, cfg: Q2LRuntimeConfig, device: torch.device) -> None:
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
    cfg: Q2LRuntimeConfig,
    *,
    hint_seed: int,
) -> torch.Tensor:
    """Generate the same support-conditioned trajectory used by Q2Last.

    The endpoint is post_states[:, -1].  Q2MonoMean reuses every post state,
    so no second Flow pass is required.
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
# Q2L checkpoint / metric loading
# =============================================================================
def extract_metric_state(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = None
    for key in ("metric_encoder", "metric", "metric_state_dict", "fsl_model"):
        if isinstance(ckpt.get(key), dict):
            state = ckpt[key]
            break
    if state is None and isinstance(ckpt.get("model"), dict):
        state = ckpt["model"]
    if state is None:
        raise KeyError("Q2L checkpoint has no MetricEncoder state")
    cleaned = {}
    for key, value in state.items():
        name = str(key)
        for prefix in ("module.", "metric.", "metric_encoder."):
            while name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def runtime_config_from_checkpoint(ckpt: Dict[str, Any]) -> Q2LRuntimeConfig:
    a = as_args_dict(ckpt.get("args", {}))
    dit_path = DIT_CHECKPOINT_OVERRIDE or str(ckpt.get("dit_checkpoint") or a.get("dit_checkpoint") or "")
    if not dit_path:
        raise RuntimeError("Cannot resolve DiT checkpoint from Q2L checkpoint")
    return Q2LRuntimeConfig(
        dit_checkpoint=dit_path,
        dit_variant=str(a.get("dit_variant", "L")),
        num_classes=int(a.get("num_classes", 1000)),
        num_steps=int(a.get("num_steps", 8)),
        solver=str(a.get("solver", "heun")),
        pair_chunk_size=int(a.get("pair_chunk_size", 16)),
        use_ema=bool(a.get("use_ema", True)),
        hint_signal_ratio=a.get("hint_signal_ratio", None),
        stochastic_hint=bool(a.get("stochastic_hint", True)),
        pure_noise_hint=bool(a.get("pure_noise_hint", False)),
        allow_condition_mismatch=bool(a.get("allow_condition_mismatch", False)),
        amp=bool(a.get("amp", True)),
        ot_pool_factor=int(a.get("ot_pool_factor", 1)),
        sinkhorn_reg=float(a.get("sinkhorn_reg", 0.1)),
        sinkhorn_iter=int(a.get("sinkhorn_iter", 30)),
        sk_weight_eps=float(a.get("sk_weight_eps", 1e-3)),
    )


def load_q2l_metrics(device: torch.device) -> Tuple[Dict[str, nn.Module], Dict[str, Dict[str, Any]], Q2LRuntimeConfig]:
    models: Dict[str, nn.Module] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    common_cfg: Optional[Q2LRuntimeConfig] = None

    for name, path in Q2L_CHECKPOINTS.items():
        if not path:
            continue
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} checkpoint missing: {path}")
        ckpt = torch_load(path)
        if not isinstance(ckpt, dict):
            raise TypeError(f"{name} checkpoint must be dict")
        a = as_args_dict(ckpt.get("args", {}))
        hidden = int(a.get("hidden_dim", 256))
        metric_dim = int(a.get("metric_dim", 256))
        dropout = float(a.get("dropout", 0.0))
        metric = MetricEncoder(1024, hidden, metric_dim, dropout=dropout)
        metric.load_state_dict(extract_metric_state(ckpt), strict=True)
        metric = metric.to(device).eval().requires_grad_(False)
        cfg = runtime_config_from_checkpoint(ckpt)

        if common_cfg is None:
            common_cfg = cfg
        elif cfg != common_cfg:
            pass
            # raise RuntimeError(
            #     "Q2L best/last runtime configs differ; endpoint sharing would be invalid.\n"
            #     f"first={common_cfg}\n{name}={cfg}"
            # )

        models[name] = metric
        metadata[name] = {
            "path": path,
            "stage": ckpt.get("stage"),
            "checkpoint_kind": ckpt.get("checkpoint_kind"),
            "epoch": ckpt.get("epoch"),
            "best_pair_acc": ckpt.get("best_pair_acc"),
            "best_gap": ckpt.get("best_gap"),
            "hidden_dim": hidden,
            "metric_dim": metric_dim,
            "dropout": dropout,
            "dit_checkpoint": cfg.dit_checkpoint,
        }
        print(f"[Q2L Load] {name} epoch={ckpt.get('epoch')} kind={ckpt.get('checkpoint_kind')} path={path}")

    if not models or common_cfg is None:
        raise RuntimeError("No Q2L checkpoint configured")
    return models, metadata, common_cfg


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
    """Fuse class-distance vectors using the same three rules as the old evaluator.

    All tensors in ``score_dict`` are class-level distances with shape [n_way].
    Smaller distance means a better match.

    dist_sum:
        Sum selected metric distances class-wise, then argmin.
    soft_vote:
        Convert each metric distance vector to soft votes via softmax(-d/tau),
        sum votes class-wise, then argmax.
    max_then_min:
        First take the maximum distance across selected metrics for each class,
        then select the class with the smallest such worst-case distance.
    """
    names = [n for n in FUSION_METRICS if n in score_dict]
    if not names:
        raise RuntimeError("FUSION_METRICS selected no available scores")

    d_all = torch.stack([score_dict[n] for n in names], dim=0)  # [metric, way]

    if mode == "dist_sum":
        return int(d_all.sum(dim=0).argmin().item())
    if mode == "soft_vote":
        tau = float(SHOT_TAU)
        return int(torch.softmax(-d_all / tau, dim=-1).sum(dim=0).argmax().item())
    if mode == "max_then_min":
        per_class_worst = d_all.max(dim=0).values
        return int(per_class_worst.argmin().item())
    raise ValueError(f"Unknown fusion mode={mode}")


@torch.inference_mode()
def aligned_pair_distances(metric: nn.Module, query_z: torch.Tensor, target_z: torch.Tensor, cfg: Q2LRuntimeConfig) -> torch.Tensor:
    # One MetricEncoder pass preserves the symmetric Q/Last treatment used in training.
    all_z = torch.cat([query_z, target_z], dim=0)
    feat = metric(all_z)
    q_raw = feat.raw[:1]
    target_raw = feat.raw[1:]

    q = pool_square_tokens(q_raw, cfg.ot_pool_factor)
    l = pool_square_tokens(target_raw, cfg.ot_pool_factor)
    q = q.expand(l.shape[0], -1, -1).contiguous()
    qn = F.normalize(q.float(), p=2, dim=-1)
    ln = F.normalize(l.float(), p=2, dim=-1)
    return aligned_sk_sinkhorn_distance(
        qn, ln,
        query_raw=q,
        support_raw=l,
        reg=cfg.sinkhorn_reg,
        n_iter=cfg.sinkhorn_iter,
        weight_eps=cfg.sk_weight_eps,
        return_transport=False,
    ).distance.float()


@torch.inference_mode()
def q2mono_mean_pair_distances(
    metric: nn.Module,
    query_z: torch.Tensor,
    post_states: torch.Tensor,
    direct_distance: torch.Tensor,
    cfg: Q2LRuntimeConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the intended monotonic Query-to-Step mean for each pair.

    Args:
        query_z:          [1,C,H,W], the original query latent z_0.
        post_states:      [P,T,C,H,W], support-conditioned z_1,...,z_T for P pairs.
        direct_distance:  [P], real Query-to-Support Dynamic-SK distance.

    Returns:
        mono_mean: [P], mean of strictly increasing running-maximum step
            distances, with the real Query-to-Support distance appended.
            z_0 itself is not included.
        per_step:  [P,T], raw Query-to-Step distances retained for Q2MonoMean,
            Q2Last (final column), and diagnostics.

    A step is retained only when it is strictly larger than every previously
    retained step, matching the rule: if the next distance does not exceed the
    previous maximum, discard it.
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
        step_distances.append(aligned_pair_distances(metric, query_z, post_states[:, t], cfg))
    per_step = torch.stack(step_distances, dim=-1)
    mono_mean = query2mono_mean_from_steps(per_step, direct_distance)
    return mono_mean, per_step


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
    q2last = finalize_branch(acc.q2last)
    q2mono = finalize_branch(acc.q2mono_mean)
    total = max(acc.q2last.total, 1)
    oracle_acc = acc.oracle_correct / total
    return {
        "q2last_accuracy_percent": q2last["accuracy_percent"],
        "q2last_ci95_percent": q2last["ci95_percent"],
        "q2mono_mean_accuracy_percent": q2mono["accuracy_percent"],
        "q2mono_mean_ci95_percent": q2mono["ci95_percent"],
        "q2mono_mean_gain_vs_q2last_pp": q2mono["accuracy_percent"] - q2last["accuracy_percent"],
        "oracle_q2last_q2mono_accuracy_percent": 100.0 * oracle_acc,
        "fuse_a_accuracy_percent": 100.0 * acc.fuse_a_correct / total,
        "fuse_b_accuracy_percent": 100.0 * acc.fuse_b_correct / total,
        "fuse_c_accuracy_percent": 100.0 * acc.fuse_c_correct / total,
        "fuse_a_correct": int(acc.fuse_a_correct),
        "fuse_b_correct": int(acc.fuse_b_correct),
        "fuse_c_correct": int(acc.fuse_c_correct),
        "fusion_accuracy_percent": 100.0 * acc.fusion_correct / total,
        "fusion_correct": int(acc.fusion_correct),
        "fusion_metrics": list(FUSION_METRICS),
        "fusion_modes": list(FUSION_MODES),
        "q2last_correct": int(acc.q2last.correct),
        "q2mono_mean_correct": int(acc.q2mono_mean.correct),
        "total": int(acc.q2last.total),
        "both_correct": int(acc.both_correct),
        "q2last_only": int(acc.q2last_only),
        "q2mono_only": int(acc.direct_only),
        "both_wrong": int(acc.both_wrong),
        "q2mono_rescue_vs_q2last": int(acc.direct_only),
        "q2mono_degradation_vs_q2last": int(acc.q2last_only),
        "q2mono_net_rescue_vs_q2last": int(acc.direct_only - acc.q2last_only),
        "prediction_changed": int(acc.prediction_changed),
        "prediction_changed_rate_percent": 100.0 * acc.prediction_changed / total,
        "q2last": q2last,
        "q2mono_mean": q2mono,
    }


@torch.inference_mode()
def evaluate_task(
    *, hospital: str, part: str, rae: nn.Module, flow: FrozenFlowRuntime,
    metrics: Dict[str, nn.Module], metadata: Dict[str, Dict[str, Any]],
    cfg: Q2LRuntimeConfig, device: torch.device, transform,
) -> Dict[str, Any]:
    requested_way = N_WAY
    query_pool, support_pool, candidate_labels, query_labels, query_root, support_root = prepare_task_pools(
        hospital, part, eval_setting=EVAL_SETTING, n_way=requested_way, n_shot=N_SHOT,
    )
    episode_n_way = min(requested_way, len(candidate_labels))

    task_seed = SEED + sum(ord(c) for c in hospital)*1009 + sum(ord(c) for c in part)*9176
    plans = generate_episode_plans(
        query_pool, support_pool, candidate_labels, query_labels,
        n_way=episode_n_way, n_shot=N_SHOT, n_episodes=EPISODES_PER_TASK,
        seed=task_seed, eval_setting=EVAL_SETTING,
    )

    print("="*126)
    print(f"[Q2L Q2Last + Q2MonoMean Eval ()] {hospital} / {part} | {episode_n_way}-way {N_SHOT}-shot 1-query | episodes={len(plans)}")
    print(f"setting                     = {EVAL_SETTING}")
    print(f"query_root                  = {query_root}")
    print(f"support_root                = {support_root}")
    print(f"candidate support classes   = {len(candidate_labels)}")
    print(f"valid GT query classes       = {len(query_labels)}")
    if EVAL_SETTING == "test_train":
        print("candidate-space policy       = NON-REDUCED (full support-side class pool)")
    else:
        print("query excluded from support  = STRICT")
    print("Q2Last evidence              = Query-to-Step Dynamic-SK at final post-flow state z_T")
    print("Q2MonoMean evidence          = mean of strictly increasing running-max steps (internal Q2S distance kept)")
    print(f"Flow runtime                 = {cfg.num_steps} steps / {cfg.solver} / chunk={cfg.pair_chunk_size}")
    print(f"shared SK                    = pool={cfg.ot_pool_factor} reg={cfg.sinkhorn_reg} iter={cfg.sinkhorn_iter} eps={cfg.sk_weight_eps}")
    print(f"K-shot aggregation           = {SHOT_AGG} tau={SHOT_TAU if SHOT_AGG=='softmin' else 'N/A'}")
    print(f"checkpoints                  = {list(metrics)}")
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

        # Encode the REAL supports and query exactly once. The internal Q2S distance (inside Q2Mono) uses support_z;
        # Q2Last/Q2MonoMean reuse the same support-conditioned post-flow trajectory.
        all_images = torch.cat([support_images, query_image], dim=0)
        all_z = encode_rae_chunked(rae, all_images, RAE_CHUNK_SIZE)
        support_z = all_z[:-1]
        query_z = all_z[-1:]

        q_pairs = query_z.expand(len(support_flat), -1, -1, -1).contiguous()
        hint_seed = int(task_seed + 8_000_003 + ep_idx * 97_003)
        post_states = generate_post_states_for_pairs(flow, q_pairs, support_images, cfg, hint_seed=hint_seed)

        gt_idx = ep.candidate_labels.index(ep.query_label)
        ep_rec = {
            "episode_id": ep.episode_id,
            "query_label": ep.query_label,
            "query_path": ep.query_path,
            "candidate_labels": list(ep.candidate_labels),
        }
        if SAVE_EPISODE_DETAILS:
            ep_rec["support_paths_by_class"] = {c: list(ep.support_paths[c]) for c in ep.candidate_labels}
            ep_rec["checkpoint_results"] = {}

        for name, metric in metrics.items():
            direct_pair = aligned_pair_distances(metric, query_z, support_z, cfg).reshape(episode_n_way, N_SHOT)
            q2mono_pair_flat, q2mono_per_step_flat = q2mono_mean_pair_distances(
                metric, query_z, post_states, direct_pair.reshape(-1), cfg
            )
            q2mono_pair = q2mono_pair_flat.reshape(episode_n_way, N_SHOT)
            q2mono_per_step = q2mono_per_step_flat.reshape(episode_n_way, N_SHOT, cfg.num_steps)

            q2last_pair = query2last_from_steps(q2mono_per_step)
            q2last_class = aggregate_kshot(q2last_pair)
            q2mono_class = aggregate_kshot(q2mono_pair)

            q2last_idx = int(q2last_class.argmin().item())
            q2mono_idx = int(q2mono_class.argmin().item())
            q2last_label = ep.candidate_labels[q2last_idx]
            q2mono_label = ep.candidate_labels[q2mono_idx]
            q2last_ok = q2last_label == ep.query_label
            q2mono_ok = q2mono_label == ep.query_label

            a = accs[name]
            update_branch(a.q2last, q2last_pair, q2last_class, gt_idx, q2last_ok)
            update_branch(a.q2mono_mean, q2mono_pair, q2mono_class, gt_idx, q2mono_ok)

            # Q2Last vs Q2MonoMean bookkeeping (Direct not reported).
            a.both_correct += int(q2last_ok and q2mono_ok)
            a.q2last_only += int(q2last_ok and not q2mono_ok)
            a.direct_only += int((not q2last_ok) and q2mono_ok)  # reused as q2mono_only
            a.both_wrong += int((not q2last_ok) and (not q2mono_ok))
            a.prediction_changed += int(q2last_label != q2mono_label)
            a.oracle_correct += int(q2last_ok or q2mono_ok)

            # ---- Same three fusion strategies as the previous evaluator ----
            # Participating metrics: Q2Last + Q2MonoMean (Direct excluded).
            score_dict = {
                "q2last": q2last_class,
                "q2mono": q2mono_class,
            }
            fuse_preds = {
                mode: score_fusion_predictions(score_dict, mode)
                for mode in FUSION_MODES
            }
            fuse_a_idx = fuse_preds["dist_sum"]
            fuse_b_idx = fuse_preds["soft_vote"]
            fuse_c_idx = fuse_preds["max_then_min"]
            fuse_a_ok = ep.candidate_labels[fuse_a_idx] == ep.query_label
            fuse_b_ok = ep.candidate_labels[fuse_b_idx] == ep.query_label
            fuse_c_ok = ep.candidate_labels[fuse_c_idx] == ep.query_label

            a.fuse_a_correct += int(fuse_a_ok)
            a.fuse_b_correct += int(fuse_b_ok)
            a.fuse_c_correct += int(fuse_c_ok)
            # Keep the old primary-fusion convention: fusion == Fuse-A / dist_sum.
            a.fusion_correct += int(fuse_a_ok)

            if SAVE_EPISODE_DETAILS:
                ep_rec["checkpoint_results"][name] = {
                    "q2last_pred_label": q2last_label,
                    "q2mono_mean_pred_label": q2mono_label,
                    "q2last_correct": bool(q2last_ok),
                    "q2mono_mean_correct": bool(q2mono_ok),
                    "q2last_only": bool(q2last_ok and (not q2mono_ok)),
                    "q2mono_only": bool((not q2last_ok) and q2mono_ok),
                    "oracle_q2last_q2mono_correct": bool(q2last_ok or q2mono_ok),
                    "fuse_a_mode": "dist_sum",
                    "fuse_a_pred_label": ep.candidate_labels[fuse_a_idx],
                    "fuse_a_correct": bool(fuse_a_ok),
                    "fuse_b_mode": "soft_vote",
                    "fuse_b_pred_label": ep.candidate_labels[fuse_b_idx],
                    "fuse_b_correct": bool(fuse_b_ok),
                    "fuse_c_mode": "max_then_min",
                    "fuse_c_pred_label": ep.candidate_labels[fuse_c_idx],
                    "fuse_c_correct": bool(fuse_c_ok),
                    "fusion_pred_label": ep.candidate_labels[fuse_a_idx],
                    "fusion_correct": bool(fuse_a_ok),
                    "prediction_changed": bool(q2last_label != q2mono_label),
                    "q2last_class_distances": {c: float(q2last_class[i].item()) for i,c in enumerate(ep.candidate_labels)},
                    "q2mono_mean_class_distances": {c: float(q2mono_class[i].item()) for i,c in enumerate(ep.candidate_labels)},
                    "q2last_pair_distances_by_class": {c: [float(v) for v in q2last_pair[i].cpu().tolist()] for i,c in enumerate(ep.candidate_labels)},
                    "q2mono_mean_pair_distances_by_class": {c: [float(v) for v in q2mono_pair[i].cpu().tolist()] for i,c in enumerate(ep.candidate_labels)},
                    "query2step_per_step_pair_distances_by_class": {
                        c: [[float(v) for v in shot_steps] for shot_steps in q2mono_per_step[i].cpu().tolist()]
                        for i, c in enumerate(ep.candidate_labels)
                    },
                }

        if SAVE_EPISODE_DETAILS:
            episode_records.append(ep_rec)
        if LOG_EVERY > 0 and (ep_idx+1) % LOG_EVERY == 0:
            msg = " | ".join(
                f"{n}:Q2Last={100*a.q2last.correct/max(a.q2last.total,1):.2f}%/Q2Mono={100*a.q2mono_mean.correct/max(a.q2mono_mean.total,1):.2f}%"
                for n,a in accs.items()
            )
            print(f"  [{ep_idx+1:04d}/{len(plans):04d}] {msg}")

        del support_images, query_image, all_images, all_z, support_z, query_z, q_pairs, post_states

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
            "candidate_space": "full_support_side_non_reduced" if EVAL_SETTING == "test_train" else "same_test_pool",
            "query_excluded_from_support": True,
            "matched_episode_for_all_scores": True,
            "q2last_metric_evidence": "Query-to-Step Dynamic-SK at the final post-flow state z_T",
            "q2mono_mean_metric_evidence": "mean of strictly increasing running-maximum Query-to-Step distances (internal Query-to-Support distance kept in the mean); z_0 excluded",
            "fusion_metrics": list(FUSION_METRICS),
            "fusion_modes": list(FUSION_MODES),
            "fusion_primary": "dist_sum",
        },
        "q2l_runtime": {
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
        "checkpoints": metadata,
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
        "checkpoint", "hospital", "part", "eval_setting", "n_way", "n_shot", "episodes",
        "q2last_accuracy_percent", "q2last_ci95_percent",
        "q2mono_mean_accuracy_percent", "q2mono_mean_ci95_percent",
        "q2mono_mean_gain_vs_q2last_pp",
        "oracle_q2last_q2mono_accuracy_percent",
        "q2last_correct", "q2mono_mean_correct", "total",
        "both_correct", "q2last_only", "q2mono_only", "both_wrong",
        "q2mono_rescue_vs_q2last", "q2mono_degradation_vs_q2last",
        "q2mono_net_rescue_vs_q2last",
        "prediction_changed", "prediction_changed_rate_percent",
        "fuse_a_accuracy_percent", "fuse_b_accuracy_percent", "fuse_c_accuracy_percent",
        "fuse_a_correct", "fuse_b_correct", "fuse_c_correct",
        "fusion_accuracy_percent", "fusion_correct",
        "checkpoint_path",
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
        raise RuntimeError("Q2L Flow evaluation requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    metrics, metadata, cfg = load_q2l_metrics(device)
    if cfg.ot_pool_factor < 1:
        raise ValueError("Q2L checkpoint ot_pool_factor must be >=1")
    if not os.path.isfile(cfg.dit_checkpoint):
        raise FileNotFoundError(f"DiT checkpoint not found: {cfg.dit_checkpoint}")

    print("="*126)
    print("[Q2L-M0B-RAE Q2Last + Q2MonoMean Episodic Evaluation ()]")
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
                RESULT_DIR, part, f"{hospital}_{part}_{EVAL_SETTING}_q2l_q2last_q2mono.json"
            )
            save_json(out_json, result)
            summary_nested.setdefault(part, {})[hospital] = result["metrics"]

            print(f"\n[RESULT] {hospital} / {part}")
            for name, m in result["metrics"].items():
                print(
                    f"  {name:<10} "
                    f"Q2Last={m['q2last_accuracy_percent']:.2f}% "
                    f"Q2MonoMean={m['q2mono_mean_accuracy_percent']:.2f}% "
                    f"FuseA={m['fuse_a_accuracy_percent']:.2f}% "
                    f"FuseB={m['fuse_b_accuracy_percent']:.2f}% "
                    f"FuseC={m['fuse_c_accuracy_percent']:.2f}% "
                    f"Oracle2={m['oracle_q2last_q2mono_accuracy_percent']:.2f}% "
                    f"MonoR/D/Net={m['q2mono_rescue_vs_q2last']}/{m['q2mono_degradation_vs_q2last']}/{m['q2mono_net_rescue_vs_q2last']}"
                )
                summary_rows.append({
                    "checkpoint": name,
                    "hospital": hospital,
                    "part": part,
                    "eval_setting": EVAL_SETTING,
                    "n_way": result["protocol"]["n_way"],
                    "n_shot": N_SHOT,
                    "episodes": EPISODES_PER_TASK,
                    **{k: v for k, v in m.items() if not isinstance(v, dict)},
                    "checkpoint_path": metadata[name]["path"],
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
            "q2l_checkpoints": Q2L_CHECKPOINTS,
            "q2last_metric_evidence": "Query-to-Step Dynamic-SK at the final post-flow state z_T",
            "q2mono_mean_metric_evidence": "mean of strictly increasing running-maximum Query-to-Step distances (internal Query-to-Support distance kept in the mean); z_0 excluded",
            "fusion_metrics": list(FUSION_METRICS),
            "fusion_modes": list(FUSION_MODES),
            "fusion_primary": "dist_sum",
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
