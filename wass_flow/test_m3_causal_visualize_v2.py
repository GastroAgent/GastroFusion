#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative visualization for M3 causal delta world model.

It samples same-class source/target pairs from the validation class pool and
saves side-by-side results for:
  1) GT history/future trajectory
  2) Flow non-causal trajectory
  3) Teacher-forced one-step causal prediction
  4) Autoregressive causal rollout

Outputs are organized as:
  output_dir/<class_name>/sample_xxx/

Author: ChatGPT (for Tan Shuyue)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageDraw
from torchvision.utils import save_image


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_name(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ('-', '_'):
            keep.append(ch)
        else:
            keep.append('_')
    return ''.join(keep).strip('_') or 'unknown_class'


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_module_from_path(module_name: str, path: str):
    import sys
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register module before execution.
    # Required by dataclass/type resolution in Python 3.11.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def decode_to_unit_image(rae, z: torch.Tensor) -> torch.Tensor:
    """z: (1,1024,16,16) -> decoded image tensor."""
    with torch.no_grad():
        img = rae.decode(z)
    return img[0].detach().cpu()


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """(3,H,W) in [0,1] -> PIL"""
    x = img_tensor.detach().cpu().clamp(0, 1)
    x = (x * 255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def save_tensor_image(img_tensor: torch.Tensor, path: str | Path) -> None:
    path = str(path)
    save_image(img_tensor, path)


def blank_like(img_tensor: torch.Tensor, value: float = 1.0) -> torch.Tensor:
    return torch.full_like(img_tensor, fill_value=value)


def make_compare_grid(
    hist_imgs: List[torch.Tensor],
    gt_imgs: List[torch.Tensor],
    flow_imgs: List[torch.Tensor],
    teacher_imgs: List[torch.Tensor],
    ar_imgs: List[torch.Tensor],
    out_path: str | Path,
) -> None:
    """Create one overview image with row labels and time-step headers."""
    # Determine size from first available image.
    ref = hist_imgs[0]
    H, W = ref.shape[1], ref.shape[2]
    left_margin = 120
    top_margin = 45
    row_gap = 10
    col_gap = 6
    bottom_margin = 10
    right_margin = 10
    n_cols = 9
    n_rows = 5
    canvas_w = left_margin + n_cols * W + (n_cols - 1) * col_gap + right_margin
    canvas_h = top_margin + n_rows * H + (n_rows - 1) * row_gap + bottom_margin
    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Headers
    for c in range(n_cols):
        x = left_margin + c * (W + col_gap) + W // 2 - 10
        draw.text((x, 12), f"t{c}", fill=(0, 0, 0))

    row_names = ["History", "GT", "Flow", "Teacher", "AR"]
    blank = blank_like(ref)
    rows: List[List[torch.Tensor]] = []
    rows.append(hist_imgs + [blank] * 5)
    rows.append([blank] * 4 + gt_imgs)
    rows.append([blank] * 4 + flow_imgs)
    rows.append([blank] * 4 + teacher_imgs)
    rows.append([blank] * 4 + ar_imgs)

    for r, (name, row_imgs) in enumerate(zip(row_names, rows)):
        y = top_margin + r * (H + row_gap)
        draw.text((15, y + H // 2 - 8), name, fill=(0, 0, 0))
        for c, img_t in enumerate(row_imgs):
            x = left_margin + c * (W + col_gap)
            canvas.paste(tensor_to_pil(img_t), (x, y))

    canvas.save(str(out_path))


# -----------------------------------------------------------------------------
# Sampling helpers
# -----------------------------------------------------------------------------

def sample_pairs_by_class(
    pool: Dict[str, List[str]],
    num_per_class: int,
    seed: int,
) -> List[Tuple[str, str, str, int]]:
    """Return list of (label, source_path, target_path, local_index)."""
    rng = random.Random(seed)
    pairs = []
    for label in sorted(pool.keys()):
        paths = list(dict.fromkeys(pool[label]))
        if len(paths) < 2:
            continue
        for local_idx in range(num_per_class):
            src, tgt = rng.sample(paths, 2)
            pairs.append((label, src, tgt, local_idx))
    return pairs


# -----------------------------------------------------------------------------
# Inference helpers
# -----------------------------------------------------------------------------

def build_gt_trajectory(flow_traj: torch.Tensor, support_z: torch.Tensor) -> torch.Tensor:
    """Create GT trajectory used by M3 visualization.

    Args:
        flow_traj: (1,9,1024,16,16), Flow states t0..t8
        support_z: (1,1024,16,16), true target latent

    Returns:
        gt_traj:   (1,9,1024,16,16), where t0..t7 come from Flow and t8=true target latent
    """
    gt_traj = flow_traj.clone()
    gt_traj[:, 8] = support_z
    return gt_traj


@torch.no_grad()
def extract_prediction_for_time(model, trajectory: torch.Tensor, observed: torch.Tensor, global_t: int) -> torch.Tensor:
    """Run model once and return predicted latent for specified global time index.

    Returns shape: (1,1024,16,16)
    """
    out = model(trajectory, observed)
    pred_masked = out["pred_masked"]        # (M,256,1024)
    masked_indices = out["masked_indices"]  # (M,2), [b, global_t]

    matches = (masked_indices[:, 0] == 0) & (masked_indices[:, 1] == int(global_t))
    if not bool(matches.any().item()):
        raise RuntimeError(f"No predicted masked frame found for global t={global_t}")

    token_pred = pred_masked[matches][0]  # (256,1024)
    frame = token_pred.view(1, 16, 16, 1024).permute(0, 3, 1, 2).contiguous()
    return frame


@torch.no_grad()
def teacher_forcing_rollout(model, gt_traj: torch.Tensor, device: torch.device) -> List[torch.Tensor]:
    """Predict t4..t8, each step using all previous GT states as known context/history."""
    preds = []
    for t in range(4, 9):
        traj_in = gt_traj.clone().to(device)
        observed = torch.zeros((1, 9), dtype=torch.bool, device=device)
        observed[:, :t] = True   # t itself and later are masked
        pred_t = extract_prediction_for_time(model, traj_in, observed, global_t=t)
        preds.append(pred_t.detach().cpu())
    return preds


@torch.no_grad()
def autoregressive_rollout(model, gt_traj: torch.Tensor, device: torch.device) -> List[torch.Tensor]:
    """Predict t4..t8 autoregressively.

    Known history starts as GT t0..t3. Thereafter newly predicted state becomes observed.
    """
    work = torch.zeros_like(gt_traj, device=device)
    work[:, :4] = gt_traj[:, :4].to(device)
    preds = []
    for t in range(4, 9):
        observed = torch.zeros((1, 9), dtype=torch.bool, device=device)
        observed[:, :t] = True
        pred_t = extract_prediction_for_time(model, work, observed, global_t=t)
        work[:, t] = pred_t
        preds.append(pred_t.detach().cpu())
    return preds


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Qualitative visualization for M3 causal world model")

    p.add_argument("--project_root", type=str, required=True,
                   help="Project root used in training, e.g. /mnt/.../wass_flow")
    p.add_argument("--m3_checkpoint", type=str, required=True,
                   help="Trained M3 checkpoint (.pt)")
    p.add_argument("--flow_checkpoint", type=str, default=None,
                   help="Flow checkpoint; defaults to args saved in M3 checkpoint if possible")
    p.add_argument("--val_json_glob", type=str, default=None,
                   help="Validation class-pool json glob/folder; defaults to args saved in M3 checkpoint")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--m3_source", type=str,
                   default="/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/fsl_multi_vae/rae_flow_metric_v1/train_m3_causal_delta_world_model.py",
                   help="Path to source file that defines CausalDeltaWorldModel/FrozenFlowRuntime")
    p.add_argument("--num_per_class", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--save_latents", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    project_root = Path(args.project_root).resolve()
    code_parent = project_root / "fsl_multi_vae"
    for p in [str(project_root), str(code_parent)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)

    # Load training/inference source module.
    m3mod = load_module_from_path("m3_world_model_src", args.m3_source)

    # Load checkpoint and recover saved args.
    ckpt = torch.load(args.m3_checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    if not ckpt_args:
        raise RuntimeError("M3 checkpoint does not contain saved args")

    project_root_from_ckpt = ckpt_args.get("project_root", None)
    if project_root_from_ckpt and str(Path(project_root_from_ckpt).resolve()) != str(project_root):
        print(f"[WARN] project_root differs: cli={project_root}, ckpt={project_root_from_ckpt}")

    val_json_glob = args.val_json_glob or ckpt_args.get("val_json_glob")
    if not val_json_glob:
        raise RuntimeError("Please provide --val_json_glob or ensure it exists in checkpoint args")

    flow_checkpoint = args.flow_checkpoint or ckpt_args.get("flow_checkpoint")
    if not flow_checkpoint:
        raise RuntimeError("Please provide --flow_checkpoint or ensure it exists in checkpoint args")

    # ------------------------------------------------------------------
    # Build frozen RAE / Flow / trainable world model
    # ------------------------------------------------------------------
    print("[INFO] Loading RAE ...")
    rae = m3mod.build_rae(str(project_root), device)
    rae.eval()

    print("[INFO] Loading Flow runtime ...")
    flow_args = SimpleNamespace(
        project_root=str(project_root),
        flow_checkpoint=flow_checkpoint,
        flow_num_steps=int(ckpt_args.get("flow_num_steps", 8)),
        flow_solver=str(ckpt_args.get("flow_solver", "heun")),
        flow_dit_variant=str(ckpt_args.get("flow_dit_variant", "L")),
        flow_endovit_path=ckpt_args.get("flow_endovit_path", None),
        flow_hint_signal_ratio=ckpt_args.get("flow_hint_signal_ratio", 0.1),
        flow_hint_dropout=float(ckpt_args.get("flow_hint_dropout", 0.0)),
        flow_stochastic_hint=False,
        flow_allow_condition_mismatch=bool(ckpt_args.get("flow_allow_condition_mismatch", False)),
        amp=bool(ckpt_args.get("amp", True)),
    )
    flow = m3mod.FrozenFlowRuntime(flow_args, device)

    print("[INFO] Loading world model ...")
    model = m3mod.CausalDeltaWorldModel(
        dim=int(ckpt_args.get("model_dim", 1024)),
        depth=int(ckpt_args.get("depth", 16)),
        encoder_depth=int(ckpt_args.get("encoder_depth", 8)),
        encoder_time=int(ckpt_args.get("encoder_time", 4)),
        num_heads=int(ckpt_args.get("num_heads", 16)),
        ffn_hidden_dim=int(ckpt_args.get("ffn_hidden_dim", 2048)),
        max_time=9,
        height=16,
        width=16,
        dropout=float(ckpt_args.get("dropout", 0.0)),
        attn_dropout=float(ckpt_args.get("attn_dropout", 0.0)),
        gradient_checkpointing=False,
    ).to(device)
    model.load_state_dict(ckpt["world_model"], strict=True)
    model.eval()

    transform = m3mod.build_transform(int(ckpt_args.get("image_size", 256)))

    # ------------------------------------------------------------------
    # Load validation pool and sample pairs
    # ------------------------------------------------------------------
    print("[INFO] Loading validation class pool ...")
    pool = m3mod.load_class_pool(root=None, json_glob=val_json_glob)
    pairs = sample_pairs_by_class(pool, num_per_class=int(args.num_per_class), seed=int(args.seed))
    print(f"[INFO] Sampled {len(pairs)} pairs across {len(pool)} classes")

    meta_all = []

    for global_idx, (label, source_path, target_path, local_idx) in enumerate(pairs):
        class_dir = ensure_dir(Path(args.output_dir) / sanitize_name(label))
        sample_dir = ensure_dir(class_dir / f"sample_{local_idx:03d}")
        print(f"[{global_idx+1}/{len(pairs)}] {label} :: {Path(source_path).name} -> {Path(target_path).name}")

        # --------------------------------------------------------------
        # Load source / target images and encode with RAE
        # --------------------------------------------------------------
        with torch.no_grad():
            source_img = m3mod.load_images([source_path], transform).to(device)
            target_img = m3mod.load_images([target_path], transform).to(device)
            both = torch.cat([source_img, target_img], dim=0)
            z = m3mod.encode_rae_chunked(rae, both, int(ckpt_args.get("rae_chunk_size", 8)))
            source_z = z[:1]
            support_z = z[1:2]

            flow_traj = flow.generate(source_z, target_img, stochastic_hint=False)  # (1,9,1024,16,16)
            gt_traj = build_gt_trajectory(flow_traj, support_z)                     # (1,9,1024,16,16)

        # --------------------------------------------------------------
        # Three settings
        # --------------------------------------------------------------
        teacher_preds = teacher_forcing_rollout(model, gt_traj, device)   # 5 tensors t4..t8
        ar_preds = autoregressive_rollout(model, gt_traj, device)         # 5 tensors t4..t8

        # --------------------------------------------------------------
        # Decode images for saving
        # --------------------------------------------------------------
        src_img_vis = source_img[0].detach().cpu()
        tgt_img_vis = target_img[0].detach().cpu()
        save_tensor_image(src_img_vis, sample_dir / "source.png")
        save_tensor_image(tgt_img_vis, sample_dir / "target.png")

        hist_imgs: List[torch.Tensor] = []
        gt_imgs: List[torch.Tensor] = []
        flow_imgs: List[torch.Tensor] = []
        teacher_imgs: List[torch.Tensor] = []
        ar_imgs: List[torch.Tensor] = []

        # Shared history t0..t3 from GT trajectory
        for t in range(4):
            img = decode_to_unit_image(rae, gt_traj[:, t].to(device))
            hist_imgs.append(img)
            save_tensor_image(img, sample_dir / f"hist_t{t}.png")

        # GT future t4..t8 (remember t8 = true target latent)
        for t in range(4, 9):
            img = decode_to_unit_image(rae, gt_traj[:, t].to(device))
            gt_imgs.append(img)
            save_tensor_image(img, sample_dir / f"gt_t{t}.png")

        # Flow future t4..t8 (non-causal target-conditioned flow results)
        for t in range(4, 9):
            img = decode_to_unit_image(rae, flow_traj[:, t].to(device))
            flow_imgs.append(img)
            save_tensor_image(img, sample_dir / f"flow_t{t}.png")

        # Teacher predictions t4..t8
        for offset, t in enumerate(range(4, 9)):
            pred = teacher_preds[offset].to(device)
            img = decode_to_unit_image(rae, pred)
            teacher_imgs.append(img)
            save_tensor_image(img, sample_dir / f"teacher_t{t}.png")

        # AR predictions t4..t8
        for offset, t in enumerate(range(4, 9)):
            pred = ar_preds[offset].to(device)
            img = decode_to_unit_image(rae, pred)
            ar_imgs.append(img)
            save_tensor_image(img, sample_dir / f"ar_t{t}.png")

        make_compare_grid(
            hist_imgs=hist_imgs,
            gt_imgs=gt_imgs,
            flow_imgs=flow_imgs,
            teacher_imgs=teacher_imgs,
            ar_imgs=ar_imgs,
            out_path=sample_dir / "compare_grid.png",
        )

        if args.save_latents:
            lat_dir = ensure_dir(sample_dir / "latents")
            torch.save({"gt_traj": gt_traj.cpu()}, lat_dir / "gt.pt")
            torch.save({"flow_traj": flow_traj.cpu()}, lat_dir / "flow.pt")
            torch.save({"teacher_preds_t4_t8": teacher_preds}, lat_dir / "teacher.pt")
            torch.save({"ar_preds_t4_t8": ar_preds}, lat_dir / "ar.pt")

        sample_meta = {
            "class_label": label,
            "source_path": source_path,
            "target_path": target_path,
            "sample_dir": str(sample_dir),
            "flow_num_steps": int(ckpt_args.get("flow_num_steps", 8)),
        }
        meta_all.append(sample_meta)

    meta_path = Path(args.output_dir) / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_all, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"[DONE] Saved qualitative results to: {args.output_dir}")
    print(f"[DONE] Meta: {meta_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
