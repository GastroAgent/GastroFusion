#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict intra-class self-excluded OT DiT Flow training.

核心重构：
  - one class per rank-local OT batch；
  - batch 内图像 identity 不重复；
  - OT 前 x0_i / x1_i 指向同一 underlying image，因此 diagonal 就是 self；
  - OTCFM 强制 intra_class_suboptimal=True，屏蔽 diagonal 后再求 constrained OT；
  - OT 后再次用 path/label 强校验：禁止 self pairing 和 cross-class pairing；
  - 保留原 DiT / RAE / EndoViT condition / DDP / EMA / GAN / spatial Sinkhorn loss。

这里的“次优 OT”更准确地说是 self-excluded constrained intra-class OT，
不是枚举全局第二优 transport plan。
"""

# import debugpy

# try:
#     print("Waiting for debugger attach")
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9504))
#     # debugpy.listen(("172.16.0.108", 9504)) 
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import argparse
import copy
import glob
import inspect
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
from torch.nn import init
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm, trange
from PIL import Image
from safetensors.torch import load_file
from timm.models.vision_transformer import VisionTransformer
from transformers import AutoTokenizer, ChineseCLIPTextModel, ChineseCLIPTextConfig
from discriminator_model import (
    build_discriminator,
    select_gan_losses,
    calculate_adaptive_weight,
    LPIPS,
)

# ──────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────
_PROJECT_ROOT = '/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow'
sys.path.append(_PROJECT_ROOT)
sys.path.append(os.path.join(_PROJECT_ROOT, 'utils'))

from train_utils import (
    find_latest_checkpoint,
    cleanup_old_checkpoints,
    ema,
    infiniteloop,
    warmup_lr,
)
from conditional_flow_matcher import (
    ConditionalFlowMatcher,
    OptimalTransportConditionalFlowMatcher,
)
from flow_models.dit_rae import DiTForRAEFlow, create_dit_rae
# dispatch_model (模型并行) 已在 DDP 改造中移除, 不再导入

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────
_UNCONDITIONAL_CLASS_ID = 999
_BLANK_CLASS_ID = 998              # x0 为高斯噪声时的特殊类别
_OP_MATCH_DISABLED_PROB = 0.25
_DROP_CONDITION_PROB = 0.0
_BLANK_NOISE_PROB = 0.0            # x0 以此概率替换为高斯噪声（Blank 模式）(原1.0→Intra/Inter永不执行)
_BLANK_NOISE_STD = 1.0             # 噪声标准差（匹配 RAE 潜变量空间，默认 N(0,1)）
_BLANK_CAPTION = '从随机噪声生成符合当前条件描述的症状表现。'

# ── OT / 内迁-互迁 配置 ──
_OT_METHOD = 'exact'               # 'exact' | 'sinkhorn' | 'unbalanced'
_OT_REG = 0.1                      # Sinkhorn 正则化（配合 adaptive_reg 自适应缩放）
_OT_ADAPTIVE_REG = True            # reg_eff = reg * M.std()
# ── Sinkhorn 迭代控制 ──
#   numItermax=100  stopThr=1e-9:  24ms (10×)  ← 推荐
_OT_NUM_ITER_MAX = 100             # 最大迭代（配合 stopThr 足够收敛）
_OT_STOP_THR = 1e-9               # marginal violation < thr 即停
_OT_WARN = False                  # 压制 POT 警告
# 本脚本只训练严格类内、排除自身的 constrained OT。
# 不再使用 mixed/inter 概率开关；以下阈值仅用于 OT 的 near-duplicate 屏蔽。
_SUBOPTIMAL_EPS = 1e-6

_IMAGE_INPUT_SIZE = 224
_ENDOVIT_MEAN = [0.3464, 0.2280, 0.2228]
_ENDOVIT_STD = [0.2520, 0.2128, 0.2093]
_ENDOVIT_PATH = os.path.join(
    _PROJECT_ROOT, 'flow_matcher_otcfm/EndoViT/pytorch_model.bin'
)
_TEXT_ENCODER_PATH = os.path.join(
    _PROJECT_ROOT, 'flow_matcher_otcfm/text_encoder'
)
_DATA_GLOB_PATTERN = "/mnt/inaisfs/data/home/tansy_criait/jmf/work/wass_flow/data/data_train_pool_imagenet_med/*.json"

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

# 无条件生成的替代 caption
_UNCONDITIONAL_CAPTION = '将当前症状表现转变为相关条件所描述出的症状。'

# ── GAN 判别器配置 ──
_DINO_CKPT_PATH = os.path.join(
    _PROJECT_ROOT, 'flow_matcher_otcfm/rae_v2/dino_vit_small_patch8_224_gan.pth'
)
_DINO_RECIPE = 'S_8'  # 根据 checkpoint 自动推断: patch8 → S_8, patch16 → S_16
_GAN_DISC_LOSS = 'hinge'         # 判别器损失类型: 'hinge' 或 'vanilla'
_GAN_GEN_LOSS = 'vanilla'        # 生成器对抗损失类型: 'vanilla'
_GAN_DIFAUG_PROB = 1.0           # DiffAug 增强概率
_GAN_DIFAUG_CUTOUT = 0.2         # DiffAug cutout 比例
_GAN_LPIPS_WEIGHT = 0.1          # LPIPS 感知损失权重 (0 表示禁用)


# ──────────────────────────────────────────────
# DDP 辅助函数
# ──────────────────────────────────────────────

def is_dist_avail() -> bool:
    """判断是否处于分布式训练环境 (torchrun 启动时设置 WORLD_SIZE)."""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """返回分布式训练的总进程数 (单进程返回 1)."""
    return dist.get_world_size() if is_dist_avail() else 1


def get_rank() -> int:
    """返回当前进程的全局 rank (单进程返回 0)."""
    return dist.get_rank() if is_dist_avail() else 0


def get_local_rank() -> int:
    """从环境变量读取 LOCAL_RANK (torchrun 自动注入, 单进程返回 0)."""
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    """判断当前是否为主进程 (rank==0)."""
    return get_rank() == 0


def setup_ddp(backend: str = 'nccl') -> int:
    """初始化 DDP 进程组.

    依赖 torchrun 注入的环境变量: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT.

    Args:
        backend: 通信后端, GPU 训练用 'nccl'

    Returns:
        local_rank: 当前节点的 GPU 编号
    """
    if not dist.is_available():
        return 0
    if dist.is_initialized():
        return get_local_rank()

    local_rank = get_local_rank()
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))

    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        # 绑定当前进程到对应 GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    return local_rank


def cleanup_ddp() -> None:
    """销毁 DDP 进程组."""
    if is_dist_avail():
        dist.destroy_process_group()


def reduce_tensor(t: torch.Tensor, average: bool = True) -> torch.Tensor:
    """跨进程 all-reduce 一个标量张量 (用于统计同步, 不在反向传播路径上).

    Args:
        t: 输入张量
        average: 是否取平均 (False 则求和)

    Returns:
        归约后的张量
    """
    if not is_dist_avail():
        return t
    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= get_world_size()
    return rt


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def init_weights(model: nn.Module) -> None:
    """对模型中的 Linear / Embedding / LayerNorm 层进行权重初始化."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            init.xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)


def sinkhorn_ot_loss(
    x_pred: torch.Tensor,
    x_target: torch.Tensor,
    reg: float = 0.1,
    n_iters: int = 20,
    normalize_cost: bool = True,
    max_tokens: int = 256,
) -> torch.Tensor:
    """逐样本 spatial-token Sinkhorn divergence，梯度仅回传到 ``x_pred``。

    与旧实现不同，这里不再把 batch 中每一整张 latent 当成一个 OT point。
    对每个样本，先把 ``(C, H, W)`` 展开为 ``H*W`` 个 C 维 spatial token，
    然后只在同一样本内部做：

        predicted spatial-token distribution
                    <->
        target spatial-token distribution

    因此：
      1. 保留明确的 paired sample correspondence，不允许跨样本错配；
      2. DDP 下每个样本的 OT 都是局部独立问题，不需要跨卡 all_gather；
      3. 当 H*W > max_tokens 时，仅为 OT 分支自适应池化，主 FM 分支不受影响。

    使用标准 entropy-regularized OT objective：
        OT_eps(a,b) = <pi, M> + eps * KL(pi || a \\otimes b)
    再做 Sinkhorn debias：
        S_eps(a,b) = OT_eps(a,b)
                     - 0.5 * OT_eps(a,a)
                     - 0.5 * OT_eps(b,b)

    Args:
        x_pred: 预测 latent，形状 (B, C, H, W)，保留梯度。
        x_target: paired 真实目标 latent，形状与 x_pred 相同，会 detach。
        reg: entropy regularization epsilon。cost 默认归一化后建议 0.05~0.5。
        n_iters: log-domain Sinkhorn 迭代次数。
        normalize_cost: 是否按每个样本的 cross-cost 最大值归一化 cost。
        max_tokens: 每个样本参与 OT 的最大 spatial token 数。
                    当前 RAE 16x16=256 会原样使用；32x32 会自动池化到 <=256。

    Returns:
        标量 loss。梯度仅经 x_pred 回传。
    """
    if x_pred.ndim != 4 or x_target.ndim != 4:
        raise ValueError(
            "spatial Sinkhorn 期望 x_pred/x_target 为 (B,C,H,W)，"
            f"当前分别为 {tuple(x_pred.shape)} / {tuple(x_target.shape)}"
        )
    if x_pred.shape != x_target.shape:
        raise ValueError(
            f"x_pred 与 x_target 形状必须一致，当前 {tuple(x_pred.shape)} "
            f"vs {tuple(x_target.shape)}"
        )
    if reg <= 0.0:
        raise ValueError(f"Sinkhorn reg 必须 > 0，当前 reg={reg}")
    if n_iters <= 0:
        raise ValueError(f"Sinkhorn n_iters 必须 > 0，当前 n_iters={n_iters}")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens 必须 > 0，当前 max_tokens={max_tokens}")

    def _pool_for_ot(x: torch.Tensor) -> torch.Tensor:
        """仅压缩 OT 分支的空间网格，使 token 数不超过 max_tokens。"""
        _, _, h, w = x.shape
        if h * w <= max_tokens:
            return x

        # 尽量保持原始宽高比，同时保证 target_h * target_w <= max_tokens。
        scale = math.sqrt(float(max_tokens) / float(h * w))
        target_h = max(1, min(h, int(math.floor(h * scale))))
        target_w = max(1, min(w, int(math.floor(w * scale))))

        # floor 通常已经满足限制；这里保留兜底，避免浮点/非方形输入导致越界。
        while target_h * target_w > max_tokens:
            if target_h >= target_w and target_h > 1:
                target_h -= 1
            elif target_w > 1:
                target_w -= 1
            else:
                break

        return F.adaptive_avg_pool2d(x, (target_h, target_w))

    pred_ot = _pool_for_ot(x_pred)
    target_ot = _pool_for_ot(x_target.detach())

    # (B,C,H,W) -> (B,N,C), N=H*W。
    # Sinkhorn/cdist 用 float32，避免 fp16/bf16 下距离与 exp/log 的数值问题；
    # cast 操作仍保留到 x_pred 的梯度链路。
    a = pred_ot.flatten(2).transpose(1, 2).float()
    b = target_ot.flatten(2).transpose(1, 2).float().detach()

    # 每个样本独立的 pairwise squared-Euclidean cost: (B,N,N)。
    M_ab = torch.cdist(a, b, p=2).square()
    M_aa = torch.cdist(a, a, p=2).square()
    with torch.no_grad():
        M_bb = torch.cdist(b, b, p=2).square()

    if normalize_cost:
        # 每个样本使用同一个 cross-cost scale 同步缩放 ab/aa/bb，
        # 避免不同 latent 尺度导致 reg 的实际含义漂移。
        cost_scale = M_ab.detach().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        M_ab = M_ab / cost_scale
        M_aa = M_aa / cost_scale
        M_bb = M_bb / cost_scale

    def _sinkhorn_objective(M: torch.Tensor) -> torch.Tensor:
        """batched log-domain Sinkhorn，返回每个样本的正则化 OT objective。"""
        batch, n, m = M.shape

        log_mu = torch.full(
            (batch, n), -math.log(float(n)), device=M.device, dtype=M.dtype
        )
        log_nu = torch.full(
            (batch, m), -math.log(float(m)), device=M.device, dtype=M.dtype
        )

        log_K = -M / reg
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)

        for _ in range(n_iters):
            u = log_mu - torch.logsumexp(
                log_K + v.unsqueeze(1), dim=2
            )
            v = log_nu - torch.logsumexp(
                log_K + u.unsqueeze(2), dim=1
            )

        log_pi = u.unsqueeze(2) + log_K + v.unsqueeze(1)
        pi = torch.exp(log_pi)

        transport = (pi * M).sum(dim=(-2, -1))

        # KL(pi || mu \\otimes nu)。由于两者总质量均为 1，
        # 完整 KL 中的 -pi + mu*nu 常数项相互抵消。
        log_ref = log_mu.unsqueeze(2) + log_nu.unsqueeze(1)
        kl = (pi * (log_pi - log_ref)).sum(dim=(-2, -1))

        return transport + reg * kl

    ot_ab = _sinkhorn_objective(M_ab)
    ot_aa = _sinkhorn_objective(M_aa)
    with torch.no_grad():
        ot_bb = _sinkhorn_objective(M_bb)

    divergence = ot_ab - 0.5 * ot_aa - 0.5 * ot_bb

    # 理论 Sinkhorn divergence 非负；有限迭代/浮点误差可能产生极小负值。
    divergence = torch.clamp(divergence, min=0.0)
    return divergence.mean()

def build_transforms(image_size: int = 256) -> dict:
    """构建训练所需的各种图像变换 pipeline.

    Returns:
        dict: 包含 transform / transform_mask / transform_A / transform_grey 四种变换
    """
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ToTensor(),
        normalize,
    ])

    transform_A = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])

    transform_grey = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.RandomGrayscale(p=1),
        normalize,
    ])

    return {
        'transform': transform,
        'transform_mask': transform_mask,
        'transform_A': transform_A,
        'transform_grey': transform_grey,
    }


class StrictIntraClassOTDataset(Dataset):
    """严格类内 OT 数据集：一个 index 对应一张真实图像 identity。

    设计原则：
      1. Dataset 不再随机构造 ``A->B`` pair；一个 index 只代表一张图像 I_i。
      2. ``x0`` 与 ``x1`` 在 OT 前具有相同 sample identity（同一路径），但允许
         使用不同的数据增强视图。
      3. 真正的 ``I_i -> I_j`` 配对交给 batch-level constrained OT 完成。
      4. ``OneClassBatchSampler`` 保证一个 rank-local OT batch 只来自一个类别，
         且 batch 内 sample identity 不重复。

    因而 OT cost matrix 的 diagonal 精确表示 self pairing；在
    ``intra_class_suboptimal=True`` 时屏蔽 diagonal 后，得到的就是
    same-class + non-self constrained OT。
    """

    def __init__(
        self,
        data_glob: str,
        *,
        transform,
        transform_A,
        transform_B,
        hint_transform,
        flip: bool = True,
        pflip: float = 0.25,
        min_class_size: int = 2,
        validate_paths: bool = True,
    ):
        super().__init__()
        self.transform = transform
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.hint_transform = hint_transform
        self.flip = bool(flip)
        self.pflip = float(pflip)
        self.min_class_size = max(2, int(min_class_size))

        json_paths = sorted(glob.glob(data_glob))
        if not json_paths:
            raise ValueError(f"No JSON files found by data_glob={data_glob}")

        class_pool = {}
        for jp in tqdm(json_paths, desc='Loading strict intra-class JSON'):
            try:
                with open(jp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as exc:
                print(f"[WARN] Failed to load {jp}: {exc}")
                continue

            if isinstance(data, dict):
                # 兼容少数 {records:[...]} / {data:[...]} 包装。
                if isinstance(data.get('records'), list):
                    data = data['records']
                elif isinstance(data.get('data'), list):
                    data = data['data']
                else:
                    values = [v for v in data.values() if isinstance(v, list)]
                    if len(values) == 1:
                        data = values[0]

            if not isinstance(data, list):
                print(f"[WARN] Skip non-list JSON: {jp}")
                continue

            for rec in data:
                if not isinstance(rec, dict):
                    continue
                path = rec.get('x') or rec.get('path') or rec.get('image_path')
                if not path:
                    continue
                path = os.path.abspath(os.path.expanduser(str(path)))
                hint_path = rec.get('hint_path', path) or path
                hint_path = os.path.abspath(os.path.expanduser(str(hint_path)))
                label = str(rec.get('label', ''))
                if not label:
                    continue
                if validate_paths and (not os.path.isfile(path) or not os.path.isfile(hint_path)):
                    continue

                item = {
                    'path': path,
                    'hint_path': hint_path,
                    'caption': str(rec.get('caption', '')),
                    'label': label,
                    'label_id': int(rec.get('label_id', -1)),
                }
                class_pool.setdefault(label, []).append(item)

        # 去掉 path 重复，避免 off-diagonal 仍然出现同一 identity。
        self.class_pool = {}
        for label, items in class_pool.items():
            unique = {}
            for item in items:
                unique[item['path']] = item
            values = list(unique.values())
            if len(values) >= self.min_class_size:
                self.class_pool[label] = values

        self.classes = sorted(self.class_pool)
        if not self.classes:
            raise ValueError(
                f"No class has >= {self.min_class_size} unique images under {data_glob}"
            )

        self.samples = []
        self.class_to_indices = {}
        for label in self.classes:
            idxs = []
            for item in self.class_pool[label]:
                idx = len(self.samples)
                self.samples.append(item)
                idxs.append(idx)
            self.class_to_indices[label] = idxs

        self._len = len(self.samples)
        sizes = [len(self.class_to_indices[c]) for c in self.classes]
        print(
            f"[INFO] StrictIntraClassOTDataset: classes={len(self.classes)}, "
            f"unique_images={self._len}, class_size[min/mean/max]="
            f"{min(sizes)}/{sum(sizes)/len(sizes):.1f}/{max(sizes)}"
        )

    def __len__(self):
        return self._len

    @staticmethod
    def _paired_flip(x1_image: Image.Image, hint_image: Image.Image, pflip: float):
        if random.random() < pflip:
            x1_image = x1_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            hint_image = hint_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < pflip:
            x1_image = x1_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            hint_image = hint_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return x1_image, hint_image

    def __getitem__(self, idx):
        item = self.samples[int(idx)]

        # x0/x1 在 identity 层面完全相同；augmentation 可以不同。
        with Image.open(item['path']) as img:
            base = img.convert('RGB')
            x0_img = base.copy()
            x1_img = base.copy()
        with Image.open(item['hint_path']) as img:
            hint_img = img.convert('RGB')

        if self.flip:
            x1_img, hint_img = self._paired_flip(x1_img, hint_img, self.pflip)

        x0_fn = self.transform_A if self.transform_A is not None else self.transform
        x1_fn = self.transform_B if self.transform_B is not None else self.transform
        x0 = x0_fn(x0_img)
        x1 = x1_fn(x1_img)
        hint = self.hint_transform(hint_img) if self.hint_transform is not None else x1_fn(hint_img)

        return {
            'caption': item['caption'],
            'x0': x0,
            'x1': x1,
            'label_A': item['label'],
            'label_B': item['label'],
            'label_A_id': item['label_id'],
            'label_B_id': item['label_id'],
            'class_id': torch.tensor(item['label_id']).long(),
            # 关键：OT 前 x0/x1 path 完全一致，diagonal 才是真正的 self pairing。
            'x0_path': item['path'],
            'x1_path': item['path'],
            'hint': hint,
            'is_intra': True,
        }


class OneClassBatchSampler(Sampler):
    """每个 rank-local minibatch 只采一个类别，并保证 batch 内 identity 唯一。

    DDP 下不同 rank 可以采不同类别；OT 只在各 rank 的本地 minibatch 内求解，
    所以每个 OT problem 始终是纯类内的。梯度随后由 DDP all-reduce。
    """

    def __init__(
        self,
        class_to_indices: dict,
        batch_size: int,
        *,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        batches_per_epoch: int = 0,
    ):
        if int(batch_size) < 2:
            raise ValueError("strict self-excluded OT requires batch_size >= 2")
        self.class_to_indices = {
            str(k): list(v) for k, v in class_to_indices.items() if len(v) >= 2
        }
        self.classes = sorted(self.class_to_indices)
        if not self.classes:
            raise ValueError("No class has at least 2 unique images for strict intra-class OT")

        self.batch_size = int(batch_size)
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self.seed = int(seed)
        total = sum(len(v) for v in self.class_to_indices.values())
        default_batches = max(1, math.ceil(total / float(self.batch_size * self.world_size)))
        self.batches_per_epoch = int(batches_per_epoch) if int(batches_per_epoch) > 0 else default_batches
        self._epoch = 0

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1
        rng = random.Random(self.seed + 1_000_003 * epoch + 10_007 * self.rank)

        # class-balanced：循环随机排列类别，避免大类垄断训练 batch。
        order = []
        while len(order) < self.batches_per_epoch:
            block = self.classes[:]
            rng.shuffle(block)
            order.extend(block)

        for label in order[:self.batches_per_epoch]:
            pool = self.class_to_indices[label]
            n = min(self.batch_size, len(pool))
            if n < 2:
                continue
            # 不放回抽样，因此 batch 内不会有 duplicate identity。
            yield rng.sample(pool, n)


def _assert_strict_pre_ot_batch(batch: dict) -> str:
    labels_a = [str(x) for x in batch['label_A']]
    labels_b = [str(x) for x in batch['label_B']]
    x0_paths = [str(x) for x in batch['x0_path']]
    x1_paths = [str(x) for x in batch['x1_path']]

    if len(labels_a) < 2:
        raise RuntimeError("strict intra-class OT batch must contain at least 2 samples")
    if len(set(labels_a)) != 1 or labels_a != labels_b:
        raise RuntimeError(
            f"OT batch is not single-class: label_A={labels_a}, label_B={labels_b}"
        )
    if x0_paths != x1_paths:
        raise RuntimeError(
            "Pre-OT x0/x1 identities must be index-aligned so diagonal means self pairing"
        )
    if len(set(x0_paths)) != len(x0_paths):
        raise RuntimeError("Duplicate image identity detected inside one OT batch")
    return labels_a[0]


def _assert_strict_post_ot_pairing(cond: dict) -> None:
    src_labels = [str(x) for x in cond['x0_label_meta']]
    tgt_labels = [str(x) for x in cond['x1_label_meta']]
    src_paths = [str(x) for x in cond['x0_path_meta']]
    tgt_paths = [str(x) for x in cond['x1_path_meta']]

    bad_class = [(a, b) for a, b in zip(src_labels, tgt_labels) if a != b]
    bad_self = [(a, b) for a, b in zip(src_paths, tgt_paths) if a == b]
    if bad_class:
        raise RuntimeError(f"Cross-class OT pairing detected: {bad_class[:3]}")
    if bad_self:
        raise RuntimeError(
            "Self pairing survived constrained OT; check conditional_flow_matcher "
            f"diagonal masking. examples={bad_self[:3]}"
        )


def build_dataloaders(
    batch_size: int,
    transforms_dict: dict,
    *,
    data_glob: str,
    num_workers: int = 0,
    distributed: bool = False,
    seed: int = 42,
    min_class_size: int = 2,
    validate_paths: bool = True,
    batches_per_epoch: int = 0,
) -> tuple:
    """构建严格类内 one-class-per-batch DataLoader。"""
    dataset = StrictIntraClassOTDataset(
        data_glob=data_glob,
        transform=transforms_dict['transform'],
        hint_transform=transforms_dict['transform_grey'],
        transform_A=transforms_dict['transform_A'],
        transform_B=transforms_dict['transform'],
        min_class_size=min_class_size,
        validate_paths=validate_paths,
    )

    batch_sampler = OneClassBatchSampler(
        dataset.class_to_indices,
        batch_size=batch_size,
        world_size=get_world_size() if distributed else 1,
        rank=get_rank() if distributed else 0,
        seed=seed,
        batches_per_epoch=batches_per_epoch,
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    print(
        f"[INFO] Strict one-class batch sampler: rank={get_rank()}, "
        f"world={get_world_size()}, max_local_batch={batch_size}, "
        f"batches_per_epoch={len(batch_sampler)}, seed={seed}"
    )
    dataloopers = [infiniteloop(dataloader)]
    return dataloopers, dataset, batch_sampler

def build_rae(device: torch.device):
    """构建 RAE (Reconstruction Autoencoder) 模型.

    Returns:
        RAE 模型 (已加载预训练权重, eval 模式)
    """
    from rae.rae import RAE
    rae_model = RAE(
        encoder_name=_RAE_ENCODER_NAME,
        resolution=_RAE_RESOLUTION,
        decoder_config_path=_RAE_DECODER_CONFIG,
        pretrained_decoder_path=_RAE_DECODER_WEIGHT,
        noise_tau=0.0,
        normalization_stat_path=_RAE_NORMALIZATION_STATS,
    ).to(device).eval()
    print(f"[INFO] RAE 模型已加载, latent_dim={rae_model.latent_dim}")
    return rae_model


def build_vision_encoder(device: torch.device):
    """构建 EndoViT 视觉编码器及对应的图像预处理函数.

    Returns:
        tuple: (vision_model, process_image_fn)
    """
    process_fn = transforms.Compose([
        transforms.Resize((_IMAGE_INPUT_SIZE, _IMAGE_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_ENDOVIT_MEAN, std=_ENDOVIT_STD),
    ])

    def process_single_image(image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB')
        return process_fn(image)

    vision_model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ).eval()
    vision_model.load_state_dict(
        torch.load(_ENDOVIT_PATH, weights_only=False)
    )
    # 显式移动到目标设备, 避免权重遗漏在 CPU
    # (timm ViT 默认 .device='cuda' 不反映真实参数设备, 需手动同步)
    vision_model = vision_model.to(device)
    vision_model.device = device

    return vision_model, process_single_image


def build_text_encoder():
    """构建中文 CLIP 文本编码器及分词器.

    Returns:
        tuple: (text_model, text_tokenizer)
    """
    config_json = json.load(open(
        os.path.join(_TEXT_ENCODER_PATH, 'config.json'), 'r'
    ))
    config = ChineseCLIPTextConfig(**config_json)
    text_model = ChineseCLIPTextModel(config, False).eval()
    text_tokenizer = AutoTokenizer.from_pretrained(
        _TEXT_ENCODER_PATH, use_fast=True,
    )
    try:
        state_dict = load_file(
            os.path.join(_TEXT_ENCODER_PATH, 'model.safetensors')
        )
        text_model.load_state_dict(state_dict)
    except Exception:
        pass

    return text_model, text_tokenizer


def _unwrap_ddp(model):
    """解包 DDP 以访问子模块方法 (如 forward_features).

    DDP 包装后只能访问 .forward(), 无法直接调用 forward_features() 等自定义方法.
    此函数兼容单进程 (原始对象) 和 DDP (包装对象) 两种情况.

    Args:
        model: 原始模型或 DDP 包装的模型

    Returns:
        解包后的原始模型
    """
    # 递归解包: 兼容嵌套 DDP (一般只有一层)
    while hasattr(model, 'module'):
        model = model.module
    return model


def encode_text(
    caption: list,
    text_model,
    text_tokenizer,
    train_encoder: bool,
) -> torch.Tensor:
    """对 caption 进行文本编码.

    Args:
        caption: 文本描述列表
        text_model: 文本编码器模型 (原始对象或 DDP 包装)
        text_tokenizer: 分词器
        train_encoder: 是否在训练模式下编码（会影响梯度计算）

    Returns:
        torch.Tensor: 文本嵌入
    """
    if text_model is None:
        return torch.empty((len(caption),))

    # 解包 DDP (如果传入的是 DDP 包装对象), 用于获取设备等属性
    text_model_raw = _unwrap_ddp(text_model)

    # 获取模型真实的参数设备 (避免依赖可能不准的 .device 字符串属性)
    text_device = next(text_model_raw.parameters()).device
    context = torch.no_grad() if not train_encoder else torch.enable_grad()

    with context:
        caption_input = text_tokenizer(
            caption, return_tensors="pt", padding=True
        ).to(text_device)
        # 标准 forward: DDP 对象和原始对象都支持 __call__, DDP 会自动同步梯度
        caption_outputs = text_model(**caption_input)
        text_embeds = caption_outputs['last_hidden_state'].to(text_device)

    return text_embeds


def encode_vision(
    image_paths: list,
    vision_model,
    process_single_image_fn,
    train_encoder: bool,
) -> torch.Tensor:
    """对图像路径列表进行视觉编码.

    Args:
        image_paths: 图像路径列表
        vision_model: 视觉编码器模型
        process_single_image_fn: 图像预处理函数
        train_encoder: 是否在训练模式下编码

    Returns:
        torch.Tensor: 视觉嵌入
    """
    if vision_model is None:
        return torch.empty((len(image_paths),))

    # 解包 DDP (如果传入的是 DDP 包装对象)
    # forward_features() 是 timm ViT 的自定义方法, 不是标准 forward(),
    # DDP 包装后无法直接访问, 必须解包到原始模型.
    # 梯度同步说明: DDP 的 grad hooks 注册在底层参数上,
    # 即使通过 .module 做前向, backward 时仍会触发 all-reduce.
    vision_model_raw = _unwrap_ddp(vision_model)

    # 获取模型真实的参数设备
    # (timm ViT 默认 .device='cuda' 字符串不反映真实参数设备, 必须用参数设备)
    vision_device = next(vision_model_raw.parameters()).device
    context = torch.no_grad() if not train_encoder else torch.enable_grad()

    with context:
        x1_images = torch.stack(
            [process_single_image_fn(p) for p in image_paths]
        )
        vision_embeds = vision_model_raw.forward_features(
            x1_images.to(vision_device)
        )

    return vision_embeds


# ──────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(
        description="DiT Flow Matching Training Script (RAE native space)"
    )

    # 模型配置
    parser.add_argument(
        "--model", type=str, default="otcfm",
        choices=["otcfm", "icfm"],
        help="Flow matching 模型类型",
    )
    parser.add_argument(
        "--dit_variant", type=str, default="L",
        choices=["S", "B", "L", "L12"],
        help="DiT 变体: S(384d), B(768d), L(1024d), L12(1024d shallow)",
    )

    # 训练超参数
    parser.add_argument("--lr", type=float, default=2e-5, help="目标学习率")
    parser.add_argument("--grad_clip", type=float, default=2.0, help="梯度裁剪范数")
    parser.add_argument("--total_steps", type=int, default=15000, help="总训练步数")
    parser.add_argument("--warmup", type=int, default=1000, help="学习率预热步数")
    parser.add_argument("--batch_size", type=int, default=8, help="每个 rank 的最大类内 OT batch 大小")
    parser.add_argument(
        "--data_glob", type=str, default=_DATA_GLOB_PATTERN,
        help="按类 JSON 数据 glob；每个 batch 只从其中一个 label 采样",
    )
    parser.add_argument("--sampler_seed", type=int, default=42, help="one-class batch sampler 随机种子")
    parser.add_argument("--min_class_size", type=int, default=2, help="参与严格类内 OT 的最小 unique image 数")
    parser.add_argument(
        "--batches_per_epoch", type=int, default=0,
        help="0=按数据量/(batch_size*world_size) 自动估计；>0 时显式指定",
    )
    parser.add_argument(
        "--validate_data_paths", action=argparse.BooleanOptionalAction, default=True,
        help="加载 JSON 时检查 image/hint path 是否存在",
    )
    parser.add_argument(
        "--suboptimal_eps", type=float, default=_SUBOPTIMAL_EPS,
        help="constrained OT 中 near-identical squared-cost 屏蔽阈值；diagonal 始终屏蔽",
    )
    parser.add_argument(
        "--allow_pairing_mismatch_resume", action="store_true",
        help="允许从非 strict-intra/self-excluded OT checkpoint 恢复；默认拒绝",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 工作线程数")
    parser.add_argument("--image_size", type=int, default=256,
                        help="RAE/hint 输入分辨率；V1 默认与 RAE resolution 对齐为 256")
    parser.add_argument("--ema_decay", type=float, default=0.95, help="EMA 衰减率")
    parser.add_argument(
        "--op_match_batch", action=argparse.BooleanOptionalAction, default=True,
        help="必须保持开启；strict intra-class 版本依赖 batch-level OT matching",
    )
    parser.add_argument("--reflow", type=bool, default=False, help="是否使用 Reflow 矫正流")
    # V1 条件配置：目标视觉 CLS + patch tokens + 弱噪声灰度 hint。
    # caption 与 class ID 默认不进入 DiT，但保留 legacy 开关用于消融。
    parser.add_argument(
        "--caption_mode", type=str, default="vision_only",
        choices=["vision_only", "cat", "only_text"],
        help="vision_only=仅目标视觉 patch；cat=视觉+文本；only_text=仅文本",
    )
    parser.add_argument(
        "--use_class_condition", action=argparse.BooleanOptionalAction, default=False,
        help="是否将 class ID embedding 输入 DiT；V1 默认关闭",
    )
    parser.add_argument(
        "--train_text_encoder", action=argparse.BooleanOptionalAction, default=False,
        help="是否训练文本编码器；vision_only 下强制关闭",
    )
    parser.add_argument(
        "--train_vision_encoder", action=argparse.BooleanOptionalAction, default=True,
        help="是否训练目标视觉编码器",
    )
    parser.add_argument(
        "--hint_signal_ratio", type=float, default=0.10,
        help="灰度 hint 信号系数 alpha，噪声系数为 sqrt(1-alpha^2)",
    )
    parser.add_argument(
        "--hint_dropout", type=float, default=0.20,
        help="逐样本将 hint 置零的概率",
    )
    parser.add_argument(
        "--pure_noise_hint", action="store_true",
        help="仅用于消融：hint 使用纯随机噪声",
    )
    parser.add_argument(
        "--use_cross_attention", action=argparse.BooleanOptionalAction, default=True,
        help="是否使用 cross-attention 注入 encoder_hidden_states",
    )
    parser.add_argument("--num_classes", type=int, default=1000,
                        help="class embedding 数量（结构兼容；V1 默认不使用）")

    # 评估与保存
    parser.add_argument("--save_step", type=int, default=5000, help="checkpoint 保存间隔（0 表示禁用）")
    parser.add_argument("--keep_n_checkpoints", type=int, default=10, help="保留最近 N 个 checkpoint")
    parser.add_argument(
        "--allow_condition_mismatch_resume", action="store_true",
        help="允许从条件配置不同或缺失的旧 checkpoint 恢复；默认拒绝，避免误续训",
    )

    # 路径配置
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(_PROJECT_ROOT, "weights_dit_flow/L_ddp"),
        help="输出目录",
    )
    parser.add_argument(
        "--image_dir", type=str, default="",
        help="训练图片目录",
    )
    parser.add_argument(
        "--log_dir", type=str,
        default=os.path.join(_PROJECT_ROOT, "logs_dit_flow"),
        help="TensorBoard 日志目录",
    )

    # ── GAN 对抗训练参数 ──
    parser.add_argument(
        "--use_gan", action="store_true", default=False,
        help="是否启用 GAN 判别器进行对抗训练 (默认关闭)",
    )
    parser.add_argument(
        "--disc_start_step", type=int, default=100,
        help="GAN 判别器开始训练的步数 (先让 Flow Matching 稳定)",
    )
    parser.add_argument(
        "--gan_every_n_steps", type=int, default=1,
        help="每隔 N 步执行一次 GAN 训练 (1=每步都训练)",
    )
    parser.add_argument(
        "--disc_lr", type=float, default=1e-5,
        help="判别器学习率",
    )

    # ── 最优传输 (OT) 正则项参数 ──
    parser.add_argument(
        "--ot_reg_weight", type=float, default=0.001,
        help="OT 正则项权重 (0=禁用, 建议先从 0.001 起). "
             "对 paired 样本内部的 RAE spatial-token 分布施加 Sinkhorn divergence",
    )
    parser.add_argument(
        "--ot_reg", type=float, default=0.1,
        help="Sinkhorn entropy 正则系数 (cost 归一化后建议 0.05~0.5)",
    )
    parser.add_argument(
        "--ot_reg_iters", type=int, default=20,
        help="Sinkhorn 迭代次数 (建议 20~50)",
    )
    parser.add_argument(
        "--ot_max_tokens", type=int, default=256,
        help="每个样本参与 spatial OT 的最大 token 数。"
             "16x16 latent 原样使用；32x32 会仅在 OT 分支自动池化到 <=256 token",
    )
    parser.add_argument(
        "--ot_reg_start_step", type=int, default=0,
        help="OT 正则项开始生效的步数 (0=从一开始启用; "
             "可设较大值让 Flow Matching 先稳定再加 OT 约束)",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────
# 训练主逻辑
# ──────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """主训练函数.

    Args:
        args: 命令参数
    """
    # ── DDP 初始化 (torchrun 启动时, 环境变量已注入) ──
    local_rank = setup_ddp()
    ddp = is_dist_avail()
    world_size = get_world_size()
    rank = get_rank()
    is_main = is_main_process()

    # ── 训练配置 ──
    # DDP 模式下每个进程绑定到自己的 GPU (cuda:local_rank)
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    caption_hidden_states_mode = args.caption_mode
    train_text_encoder = bool(args.train_text_encoder and caption_hidden_states_mode != 'vision_only')
    train_vision_encoder = bool(args.train_vision_encoder)
    train_dit = True          # 是否训练 DiT
    use_gan = args.use_gan     # 是否启用 GAN 对抗训练

    if args.model != "otcfm":
        raise ValueError(
            "strict intra-class self-excluded 版本只支持 --model otcfm；"
            "因为训练协议必须使用 batch OT matching。"
        )
    if not bool(args.op_match_batch):
        raise ValueError("strict intra-class 版本要求 --op_match_batch=True")
    if args.batch_size < 2:
        raise ValueError("strict self-excluded OT requires --batch_size >= 2")

    prefix = f"[rank {rank}]" if ddp else ""
    print(f"{prefix}[INFO] DiT Flow Matching 训练"
          f"{' (DDP, world_size=' + str(world_size) + ')' if ddp else ' (单进程)'}")
    print(f"{prefix}[INFO] DiT 变体: {args.dit_variant}")
    print(f"{prefix}[INFO] 设备: {device}")
    print(f"{prefix}[INFO] 训练开关: DiT={train_dit}, Text={train_text_encoder}, "
          f"Vision={train_vision_encoder}, GAN={use_gan}")
    print(f"{prefix}[INFO] Caption 模式: {caption_hidden_states_mode}")
    print(f"{prefix}[INFO] Class ID condition: {args.use_class_condition}")
    print(f"{prefix}[INFO] Hint: alpha={args.hint_signal_ratio}, "
          f"dropout={args.hint_dropout}, pure_noise={args.pure_noise_hint}")
    print(f"{prefix}[INFO] RAE input: [0,1], image_size={args.image_size}")
    print(f"{prefix}[INFO] Cross-Attention: {args.use_cross_attention}")
    print(
        f"{prefix}[INFO] Pairing protocol: STRICT_INTRA_CLASS_SELF_EXCLUDED_OT | "
        f"one_class_per_local_batch=True | identity_aligned_pre_ot=True | "
        f"suboptimal_eps={args.suboptimal_eps}"
    )
    if args.ot_reg_weight > 0.0:
        print(f"{prefix}[INFO] Spatial Sinkhorn OT: weight={args.ot_reg_weight}, "
              f"reg={args.ot_reg}, iters={args.ot_reg_iters}, "
              f"max_tokens={args.ot_max_tokens}, "
              f"start_step={args.ot_reg_start_step}")
    else:
        print(f"{prefix}[INFO] Spatial Sinkhorn OT: 禁用")

    # ── 创建输出/日志目录 ──
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    # SummaryWriter 仅在主进程创建, 避免多进程同时写 TB 日志冲突
    writer = SummaryWriter(log_dir=args.log_dir) if is_main else None

    # ── 数据加载 ──
    transforms_dict = build_transforms(args.image_size)
    dataloopers, dataset, sampler = build_dataloaders(
        args.batch_size, transforms_dict,
        data_glob=args.data_glob,
        num_workers=args.num_workers,
        distributed=ddp,
        seed=args.sampler_seed,
        min_class_size=args.min_class_size,
        validate_paths=args.validate_data_paths,
        batches_per_epoch=args.batches_per_epoch,
    )

    # ── DiT 模型初始化 ──
    dit_model = create_dit_rae(
        variant=args.dit_variant,
        rae_latent_dim=1024,
        rae_spatial_size=16,
        cross_attn_dim=768,
        num_classes=args.num_classes,
        use_cross_attention=args.use_cross_attention,
    ).to(device)

    dit_info = dit_model.get_model_info()
    print(f"[INFO] DiT 模型: hidden_dim={dit_info['hidden_dim']}, "
          f"depth={dit_info['depth']}, heads={dit_info['num_heads']}, "
          f"params={dit_info['total_parameters']:,}")

    # ── 视觉编码器 (EndoViT) ──
    vision_model, process_single_image_fn = build_vision_encoder(device)

    # ── 文本编码器：V1 vision_only 不加载，避免无效显存/优化器开销 ──
    if caption_hidden_states_mode == 'vision_only':
        text_model, text_tokenizer = None, None
    else:
        text_model, text_tokenizer = build_text_encoder()

    # ── RAE (Reconstruction Autoencoder) ──
    rae_model = build_rae(device)

    # ── GAN 判别器 (可选) ──
    discriminator = None
    disc_aug = None
    lpips_model = None
    disc_loss_fn = None
    gen_loss_fn = None
    if use_gan:
        # 确定并验证 DINO checkpoint 路径
        dino_ckpt = _DINO_CKPT_PATH
        if not os.path.exists(dino_ckpt):
            # 尝试备选路径
            alt_path = os.path.join(
                _PROJECT_ROOT, 'flow_matcher_otcfm', 'rae_v2',
                'dino_deitsmall8_300ep_pretrain.pth'
            )
            if os.path.exists(alt_path):
                dino_ckpt = alt_path
            else:
                raise FileNotFoundError(
                    f"DINO checkpoint 未找到: {_DINO_CKPT_PATH} 及备选路径 {alt_path}"
                )

        # 根据 checkpoint 文件名推断 recipe
        recipe = 'S_8' if ('patch8' in dino_ckpt or 'small8' in dino_ckpt) else 'S_16'

        discriminator, disc_aug = build_discriminator(
            dino_ckpt_path=dino_ckpt,
            device=device,
            ks=4,
            recipe=recipe,
            aug_prob=_GAN_DIFAUG_PROB,
            aug_cutout=_GAN_DIFAUG_CUTOUT,
        )
        discriminator.train()
        print(f"[INFO] GAN 判别器已构建 (recipe={recipe}), "
              f"disc_start_step={args.disc_start_step}, "
              f"gan_every_n_steps={args.gan_every_n_steps}")

        # 选择 GAN 损失函数
        disc_loss_fn, gen_loss_fn = select_gan_losses(_GAN_DISC_LOSS, _GAN_GEN_LOSS)
        print(f"[INFO] GAN 损失函数: disc={disc_loss_fn.__name__}, gen={gen_loss_fn.__name__}")

        # LPIPS 感知损失 (可选)
        if _GAN_LPIPS_WEIGHT > 0:
            lpips_model = LPIPS().to(device).eval()
            print(f"[INFO] LPIPS 感知损失已加载, weight={_GAN_LPIPS_WEIGHT}")

    # ── DiT 冻结/训练设置 ──
    if train_dit:
        dit_model.train()
    else:
        dit_model.eval()
        for param in dit_model.parameters():
            param.requires_grad = False
        print("[INFO] DiT 已冻结, 不参与训练")

    # ── EMA 模型 ──
    ema_model = copy.deepcopy(dit_model)

    # ── DiT 优化器 ──
    dit_optim = dit_sched = None
    if train_dit:
        dit_optim = torch.optim.AdamW(
            dit_model.parameters(), lr=args.lr, weight_decay=0.001
        )
        dit_sched = torch.optim.lr_scheduler.LambdaLR(
            dit_optim, lr_lambda=lambda step: warmup_lr(step, args.warmup)
        )
    else:
        print("[INFO] DiT 优化器已跳过")

    text_optim = text_sched = None
    if train_text_encoder and text_model is not None:
        text_model.train()
        text_optim = torch.optim.AdamW(
            text_model.parameters(), lr=args.lr * 0.5, weight_decay=0.001
        )
        text_sched = torch.optim.lr_scheduler.LambdaLR(
            text_optim, lr_lambda=lambda step: warmup_lr(step, args.warmup)
        )

    vision_optim = vision_sched = None
    if train_vision_encoder and vision_model is not None:
        vision_model.train()
        vision_optim = torch.optim.AdamW(
            vision_model.parameters(), lr=args.lr * 0.5, weight_decay=0.001
        )
        vision_sched = torch.optim.lr_scheduler.LambdaLR(
            vision_optim, lr_lambda=lambda step: warmup_lr(step, args.warmup)
        )

    # ── GAN 判别器优化器 (独立于生成器) ──
    disc_optim = disc_sched = None
    if use_gan and discriminator is not None:
        # 只训练判别器头 (DINO backbone 已冻结)
        disc_trainable_params = list(discriminator.heads.parameters())
        disc_optim = torch.optim.AdamW(
            disc_trainable_params, lr=args.disc_lr, weight_decay=0.001
        )
        disc_sched = torch.optim.lr_scheduler.LambdaLR(
            disc_optim, lr_lambda=lambda step: warmup_lr(step, args.warmup)
        )
        num_disc_params = sum(p.numel() for p in disc_trainable_params)
        print(f"[INFO] 判别器优化器已创建, 可训练参数量: {num_disc_params:,}")

    # ── Flow Matcher 初始化 ──
    sigma = 0.1
    if args.model == "otcfm":
        FM = OptimalTransportConditionalFlowMatcher(
            sigma=sigma,
            ot_method=_OT_METHOD,
            ot_reg=_OT_REG,
            ot_adaptive_reg=_OT_ADAPTIVE_REG,
            ot_num_iter_max=_OT_NUM_ITER_MAX,
            ot_stop_thr=_OT_STOP_THR,
            ot_warn=_OT_WARN,
        )
    elif args.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {args.model}, must be one of ['otcfm', 'icfm']"
        )

    fm_call_sig = inspect.signature(FM.get_sample_location_and_conditional_flow)
    for required_name in ("intra_class_suboptimal", "suboptimal_eps"):
        if required_name not in fm_call_sig.parameters:
            raise RuntimeError(
                "Current conditional_flow_matcher is too old for strict intra-class OT: "
                f"missing argument {required_name!r}"
            )

    # ── 检查点保存目录 ──
    savedir = os.path.join(args.output_dir, args.model)
    os.makedirs(savedir, exist_ok=True)

    # ── 恢复训练 ──
    latest_model = find_latest_checkpoint(savedir)
    start_step = 1
    if latest_model:
        print("加载 latest model.")
        checkpoint = torch.load(latest_model, map_location=device, weights_only=True)
        expected_condition = {
            "condition_mode": (
                "vision_hint_v1"
                if caption_hidden_states_mode == "vision_only" and not args.use_class_condition
                else "legacy_or_ablation"
            ),
            "caption_mode": caption_hidden_states_mode,
            "use_class_id": bool(args.use_class_condition),
            "hint_signal_ratio": float(args.hint_signal_ratio),
            "pure_noise_hint": bool(args.pure_noise_hint),
            "rae_input_range": "[0,1]",
            "image_size": int(args.image_size),
        }
        saved_condition = checkpoint.get("condition_config")
        expected_pairing = {
            "pairing_mode": "strict_intra_class_self_excluded_ot",
            "one_class_per_local_batch": True,
            "identity_aligned_pre_ot": True,
            "unique_identity_within_batch": True,
            "ot_sample_plan": True,
        }
        saved_pairing = checkpoint.get("pairing_config")
        pairing_mismatch = (
            saved_pairing is None
            or saved_pairing.get("pairing_mode") != expected_pairing["pairing_mode"]
            or not bool(saved_pairing.get("one_class_per_local_batch", False))
            or not bool(saved_pairing.get("identity_aligned_pre_ot", False))
            or not bool(saved_pairing.get("unique_identity_within_batch", False))
            or not bool(saved_pairing.get("ot_sample_plan", False))
        )
        if pairing_mismatch and not args.allow_pairing_mismatch_resume:
            raise RuntimeError(
                "检测到已有 checkpoint 不是 strict intra-class self-excluded OT 协议。\n"
                f"  checkpoint={latest_model}\n"
                f"  saved_pairing={saved_pairing}\n"
                f"  expected_pairing={expected_pairing}\n"
                "为避免把旧的 mixed-batch OT 权重误当成严格类内 OT 继续训练，默认拒绝恢复。"
                "如果只是明确想用旧权重初始化，请显式加入 --allow_pairing_mismatch_resume。"
            )
        if pairing_mismatch:
            print(f"[WARN] 允许 pairing protocol 不一致恢复: saved={saved_pairing}")
        mismatch = (
            saved_condition is None
            or saved_condition.get("condition_mode") != expected_condition["condition_mode"]
            or saved_condition.get("caption_mode") != expected_condition["caption_mode"]
            or bool(saved_condition.get("use_class_id", False)) != expected_condition["use_class_id"]
            or abs(float(saved_condition.get("hint_signal_ratio", -1.0)) - expected_condition["hint_signal_ratio"]) > 1e-8
            or bool(saved_condition.get("pure_noise_hint", False)) != expected_condition["pure_noise_hint"]
            or saved_condition.get("rae_input_range") != expected_condition["rae_input_range"]
            or int(saved_condition.get("image_size", -1)) != expected_condition["image_size"]
        )
        if mismatch and not args.allow_condition_mismatch_resume:
            raise RuntimeError(
                "检测到输出目录中已有 checkpoint，但其 DiT 条件配置与当前 V1 不一致。\n"
                f"  checkpoint={latest_model}\n"
                f"  saved={saved_condition}\n"
                f"  current={expected_condition}\n"
                "请使用新的 output_dir；确需旧权重初始化时显式加入 "
                "--allow_condition_mismatch_resume。"
            )
        if mismatch:
            print(f"[WARN] 允许条件不一致恢复: saved={saved_condition}, current={expected_condition}")
        dit_model.load_state_dict(checkpoint['dit_model'], strict=False)
        ema_model.load_state_dict(checkpoint['ema_model'], strict=False)
        start_step = checkpoint['step']
        if checkpoint.get('text_model') is not None and text_model is not None:
            text_model.load_state_dict(checkpoint['text_model'], strict=False)
        if checkpoint.get('vision_model') is not None and vision_model is not None:
            vision_model.load_state_dict(checkpoint['vision_model'], strict=False)
        # 恢复判别器权重 (如果存在且启用 GAN)
        if use_gan and 'discriminator' in checkpoint and discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'], strict=False)
            print(f"[INFO] 判别器权重已从检查点恢复")
        print(f"[INFO] 从 step {start_step} 恢复训练")

    # ── 模型设备放置 (DDP 下每个进程的模型需显式放到对应 GPU) ──
    # 注意: build_text_encoder() 不做 .to(device), 必须在此处统一迁移,
    #       否则 DDP 包装时会报 "module parameters {device(type='cpu')}" 错误.
    rae_model = rae_model.to(device)
    dit_model = dit_model.to(device)
    if text_model is not None:
        text_model = text_model.to(device)
    if vision_model is not None:
        vision_model = vision_model.to(device)
    if discriminator is not None:
        discriminator = discriminator.to(device)

    # 为 DiT 设置 .device 属性 (前向时多处使用 dit_model.device)
    dit_model.device = device

    # ── DDP 包装 (数据并行, 取代旧的 dispatch_model 模型并行) ──
    # 每个进程持有完整模型副本, 处理不同 batch, 反向传播时自动 all-reduce 梯度.
    # 只包装需要梯度同步的可训练模型; RAE 冻结不包装.
    # find_unused_parameters=True: 因为不是所有参数每步都参与 (如条件分支)
    dit_ddp = text_ddp = vision_ddp = disc_ddp = None

    if ddp:
        if train_dit:
            dit_ddp = DDP(
                dit_model, device_ids=[local_rank],
                find_unused_parameters=True,
            )
        if train_text_encoder and text_model is not None:
            text_ddp = DDP(
                text_model, device_ids=[local_rank],
                find_unused_parameters=True,
            )
        if train_vision_encoder and vision_model is not None:
            vision_ddp = DDP(
                vision_model, device_ids=[local_rank],
                find_unused_parameters=True,
            )
        if use_gan and discriminator is not None:
            disc_ddp = DDP(
                discriminator, device_ids=[local_rank],
                find_unused_parameters=True,
            )
        print(f"[rank {rank}][INFO] DDP 包装完成: "
              f"DiT={'Y' if dit_ddp else 'N'}, Text={'Y' if text_ddp else 'N'}, "
              f"Vision={'Y' if vision_ddp else 'N'}, Disc={'Y' if disc_ddp else 'N'}")

    # 前向传播用的模型引用: DDP 模式下用 *_ddp, 否则用原始对象
    # 注意: state_dict/EMA/grad_clip 仍用原始对象 (dit_model 等), 避免前缀问题
    dit_fwd = dit_ddp if dit_ddp is not None else dit_model
    text_fwd = text_ddp if text_ddp is not None else text_model
    vision_fwd = vision_ddp if vision_ddp is not None else vision_model
    disc_fwd = disc_ddp if disc_ddp is not None else discriminator

    global_step = start_step
    # strict 版本永远启用 batch OT；不存在 Blank / Inter step。
    op_match = bool(args.op_match_batch)

    # ══════════════════════════════════════════
    # 训练循环
    # ══════════════════════════════════════════
    with trange(
        start_step, args.total_steps,
        initial=start_step, total=args.total_steps,
        dynamic_ncols=True,
    ) as step_pbar:
        for step in step_pbar:
            global_step += 1

            # 判断当前步是否执行 GAN 训练
            do_gan = (
                use_gan
                and global_step >= args.disc_start_step
                and (global_step - args.disc_start_step) % args.gan_every_n_steps == 0
            )

            # ── 梯度清零 ──
            if dit_optim is not None:
                dit_optim.zero_grad()
            if text_optim is not None:
                text_optim.zero_grad()
            if vision_optim is not None:
                vision_optim.zero_grad()
            if disc_optim is not None and do_gan:
                disc_optim.zero_grad()

            # ── 严格协议：每一步都只做类内、排除自身的 constrained OT ──
            is_blank_step = False
            is_intra_step = True

            # ── 获取训练批次 ──
            datalooper = random.choice(dataloopers)
            batch = next(datalooper)
            batch_class = _assert_strict_pre_ot_batch(batch)

            x0 = batch['x0']
            x1 = batch['x1']
            x1_path = batch['x1_path']
            caption = batch['caption']

            hint_base = batch['hint']
            if not 0.0 <= args.hint_signal_ratio <= 1.0:
                raise ValueError(f"hint_signal_ratio 必须在 [0,1]，当前={args.hint_signal_ratio}")
            if not 0.0 <= args.hint_dropout <= 1.0:
                raise ValueError(f"hint_dropout 必须在 [0,1]，当前={args.hint_dropout}")
            if args.pure_noise_hint:
                hint = torch.randn_like(hint_base)
            else:
                alpha = float(args.hint_signal_ratio)
                hint = (
                    alpha * hint_base
                    + math.sqrt(max(1.0 - alpha * alpha, 0.0))
                    * torch.randn_like(hint_base)
                )
                if args.hint_dropout > 0.0:
                    keep = (
                        torch.rand(hint.shape[0], 1, 1, 1, device=hint.device)
                        >= args.hint_dropout
                    ).to(hint.dtype)
                    hint = hint * keep

            y = batch['class_id']

            # ── RAE 编码 (直接在 RAE 空间, 无需 Mapper) ──
            # RAE 编码: (B, 3, H, W) -> (B, 1024, 16, 16)
            # DiT 直接在此空间运行, 无需维度映射
            images_for_rae = torch.cat([
                batch['x0'].to(device),
                batch['x1'].to(device),
            ], dim=0)
            # Dataset transform normalizes to [-1,1], while RAE.encode is used
            # with [0,1] inputs in the RAE training and trajectory scripts.
            images_for_rae = ((images_for_rae + 1.0) / 2.0).clamp(0.0, 1.0)
            with torch.no_grad():
                z_rae = rae_model.encode(images_for_rae)  # (2B, 1024, 16, 16)
            z_rae_0, z_rae_1 = z_rae.chunk(2, dim=0)

            # Flow Matching 直接在 RAE 空间: (B, 1024, 16, 16)
            x0 = z_rae_0  # (B, 1024, 16, 16)
            x1 = z_rae_1  # (B, 1024, 16, 16)

            # ── 文本条件：V1 vision_only 完全关闭，不用空字符串 embedding 代替 ──
            if caption_hidden_states_mode == 'vision_only':
                text_embeds = None
            else:
                text_embeds = encode_text(
                    caption, text_fwd, text_tokenizer, train_text_encoder
                )

            # ── 目标视觉条件：CLS 用于 AdaLN，patch tokens 用于 Cross-Attention ──
            vision_embeds = encode_vision(
                x1_path, vision_fwd, process_single_image_fn, train_vision_encoder
            )

            if caption_hidden_states_mode == 'vision_only':
                caption_hidden_states = vision_embeds[:, 1:, ...].to(dit_model.device)
            elif caption_hidden_states_mode == 'cat':
                caption_hidden_states = torch.cat([
                    vision_embeds[:, 1:, ...].to(dit_model.device),
                    text_embeds[:, 1:, ...].to(dit_model.device),
                ], dim=1)
            elif caption_hidden_states_mode == 'only_text':
                caption_hidden_states = text_embeds[:, 1:, ...].to(dit_model.device)
            else:
                raise ValueError(
                    f"Unknown caption_hidden_states_mode: {caption_hidden_states_mode}"
                )

            # FM matcher 仍保留 y 用于样本重排/元数据；是否输入 DiT 由 use_class_condition 决定。
            text_cls_for_cond = (
                text_embeds[:, 0].to(dit_model.device)
                if text_embeds is not None
                else torch.zeros_like(vision_embeds[:, 0].to(dit_model.device))
            )
            cond = {
                'x0': x0.to(dit_model.device),
                'x1': x1.to(dit_model.device),
                'caption': caption,
                'caption_hidden_states': caption_hidden_states.to(dit_model.device),
                'y': y.to(dit_model.device),
                'hint': hint.to(dit_model.device),
                'text_embeds': text_cls_for_cond,
                'image_embeds': vision_embeds[:, 0].to(dit_model.device),
                'x1_pixel': batch['x1'].to(dit_model.device),
                # 仅用于 OT 后强校验；matcher 会按 source i / target j 同步重排。
                'x0_path_meta': list(batch['x0_path']),
                'x1_path_meta': list(batch['x1_path']),
                'x0_label_meta': list(batch['label_A']),
                'x1_label_meta': list(batch['label_B']),
            }

            t, xt, ut = FM.get_sample_location_and_conditional_flow(
                x0, x1,
                sample_plan=op_match,
                cond=cond,
                print_info=False,
                replace=False,
                intra_class_suboptimal=True,
                suboptimal_eps=float(args.suboptimal_eps),
            )
            _assert_strict_post_ot_pairing(cond)

            # ── 前向传播 ──
            xt = xt.to(dit_model.device)
            t = t.to(dit_model.device)
            ut = ut.to(dit_model.device)
            caption_hidden_states = cond['caption_hidden_states'].to(dit_model.device)
            y = cond['y']

            # DiT 前向: 冻结时用 no_grad, 训练时计算梯度
            # DDP 模式下用 dit_fwd (DDP 包装), 保证反向传播时梯度 all-reduce
            context_dit = torch.enable_grad() if train_dit else torch.no_grad()
            with context_dit:
                vt_pred = dit_fwd(
                    xt=xt,
                    timestep=t.reshape(-1),
                    class_labels=(y if args.use_class_condition else None),
                    encoder_hidden_states=caption_hidden_states,
                    hint=cond['hint'],
                    text_embeds=(
                        cond['text_embeds']
                        if caption_hidden_states_mode != 'vision_only'
                        else None
                    ),
                    image_embeds=cond['image_embeds'],
                )['sample']

            # ── Flow Matching 重建损失 ──
            loss_fm = torch.mean((vt_pred - ut) ** 2)

            # ── Spatial Sinkhorn OT 正则项 (可选) ──
            # v-space: 继续由 paired FM MSE 精确监督。
            # x-space: 用同一样本内部的 spatial-token Sinkhorn 约束结构分布。
            #
            # 这里不直接使用 xt + (1-t)*vt_pred 作为 clean endpoint，原因是
            # CFM/OT-CFM 训练中的 xt 可能含 sigma 噪声。改用训练期等价的
            # clean endpoint surrogate：
            #   x1_pred_clean = x1 + (1-t) * (vt_pred - ut)
            # 因而当 vt_pred == ut 时，严格有 x1_pred_clean == x1，
            # OT 与主 FM loss 在最优点不会互相冲突。
            loss_ot = torch.tensor(0.0, device=device)
            do_ot = (
                args.ot_reg_weight > 0.0
                and global_step >= args.ot_reg_start_step
            )
            if do_ot:
                t_val_ot = t.reshape(-1)
                t_expand_ot = t_val_ot.view(-1, 1, 1, 1)

                # cond['x1'] 已随 OT batch matching 重排，与 xt/ut/vt_pred 顺序对齐。
                x1_target_for_ot = cond['x1'].to(vt_pred.device).detach()
                x1_pred_clean_for_ot = (
                    x1_target_for_ot
                    + (1.0 - t_expand_ot) * (vt_pred - ut)
                )

                loss_ot = sinkhorn_ot_loss(
                    x1_pred_clean_for_ot,
                    x1_target_for_ot,
                    reg=args.ot_reg,
                    n_iters=args.ot_reg_iters,
                    max_tokens=args.ot_max_tokens,
                )

            # ── GAN 对抗损失 (在 disc_start_step 之后才启用) ──
            loss_gan_gen = torch.tensor(0.0, device=device)
            loss_gan_disc = torch.tensor(0.0, device=device)
            adaptive_w = torch.tensor(1.0, device=device)

            if do_gan:
                # ── 1) 从 DiT 预测的速度场估算生成的目标潜变量 (保留梯度) ──
                # Flow Matching: xt = t * x1 + (1 - t) * x0
                # vt = x1 - x0 (在标准 CFM 中), 所以 x1_pred ≈ xt + (1 - t) * vt
                #
                # 注意: DiT 直接在 RAE 空间运行, 输出 vt_pred 已是 RAE 潜变量 (B, 1024, 16, 16),
                # 无需 mapper.inverse() 转换, 直接用 RAE 解码即可.
                # 梯度穿过 rae_model.decode() 回传到 DiT 参数.
                t_val = t.squeeze()
                if t_val.dim() == 0:
                    t_val = t_val.unsqueeze(0)
                t_expand = t_val.view(-1, 1, 1, 1)

                # 估算目标潜变量: x1_pred = xt + (1 - t) * vt (梯度可流回 DiT)
                x1_pred = xt + (1.0 - t_expand) * vt_pred  # (B, 1024, 16, 16)

                # RAE 解码 (DiT 输出已在 RAE 空间, 无需 mapper.inverse())
                fake_images = rae_model.decode(x1_pred)  # (B, 3, H, W)

                # 归一化到 [-1, 1] 范围 (判别器期望)
                if fake_images.max() > 1.0:
                    fake_images = fake_images / 255.0
                fake_images = fake_images * 2.0 - 1.0   # [0,1] -> [-1,1]
                fake_images = fake_images.clamp(-1.0, 1.0)

                # 真实图像 (已通过 transforms.Normalize 在 [-1, 1])
                real_images = cond['x1_pixel']  # 已经过 OT 重排，与 fake_images 顺序一致

                # 调整尺寸到判别器期望的大小 (224x224)
                fake_images_resized = F.interpolate(
                    fake_images, size=(224, 224),
                    mode='bilinear', align_corners=False,
                )
                real_images_resized = F.interpolate(
                    real_images, size=(224, 224),
                    mode='bilinear', align_corners=False,
                )

                # ── 2) 判别器训练步骤 (使用 detached fake_images) ──
                # 判别器梯度已在循环开头清零
                # DDP: 判别器紧接着会 step(), 梯度同步无意义, 用 no_sync 跳过 all-reduce
                #      (DDP 模式下用 disc_fwd 以保证 DDP 前向 hook 正常注册)
                disc_sync_ctx = disc_ddp.no_sync() if disc_ddp is not None else nullcontext()
                with disc_sync_ctx:
                    fake_augmented_d = disc_aug.aug(fake_images_resized.detach())
                    real_augmented = disc_aug.aug(real_images_resized)

                    logits_fake_d, logits_real_d = disc_fwd(
                        fake_augmented_d, real_augmented,
                    )
                    loss_gan_disc = disc_loss_fn(logits_real_d, logits_fake_d)
                    loss_gan_disc.backward()

                # 判别器梯度裁剪 & 步进
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(), args.grad_clip,
                )
                disc_optim.step()
                disc_sched.step()

                # ── 3) 生成器对抗损失 (梯度穿过 fake_images 回传到 DiT) ──
                # 临时冻结判别器参数, 避免 total_loss.backward() 时
                # 浪费显存计算判别器参数的梯度
                disc_heads_requires_grad = [
                    p.requires_grad for p in discriminator.heads.parameters()
                ]
                discriminator.dino_proxy[0].requires_grad_(False)
                for p in discriminator.heads.parameters():
                    p.requires_grad_(False)

                fake_augmented_g = disc_aug.aug(fake_images_resized)
                logits_fake_g, _ = disc_fwd(fake_augmented_g, None)
                loss_gan_gen_raw = gen_loss_fn(logits_fake_g)

                # 计算自适应权重 (平衡 Flow Matching loss 和 GAN loss 的梯度)
                ref_layer = None
                if train_dit:
                    for p in dit_model.parameters():
                        if p.requires_grad:
                            ref_layer = p
                            break

                if ref_layer is not None:
                    try:
                        adaptive_w = calculate_adaptive_weight(
                            loss_fm, loss_gan_gen_raw, ref_layer,
                        )
                    except Exception:
                        adaptive_w = torch.tensor(1.0, device=device)

                loss_gan_gen = adaptive_w * loss_gan_gen_raw

                # LPIPS 感知损失 (可选)
                if lpips_model is not None and _GAN_LPIPS_WEIGHT > 0:
                    with torch.no_grad():
                        loss_lpips = lpips_model(
                            fake_images_resized.detach(),
                            real_images_resized,
                        )
                    loss_gan_gen = loss_gan_gen + _GAN_LPIPS_WEIGHT * loss_lpips

                # 恢复判别器 heads 的可训练状态 (供下一步使用)
                for p, rg in zip(
                    discriminator.heads.parameters(), disc_heads_requires_grad
                ):
                    p.requires_grad_(rg)

            # ── 总损失 = FM loss + 加权 GAN 生成器损失 + 加权 OT 正则项 ──
            total_loss = loss_fm + loss_gan_gen
            if do_ot:
                total_loss = total_loss + args.ot_reg_weight * loss_ot

            # ── 反向传播 ──
            total_loss.backward()

            dit_grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    dit_model.parameters(), args.grad_clip
                ) if dit_optim is not None else torch.tensor(-1.0)
            )
            text_grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    text_model.parameters(), args.grad_clip * 0.2
                ) if text_optim is not None else torch.tensor(-1.0)
            )
            vision_grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    vision_model.parameters(), args.grad_clip * 0.2
                ) if vision_optim is not None else torch.tensor(-1.0)
            )

            # ── 优化器步进 (各自独立, 判别器已在上方步进) ──
            if dit_optim is not None:
                dit_optim.step()
                dit_sched.step()
            if text_optim is not None:
                text_optim.step()
                text_sched.step()
            if vision_optim is not None:
                vision_optim.step()
                vision_sched.step()

            # ── EMA 更新 (仅 DiT 训练时) ──
            if train_dit:
                ema(dit_model, ema_model, args.ema_decay)
            ema_model = ema_model.to(dit_model.device)

            # ── TensorBoard 日志 (仅主进程写入, 避免多进程写冲突) ──
            if is_main and writer is not None:
                writer.add_scalar('Training/Loss_FM', loss_fm.item(), global_step)
                writer.add_scalar('Training/TotalLoss', total_loss.item(), global_step)
                if args.ot_reg_weight > 0.0:
                    writer.add_scalar('Training/Loss_OT', loss_ot.item(), global_step)
                    writer.add_scalar('Training/OTWeighted', (args.ot_reg_weight * loss_ot).item(), global_step)
                writer.add_scalar('Training/DiTGradNorm', dit_grad_norm.item(), global_step)
                writer.add_scalar('Training/TextGradNorm', text_grad_norm.item(), global_step)
                writer.add_scalar('Training/VisionGradNorm', vision_grad_norm.item(), global_step)
                if dit_optim is not None:
                    writer.add_scalar('LearningRate/DiT', dit_optim.param_groups[0]['lr'], global_step)

                # GAN 相关日志
                if use_gan:
                    writer.add_scalar('GAN/DiscLoss', loss_gan_disc.item(), global_step)
                    writer.add_scalar('GAN/GenLoss', loss_gan_gen.item(), global_step)
                    writer.add_scalar('GAN/AdaptiveWeight', adaptive_w.item(), global_step)
                    if disc_optim is not None:
                        writer.add_scalar('LearningRate/Disc', disc_optim.param_groups[0]['lr'], global_step)

            # ── 保存 checkpoint (仅主进程保存, 避免多进程写冲突) ──
            if is_main and args.save_step > 0 and global_step % args.save_step == 0:
                ckpt = {
                    "dit_model": dit_model.state_dict(),
                    "ema_model": ema_model.state_dict(),
                    "text_model": (
                        text_model.state_dict() if text_model is not None else None
                    ),
                    "vision_model": (
                        vision_model.state_dict() if vision_model is not None else None
                    ),
                    "step": global_step,
                    "dit_variant": args.dit_variant,
                    "num_classes": args.num_classes,
                    "pairing_config": {
                        "pairing_mode": "strict_intra_class_self_excluded_ot",
                        "one_class_per_local_batch": True,
                        "identity_aligned_pre_ot": True,
                        "unique_identity_within_batch": True,
                        "self_exclusion": "diagonal_and_near_identical_cost_mask",
                        "suboptimal_eps": float(args.suboptimal_eps),
                        "ot_sample_plan": True,
                        "ot_method": _OT_METHOD,
                        "ot_reg": _OT_REG,
                        "ot_adaptive_reg": _OT_ADAPTIVE_REG,
                        "local_batch_max_size": int(args.batch_size),
                        "sampler_seed": int(args.sampler_seed),
                        "data_glob": str(args.data_glob),
                    },
                    "condition_config": {
                        "condition_mode": (
                            "vision_hint_v1"
                            if caption_hidden_states_mode == "vision_only" and not args.use_class_condition
                            else "legacy_or_ablation"
                        ),
                        "caption_mode": caption_hidden_states_mode,
                        "use_caption": caption_hidden_states_mode != "vision_only",
                        "use_class_id": bool(args.use_class_condition),
                        "use_target_image_cls": True,
                        "use_target_image_tokens": caption_hidden_states_mode in ("vision_only", "cat"),
                        "hint_signal_ratio": float(args.hint_signal_ratio),
                        "hint_dropout": float(args.hint_dropout),
                        "pure_noise_hint": bool(args.pure_noise_hint),
                        "rae_input_range": "[0,1]",
                        "image_size": int(args.image_size),
                    },
                    "args": vars(args),
                }
                # 保存判别器权重 (如果启用 GAN)
                if use_gan and discriminator is not None:
                    ckpt["discriminator"] = discriminator.state_dict()

                torch.save(
                    ckpt,
                    os.path.join(savedir, f"{args.model}_dit_{args.dit_variant}_step_{global_step}.pt"),
                )
                cleanup_old_checkpoints(savedir, args.keep_n_checkpoints)
                print(f"[SAVE] Checkpoint saved at step {global_step}")

            # DDP: 保存 checkpoint 后同步所有进程, 确保下一步数据一致
            if ddp and args.save_step > 0 and global_step % args.save_step == 0:
                dist.barrier()

            # ── 进度条更新 ──
            desc = (
                f"fm:{loss_fm.item():.4f} "
                f"total:{total_loss.item():.4f} "
                f"IntraOT[{batch_class}] "
                f"DiT:{dit_grad_norm.item():.4f} "
                f"Text:{text_grad_norm.item():.4f} "
                f"Vis:{vision_grad_norm.item():.4f} "
            )
            if args.ot_reg_weight > 0.0:
                ot_tag = "ON" if do_ot else "OFF"
                desc += f"ot:{loss_ot.item():.4f} [OT:{ot_tag}] "
            if use_gan:
                gan_active = "ON" if do_gan else "OFF"
                desc += (
                    f"dLoss:{loss_gan_disc.item():.4f} "
                    f"gLoss:{loss_gan_gen.item():.4f} "
                    f"[GAN:{gan_active}] "
                )
            desc += f"[{args.dit_variant} DiT:{'T' if train_dit else 'F'}]"
            step_pbar.set_description(desc)

    if is_main and writer is not None:
        writer.close()

    # DDP: 等待所有进程完成, 然后销毁进程组
    if ddp:
        dist.barrier()
        cleanup_ddp()

    if is_main:
        print("[DONE] DiT Flow Matching 训练完成")


def main():
    """程序入口."""
    args = parse_arguments()
    train(args)


if __name__ == "__main__":
    main()