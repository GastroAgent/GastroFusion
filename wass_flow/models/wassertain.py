"""
FSL 孪生度量网络 (Siamese Wasserstein Network) — ResNet12-lite + 多层注意力

================================================================
设计要点
================================================================
    1. 孪生: query/support 共享同一个主干 (ResNet12-lite)
    2. 容量: ~7-8M 参数 (ResNet12 量级), 远大于原 2-ConvBlock 方案
    3. 多层注意力: 每个 ResBlock 后挂一个 SpatialAttention, 共 5 层
       - 浅层 Attn 学低级空间模式
       - 中层 Attn 学结构关系
       - 深层 Attn 学语义显著性
    4. 禁用 BatchNorm: 全部 GroupNorm (适合小 batch / FSL 场景)
    5. 保留空间结构: 取最后一个 stage 出口 (8×8=64 点 × out_channels 维) 作为点云
       (Wasserstein 需要分布而非单一向量)
    6. Sinkhorn 纯 torch 实现, log-domain 数值稳定, 完全可微
    7. 与项目主线 (wass_flow) 一致: OT 距离度量

================================================================
架构
================================================================
    VAE latent (B, 4, 32, 32)
        │
        ├─ Stem:        Conv3×3(4→64) + GN + GELU              32×32
        ├─ Stage1(1块): ResBlock(64)      + SpatialAttn①       32×32
        ├─ Stage2(2块): ResBlock(128)×2   + SpatialAttn②③      16×16 (stride2)
        ├─ Stage3(2块): ResBlock(256)×2   + SpatialAttn④⑤       8×8 (stride2)
        │
        └─ 取 Stage3 出口 (8×8=64 点) → 1×1 Conv proj → out_channels 维 → flatten
                                                                        ↓
                                                              (B, 64, out_channels)
                                                                        ↓
    query 点云 (Q,64,D)  vs  support 点云 (N,64,D)
        → 可选 OT 求解器（默认均匀 Sinkhorn）→ 距离 (Q, N)
        → -距离 × scale / temperature → logits → CE loss

================================================================
接口 (与旧版完全一致, 训练/评估代码无需改动)
================================================================
    SiameseWassNet.forward(support_latent, support_labels, query_latent, n_way) → logits
    SiameseWassNet.forward_batch(support_latent, query_latent, return_points)
    SiameseWassNet.latent_to_points(latent) → (B, 64, out_channels)
    SiameseWassNet.compute_logits(...)  # 推理别名 (no_grad)
    build_fsl_model(...) / load_fsl_model(...)
"""

import math
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# OT 距离：均匀 Sinkhorn / 动态 Sinkhorn / DeepEMD QPTH / OpenCV
# ============================================================
def _broadcast_episode_points(q_points: torch.Tensor,
                              s_points: torch.Tensor):
    """广播 q/s 的 episode 前导维。"""
    if q_points.dim() < 3 or s_points.dim() < 3:
        raise ValueError(
            f"q_points/s_points 至少需要 3 维，得到 "
            f"q={tuple(q_points.shape)}, s={tuple(s_points.shape)}")
    if q_points.shape[-1] != s_points.shape[-1]:
        raise ValueError(
            f"点特征维不一致：q={tuple(q_points.shape)}, "
            f"s={tuple(s_points.shape)}")

    q_leading = q_points.shape[:-3]
    s_leading = s_points.shape[:-3]
    target_leading = torch.broadcast_shapes(q_leading, s_leading)

    def _expand(x: torch.Tensor) -> torch.Tensor:
        current_leading = x.shape[:-3]
        tail = x.shape[-3:]
        pad = len(target_leading) - len(current_leading)
        if pad < 0:
            raise RuntimeError("内部广播维数计算错误")
        x = x.reshape((1,) * pad + current_leading + tail)
        return x.expand(target_leading + tail)

    return _expand(q_points), _expand(s_points)


def _pairwise_squared_euclidean_cost(q_points: torch.Tensor,
                                     s_points: torch.Tensor) -> torch.Tensor:
    """返回 (..., Q, N, Mq, Ms) 的 squared-Euclidean ground cost。"""
    q = q_points.unsqueeze(-3)
    s = s_points.unsqueeze(-4)
    q_norm = (q * q).sum(dim=-1, keepdim=True)
    s_norm = (s * s).sum(dim=-1).unsqueeze(-2)
    cost = q_norm + s_norm - 2.0 * (q @ s.transpose(-1, -2))
    return cost.clamp_min(0.0)


def _pairwise_cosine_cost(q_points: torch.Tensor,
                          s_points: torch.Tensor):
    """DeepEMD ground cost: 1 - cosine similarity。"""
    q_unit = F.normalize(q_points, p=2, dim=-1, eps=1e-12)
    s_unit = F.normalize(s_points, p=2, dim=-1, eps=1e-12)
    q = q_unit.unsqueeze(-3)
    s = s_unit.unsqueeze(-4)
    similarity = (q @ s.transpose(-1, -2)).clamp(-1.0, 1.0)
    return (1.0 - similarity).clamp_min(0.0), q_unit, s_unit


def _uniform_node_weights(cost: torch.Tensor):
    """均匀边缘质量。"""
    mq, ms = cost.shape[-2], cost.shape[-1]
    weight_q = torch.full(
        cost.shape[:-1], 1.0 / float(mq),
        device=cost.device, dtype=cost.dtype)
    weight_s = torch.full(
        cost.shape[:-2] + (ms,), 1.0 / float(ms),
        device=cost.device, dtype=cost.dtype)
    return weight_q, weight_s


def _dynamic_node_weights(q_weight_points: torch.Tensor,
                          s_weight_points: torch.Tensor,
                          weight_eps: float = 1e-3):
    """DeepEMD 风格的 pair-conditioned 动态节点权重。"""
    if weight_eps <= 0:
        raise ValueError(f"weight_eps 必须 > 0，当前为 {weight_eps}")

    q_global = q_weight_points.mean(dim=-2)
    s_global = s_weight_points.mean(dim=-2)

    q_scores = torch.einsum(
        '...qmd,...nd->...qnm', q_weight_points, s_global)
    s_scores = torch.einsum(
        '...nkd,...qd->...qnk', s_weight_points, q_global)

    weight_q = F.relu(q_scores) + float(weight_eps)
    weight_s = F.relu(s_scores) + float(weight_eps)

    weight_q = weight_q / weight_q.sum(
        dim=-1, keepdim=True).clamp_min(weight_eps)
    weight_s = weight_s / weight_s.sum(
        dim=-1, keepdim=True).clamp_min(weight_eps)
    return weight_q, weight_s


def _sinkhorn_from_cost(cost: torch.Tensor,
                        weight_q: torch.Tensor,
                        weight_s: torch.Tensor,
                        reg: float,
                        n_iter: int) -> torch.Tensor:
    """给定 cost 和边缘质量，返回熵正则 OT cost。"""
    if reg <= 0:
        raise ValueError(f"Sinkhorn reg 必须 > 0，当前为 {reg}")
    if n_iter <= 0:
        raise ValueError(f"Sinkhorn n_iter 必须 > 0，当前为 {n_iter}")

    expected_q = cost.shape[:-1]
    expected_s = cost.shape[:-2] + (cost.shape[-1],)
    if weight_q.shape != expected_q:
        raise ValueError(
            f"weight_q 形状错误，预期={expected_q}, 实际={weight_q.shape}")
    if weight_s.shape != expected_s:
        raise ValueError(
            f"weight_s 形状错误，预期={expected_s}, 实际={weight_s.shape}")

    log_a = torch.log(weight_q.clamp_min(1e-12)).unsqueeze(-1)
    log_b = torch.log(weight_s.clamp_min(1e-12)).unsqueeze(-2)
    log_k = -cost / float(reg)

    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(int(n_iter)):
        u = log_a - torch.logsumexp(log_k + v, dim=-1, keepdim=True)
        v = log_b - torch.logsumexp(log_k + u, dim=-2, keepdim=True)

    transport = torch.exp(u + log_k + v)
    return (transport * cost).sum(dim=(-2, -1))



def _pool_square_point_grid(points: torch.Tensor,
                            pool_factor: int) -> torch.Tensor:
    """对由 H×W 特征图展平得到的点云做空间平均池化。

    points: (..., K, D)，要求 K 为完全平方数。
    pool_factor=1 时原样返回；例如 K=64、factor=2 时 8×8→4×4，
    点数由 64 降为 16。该选项主要用于降低 QPTH 的二次规划规模。
    """
    pool_factor = int(pool_factor)
    if pool_factor < 1:
        raise ValueError(f"qpth_point_pool 必须 >= 1，当前为 {pool_factor}")
    if pool_factor == 1:
        return points

    n_points = points.shape[-2]
    side = int(round(math.sqrt(n_points)))
    if side * side != n_points:
        raise ValueError(
            "qpth_point_pool 仅支持由方形特征图展平的点云，"
            f"当前点数={n_points} 不是完全平方数")
    if side % pool_factor != 0:
        raise ValueError(
            f"点云边长 {side} 不能被 qpth_point_pool={pool_factor} 整除")

    leading = points.shape[:-2]
    dim = points.shape[-1]
    flat = points.reshape(-1, side, side, dim).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(
        flat, kernel_size=pool_factor, stride=pool_factor)
    new_side = side // pool_factor
    return pooled.permute(0, 2, 3, 1).reshape(
        leading + (new_side * new_side, dim))


def _deepemd_qpth_distance(cost: torch.Tensor,
                            weight_q: torch.Tensor,
                            weight_s: torch.Tensor,
                            form: str = 'QP',
                            l2_strength: float = 1e-4,
                            qp_jitter: float = 1e-4,
                            chunk_size: int = 1) -> torch.Tensor:
    """按项目 ``emd_utils.py`` 的约束构造计算可微 DeepEMD-QPTH。

    Args:
        cost:
            (..., Q, N, Mq, Ms)，DeepEMD ground cost，通常为 1-cosine。
        weight_q:
            (..., Q, N, Mq)，pair-conditioned query 节点权重。
        weight_s:
            (..., Q, N, Ms)，pair-conditioned support 节点权重。
        form:
            ``'QP'``：Q = c c^T + qp_jitter I，p = 0。
            ``'L2'``：Q = l2_strength I，p = c。
        l2_strength:
            ``form='L2'`` 时的二次正则系数。
        qp_jitter:
            ``form='QP'`` 时保证 Q 正定的小对角项。原实现为 1e-4。
        chunk_size:
            每次送入 QPTH 的 query-support 对数量，降低批量显存峰值。

    Returns:
        (..., Q, N) 的平均运输代价，越小越相似。

    说明：
        1. 完整保留用户 ``emd_inference_qpth`` 的质量缩放与约束形式：
           x>=0、行/列运输量不超过节点质量、总运输质量固定。
        2. 原 DeepEMD 最后对 flow-weighted similarity 除以 num_node；这里
           等价地返回 ``sum(flow * cost) / transported_mass``，从而与其他
           三种模式统一为“越小越相似”的平均 OT 距离。
        3. QPTH 使用稠密 Q/G/KKT 矩阵。64×64 节点会产生 4096 个变量，
           显存开销很高；可通过 ``qpth_point_pool=2`` 将 8×8 池化为 4×4。
    """
    try:
        from qpth.qp import QPFunction
    except ImportError as exc:
        raise ImportError(
            "deepemd_qpth 需要 qpth。请先在当前环境安装 qpth。"
        ) from exc

    form = str(form).upper()
    if form not in {'L2', 'QP'}:
        raise ValueError(f"qpth_form 必须为 'L2' 或 'QP'，当前为 {form}")
    if l2_strength <= 0:
        raise ValueError(f"qpth_l2_strength 必须 > 0，当前为 {l2_strength}")
    if qp_jitter <= 0:
        raise ValueError(f"qpth_qp_jitter 必须 > 0，当前为 {qp_jitter}")
    if int(chunk_size) <= 0:
        raise ValueError(f"qpth_chunk_size 必须 > 0，当前为 {chunk_size}")

    expected_q = cost.shape[:-1]
    expected_s = cost.shape[:-2] + (cost.shape[-1],)
    if weight_q.shape != expected_q or weight_s.shape != expected_s:
        raise ValueError(
            "QPTH 权重形状不匹配："
            f"cost={tuple(cost.shape)}, weight_q={tuple(weight_q.shape)}, "
            f"weight_s={tuple(weight_s.shape)}"
        )

    pair_shape = cost.shape[:-2]
    mq, ms = cost.shape[-2], cost.shape[-1]
    n_var = mq * ms

    flat_cost = cost.reshape(-1, mq, ms)
    flat_wq = weight_q.reshape(-1, mq)
    flat_ws = weight_s.reshape(-1, ms)

    if n_var > 1600:
        warnings.warn(
            "deepemd_qpth 当前每个图像对包含 "
            f"{mq}×{ms}={n_var} 个 QP 变量。QPTH 使用稠密矩阵，"
            "64×64 点通常显存开销很高；建议设置 qpth_point_pool=2。",
            RuntimeWarning,
        )

    outputs = []
    chunk_size = int(chunk_size)
    device = cost.device

    # 与用户 emd_utils.py 一致，QPTH 内部使用 double。
    for start in range(0, flat_cost.shape[0], chunk_size):
        end = min(start + chunk_size, flat_cost.shape[0])
        distance_matrix = flat_cost[start:end].double()
        weight1 = flat_wq[start:end].double()
        weight2 = flat_ws[start:end].double()
        nbatch = distance_matrix.shape[0]

        # 与原实现一致：每侧总质量缩放到各自节点数。
        weight1 = (
            weight1 * float(mq)
            / weight1.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )
        weight2 = (
            weight2 * float(ms)
            / weight2.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )

        q_1 = distance_matrix.reshape(nbatch, 1, n_var)
        eye = torch.eye(n_var, device=device, dtype=torch.double)

        if form == 'QP':
            # 用户实现中的 QTQ 版本：Q = c c^T + 1e-4 I, p = 0。
            Q = torch.bmm(q_1.transpose(2, 1), q_1)
            Q = Q + float(qp_jitter) * eye.unsqueeze(0)
            p = torch.zeros(
                nbatch, n_var, device=device, dtype=torch.double)
        else:
            # 用户实现中的 regularizer 版本：
            # 0.5*l2_strength*||x||^2 + c^T x。
            Q = (
                float(l2_strength) * eye
            ).unsqueeze(0).expand(nbatch, n_var, n_var).contiguous()
            p = distance_matrix.reshape(nbatch, n_var)

        h_nonneg = torch.zeros(
            nbatch, n_var, device=device, dtype=torch.double)
        h_marginal = torch.cat([weight1, weight2], dim=1)
        h = torch.cat([h_nonneg, h_marginal], dim=1)

        # -x <= 0  =>  x >= 0。
        G_nonneg = (-eye).unsqueeze(0).expand(
            nbatch, n_var, n_var).contiguous()

        # 行和/列和上界，与 emd_utils.py 中 G_2 完全同义。
        row_constraints = torch.zeros(
            mq, n_var, device=device, dtype=torch.double)
        row_constraints[
            torch.arange(mq, device=device).unsqueeze(1),
            torch.arange(n_var, device=device).reshape(mq, ms),
        ] = 1.0

        col_constraints = torch.zeros(
            ms, n_var, device=device, dtype=torch.double)
        col_constraints[
            torch.arange(ms, device=device).unsqueeze(1),
            torch.arange(n_var, device=device).reshape(mq, ms).transpose(0, 1),
        ] = 1.0

        G_marginal_base = torch.cat(
            [row_constraints, col_constraints], dim=0)
        G_marginal = G_marginal_base.unsqueeze(0).expand(
            nbatch, mq + ms, n_var).contiguous()
        G = torch.cat([G_nonneg, G_marginal], dim=1)

        # 与原实现一致：总运输质量取两侧总质量的较小值。
        A = torch.ones(
            nbatch, 1, n_var, device=device, dtype=torch.double)
        transported_mass = torch.minimum(
            weight1.sum(dim=1), weight2.sum(dim=1))
        b = transported_mass.unsqueeze(1)

        flow = QPFunction(verbose=-1)(Q, p, G, h, A, b)

        # 统一返回平均 transport cost，避免节点数改变 logits 尺度。
        transport_cost = (
            flow * distance_matrix.reshape(nbatch, n_var)
        ).sum(dim=1)
        outputs.append(
            transport_cost / transported_mass.clamp_min(1e-12)
        )

    distance = torch.cat(outputs, dim=0).reshape(pair_shape)
    return distance.to(dtype=cost.dtype)


@torch.no_grad()
def _deepemd_opencv_distance(cost: torch.Tensor,
                             weight_q: torch.Tensor,
                             weight_s: torch.Tensor,
                             q_unit: torch.Tensor,
                             s_unit: torch.Tensor) -> torch.Tensor:
    """OpenCV EMD，统一返回越小越相似的 transport cost。"""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "deepemd_opencv 需要 OpenCV：pip install opencv-python") from exc

    pair_shape = cost.shape[:-2]
    mq, ms = cost.shape[-2], cost.shape[-1]
    dim = q_unit.shape[-1]

    q_pair = q_unit.unsqueeze(-3).expand(pair_shape + (mq, dim))
    s_pair = s_unit.unsqueeze(-4).expand(pair_shape + (ms, dim))

    flat_cost = cost.reshape(-1, mq, ms)
    flat_wq = weight_q.reshape(-1, mq)
    flat_ws = weight_s.reshape(-1, ms)
    flat_q = q_pair.reshape(-1, mq, dim)
    flat_s = s_pair.reshape(-1, ms, dim)

    distances = []
    for idx in range(flat_cost.shape[0]):
        cost_np = flat_cost[idx].detach().cpu().numpy().astype(np.float32)
        wq_np = flat_wq[idx].detach().cpu().numpy().astype(np.float32)
        ws_np = flat_ws[idx].detach().cpu().numpy().astype(np.float32)
        q_np = flat_q[idx].detach().cpu().numpy().astype(np.float32)
        s_np = flat_s[idx].detach().cpu().numpy().astype(np.float32)

        signature_q = np.concatenate([wq_np[:, None], q_np], axis=1)
        signature_s = np.concatenate([ws_np[:, None], s_np], axis=1)

        _, _, flow = cv2.EMD(
            signature_q, signature_s, cv2.DIST_USER, cost_np)
        distances.append(float((flow * cost_np).sum()))

    return torch.tensor(
        distances, device=cost.device, dtype=cost.dtype).reshape(pair_shape)


def wassertain_distance(q_points: torch.Tensor,
                      s_points: torch.Tensor,
                      reg: float = 0.1,
                      n_iter: int = 20,
                      method: str = 'deepemd_opencv',
                      q_weight_points: Optional[torch.Tensor] = None,
                      s_weight_points: Optional[torch.Tensor] = None,
                      weight_eps: float = 1e-3,
                      qpth_form: str = 'QP',
                      qpth_l2_strength: float = 1e-4,
                      qpth_qp_jitter: float = 1e-4,
                      qpth_chunk_size: int = 1,
                      qpth_point_pool: int = 1) -> torch.Tensor:
    """统一 OT 接口，返回 (..., Q, N) 距离，越小越相似。

    method:
        uniform_sinkhorn:
            均匀节点权重 + squared-Euclidean + Sinkhorn。
        dynamic_sinkhorn:
            DeepEMD 动态节点权重 + squared-Euclidean + Sinkhorn。
        deepemd_qpth:
            DeepEMD 动态节点权重 + (1-cosine) + 可微 QPTH。
        deepemd_opencv:
            DeepEMD 动态节点权重 + (1-cosine) + OpenCV EMD。

    不传 method 时保持原始均匀 Sinkhorn 行为，因此单路径旧代码不受影响。
    """
    valid_methods = {
        'uniform_sinkhorn', 'dynamic_sinkhorn',
        'deepemd_qpth', 'deepemd_opencv'}
    if method not in valid_methods:
        raise ValueError(
            f"未知 OT method={method}，可选值={sorted(valid_methods)}")

    q_points = q_points.float()
    s_points = s_points.float()
    q_points, s_points = _broadcast_episode_points(q_points, s_points)

    if method == 'uniform_sinkhorn':
        cost = _pairwise_squared_euclidean_cost(q_points, s_points)
        weight_q, weight_s = _uniform_node_weights(cost)
        return _sinkhorn_from_cost(
            cost, weight_q, weight_s, reg=reg, n_iter=n_iter)

    if q_weight_points is None:
        q_weight_points = q_points
    else:
        q_weight_points = q_weight_points.float()
    if s_weight_points is None:
        s_weight_points = s_points
    else:
        s_weight_points = s_weight_points.float()

    q_weight_points, s_weight_points = _broadcast_episode_points(
        q_weight_points, s_weight_points)
    if q_weight_points.shape != q_points.shape:
        raise ValueError(
            f"q_weight_points 与 q_points 形状不一致："
            f"{q_weight_points.shape} vs {q_points.shape}")
    if s_weight_points.shape != s_points.shape:
        raise ValueError(
            f"s_weight_points 与 s_points 形状不一致："
            f"{s_weight_points.shape} vs {s_points.shape}")

    # QPTH 在节点数较大时开销极高，允许只在该分支做空间平均池化。
    if method == 'deepemd_qpth' and int(qpth_point_pool) > 1:
        q_points = _pool_square_point_grid(q_points, qpth_point_pool)
        s_points = _pool_square_point_grid(s_points, qpth_point_pool)
        q_weight_points = _pool_square_point_grid(
            q_weight_points, qpth_point_pool)
        s_weight_points = _pool_square_point_grid(
            s_weight_points, qpth_point_pool)

    weight_q, weight_s = _dynamic_node_weights(
        q_weight_points, s_weight_points, weight_eps=weight_eps)

    if method == 'dynamic_sinkhorn':
        cost = _pairwise_squared_euclidean_cost(q_points, s_points)
        return _sinkhorn_from_cost(
            cost, weight_q, weight_s, reg=reg, n_iter=n_iter)

    cost, q_unit, s_unit = _pairwise_cosine_cost(q_points, s_points)
    if method == 'deepemd_qpth':
        return _deepemd_qpth_distance(
            cost=cost,
            weight_q=weight_q,
            weight_s=weight_s,
            form=qpth_form,
            l2_strength=qpth_l2_strength,
            qp_jitter=qpth_qp_jitter,
            chunk_size=qpth_chunk_size,
        )

    return _deepemd_opencv_distance(
        cost, weight_q, weight_s, q_unit, s_unit)


# ============================================================
# SiameseWassNet: 孪生 ResNet10-lite, Wasserstein 度量
# ============================================================
class SiameseWassNet(FSLModelBase):
    """孪生度量网络: 共享 ResNet10-lite 主干 + 多层注意力 → 点云 → Sinkhorn OT。

    输入: VAE latent (B, 4, 32, 32)
    输出: 点云 (B, 64, out_channels) — 8×8=64 个空间位置 × out_channels 维

    forward 返回 logits (Q, n_way), 可直接 CE loss。
    """

    def __init__(self,
                 in_channels: int = 4,
                 hidden_channels: int = 128,
                 out_channels: int = 256,
                 dropout: float = 0.1,
                 sinkhorn_reg: float = 0.1,
                 sinkhorn_iter: int = 20,
                 scale: float = 10.0,
                 temperature: float = 1.0,
                 blocks_per_stage=(1, 2, 2),
                 attn_kernel: int = 7):
        """
            Args:
            in_channels:     VAE latent 通道数 (默认 4)
            hidden_channels: 主干基础通道数 (Stem/Stage1=128, Stage2/3 = 2/4×)
            out_channels:    点云最终特征维度 (默认 256)
            dropout:         dropout 概率
            sinkhorn_reg:    Sinkhorn 正则化系数 ε
            sinkhorn_iter:   Sinkhorn 迭代次数
            scale:           logits 缩放因子 (距离乘 scale 放大梯度)
            temperature:     softmax 温度
            blocks_per_stage: 每个 stage 的 ResBlock 数 (默认 1,2,2 → ResNet10-lite)
            attn_kernel:     SpatialAttention 卷积核大小
        """
        super().__init__()
        self.out_channels = out_channels
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_iter = sinkhorn_iter
        self.scale = scale
        self.temperature = temperature

        # ── 共享主干 (ResNet10-lite + 5 层注意力) ──
        self.backbone = ResNet10Lite(
            in_channels=in_channels,
            base_ch=hidden_channels,
            out_ch=out_channels,
            blocks_per_stage=blocks_per_stage,
            attn_kernel=attn_kernel,
        )
        self.dropout = nn.Dropout(dropout)

    def latent_to_points(self, latent: torch.Tensor,
                         return_raw: bool = False):
        """VAE latent → point features。

        Wasserstein 始终使用 L2-normalized points；VISReg 若启用，则额外
        使用归一化前的 raw point features 形成 cloud-level representation。
        这样不会让 VISReg 的 scale term 与单位球约束发生冲突。
        """
        feat = self.backbone(latent)         # (B, out_ch, 8, 8)
        feat = self.dropout(feat)
        B, C, H, W = feat.shape
        raw_points = feat.flatten(2).transpose(1, 2)  # (B,64,D)
        points = F.normalize(raw_points, dim=-1)
        if return_raw:
            return points, raw_points
        return points

    def forward_episode(self,
                        support_latent: torch.Tensor,
                        support_labels: torch.Tensor,
                        query_latent: torch.Tensor,
                        n_way: int) -> torch.Tensor:
        """一个 episode 的前向 (FSL 推理接口, no_grad 调用)。

        Args:
            support_latent: (N, 4, 32, 32) — 每类 1 个聚合 latent
            support_labels: (N,) 兼容接口 (本实现不使用)
            query_latent:   (Q, 4, 32, 32)
            n_way:          类别数
        Returns:
            logits: (Q, n_way)
        """
        s_points = self.latent_to_points(support_latent)  # (N, 64, D)
        q_points = self.latent_to_points(query_latent)    # (Q, 64, D)

        # Sinkhorn OT 距离 (Q, N)
        dist = wassertain_distance(q_points, s_points,
                                 reg=self.sinkhorn_reg,
                                 n_iter=self.sinkhorn_iter)

        # 距离 → logits (越近分越大)
        logits = -dist * self.scale / self.temperature
        return logits

    def forward(self,
                s_z: torch.Tensor,
                q_z: torch.Tensor,
                return_points: bool = False,
                return_cloud_features: bool = False):
        """统一前向入口 (DDP 友好)。

        自动识别单 episode (4D) / batch (5D), 内部统一走 batch 实现。
        DDP 包装后会 hook 这个 forward 来同步梯度。

        Args:
            s_z: (E, N, 4, 32, 32) batch 或 (N, 4, 32, 32) 单 episode
            q_z: (E, Q, 4, 32, 32) batch 或 (Q, 4, 32, 32) 单 episode
            return_points: 是否返回点云 (供 SIGReg 使用)
        Returns:
            若 return_points=False: logits (与输入同维数, 单 episode 会 squeeze)
            若 return_points=True: (logits, s_points, q_points)
                - 单 episode: s_points (N,64,D), q_points (Q,64,D)
                - batch:      s_points (E,N,64,D), q_points (E,Q,64,D)
        """
        single = (s_z.dim() == 4)
        if single:
            s_z = s_z.unsqueeze(0)   # (1, N, 4, 32, 32)
            q_z = q_z.unsqueeze(0)   # (1, Q, 4, 32, 32)

        out = self.forward_batch(
            s_z, q_z, return_points=True,
            return_cloud_features=return_cloud_features)
        if return_cloud_features:
            logits, s_points, q_points, s_cloud, q_cloud = out
        else:
            logits, s_points, q_points = out

        if single:
            logits = logits.squeeze(0)         # (Q, N)
            s_points = s_points.squeeze(0)     # (N, 64, D)
            q_points = q_points.squeeze(0)     # (Q, 64, D)
            if return_cloud_features:
                s_cloud = s_cloud.squeeze(0)   # (N,D)
                q_cloud = q_cloud.squeeze(0)   # (Q,D)

        if return_points and return_cloud_features:
            return logits, s_points, q_points, s_cloud, q_cloud
        if return_points:
            return logits, s_points, q_points
        if return_cloud_features:
            return logits, s_cloud, q_cloud
        return logits

    def forward_batch(self,
                      support_latent: torch.Tensor,
                      query_latent: torch.Tensor,
                      return_points: bool = False,
                      return_cloud_features: bool = False):
        """Batch 前向 (同时处理 E 个 episode)。

        用于 meta-training 时把多个 episode 打包成一个 batch,
        共享一次 forward 的算子级并行, 显著提升吞吐。

        Args:
            support_latent: (E, N, 4, 32, 32) — E 个 episode, 每个 N 个类
            query_latent:   (E, Q, 4, 32, 32) — 每个 episode Q 个 query
            return_points:  是否返回点云 (供 SIGReg 使用)
        Returns:
            logits: (E, Q, N)
            (可选) s_points: (E, N, 64, D), q_points: (E, Q, 64, D)
        """
        E, N, C, H, W = support_latent.shape
        Eq, Q, C2, H2, W2 = query_latent.shape
        assert E == Eq, f"episode 数不一致: support={E}, query={Eq}"

        # 展平成 (E*N, C, H, W) / (E*Q, C, H, W) 共享编码器
        s_flat = support_latent.reshape(E * N, C, H, W)
        q_flat = query_latent.reshape(E * Q, C2, H2, W2)
        if return_cloud_features:
            s_points_flat, s_raw_flat = self.latent_to_points(
                s_flat, return_raw=True)
            q_points_flat, q_raw_flat = self.latent_to_points(
                q_flat, return_raw=True)
        else:
            s_points_flat = self.latent_to_points(s_flat)
            q_points_flat = self.latent_to_points(q_flat)

        # 还原成 5D 点云。Wasserstein 只使用 normalized points。
        D = s_points_flat.shape[-1]
        s_points = s_points_flat.reshape(E, N, -1, D)
        q_points = q_points_flat.reshape(E, Q, -1, D)

        # VISReg 使用每朵独立 point cloud 的 pre-L2 mean representation。
        if return_cloud_features:
            s_cloud = s_raw_flat.mean(dim=1).reshape(E, N, D)
            q_cloud = q_raw_flat.mean(dim=1).reshape(E, Q, D)

        # Batch Sinkhorn OT: (E, Q, N)
        dist = wassertain_distance(q_points, s_points,
                                 reg=self.sinkhorn_reg,
                                 n_iter=self.sinkhorn_iter)

        logits = -dist * self.scale / self.temperature
        if return_points and return_cloud_features:
            return logits, s_points, q_points, s_cloud, q_cloud
        if return_points:
            return logits, s_points, q_points
        if return_cloud_features:
            return logits, s_cloud, q_cloud
        return logits


# ============================================================
# FiLM-B: 内容相关 Feature-wise Linear Modulation (零初始化恒等启动)
# ============================================================
class FiLM(nn.Module):
    """γ, β = MLP(GAP(h));  h' = h * (1 + γ) + β。

    MLP 最后一层零初始化 → 训练初期 γ=β=0 → h'=h (恒等启动),
    后期慢慢学出偏移以对齐多图VAE与标准VAE两种 latent 分布。
    """

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 4, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * channels),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        pooled = h.mean(dim=(2, 3))                         # (B, C)
        gamma, beta = self.mlp(pooled).chunk(2, dim=-1)     # 各 (B, C)
        return h * (1.0 + gamma[..., None, None]) + beta[..., None, None]


# ============================================================
# Dual-Path Backbone (双 stem + 共享 ResNet10-lite + 逐层 FiLM)
# ============================================================
class DualPathBackbone(nn.Module):
    """双 stem + 共享 ResNet10-lite 主干。

    forward(source):
        'multi' → stem_multi + 逐层 FiLM  (多图VAE 路径)
        'std'   → stem_std  (标准VAE 路径, 无 FiLM)

    其余结构与 ResNet10Lite 完全一致 (Stem + 3 Stage, 取最后 Stage 出口 8×8)。
    """

    def __init__(self,
                 in_channels: int = 4,
                 base_ch: int = 128,
                 out_ch: int = 256,
                 blocks_per_stage=(1, 2, 2),
                 attn_kernel: int = 7,
                 use_film: bool = True):
        super().__init__()
        assert len(blocks_per_stage) == 3
        channels = [base_ch, base_ch * 2, base_ch * 4]
        self.stage3_ch = channels[2]
        self.out_channels = out_ch
        self.use_film = use_film

        # 双 Stem (不共享, 隔离两种 latent 分布)
        ng_stem = _gn_groups(channels[0], 8)
        self.stem_multi = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, 1, 1, bias=False),
            nn.GroupNorm(ng_stem, channels[0]),
            nn.GELU(),
        )
        self.stem_std = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, 1, 1, bias=False),
            nn.GroupNorm(ng_stem, channels[0]),
            nn.GELU(),
        )

        # 共享 4 个 Stage
        self.stages = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.films = nn.ModuleList() if use_film else None
        in_ch = channels[0]
        for s, (out_c, n_blk) in enumerate(zip(channels, blocks_per_stage)):
            stage_blocks, stage_attns, stage_films = [], [], []
            for b in range(n_blk):
                stride = 2 if (b == 0 and s > 0) else 1
                stage_blocks.append(ResBlock(in_ch, out_c, stride=stride))
                stage_attns.append(SpatialAttention(kernel_size=attn_kernel))
                if use_film:
                    stage_films.append(FiLM(out_c))
                in_ch = out_c
            self.stages.append(nn.Sequential(*stage_blocks))
            self.attentions.append(nn.ModuleList(stage_attns))
            if use_film:
                self.films.append(nn.ModuleList(stage_films))

        ng_out = _gn_groups(out_ch, 8)
        self.proj = nn.Sequential(
            nn.Conv2d(self.stage3_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(ng_out, out_ch),
        )

    def forward(self, x: torch.Tensor, source: str = 'multi') -> torch.Tensor:
        """x: (B, 4, 32, 32) → feat: (B, out_ch, 8, 8)。"""
        if source == 'multi':
            x = self.stem_multi(x)
        elif source == 'std':
            x = self.stem_std(x)
        else:
            raise ValueError(f"source 必须是 'multi' 或 'std', 得到 '{source}'")

        last_idx = len(self.stages) - 1
        for s, (stage, attns) in enumerate(zip(self.stages, self.attentions)):
            films = self.films[s] if (self.use_film and self.films is not None) else None
            blocks = list(stage)
            for i, (blk, attn) in enumerate(zip(blocks, attns)):
                x = attn(blk(x))
                if source == 'multi' and films is not None:
                    x = films[i](x)
            if s == last_idx:
                feat = x
        return self.proj(feat)


# ============================================================
# DualPathSiameseWassNet: 双 stem 孪生度量网络
# ============================================================
class DualPathSiameseWassNet(FSLModelBase):
    """Dual-Path 孪生度量网络。

    对外接口与 SiameseWassNet 完全一致 (forward / forward_episode /
    forward_batch / compute_logits), 内部根据 self.exp 自动决定 source:
        exp=1: s_source='multi', q_source='std'  (异构空间)
        exp=2: s_source='multi', q_source='multi' (同构空间)
    support 永远来自多图VAE → s_source='multi'。

    Dual 路径支持四种 ot_method：
        uniform_sinkhorn（默认）、dynamic_sinkhorn、
        deepemd_qpth、deepemd_opencv。
    """

    def __init__(self,
                 in_channels: int = 4,
                 hidden_channels: int = 128,
                 out_channels: int = 256,
                 dropout: float = 0.1,
                 sinkhorn_reg: float = 0.1,
                 sinkhorn_iter: int = 20,
                 scale: float = 10.0,
                 temperature: float = 1.0,
                 blocks_per_stage=(1, 2, 2),
                 attn_kernel: int = 7,
                 use_film: bool = True,
                 exp: int = 1,
                 ot_method: str = 'uniform_sinkhorn',
                 dynamic_weight_eps: float = 1e-3,
                 qpth_form: str = 'QP',
                 qpth_l2_strength: float = 1e-4,
                 qpth_qp_jitter: float = 1e-4,
                 qpth_chunk_size: int = 1,
                 qpth_point_pool: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_iter = sinkhorn_iter
        self.scale = scale
        self.temperature = temperature
        self.exp = exp
        self.s_source = 'multi'
        self.q_source = 'std' if exp == 1 else 'multi'
        self.ot_method = ot_method
        self.dynamic_weight_eps = float(dynamic_weight_eps)
        self.qpth_form = str(qpth_form).upper()
        self.qpth_l2_strength = float(qpth_l2_strength)
        self.qpth_qp_jitter = float(qpth_qp_jitter)
        self.qpth_chunk_size = int(qpth_chunk_size)
        self.qpth_point_pool = int(qpth_point_pool)
        valid_ot_methods = {
            'uniform_sinkhorn', 'dynamic_sinkhorn',
            'deepemd_qpth', 'deepemd_opencv'}
        if self.ot_method not in valid_ot_methods:
            raise ValueError(
                f"未知 ot_method={self.ot_method}，"
                f"可选值={sorted(valid_ot_methods)}")
        if self.dynamic_weight_eps <= 0:
            raise ValueError('dynamic_weight_eps 必须 > 0')
        if self.qpth_form not in {'L2', 'QP'}:
            raise ValueError("qpth_form 必须为 'L2' 或 'QP'")
        if self.qpth_l2_strength <= 0:
            raise ValueError('qpth_l2_strength 必须 > 0')
        if self.qpth_qp_jitter <= 0:
            raise ValueError('qpth_qp_jitter 必须 > 0')
        if self.qpth_chunk_size <= 0:
            raise ValueError('qpth_chunk_size 必须 > 0')
        if self.qpth_point_pool <= 0:
            raise ValueError('qpth_point_pool 必须 >= 1')

        self.backbone = DualPathBackbone(
            in_channels=in_channels,
            base_ch=hidden_channels,
            out_ch=out_channels,
            blocks_per_stage=blocks_per_stage,
            attn_kernel=attn_kernel,
            use_film=use_film,
        )
        self.dropout = nn.Dropout(dropout)

    def latent_to_points(self, latent: torch.Tensor,
                         source: str = 'multi',
                         return_raw: bool = False):
        feat = self.backbone(latent, source=source)
        feat = self.dropout(feat)
        B, C, H, W = feat.shape
        raw_points = feat.flatten(2).transpose(1, 2)
        points = raw_points
        points = F.normalize(raw_points, dim=-1)
        if return_raw:
            return points, raw_points
        return points

    def _compute_ot_distance(self,
                             q_points: torch.Tensor,
                             s_points: torch.Tensor,
                             q_raw_points: Optional[torch.Tensor] = None,
                             s_raw_points: Optional[torch.Tensor] = None):
        """Dual-path 专用 OT 调用。四种模式统一返回距离。"""
        if self.ot_method == 'deepemd_opencv' and self.training:
            raise RuntimeError(
                "deepemd_opencv 不可微，只能在 model.eval() 推理时使用；"
                "训练请改用 dynamic_sinkhorn。")
        return wassertain_distance(
            q_points=q_points,
            s_points=s_points,
            reg=self.sinkhorn_reg,
            n_iter=self.sinkhorn_iter,
            method=self.ot_method,
            q_weight_points=q_raw_points,
            s_weight_points=s_raw_points,
            weight_eps=self.dynamic_weight_eps,
            qpth_form=self.qpth_form,
            qpth_l2_strength=self.qpth_l2_strength,
            qpth_qp_jitter=self.qpth_qp_jitter,
            qpth_chunk_size=self.qpth_chunk_size,
            qpth_point_pool=self.qpth_point_pool,
        )

    def forward_episode(self,
                        support_latent: torch.Tensor,
                        support_labels: torch.Tensor,
                        query_latent: torch.Tensor,
                        n_way: int) -> torch.Tensor:
        need_raw = self.ot_method != 'uniform_sinkhorn'
        if need_raw:
            s_points, s_raw = self.latent_to_points(
                support_latent, source=self.s_source, return_raw=True)
            q_points, q_raw = self.latent_to_points(
                query_latent, source=self.q_source, return_raw=True)
        else:
            s_points = self.latent_to_points(
                support_latent, source=self.s_source)
            q_points = self.latent_to_points(
                query_latent, source=self.q_source)
            s_raw = q_raw = None

        dist = self._compute_ot_distance(
            q_points, s_points, q_raw_points=q_raw, s_raw_points=s_raw)
        return -dist * self.scale / self.temperature

    def forward(self,
                s_z: torch.Tensor,
                q_z: torch.Tensor,
                return_points: bool = False,
                return_cloud_features: bool = False):
        """统一前向入口 (DDP 友好, 接口与 SiameseWassNet 完全一致)。

        自动识别单 episode (4D) / batch (5D), 内部根据 self.exp 选择 source。
        """
        single = (s_z.dim() == 4)
        if single:
            s_z = s_z.unsqueeze(0)
            q_z = q_z.unsqueeze(0)
        out = self.forward_batch(
            s_z, q_z, return_points=True,
            return_cloud_features=return_cloud_features)
        if return_cloud_features:
            logits, s_points, q_points, s_cloud, q_cloud = out
        else:
            logits, s_points, q_points = out
        if single:
            logits = logits.squeeze(0)
            s_points = s_points.squeeze(0)
            q_points = q_points.squeeze(0)
            if return_cloud_features:
                s_cloud = s_cloud.squeeze(0)
                q_cloud = q_cloud.squeeze(0)
        if return_points and return_cloud_features:
            return logits, s_points, q_points, s_cloud, q_cloud
        if return_points:
            return logits, s_points, q_points
        if return_cloud_features:
            return logits, s_cloud, q_cloud
        return logits

    def forward_batch(self,
                      support_latent: torch.Tensor,
                      query_latent: torch.Tensor,
                      return_points: bool = False,
                      return_cloud_features: bool = False):
        E, N, C, H, W = support_latent.shape
        Eq, Q, C2, H2, W2 = query_latent.shape
        assert E == Eq

        s_flat = support_latent.reshape(E * N, C, H, W)
        q_flat = query_latent.reshape(E * Q, C2, H2, W2)

        # 动态权重和 VISReg 都需要 pre-L2 raw points。
        need_raw = return_cloud_features or self.ot_method != 'uniform_sinkhorn'
        if need_raw:
            s_points_flat, s_raw_flat = self.latent_to_points(
                s_flat, source=self.s_source, return_raw=True)
            q_points_flat, q_raw_flat = self.latent_to_points(
                q_flat, source=self.q_source, return_raw=True)
        else:
            s_points_flat = self.latent_to_points(s_flat, source=self.s_source)
            q_points_flat = self.latent_to_points(q_flat, source=self.q_source)
            s_raw_flat = q_raw_flat = None

        D = s_points_flat.shape[-1]
        s_points = s_points_flat.reshape(E, N, -1, D)
        q_points = q_points_flat.reshape(E, Q, -1, D)

        if need_raw:
            s_raw_points = s_raw_flat.reshape(E, N, -1, D)
            q_raw_points = q_raw_flat.reshape(E, Q, -1, D)
        else:
            s_raw_points = q_raw_points = None

        if return_cloud_features:
            s_cloud = s_raw_flat.mean(dim=1).reshape(E, N, D)
            q_cloud = q_raw_flat.mean(dim=1).reshape(E, Q, D)

        dist = self._compute_ot_distance(
            q_points, s_points,
            q_raw_points=q_raw_points,
            s_raw_points=s_raw_points)
        logits = -dist * self.scale / self.temperature
        if return_points and return_cloud_features:
            return logits, s_points, q_points, s_cloud, q_cloud
        if return_points:
            return logits, s_points, q_points
        if return_cloud_features:
            return logits, s_cloud, q_cloud
        return logits


# ============================================================
# 工具: 构建/加载模型
# ============================================================
def build_fsl_model(model_type: str = 'dual_siamese_wass',
                    in_channels: int = 4,
                    hidden_channels: int = 128,
                    out_channels: int = 256,
                    dropout: float = 0.1,
                    sinkhorn_reg: float = 0.1,
                    sinkhorn_iter: int = 20,
                 scale: float = 10.0,
                 temperature: float = 1.0,
                 blocks_per_stage=(1, 2, 2),
                 attn_kernel: int = 7,
                 use_film: bool = True,
                 exp: int = 1,
                 ot_method: str = 'uniform_sinkhorn',
                 dynamic_weight_eps: float = 1e-3,
                 qpth_form: str = 'QP',
                 qpth_l2_strength: float = 1e-4,
                 qpth_qp_jitter: float = 1e-4,
                 qpth_chunk_size: int = 1,
                 qpth_point_pool: int = 1,
                 device: str = 'cuda', **kwargs) -> FSLModelBase:
    """按 model_type 构建模型。

    Args:
        model_type: 'dual_siamese_wass' (双 stem + FiLM, 默认) 或
                    'siamese_wass'      (单 stem, 原版兼容)
        exp:        1=异构空间(q走std), 2=同构空间(q走multi);
                    仅 dual_siamese_wass 用到, siamese_wass 忽略
        use_film:   是否启用逐层 FiLM (仅 dual_siamese_wass)
        其余参数见对应网络 __init__
    """
    if model_type == 'dual_siamese_wass':
        model = DualPathSiameseWassNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
            sinkhorn_reg=sinkhorn_reg,
            sinkhorn_iter=sinkhorn_iter,
            scale=scale,
            temperature=temperature,
            blocks_per_stage=blocks_per_stage,
            attn_kernel=attn_kernel,
            use_film=use_film,
            exp=exp,
            ot_method=ot_method,
            dynamic_weight_eps=dynamic_weight_eps,
            qpth_form=qpth_form,
            qpth_l2_strength=qpth_l2_strength,
            qpth_qp_jitter=qpth_qp_jitter,
            qpth_chunk_size=qpth_chunk_size,
            qpth_point_pool=qpth_point_pool,
        )
    elif model_type == 'siamese_wass':
        model = SiameseWassNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout=dropout,
            sinkhorn_reg=sinkhorn_reg,
            sinkhorn_iter=sinkhorn_iter,
            scale=scale,
            temperature=temperature,
            blocks_per_stage=blocks_per_stage,
            attn_kernel=attn_kernel,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. "
                         f"当前支持: 'dual_siamese_wass', 'siamese_wass'")
    return model.to(device)


def load_fsl_model(ckpt_path: str,
                   model_type: str = 'siamese_wass',
                   device: str = 'cuda',
                   **build_kwargs) -> FSLModelBase:
    """加载 checkpoint 到 FSL 模型。"""
    model = build_fsl_model(model_type=model_type, device='cpu', **build_kwargs)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    if 'model' in state:
        state = state['model']
    elif 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_fsl_model(device=device,
                            hidden_channels=128,
                            out_channels=256)
    print(f"[Test] device={device}")

    # 模拟 episode: 6-way (1 query + 5 distractor), 每类 1 support latent, 1 query
    N, Q = 6, 1
    support_latent = torch.randn(N, 4, 32, 32, device=device)
    support_labels = torch.arange(N, device=device)
    query_latent = torch.randn(Q, 4, 32, 32, device=device)

    logits = model.forward_episode(support_latent, support_labels,
                                   query_latent, N)
    print(f"logits shape: {logits.shape}")   # (1, 6)
    assert logits.shape == (Q, N)

    # 验证可微性
    loss = F.cross_entropy(logits, torch.tensor([0], device=device))
    loss.backward()
    grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad)
    print(f"梯度检查: {'✓ 通过' if grad_ok else '✗ 失败'}")

    # 参数量统计
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SiameseWassNet (ResNet10-lite) 参数量: {n_param:,} ({n_param/1e6:.2f}M)")

    # 主干 / 注意力 / proj 分模块统计
    bb = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print(f"  ├─ backbone (ResNet10-lite + 5 attn): {bb:,} ({bb/1e6:.2f}M)")
    n_attn = 0
    for attns in model.backbone.attentions:
        for a in attns:
            n_attn += sum(p.numel() for p in a.parameters())
    print(f"  │  └─ 其中 5 个 SpatialAttention: {n_attn:,}")

    # 禁用 BN 校验
    has_bn = any(isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)
                 for m in model.modules())
    print(f"BatchNorm 检查: {'✗ 含 BN!' if has_bn else '✓ 无 BN (全 GroupNorm)'}")

    # 测 batch forward
    E = 4
    s_batch = torch.randn(E, N, 4, 32, 32, device=device)
    q_batch = torch.randn(E, Q, 4, 32, 32, device=device)
    logits_b, s_pts, q_pts = model.forward_batch(s_batch, q_batch,
                                                 return_points=True)
    print(f"batch logits: {logits_b.shape}, s_pts: {s_pts.shape}, "
          f"q_pts: {q_pts.shape}")
    assert logits_b.shape == (E, Q, N)
    assert s_pts.shape == (E, N, 64, 256)
    print("✓ FSL 网络自测通过")
