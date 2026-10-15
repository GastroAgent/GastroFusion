#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic-SK weighted entropic Sinkhorn OT for the RAE-Flow-Metric V1 pipeline.

V1 canonical definition
-----------------------
1) Dynamic node masses are computed from *raw* metric tokens using a
   DeepEMD-style pair-conditioned weighting rule.
2) The OT ground cost is squared Euclidean distance on L2-normalized metric
   tokens:

       F = normalize(G_raw)
       C_ij = ||F_i - F_j||_2^2

3) Entropic Sinkhorn solves the transport problem with the dynamic marginals.
   The returned value is the regularized transport cost, not exact unregularized W2.
4) In Flow trajectory training (F1), dynamic SK masses should be computed once
   from the direct state t=0 and then reused for every trajectory step.

This module intentionally exposes both:
- dynamic convenience APIs (compute SK + Sinkhorn in one call), and
- fixed-mass APIs (reuse previously computed SK masses).

The latter is required for "fixed SK across trajectory".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Results / containers
# -----------------------------------------------------------------------------


@dataclass
class SKWeights:
    """Pair-conditioned OT marginal masses.

    For aligned pairs:
        query:   (..., Mq)
        support: (..., Ms)

    For Cartesian query-support pairs:
        query:   (..., Q, N, Mq)
        support: (..., Q, N, Ms)
    """

    query: torch.Tensor
    support: torch.Tensor


@dataclass
class SinkhornResult:
    """Sinkhorn output.

    Attributes
    ----------
    distance:
        Entropy-regularized OT transport cost.
    weight_q / weight_s:
        Marginal masses actually used by Sinkhorn.
    transport:
        Optional transport plan. ``None`` unless requested.
    cost:
        Optional ground-cost matrix. ``None`` unless requested.
    """

    distance: torch.Tensor
    weight_q: torch.Tensor
    weight_s: torch.Tensor
    transport: Optional[torch.Tensor] = None
    cost: Optional[torch.Tensor] = None


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def _check_point_tensor(x: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} 必须是 torch.Tensor")
    if x.ndim < 2:
        raise ValueError(
            f"{name} 至少应为 (..., num_points, dim)，实际 shape={tuple(x.shape)}"
        )
    if x.shape[-2] <= 0 or x.shape[-1] <= 0:
        raise ValueError(f"{name} 含空维度: shape={tuple(x.shape)}")


def _check_same_feature_dim(q: torch.Tensor, s: torch.Tensor) -> None:
    if q.shape[-1] != s.shape[-1]:
        raise ValueError(
            "query/support 特征维不一致: "
            f"q={tuple(q.shape)}, s={tuple(s.shape)}"
        )


def _safe_float_points(x: torch.Tensor) -> torch.Tensor:
    # Sinkhorn/logsumexp is markedly more stable in fp32 than fp16/bf16.
    return x.float()


def l2_normalize_points(points: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize local metric tokens along the feature dimension."""
    _check_point_tensor(points, "points")
    return F.normalize(points, p=2, dim=-1, eps=float(eps))


# -----------------------------------------------------------------------------
# Ground cost
# -----------------------------------------------------------------------------


def aligned_squared_euclidean_cost(
    query_points: torch.Tensor,
    support_points: torch.Tensor,
    *,
    normalize: bool = True,
    norm_eps: float = 1e-12,
) -> torch.Tensor:
    """Squared-Euclidean cost for aligned query-support pairs.

    Parameters
    ----------
    query_points:
        (..., Mq, D)
    support_points:
        (..., Ms, D)
        Leading dimensions must be broadcastable with query_points.
    normalize:
        V1 default=True. Each node is first L2 normalized, then
        C_ij = ||q_i - s_j||^2.

    Returns
    -------
    cost:
        Broadcasted shape (..., Mq, Ms).
    """
    _check_point_tensor(query_points, "query_points")
    _check_point_tensor(support_points, "support_points")
    _check_same_feature_dim(query_points, support_points)

    q = _safe_float_points(query_points)
    s = _safe_float_points(support_points)
    if normalize:
        q = l2_normalize_points(q, eps=norm_eps)
        s = l2_normalize_points(s, eps=norm_eps)

    # (..., Mq, 1)
    q_norm = (q * q).sum(dim=-1, keepdim=True)
    # (..., 1, Ms)
    s_norm = (s * s).sum(dim=-1).unsqueeze(-2)
    cost = q_norm + s_norm - 2.0 * torch.matmul(q, s.transpose(-1, -2))
    return cost.clamp_min(0.0)


def pairwise_squared_euclidean_cost(
    query_points: torch.Tensor,
    support_points: torch.Tensor,
    *,
    normalize: bool = True,
    norm_eps: float = 1e-12,
) -> torch.Tensor:
    """Cartesian query-support squared-Euclidean ground cost.

    Parameters
    ----------
    query_points:
        (..., Q, Mq, D)
    support_points:
        (..., N, Ms, D)
    normalize:
        V1 default=True.

    Returns
    -------
    cost:
        (..., Q, N, Mq, Ms)
    """
    if query_points.ndim < 3 or support_points.ndim < 3:
        raise ValueError(
            "pairwise 输入必须至少为 (..., set, point, dim)，"
            f"q={tuple(query_points.shape)}, s={tuple(support_points.shape)}"
        )
    _check_same_feature_dim(query_points, support_points)

    q = _safe_float_points(query_points)
    s = _safe_float_points(support_points)
    if normalize:
        q = l2_normalize_points(q, eps=norm_eps)
        s = l2_normalize_points(s, eps=norm_eps)

    # Insert the candidate-support axis for q and the query axis for s.
    # q: (..., Q, 1, Mq, D)
    # s: (..., 1, N, Ms, D)
    q = q.unsqueeze(-3)
    s = s.unsqueeze(-4)

    q_norm = (q * q).sum(dim=-1, keepdim=True)
    s_norm = (s * s).sum(dim=-1).unsqueeze(-2)
    cost = q_norm + s_norm - 2.0 * torch.matmul(q, s.transpose(-1, -2))
    return cost.clamp_min(0.0)


# -----------------------------------------------------------------------------
# Dynamic SK node masses (DeepEMD-style, pair-conditioned)
# -----------------------------------------------------------------------------


def compute_aligned_dynamic_sk_weights(
    query_raw: torch.Tensor,
    support_raw: torch.Tensor,
    *,
    weight_eps: float = 1e-3,
) -> SKWeights:
    """Dynamic node masses for aligned query-support pairs.

    This follows the supplied DeepEMD-style implementation:

        q_global = mean_i q_i
        s_global = mean_j s_j
        score_q_i = relu(q_i^T s_global) + eps
        score_s_j = relu(s_j^T q_global) + eps
        a = score_q / sum(score_q)
        b = score_s / sum(score_s)

    Crucially, ``query_raw`` / ``support_raw`` are pre-L2 metric tokens.

    Shapes
    ------
    query_raw:
        (..., Mq, D)
    support_raw:
        (..., Ms, D)
    returns:
        weight_q (..., Mq), weight_s (..., Ms)
    """
    if weight_eps <= 0:
        raise ValueError(f"weight_eps 必须 > 0，当前={weight_eps}")
    _check_point_tensor(query_raw, "query_raw")
    _check_point_tensor(support_raw, "support_raw")
    _check_same_feature_dim(query_raw, support_raw)

    q = _safe_float_points(query_raw)
    s = _safe_float_points(support_raw)

    q_global = q.mean(dim=-2)  # (..., D)
    s_global = s.mean(dim=-2)  # (..., D)

    q_scores = (q * s_global.unsqueeze(-2)).sum(dim=-1)
    s_scores = (s * q_global.unsqueeze(-2)).sum(dim=-1)

    weight_q = F.relu(q_scores) + float(weight_eps)
    weight_s = F.relu(s_scores) + float(weight_eps)

    weight_q = weight_q / weight_q.sum(dim=-1, keepdim=True).clamp_min(weight_eps)
    weight_s = weight_s / weight_s.sum(dim=-1, keepdim=True).clamp_min(weight_eps)

    return SKWeights(query=weight_q, support=weight_s)


def compute_pairwise_dynamic_sk_weights(
    query_raw: torch.Tensor,
    support_raw: torch.Tensor,
    *,
    weight_eps: float = 1e-3,
) -> SKWeights:
    """Dynamic SK masses for all query-support candidates.

    Parameters
    ----------
    query_raw:
        (..., Q, Mq, D)
    support_raw:
        (..., N, Ms, D)

    Returns
    -------
    SKWeights:
        query   (..., Q, N, Mq)
        support (..., Q, N, Ms)

    This is the Cartesian/batched form of the uploaded implementation's
    ``_dynamic_node_weights``.
    """
    if weight_eps <= 0:
        raise ValueError(f"weight_eps 必须 > 0，当前={weight_eps}")
    if query_raw.ndim < 3 or support_raw.ndim < 3:
        raise ValueError(
            "pairwise raw points 至少应为 (..., set, point, dim)，"
            f"q={tuple(query_raw.shape)}, s={tuple(support_raw.shape)}"
        )
    _check_same_feature_dim(query_raw, support_raw)

    q = _safe_float_points(query_raw)
    s = _safe_float_points(support_raw)

    q_global = q.mean(dim=-2)  # (..., Q, D)
    s_global = s.mean(dim=-2)  # (..., N, D)

    # (..., Q, N, Mq)
    weight_q = torch.einsum("...qmd,...nd->...qnm", q, s_global)
    # (..., Q, N, Ms)
    weight_s = torch.einsum("...nkd,...qd->...qnk", s, q_global)

    weight_q = F.relu(weight_q) + float(weight_eps)
    weight_s = F.relu(weight_s) + float(weight_eps)

    weight_q = weight_q / weight_q.sum(dim=-1, keepdim=True).clamp_min(weight_eps)
    weight_s = weight_s / weight_s.sum(dim=-1, keepdim=True).clamp_min(weight_eps)

    return SKWeights(query=weight_q, support=weight_s)


# -----------------------------------------------------------------------------
# Sinkhorn solver from a precomputed cost + fixed marginals
# -----------------------------------------------------------------------------


def sinkhorn_from_cost(
    cost: torch.Tensor,
    weight_q: torch.Tensor,
    weight_s: torch.Tensor,
    *,
    reg: float = 0.1,
    n_iter: int = 30,
    return_transport: bool = False,
    return_cost: bool = False,
) -> SinkhornResult:
    """Log-domain entropy-regularized Sinkhorn with supplied marginals.

    ``cost`` must have shape (..., Mq, Ms).
    ``weight_q`` must be broadcast-compatible with (..., Mq).
    ``weight_s`` must be broadcast-compatible with (..., Ms).

    This function is the key F1 primitive: compute dynamic SK weights once at
    t=0, then call this function with each trajectory step's ground cost while
    reusing the same weights.
    """
    if reg <= 0:
        raise ValueError(f"Sinkhorn reg 必须 > 0，当前={reg}")
    if int(n_iter) <= 0:
        raise ValueError(f"Sinkhorn n_iter 必须 > 0，当前={n_iter}")
    if cost.ndim < 2:
        raise ValueError(f"cost 至少应为 (..., Mq, Ms)，实际={tuple(cost.shape)}")

    cost = _safe_float_points(cost)
    weight_q = _safe_float_points(weight_q)
    weight_s = _safe_float_points(weight_s)

    target_q_shape = cost.shape[:-1]
    target_s_shape = cost.shape[:-2] + (cost.shape[-1],)

    try:
        weight_q = torch.broadcast_to(weight_q, target_q_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"weight_q 无法广播到 {target_q_shape}，实际={tuple(weight_q.shape)}"
        ) from exc

    try:
        weight_s = torch.broadcast_to(weight_s, target_s_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"weight_s 无法广播到 {target_s_shape}，实际={tuple(weight_s.shape)}"
        ) from exc

    # Re-normalize defensively; fixed weights may have passed through pooling,
    # slicing, checkpoint I/O, etc.
    weight_q = weight_q.clamp_min(1e-12)
    weight_s = weight_s.clamp_min(1e-12)
    weight_q = weight_q / weight_q.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    weight_s = weight_s / weight_s.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    log_a = torch.log(weight_q).unsqueeze(-1)  # (..., Mq, 1)
    log_b = torch.log(weight_s).unsqueeze(-2)  # (..., 1, Ms)
    log_k = -cost / float(reg)

    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)

    for _ in range(int(n_iter)):
        u = log_a - torch.logsumexp(log_k + v, dim=-1, keepdim=True)
        v = log_b - torch.logsumexp(log_k + u, dim=-2, keepdim=True)

    transport = torch.exp(u + log_k + v)
    distance = (transport * cost).sum(dim=(-2, -1))

    return SinkhornResult(
        distance=distance,
        weight_q=weight_q,
        weight_s=weight_s,
        transport=transport if return_transport else None,
        cost=cost if return_cost else None,
    )


# -----------------------------------------------------------------------------
# Aligned-pair APIs (M0-B / M1)
# -----------------------------------------------------------------------------


def aligned_sinkhorn_from_weights(
    query_points: torch.Tensor,
    support_points: torch.Tensor,
    weight_q: torch.Tensor,
    weight_s: torch.Tensor,
    *,
    reg: float = 0.1,
    n_iter: int = 30,
    normalize_cost_points: bool = True,
    norm_eps: float = 1e-12,
    return_transport: bool = False,
    return_cost: bool = False,
) -> SinkhornResult:
    """Aligned Sinkhorn using externally supplied/fixed node masses."""
    cost = aligned_squared_euclidean_cost(
        query_points,
        support_points,
        normalize=normalize_cost_points,
        norm_eps=norm_eps,
    )
    return sinkhorn_from_cost(
        cost,
        weight_q,
        weight_s,
        reg=reg,
        n_iter=n_iter,
        return_transport=return_transport,
        return_cost=return_cost,
    )


def aligned_sk_sinkhorn_distance(
    query_points: torch.Tensor,
    support_points: torch.Tensor,
    *,
    query_raw: Optional[torch.Tensor] = None,
    support_raw: Optional[torch.Tensor] = None,
    reg: float = 0.1,
    n_iter: int = 30,
    weight_eps: float = 1e-3,
    normalize_cost_points: bool = True,
    norm_eps: float = 1e-12,
    return_transport: bool = False,
    return_cost: bool = False,
) -> SinkhornResult:
    """Dynamic-SK Sinkhorn for aligned image pairs.

    Intended for M0-B/M1 triplet/ranking batches.

    V1 recommendation:
        query_points/support_points = metric tokens used for OT geometry
        query_raw/support_raw       = pre-L2 raw metric tokens used for SK masses

    For safety, if raw tensors are omitted, the original point tensors are used
    as SK inputs. In the V1 pipeline, callers should explicitly pass raw tokens.
    """
    q_raw = query_points if query_raw is None else query_raw
    s_raw = support_points if support_raw is None else support_raw

    weights = compute_aligned_dynamic_sk_weights(
        q_raw,
        s_raw,
        weight_eps=weight_eps,
    ) # 重要性
    return aligned_sinkhorn_from_weights(
        query_points,
        support_points,
        weights.query,
        weights.support,
        reg=reg,
        n_iter=n_iter,
        normalize_cost_points=normalize_cost_points,
        norm_eps=norm_eps,
        return_transport=return_transport,
        return_cost=return_cost,
    )


# -----------------------------------------------------------------------------
# Cartesian pair APIs (M2 / F1)
# -----------------------------------------------------------------------------


def pairwise_sinkhorn_from_weights(
    query_points: torch.Tensor,
    support_points: torch.Tensor,
    weight_q: torch.Tensor,
    weight_s: torch.Tensor,
    *,
    reg: float = 0.1,
    n_iter: int = 30,
    normalize_cost_points: bool = True,
    norm_eps: float = 1e-12,
    return_transport: bool = False,
    return_cost: bool = False,
) -> SinkhornResult:
    """All query-support Sinkhorn distances with supplied/fixed SK masses.

    Parameters
    ----------
    query_points:
        (..., Q, Mq, D)
    support_points:
        (..., N, Ms, D)
    weight_q:
        (..., Q, N, Mq)
    weight_s:
        (..., Q, N, Ms)

    Returns
    -------
    distance:
        (..., Q, N)
    """
    cost = pairwise_squared_euclidean_cost(
        query_points,
        support_points,
        normalize=normalize_cost_points,
        norm_eps=norm_eps,
    )
    return sinkhorn_from_cost(
        cost,
        weight_q,
        weight_s,
        reg=reg,
        n_iter=n_iter,
        return_transport=return_transport,
        return_cost=return_cost,
    )


def pairwise_sk_sinkhorn_distance(
    query_points: torch.Tensor,
    support_points: torch.Tensor,
    *,
    query_raw: Optional[torch.Tensor] = None,
    support_raw: Optional[torch.Tensor] = None,
    reg: float = 0.1,
    n_iter: int = 30,
    weight_eps: float = 1e-3,
    normalize_cost_points: bool = True,
    norm_eps: float = 1e-12,
    return_transport: bool = False,
    return_cost: bool = False,
) -> SinkhornResult:
    """Dynamic-SK Sinkhorn for all query-support candidate pairs.

    Intended for M2 direct episodic classification. For F1 trajectory training,
    do *not* call this independently at every t; instead compute weights once
    from t=0 with ``compute_pairwise_dynamic_sk_weights`` and reuse them through
    ``pairwise_sinkhorn_from_weights``.
    """
    q_raw = query_points if query_raw is None else query_raw
    s_raw = support_points if support_raw is None else support_raw

    weights = compute_pairwise_dynamic_sk_weights(
        q_raw,
        s_raw,
        weight_eps=weight_eps,
    )
    return pairwise_sinkhorn_from_weights(
        query_points,
        support_points,
        weights.query,
        weights.support,
        reg=reg,
        n_iter=n_iter,
        normalize_cost_points=normalize_cost_points,
        norm_eps=norm_eps,
        return_transport=return_transport,
        return_cost=return_cost,
    )


# -----------------------------------------------------------------------------
# Compatibility aliases / small utilities
# -----------------------------------------------------------------------------


def dynamic_node_weights(
    query_raw: torch.Tensor,
    support_raw: torch.Tensor,
    *,
    weight_eps: float = 1e-3,
    pairwise: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compatibility helper returning ``(weight_q, weight_s)``.

    Set ``pairwise=False`` for aligned pairs (..., M, D), and ``pairwise=True``
    for Cartesian sets (..., Q, M, D) vs (..., N, K, D).
    """
    if pairwise:
        weights = compute_pairwise_dynamic_sk_weights(
            query_raw, support_raw, weight_eps=weight_eps
        )
    else:
        weights = compute_aligned_dynamic_sk_weights(
            query_raw, support_raw, weight_eps=weight_eps
        )
    return weights.query, weights.support


def sk_sinkhorn_distance(
    query_points: torch.Tensor,
    support_points: torch.Tensor,
    *,
    query_raw: Optional[torch.Tensor] = None,
    support_raw: Optional[torch.Tensor] = None,
    reg: float = 0.1,
    n_iter: int = 30,
    weight_eps: float = 1e-3,
    pairwise: bool = False,
    return_transport: bool = False,
) -> SinkhornResult:
    """Unified convenience alias.

    ``pairwise=False`` -> aligned_sk_sinkhorn_distance
    ``pairwise=True``  -> pairwise_sk_sinkhorn_distance
    """
    if pairwise:
        return pairwise_sk_sinkhorn_distance(
            query_points,
            support_points,
            query_raw=query_raw,
            support_raw=support_raw,
            reg=reg,
            n_iter=n_iter,
            weight_eps=weight_eps,
            return_transport=return_transport,
        )
    return aligned_sk_sinkhorn_distance(
        query_points,
        support_points,
        query_raw=query_raw,
        support_raw=support_raw,
        reg=reg,
        n_iter=n_iter,
        weight_eps=weight_eps,
        return_transport=return_transport,
    )


__all__ = [
    "SKWeights",
    "SinkhornResult",
    "l2_normalize_points",
    "aligned_squared_euclidean_cost",
    "pairwise_squared_euclidean_cost",
    "compute_aligned_dynamic_sk_weights",
    "compute_pairwise_dynamic_sk_weights",
    "dynamic_node_weights",
    "sinkhorn_from_cost",
    "aligned_sinkhorn_from_weights",
    "pairwise_sinkhorn_from_weights",
    "aligned_sk_sinkhorn_distance",
    "pairwise_sk_sinkhorn_distance",
    "sk_sinkhorn_distance",
]