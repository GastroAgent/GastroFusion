from __future__ import annotations

import torch
import torch.nn.functional as F


def trajectory_step_profile(tokens: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalized adjacent-step motion profile.

    Args:
        tokens: (..., V, N, D)
    Returns:
        (..., V-1), sums to one along the final dimension.
    """
    if tokens.ndim < 4 or tokens.shape[-3] < 2:
        raise ValueError(f"Expected (..., V, N, D), got {tuple(tokens.shape)}")
    delta = (tokens[..., 1:, :, :] - tokens[..., :-1, :, :]).norm(dim=-1).mean(dim=-1)
    return delta / delta.sum(dim=-1, keepdim=True).clamp_min(eps)


def trajectory_geometry_loss(
    rae_tokens: torch.Tensor,
    metric_tokens: torch.Tensor,
    loss: str = "l1",
) -> torch.Tensor:
    """Keep the relative temporal motion profile after dimensionality reduction."""
    p_rae = trajectory_step_profile(rae_tokens).detach()
    p_metric = trajectory_step_profile(metric_tokens)
    if loss == "l1":
        return F.l1_loss(p_metric, p_rae)
    if loss == "mse":
        return F.mse_loss(p_metric, p_rae)
    raise ValueError(f"Unknown geometry loss: {loss}")


def sk_weighted_path_length(metric_tokens: torch.Tensor, query_mass: torch.Tensor) -> torch.Tensor:
    """Diagnostic SK-weighted path length.

    metric_tokens: (..., V, N, D)
    query_mass: (..., N), broadcast-compatible with the leading dimensions.
    """
    delta = (metric_tokens[..., 1:, :, :] - metric_tokens[..., :-1, :, :]).norm(dim=-1)
    return (delta * query_mass.unsqueeze(-2)).sum(dim=-1).sum(dim=-1)
