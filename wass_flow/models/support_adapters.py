from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class IdentitySupportAdapter(nn.Module):
    """Keep K individual support clouds. Core V1 uses this by default."""

    mode = "none"

    def forward(self, support_tokens: torch.Tensor) -> torch.Tensor:
        return support_tokens


class MeanSupportAdapter(nn.Module):
    """Simple class prototype in metric space: (..., N, K, M, D) -> (..., N, 1, M, D)."""

    mode = "mean"

    def forward(self, support_tokens: torch.Tensor) -> torch.Tensor:
        if support_tokens.ndim < 5:
            raise ValueError("Expected (..., n_way, n_shot, n_nodes, dim)")
        return support_tokens.mean(dim=-3, keepdim=True)


class ExternalSFCAdapter(nn.Module):
    """Thin, removable wrapper around an externally supplied SFC implementation.

    The uploaded files only show calls to get_sfc(), not its implementation. This
    adapter therefore makes no assumptions about optimization or differentiability.
    The callable must accept one episode shaped (n_way, n_shot, n_nodes, dim) and
    return (n_way, n_nodes, dim) or (n_way, 1, n_nodes, dim).
    """

    mode = "sfc"

    def __init__(self, sfc_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.sfc_fn = sfc_fn

    def forward(self, support_tokens: torch.Tensor) -> torch.Tensor:
        if support_tokens.ndim != 5:
            raise ValueError(
                "ExternalSFCAdapter currently expects batched episodes "
                "(B,n_way,n_shot,n_nodes,dim); "
                f"got {tuple(support_tokens.shape)}"
            )
        outputs = []
        for episode in support_tokens:
            result = self.sfc_fn(episode)
            if result.ndim == 3:
                result = result.unsqueeze(1)
            if result.ndim != 4 or result.shape[1] != 1:
                raise RuntimeError(f"SFC returned unsupported shape {tuple(result.shape)}")
            outputs.append(result)
        return torch.stack(outputs)
