"""
VISReg adapted for FSL point embeddings.

Core objective follows VISReg (Variance-Invariance-Sketching Regularization):
  1) center: drive embedding mean toward 0
  2) scale: drive per-dimension standard deviation toward 1
  3) shape: after detached standardization, match random 1-D projections
     to standard-normal quantiles (sliced-Wasserstein style)

Expected input:
    z: (V, B, D)

For this FSL project, VISReg is applied at the independent point-cloud level:
each support class cloud / query cloud is first represented by the mean of its
pre-L2 point features. A meta-batch then forms a cloud-embedding population
(1, E*(N+Q), D). Wasserstein itself still uses normalized point clouds and
compares query vs each support class independently.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VISReg(nn.Module):
    """Variance-Invariance-Sketching regularizer.

    ``projection_chunk_size`` is an engineering adaptation for this project:
    shape projections are evaluated in chunks to reduce peak GPU memory. It does
    not change the objective; the loss is averaged across the same number of
    random projection directions.
    """

    def __init__(
        self,
        num_projections: int = 1024,
        scale_weight: float = 1.0,
        shape_weight: float = 1.0,
        center_weight: float = 1.0,
        projection_chunk_size: int = 128,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_projections <= 0:
            raise ValueError("num_projections 必须 > 0")
        if projection_chunk_size <= 0:
            raise ValueError("projection_chunk_size 必须 > 0")

        self.num_projections = int(num_projections)
        self.scale_weight = float(scale_weight)
        self.shape_weight = float(shape_weight)
        self.center_weight = float(center_weight)
        self.projection_chunk_size = int(projection_chunk_size)
        self.eps = float(eps)

        self._cached_b: int = -1
        self._cached_target: torch.Tensor | None = None

    def _normal_quantiles(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return standard-normal quantiles with float32 target computation."""
        if batch_size < 2:
            raise ValueError(
                f"VISReg 至少需要 2 个 embedding sample，当前 B={batch_size}。"
            )
        if self._cached_b != batch_size or self._cached_target is None:
            # q_i = i/(B+1), i=1..B. Compute in fp32 for AMP stability.
            q = torch.arange(
                1, batch_size + 1, device=device, dtype=torch.float32
            ).div_(batch_size + 1)
            target = torch.erfinv(q.mul(2.0).sub(1.0)).mul_(math.sqrt(2.0))
            self._cached_target = target
            self._cached_b = batch_size
        return self._cached_target.to(device=device)

    def compute_terms(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(scale_loss, shape_loss, center_loss)``."""
        if z.ndim != 3:
            raise ValueError(f"VISReg 输入必须是 (V,B,D)，实际 shape={tuple(z.shape)}")

        _, batch_size, feat_dim = z.shape
        if batch_size < 2:
            raise ValueError(
                f"VISReg sample 维 B 必须 >=2，实际 B={batch_size}。"
            )

        # Center term.
        mean = z.mean(dim=1, keepdim=True)
        center_loss = mean.square().mean()

        # Scale term. This follows VISReg's RMS standard deviation convention.
        centered = z - mean
        std = centered.norm(p=2, dim=1).div(math.sqrt(batch_size)).clamp_min(self.eps)
        scale_loss = (std - 1.0).square().mean()

        # Shape term. Detaching std decouples shape optimization from scale.
        normalized = centered / std.detach().unsqueeze(1)
        target = self._normal_quantiles(batch_size, z.device).view(1, batch_size, 1)

        shape_sum = z.new_zeros(())
        done = 0
        while done < self.num_projections:
            k = min(self.projection_chunk_size, self.num_projections - done)
            directions = torch.randn(
                feat_dim, k, device=z.device, dtype=z.dtype
            )
            directions = F.normalize(directions, p=2, dim=0)

            projected = torch.matmul(normalized, directions)
            projected = projected.sort(dim=1).values

            # target is float32 by design. Our FSL training feeds fp32 point
            # embeddings; for other dtypes, compare in fp32 for stability.
            chunk_loss = (projected.float() - target).square().mean()
            shape_sum = shape_sum + chunk_loss.to(dtype=z.dtype) * k
            done += k

        shape_loss = shape_sum / float(self.num_projections)
        return scale_loss, shape_loss, center_loss

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        scale_loss, shape_loss, center_loss = self.compute_terms(z)
        return (
            self.scale_weight * scale_loss
            + self.shape_weight * shape_loss
            + self.center_weight * center_loss
        )


def cloud_features_to_visreg_input(
    support_cloud: torch.Tensor,
    query_cloud: torch.Tensor,
) -> torch.Tensor:
    """Convert independent cloud-level embeddings to VISReg ``(V,B,D)``.

    Batch mode:
        support_cloud: (E, n_way, D)
        query_cloud:   (E, n_query, D)

        Each support class and each query is an independent point cloud.
        VISReg treats those cloud representations as the sample population:
            (E, n_way+n_query, D) -> (1, E*(n_way+n_query), D)

    Single-episode mode:
        support_cloud: (n_way, D)
        query_cloud:   (n_query, D)

        output: (1, n_way+n_query, D)

    Wasserstein never uses this pooled tensor. It still compares each query
    point cloud against each support-class point cloud independently.
    """
    if support_cloud.ndim != query_cloud.ndim:
        raise ValueError(
            "support_cloud/query_cloud 维数不一致: "
            f"{support_cloud.ndim} vs {query_cloud.ndim}"
        )

    if support_cloud.ndim == 3:
        # (E,N,D)+(E,Q,D) -> (1,E*(N+Q),D)
        clouds = torch.cat([support_cloud, query_cloud], dim=1)
        feat_dim = clouds.shape[-1]
        return clouds.reshape(1, -1, feat_dim).contiguous()

    if support_cloud.ndim == 2:
        # (N,D)+(Q,D) -> (1,N+Q,D)
        clouds = torch.cat([support_cloud, query_cloud], dim=0)
        return clouds.unsqueeze(0).contiguous()

    raise ValueError(
        "cloud_features_to_visreg_input 仅支持 2D/3D tensors，实际 "
        f"support={tuple(support_cloud.shape)}, query={tuple(query_cloud.shape)}"
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reg = VISReg(num_projections=256, projection_chunk_size=64).to(device)

    sp = torch.randn(4, 7, 256, device=device, requires_grad=True)
    qp = torch.randn(4, 1, 256, device=device, requires_grad=True)
    z = cloud_features_to_visreg_input(sp, qp)
    assert z.shape == (1, 32, 256), z.shape
    loss = reg(z)
    loss.backward()

    z_single = cloud_features_to_visreg_input(sp[0], qp[0])
    assert z_single.shape == (1, 8, 256), z_single.shape

    print(f"batch_input={tuple(z.shape)}, single_input={tuple(z_single.shape)}, "
          f"loss={float(loss.detach()):.6f}")
    print("VISReg cloud-level batch regularization smoke test passed")
