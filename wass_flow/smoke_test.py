#!/usr/bin/env python3
"""CPU synthetic test for metric/OT/trajectory/VISReg modules (no RAE/DiT weights)."""
from __future__ import annotations

import torch

from losss.geometry import trajectory_geometry_loss
from losss.visreg import VISReg
from models.metric_encoder import MetricEncoder
from models.v1_model import RAEMetricV1


def main():
    torch.manual_seed(0)
    metric = MetricEncoder(in_dim=1024, hidden_dim=64, out_dim=32)
    core = RAEMetricV1(
        metric_encoder=metric,
        sinkhorn_reg=0.2,
        sinkhorn_iter=5,
        ot_pool_factor=4,
    )
    b, q, n, k, v = 1, 2, 3, 2, 4
    query = torch.randn(b, q, 1024, 16, 16)
    support = torch.randn(b, n, k, 1024, 16, 16)
    direct = core("direct", query, support)
    assert direct.logits.shape == (b, q, n)
    direct.logits.sum().backward()
    assert metric.net[0].weight.grad is not None
    metric.zero_grad(set_to_none=True)

    trajectory = torch.randn(b, q, n, k, v, 1024, 16, 16)
    flow = core("flow", trajectory, support)
    assert flow.distances.shape == (b, q, n, k, v)
    assert flow.score.logits.shape == (b, q, n)
    flow.score.logits.sum().backward()
    assert metric.net[0].weight.grad is not None

    visreg = VISReg(num_projections=8, projection_chunk_size=4)
    z = torch.randn(v, 10, 32, requires_grad=True)
    vis_loss = visreg(z)
    vis_loss.backward()
    assert z.grad is not None

    rae_tokens = torch.randn(5, v, 16, 1024)
    metric_tokens = torch.randn(5, v, 16, 32, requires_grad=True)
    geom = trajectory_geometry_loss(rae_tokens, metric_tokens)
    geom.backward()
    assert metric_tokens.grad is not None
    print("RAE–Flow–Metric V1 synthetic smoke test passed")


if __name__ == "__main__":
    main()
