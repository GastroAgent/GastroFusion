"""
Time-conditioned Metric Encoder v3

Initialization:
Q2L MetricEncoder checkpoint -> backbone
Time branch -> zero initialized

Guarantee:
M_tc(z,t) ~= M_q2l(z) at initialization.
"""

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]
        return self.net(t.float())


class AdaLNZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, t_emb):
        gamma, beta = self.proj(t_emb).chunk(2, dim=-1)
        return x * (1 + gamma) + beta


class TimeConditionedMetricEncoder(nn.Module):
    def __init__(self, q2l_metric_encoder, metric_dim=256):
        super().__init__()
        self.backbone = q2l_metric_encoder
        self.time_embedding = TimeEmbedding(metric_dim)
        self.time_modulation = AdaLNZero(metric_dim)

    def forward(self, x, t):
        feat = self.backbone(x)
        if hasattr(feat, "raw"):
            raw = self.time_modulation(
                feat.raw,
                self.time_embedding(t)
            )
            return raw, feat.cloud
        return self.time_modulation(
            feat,
            self.time_embedding(t)
        )
