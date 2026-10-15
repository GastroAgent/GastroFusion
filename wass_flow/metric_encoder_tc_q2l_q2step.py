import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Zero initialized timestep branch. Init output is exactly zero."""
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

    def forward(self, h, t_emb):
        gamma, beta = self.proj(t_emb).chunk(2, dim=-1)
        return h * (1 + gamma[:, None, :]) + beta[:, None, :]


class TimeConditionedMetricEncoder(nn.Module):
    """
    Wrapper around pretrained Q2L MetricEncoder.

    Initialization guarantee:
        TimeConditionedMetricEncoder(z,t)
        == MetricEncoder(z)

    because all timestep modulation parameters are zero initialized.
    """
    def __init__(self, q2l_metric_encoder):
        super().__init__()
        self.backbone = q2l_metric_encoder
        dim = getattr(q2l_metric_encoder, "metric_dim", None) or 256

        self.time_embedding = TimeEmbedding(dim)
        self.time_adaln = AdaLNZero(dim)

    def forward(self, z, t):
        base = self.backbone(z)

        if hasattr(base, "raw"):
            raw = base.raw
            cloud = base.cloud
            raw = self._apply_time(raw, t)
            return raw, cloud

        return self._apply_time(base, t)

    def _apply_time(self, x, t):
        emb = self.time_embedding(t)
        if x.ndim == 2:
            return self.time_adaln(x[:, None, :], emb)[:, 0, :]
        return self.time_adaln(x, emb)
