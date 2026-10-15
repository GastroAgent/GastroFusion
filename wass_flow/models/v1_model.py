from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .metric_encoder import MetricEncoder, MetricFeatures, pool_square_tokens
from .sk_sinkhorn import (
    OTResult,
    aligned_sk_marginals,
    aligned_sk_sinkhorn_distance,
    sk_sinkhorn_distance,
)
from .trajectory_scorer import (
    TrajectoryScore,
    TrajectoryScorer,
    direct_shot_softmin_logits,
)


@dataclass
class DirectOutput:
    logits: torch.Tensor
    class_energy: torch.Tensor
    shot_distances: torch.Tensor
    query_features: MetricFeatures
    support_features: MetricFeatures


@dataclass
class FlowOutput:
    score: TrajectoryScore
    distances: torch.Tensor
    direct_logits: torch.Tensor
    direct_class_energy: torch.Tensor
    trajectory_features: MetricFeatures
    support_features: MetricFeatures
    query_mass: torch.Tensor
    support_mass: torch.Tensor


class RAEMetricV1(nn.Module):
    """Metric/OT/scoring core; RAE and DiT remain external frozen modules."""

    def __init__(
        self,
        metric_encoder: MetricEncoder | None = None,
        sinkhorn_reg: float = 0.1,
        sinkhorn_iter: int = 30,
        sk_eps: float = 1e-3,
        ot_pool_factor: int = 1,
        trajectory_scorer: TrajectoryScorer | None = None,
    ) -> None:
        super().__init__()
        self.metric_encoder = metric_encoder or MetricEncoder()
        self.trajectory_scorer = trajectory_scorer or TrajectoryScorer()
        self.sinkhorn_reg = float(sinkhorn_reg)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.sk_eps = float(sk_eps)
        self.ot_pool_factor = int(ot_pool_factor)


    def forward(self, mode: str, *args, **kwargs):
        if mode == "direct":
            return self.direct(*args, **kwargs)
        if mode == "flow":
            return self.flow(*args, **kwargs)
        raise ValueError(f"Unknown RAEMetricV1 mode: {mode}")

    def _pool_features(self, features: MetricFeatures) -> MetricFeatures:
        if self.ot_pool_factor == 1:
            return features
        raw = pool_square_tokens(features.raw, self.ot_pool_factor)
        normalized = torch.nn.functional.normalize(raw, p=2, dim=-1)
        return MetricFeatures(raw=raw, normalized=normalized, cloud=raw.mean(dim=-2))

    def direct(
        self,
        query_latents: torch.Tensor,
        support_latents: torch.Tensor,
    ) -> DirectOutput:
        """Direct N-way K-shot metric classification.

        Args:
            query_latents: (B, Q, C, H, W)
            support_latents: (B, N, K, C, H, W)
        """
        if query_latents.ndim != 5 or support_latents.ndim != 6:
            raise ValueError(
                f"Expected query (B,Q,C,H,W), support (B,N,K,C,H,W); got "
                f"{query_latents.shape}, {support_latents.shape}"
            )
        b, q = query_latents.shape[:2]
        n_way, n_shot = support_latents.shape[1:3]
        q_feat_full = self.metric_encoder(query_latents)
        s_feat_full = self.metric_encoder(support_latents)
        q_feat = self._pool_features(q_feat_full)
        s_feat = self._pool_features(s_feat_full)
        s_raw = s_feat.raw.reshape(b, n_way * n_shot, *s_feat.raw.shape[-2:])
        s_norm = s_feat.normalized.reshape(b, n_way * n_shot, *s_feat.normalized.shape[-2:])
        result = sk_sinkhorn_distance(
            q_feat.normalized,
            s_norm,
            query_raw=q_feat.raw,
            support_raw=s_raw,
            reg=self.sinkhorn_reg,
            n_iter=self.sinkhorn_iter,
            weight_eps=self.sk_eps,
        )
        shot_dist = result.distance.reshape(b, q, n_way, n_shot)
        class_energy, logits = direct_shot_softmin_logits(
            shot_dist,
            self.trajectory_scorer.shot_temperature,
            self.trajectory_scorer.class_temperature,
        )
        return DirectOutput(
            logits=logits,
            class_energy=class_energy,
            shot_distances=shot_dist,
            query_features=q_feat_full,
            support_features=s_feat_full,
        )

    def flow(
        self,
        trajectories: torch.Tensor,
        support_latents: torch.Tensor,
    ) -> FlowOutput:
        """Classify pair-conditioned trajectories with fixed Direct SK masses.

        Args:
            trajectories: (B, Q, N, K, V, C, H, W), including x0 and predicted
                states only. Do not append the GT target as a time step.
            support_latents: (B, N, K, C, H, W)
        """
        if trajectories.ndim != 8 or support_latents.ndim != 6:
            raise ValueError(
                f"Expected trajectory (B,Q,N,K,V,C,H,W) and support (B,N,K,C,H,W), "
                f"got {trajectories.shape}, {support_latents.shape}"
            )
        b, q, n_way, n_shot, v = trajectories.shape[:5]
        if support_latents.shape[:3] != (b, n_way, n_shot):
            raise ValueError("trajectory/support leading shapes do not match")

        traj_full = self.metric_encoder(trajectories)
        sup_full = self.metric_encoder(support_latents)
        traj = self._pool_features(traj_full)
        sup = self._pool_features(sup_full)

        # Expand each support over query and time; all tensors are aligned pairs.
        sup_raw_pair = sup.raw[:, None, :, :, None].expand(
            b, q, n_way, n_shot, v, *sup.raw.shape[-2:]
        )
        sup_norm_pair = sup.normalized[:, None, :, :, None].expand_as(sup_raw_pair)

        # Fixed SK masses are computed once from the Direct state t=0.
        q_mass, s_mass = aligned_sk_marginals(
            traj.raw[..., 0, :, :],
            sup.raw[:, None].expand(b, q, n_way, n_shot, *sup.raw.shape[-2:]),
            eps=self.sk_eps,
        )
        q_mass_t = q_mass.unsqueeze(-2).expand(*q_mass.shape[:-1], v, q_mass.shape[-1])
        s_mass_t = s_mass.unsqueeze(-2).expand(*s_mass.shape[:-1], v, s_mass.shape[-1])
        result = aligned_sk_sinkhorn_distance(
            traj.normalized,
            sup_norm_pair,
            fixed_marginals=(q_mass_t, s_mass_t),
            reg=self.sinkhorn_reg,
            n_iter=self.sinkhorn_iter,
        )
        distances = result.distance  # (B,Q,N,K,V)
        score = self.trajectory_scorer(distances)
        direct_energy, direct_logits = direct_shot_softmin_logits(
            distances[..., 0],
            self.trajectory_scorer.shot_temperature,
            self.trajectory_scorer.class_temperature,
        )
        return FlowOutput(
            score=score,
            distances=distances,
            direct_logits=direct_logits,
            direct_class_energy=direct_energy,
            trajectory_features=traj_full,
            support_features=sup_full,
            query_mass=q_mass,
            support_mass=s_mass,
        )
