from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrajectoryScore:
    pair_energy: torch.Tensor
    class_energy: torch.Tensor
    logits: torch.Tensor
    direct: torch.Tensor
    endpoint: torch.Tensor
    area: torch.Tensor
    monotonic_violation: torch.Tensor


def _inv_softplus(x: float) -> float:
    x_t = torch.tensor(float(x))
    return float(torch.log(torch.expm1(x_t)).item())


class TrajectoryScorer(nn.Module):
    """Positive linear trajectory energy followed by differentiable shot soft-min.

    Expected distance shape: (..., n_way, n_shot, T+1).
    """

    def __init__(
        self,
        init_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        shot_temperature: float = 0.25,
        class_temperature: float = 1.0,
        ratio_clip: float = 10.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.raw_weights = nn.Parameter(torch.tensor([_inv_softplus(v) for v in init_weights]))
        self.raw_shot_temperature = nn.Parameter(torch.tensor(_inv_softplus(shot_temperature)))
        self.raw_class_temperature = nn.Parameter(torch.tensor(_inv_softplus(class_temperature)))
        self.ratio_clip = float(ratio_clip)
        self.eps = float(eps)

    @property
    def weights(self) -> torch.Tensor:
        return F.softplus(self.raw_weights)

    @property
    def shot_temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_shot_temperature).clamp_min(1e-4)

    @property
    def class_temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_class_temperature).clamp_min(1e-4)

    def forward(self, distances: torch.Tensor) -> TrajectoryScore:
        if distances.ndim < 3 or distances.shape[-1] < 2:
            raise ValueError(
                "distances must be (..., n_way, n_shot, T+1), "
                f"got {tuple(distances.shape)}"
            )
        direct = distances[..., 0]
        endpoint = distances[..., -1]
        area = distances[..., 1:].mean(dim=-1)
        monotonic = F.relu(distances[..., 1:] - distances[..., :-1]).mean(dim=-1)

        denom = direct.detach().clamp_min(self.eps)
        endpoint_rel = (endpoint / denom).clamp(max=self.ratio_clip)
        area_rel = (area / denom).clamp(max=self.ratio_clip)
        mono_rel = (monotonic / denom).clamp(max=self.ratio_clip)
        components = torch.stack([direct, endpoint_rel, area_rel, mono_rel], dim=-1)
        pair_energy = (components * self.weights).sum(dim=-1)

        tau_s = self.shot_temperature
        # soft-min with 1/K normalization so its scale is stable across shot count.
        class_energy = -tau_s * (
            torch.logsumexp(-pair_energy / tau_s, dim=-1)
            - torch.log(torch.tensor(pair_energy.shape[-1], device=pair_energy.device, dtype=pair_energy.dtype))
        )
        logits = -class_energy / self.class_temperature
        return TrajectoryScore(
            pair_energy=pair_energy,
            class_energy=class_energy,
            logits=logits,
            direct=direct,
            endpoint=endpoint,
            area=area,
            monotonic_violation=monotonic,
        )


def direct_shot_softmin_logits(
    direct_distances: torch.Tensor,
    shot_temperature: torch.Tensor | float = 0.25,
    class_temperature: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct-only class energy/logits from (..., n_way, n_shot) distances."""
    tau_s = torch.as_tensor(shot_temperature, device=direct_distances.device, dtype=direct_distances.dtype).clamp_min(1e-4)
    tau_c = torch.as_tensor(class_temperature, device=direct_distances.device, dtype=direct_distances.dtype).clamp_min(1e-4)
    class_energy = -tau_s * (
        torch.logsumexp(-direct_distances / tau_s, dim=-1)
        - torch.log(torch.tensor(direct_distances.shape[-1], device=direct_distances.device, dtype=direct_distances.dtype))
    )
    return class_energy, -class_energy / tau_c
