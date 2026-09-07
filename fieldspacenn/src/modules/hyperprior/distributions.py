"""Diagonal Gaussian distributions for multi-grid FieldSpace latents."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence

import torch


class FieldSpaceDiagonalGaussianDistribution:
    """Diagonal Gaussian over a single FieldSpace tensor.

    The input tensor stores moments as ``(..., 2 * latent_features)`` and is split
    into mean and log-variance along the last dimension.
    """

    def __init__(self, moments: torch.Tensor) -> None:
        if moments.shape[-1] % 2 != 0:
            raise ValueError(
                "FieldSpaceDiagonalGaussianDistribution expects an even last "
                f"dimension, got shape {tuple(moments.shape)}."
            )
        self.moments = moments
        self.mean, self.logvar = torch.chunk(moments, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        self.dims: List[int] = [idx for idx in range(1, self.mean.dim())]

    def sample(self) -> torch.Tensor:
        """Sample with the reparameterization trick."""

        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self) -> torch.Tensor:
        """Return the posterior mode."""

        return self.mean

    def kl(self, other: Optional["FieldSpaceDiagonalGaussianDistribution"] = None) -> torch.Tensor:
        """Return per-batch KL divergence to ``other`` or ``N(0, I)``."""

        if other is None:
            return 0.5 * torch.sum(self.mean.pow(2) + self.var - 1.0 - self.logvar, dim=self.dims)
        return 0.5 * torch.sum(
            (self.mean - other.mean).pow(2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=self.dims,
        )

    def nll(self, sample: torch.Tensor) -> torch.Tensor:
        """Return per-batch negative log-likelihood of ``sample``."""

        logtwopi = math.log(2.0 * math.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + (sample - self.mean).pow(2) / self.var, dim=self.dims)

    def to(self, device: torch.device | str) -> "FieldSpaceDiagonalGaussianDistribution":
        """Move moments to ``device`` and rebuild cached tensors."""

        return FieldSpaceDiagonalGaussianDistribution(self.moments.to(device=device))


class MGDiagonalGaussianDistribution:
    """Diagonal Gaussian over a nested zoom-group FieldSpace structure."""

    def __init__(self, moments_groups: Sequence[Optional[Mapping[int, torch.Tensor]]]) -> None:
        self.distributions: List[Optional[Dict[int, FieldSpaceDiagonalGaussianDistribution]]] = []
        for group in moments_groups:
            if group is None:
                self.distributions.append(None)
                continue
            self.distributions.append(
                {int(zoom): FieldSpaceDiagonalGaussianDistribution(tensor) for zoom, tensor in group.items()}
            )

    def sample(self) -> List[Optional[Dict[int, torch.Tensor]]]:
        """Sample every tensor and preserve the nested structure."""

        return self._map(lambda dist: dist.sample())

    def mode(self) -> List[Optional[Dict[int, torch.Tensor]]]:
        """Return posterior modes with the nested structure preserved."""

        return self._map(lambda dist: dist.mode())

    def kl(self, reduction: str = "mean") -> torch.Tensor:
        """Compute KL to the standard normal across all groups and zooms."""

        values = []
        for group in self.distributions:
            if not group:
                continue
            values.extend(dist.kl() for dist in group.values())
        if not values:
            return torch.tensor(0.0)
        stacked = torch.cat([value.reshape(-1) for value in values])
        if reduction == "mean":
            return stacked.mean()
        if reduction == "sum":
            return stacked.sum()
        if reduction == "none":
            return stacked
        raise ValueError(f"Unknown KL reduction '{reduction}'.")

    def kl_dict(self) -> Dict[str, torch.Tensor]:
        """Return diagnostic mean KL values keyed by group and zoom."""

        result = {}
        for group_idx, group in enumerate(self.distributions):
            if not group:
                continue
            for zoom, dist in group.items():
                result[f"group{group_idx}/zoom{zoom}"] = dist.kl().mean()
        return result

    def _map(self, fn) -> List[Optional[Dict[int, torch.Tensor]]]:
        output: List[Optional[Dict[int, torch.Tensor]]] = []
        for group in self.distributions:
            if group is None:
                output.append(None)
                continue
            output.append({zoom: fn(dist) for zoom, dist in group.items()})
        return output
