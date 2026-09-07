"""Hyper-prior compression utilities for FieldSpace tensors."""

from .distributions import FieldSpaceDiagonalGaussianDistribution, MGDiagonalGaussianDistribution
from .entropy_adapters import FieldSpaceEntropyBottleneckAdapter, FieldSpaceGaussianConditionalAdapter
from .losses import HEALPixCRA5RateDistortionLoss
from .projections import MultiZoomPointwiseProjection

__all__ = [
    "FieldSpaceDiagonalGaussianDistribution",
    "FieldSpaceEntropyBottleneckAdapter",
    "FieldSpaceGaussianConditionalAdapter",
    "HEALPixCRA5RateDistortionLoss",
    "MGDiagonalGaussianDistribution",
    "MultiZoomPointwiseProjection",
]
