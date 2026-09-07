"""Pointwise projections for nested FieldSpace tensors."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn


class MultiZoomPointwiseProjection(nn.Module):
    """Project the last feature dimension of every zoom-group tensor.

    Inputs and outputs keep the FieldSpace layout ``(b, v, t, n, d, f)``. The
    projection can use per-zoom ``nn.Linear`` layers, with an optional shared
    fallback for zooms that are not configured explicitly.
    """

    def __init__(
        self,
        zooms: Optional[Sequence[int]] = None,
        in_features: int | Sequence[int] | Mapping[int, int] = 1,
        out_features: int = 1,
        shared: bool = False,
    ) -> None:
        super().__init__()
        self.out_features = int(out_features)
        self.projections = nn.ModuleDict()
        self.shared_projection: Optional[nn.Linear] = None

        if shared:
            if isinstance(in_features, Mapping) or isinstance(in_features, Sequence) and not isinstance(in_features, (str, bytes)):
                raise ValueError("Shared projection requires a single integer in_features value.")
            self.shared_projection = nn.Linear(int(in_features), self.out_features)
            return

        if zooms is None:
            if not isinstance(in_features, int):
                raise ValueError("Per-zoom projections require explicit zooms when in_features is not an int.")
            self.shared_projection = nn.Linear(int(in_features), self.out_features)
            return

        feature_map = self._feature_map(zooms, in_features)
        for zoom, feature_count in feature_map.items():
            if int(feature_count) == self.out_features:
                self.projections[str(int(zoom))] = nn.Identity()
            else:
                self.projections[str(int(zoom))] = nn.Linear(int(feature_count), self.out_features)

    def forward(
        self,
        groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """Project every tensor in ``groups``."""

        output: List[Optional[Dict[int, torch.Tensor]]] = []
        for group in groups:
            if group is None:
                output.append(None)
                continue
            projected_group: Dict[int, torch.Tensor] = {}
            for zoom, tensor in group.items():
                projected_group[int(zoom)] = self._project_tensor(int(zoom), tensor)
            output.append(projected_group)
        return output

    def _project_tensor(self, zoom: int, tensor: torch.Tensor) -> torch.Tensor:
        projection = self.projections[str(int(zoom))] if str(int(zoom)) in self.projections else self.shared_projection
        if projection is None:
            raise KeyError(f"No projection configured for zoom {zoom} and no shared projection is available.")
        return projection(tensor)

    @staticmethod
    def _feature_map(
        zooms: Sequence[int],
        in_features: int | Sequence[int] | Mapping[int, int],
    ) -> Dict[int, int]:
        if isinstance(in_features, Mapping):
            return {int(zoom): int(in_features[int(zoom)]) for zoom in zooms}
        if isinstance(in_features, int):
            return {int(zoom): int(in_features) for zoom in zooms}
        if len(in_features) != len(zooms):
            raise ValueError("in_features sequence must have the same length as zooms.")
        return {int(zoom): int(feature_count) for zoom, feature_count in zip(zooms, in_features)}
