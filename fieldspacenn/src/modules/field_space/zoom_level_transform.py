import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from .field_space_base import coarsen_zoom, refine_zoom


class ZoomLevelTransformConfig:
    """Base configuration for one-way zoom-level transformation layers."""

    transform_kind = "base"

    def __init__(self, zoom_mapping: Mapping[int, int]) -> None:
        self.zoom_mapping = {
            int(reference_zoom): int(out_zoom)
            for reference_zoom, out_zoom in zoom_mapping.items()
        }
        if not self.zoom_mapping:
            raise ValueError(f"{type(self).__name__} requires a non-empty zoom mapping.")


class RefineZoomsLayerConfig(ZoomLevelTransformConfig):
    transform_kind = "refine"

    def __init__(self, refine_zooms: Mapping[int, int]) -> None:
        self.refine_zooms = {
            int(reference_zoom): int(out_zoom)
            for reference_zoom, out_zoom in refine_zooms.items()
        }
        super().__init__(self.refine_zooms)


class CoarsenZoomsLayerConfig(ZoomLevelTransformConfig):
    transform_kind = "coarsen"

    def __init__(self, coarsen_zooms: Mapping[int, int]) -> None:
        self.coarsen_zooms = {
            int(reference_zoom): int(out_zoom)
            for reference_zoom, out_zoom in coarsen_zooms.items()
        }
        super().__init__(self.coarsen_zooms)


class CreateZeroZoomsLayerConfig(ZoomLevelTransformConfig):
    transform_kind = "zeros"

    def __init__(self, zero_zooms: Mapping[int, int]) -> None:
        self.zero_zooms = {
            int(reference_zoom): int(out_zoom)
            for reference_zoom, out_zoom in zero_zooms.items()
        }
        super().__init__(self.zero_zooms)


class ZoomLevelTransformLayer(nn.Module):
    """Apply a one-way refine, coarsen, or zero-allocation zoom transform."""

    def __init__(
        self,
        config: ZoomLevelTransformConfig,
        in_zooms: Sequence[int],
        in_features: Sequence[int],
    ) -> None:
        super().__init__()
        self.transform_kind = config.transform_kind
        self.zoom_mapping = dict(config.zoom_mapping)

        feature_by_zoom = {
            int(zoom): int(feature)
            for zoom, feature in zip(in_zooms, in_features)
        }
        for reference_zoom, out_zoom in self.zoom_mapping.items():
            if reference_zoom not in feature_by_zoom:
                raise ValueError(
                    f"{type(config).__name__} requires reference zoom {reference_zoom} before "
                    f"creating zoom {out_zoom}."
                )
            if out_zoom in feature_by_zoom:
                raise ValueError(
                    f"{type(config).__name__} will not overwrite existing zoom {out_zoom}."
                )
            self._validate_direction(reference_zoom, out_zoom)
            feature_by_zoom[out_zoom] = feature_by_zoom[reference_zoom]

        self.out_zooms: List[int] = sorted(feature_by_zoom)
        self.out_features: List[int] = [feature_by_zoom[zoom] for zoom in self.out_zooms]

    def _validate_direction(self, reference_zoom: int, out_zoom: int) -> None:
        if self.transform_kind == "refine" and out_zoom <= reference_zoom:
            raise ValueError(
                f"Refinement requires a higher output zoom, got {reference_zoom}->{out_zoom}."
            )
        if self.transform_kind == "coarsen" and out_zoom >= reference_zoom:
            raise ValueError(
                f"Coarsening requires a lower output zoom, got {reference_zoom}->{out_zoom}."
            )
        if self.transform_kind == "zeros" and out_zoom == reference_zoom:
            raise ValueError("A zero output zoom must differ from its reference zoom.")

    @staticmethod
    def _zero_spatial_size(reference_size: int, reference_zoom: int, out_zoom: int) -> int:
        if out_zoom > reference_zoom:
            return reference_size * 4 ** (out_zoom - reference_zoom)

        divisor = 4 ** (reference_zoom - out_zoom)
        if reference_size % divisor:
            raise ValueError(
                f"Cannot create zoom {out_zoom} from zoom {reference_zoom}: spatial size "
                f"{reference_size} is not divisible by {divisor}."
            )
        return reference_size // divisor

    def _transform_tensor(
        self,
        reference: torch.Tensor,
        reference_zoom: int,
        out_zoom: int,
    ) -> torch.Tensor:
        if self.transform_kind == "refine":
            return refine_zoom(reference, reference_zoom, out_zoom)
        if self.transform_kind == "coarsen":
            return coarsen_zoom(reference, reference_zoom, out_zoom)
        if self.transform_kind == "zeros":
            output_shape = list(reference.shape)
            output_shape[3] = self._zero_spatial_size(
                reference.shape[3],
                reference_zoom,
                out_zoom,
            )
            return reference.new_zeros(output_shape)
        raise ValueError(f"Unsupported zoom transform kind: {self.transform_kind}.")

    def _transform_mask(
        self,
        reference: torch.Tensor,
        reference_zoom: int,
        out_zoom: int,
        output_shape: Sequence[int],
    ) -> torch.Tensor:
        if self.transform_kind == "refine":
            return refine_zoom(reference, reference_zoom, out_zoom)
        if self.transform_kind == "coarsen":
            if reference.dtype != torch.bool:
                return coarsen_zoom(reference, reference_zoom, out_zoom)

            divisor = 4 ** (reference_zoom - out_zoom)
            grouped_shape = [*reference.shape[:3], -1, divisor, *reference.shape[4:]]
            return reference.reshape(grouped_shape).all(dim=4)
        if self.transform_kind == "zeros":
            fill_value = True if reference.dtype == torch.bool else 1.0
            return reference.new_full(tuple(output_shape), fill_value)
        raise ValueError(f"Unsupported zoom transform kind: {self.transform_kind}.")

    def forward(
        self,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any],
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Sequence[Optional[Dict[int, torch.Tensor]]]:
        del emb_groups

        for group_idx, x_zooms in enumerate(x_zooms_groups):
            if x_zooms is None:
                continue
            for reference_zoom, out_zoom in self.zoom_mapping.items():
                if reference_zoom not in x_zooms:
                    raise ValueError(
                        f"{type(self).__name__} requires reference zoom {reference_zoom} "
                        f"in group {group_idx}."
                    )
                if out_zoom in x_zooms:
                    raise ValueError(
                        f"{type(self).__name__} will not overwrite existing zoom {out_zoom} "
                        f"in group {group_idx}."
                    )
                x_zooms[out_zoom] = self._transform_tensor(
                    x_zooms[reference_zoom],
                    reference_zoom,
                    out_zoom,
                )

                if out_zoom not in sample_configs:
                    if reference_zoom not in sample_configs:
                        raise ValueError(
                            f"{type(self).__name__} requires sample_configs for reference zoom "
                            f"{reference_zoom}."
                        )
                    sample_configs[out_zoom] = copy.deepcopy(sample_configs[reference_zoom])

                if mask_groups is not None and group_idx < len(mask_groups):
                    mask_zooms = mask_groups[group_idx]
                    if (
                        mask_zooms is not None
                        and out_zoom not in mask_zooms
                        and reference_zoom in mask_zooms
                    ):
                        mask_zooms[out_zoom] = self._transform_mask(
                            mask_zooms[reference_zoom],
                            reference_zoom,
                            out_zoom,
                            x_zooms[out_zoom].shape,
                        )

        return x_zooms_groups
