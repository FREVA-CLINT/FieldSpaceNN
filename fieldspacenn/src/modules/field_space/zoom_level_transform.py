import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from .field_space_base import coarsen_zoom, refine_zoom
from ..grids.grid_utils import decode_zooms, encode_zooms, to_zoom


def _normalize_unique_sorted_zooms(zooms: Optional[Sequence[int]]) -> List[int]:
    if zooms is None:
        return []
    return sorted(dict.fromkeys(int(zoom) for zoom in zooms))


def _extract_patch_index_zooms(sample_configs: Mapping[int, Mapping[str, Any]]) -> Dict[int, Any]:
    return {
        int(zoom): config["patch_index"]
        for zoom, config in sample_configs.items()
        if isinstance(zoom, int) and "patch_index" in config
    }


def _validate_matching_timestep_counts(
    *,
    zooms: Sequence[int],
    sample_configs: Mapping[int, Mapping[str, Any]],
) -> None:
    if len(zooms) <= 1:
        return

    timestep_counts = {
        int(zoom): int(sample_configs[int(zoom)]["n_past_ts"])
        + int(sample_configs[int(zoom)]["n_future_ts"])
        + 1
        for zoom in zooms
    }
    reference_count = next(iter(timestep_counts.values()))
    if any(count != reference_count for count in timestep_counts.values()):
        raise ValueError(
            "ReencodeZoomsLayer requires matching timestep counts across output zooms, "
            f"got {timestep_counts}."
        )


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


class ReencodeZoomsLayerConfig:
    """Configure one-way decoding and residual re-encoding of a zoom pyramid."""

    def __init__(
        self,
        decode_zoom: Optional[int] = None,
        out_zooms: Optional[Sequence[int]] = None,
    ) -> None:
        self.decode_zoom = None if decode_zoom is None else int(decode_zoom)
        self.out_zooms = None if out_zooms is None else [int(zoom) for zoom in out_zooms]

    def partition_input_zooms(self, in_zooms: Sequence[int]) -> tuple[List[int], List[int]]:
        input_zooms = sorted(dict.fromkeys(int(zoom) for zoom in in_zooms))
        if self.decode_zoom is None:
            return input_zooms, []
        return (
            [zoom for zoom in input_zooms if zoom <= self.decode_zoom],
            [zoom for zoom in input_zooms if zoom > self.decode_zoom],
        )

    def resolve_reencoded_highest_zoom(self, in_zooms: Sequence[int]) -> Optional[int]:
        reencoded_zooms, _ = self.partition_input_zooms(in_zooms)
        return max(reencoded_zooms) if reencoded_zooms else None

    def resolve_output_zooms(self, in_zooms: Sequence[int]) -> List[int]:
        _, passthrough_zooms = self.partition_input_zooms(in_zooms)
        reencoded_highest_zoom = self.resolve_reencoded_highest_zoom(in_zooms)
        requested_output_zooms = _normalize_unique_sorted_zooms(self.out_zooms)
        if not requested_output_zooms:
            default_output_zooms = []
            if reencoded_highest_zoom is not None:
                default_output_zooms.append(reencoded_highest_zoom)
            default_output_zooms.extend(passthrough_zooms)
            return sorted(dict.fromkeys(default_output_zooms))

        if reencoded_highest_zoom is None:
            invalid_output_zooms = [
                zoom for zoom in requested_output_zooms if zoom not in passthrough_zooms
            ]
        else:
            invalid_output_zooms = [
                zoom
                for zoom in requested_output_zooms
                if zoom not in passthrough_zooms and zoom > reencoded_highest_zoom
            ]
        if invalid_output_zooms:
            raise ValueError(
                "ReencodeZoomsLayerConfig requires requested output zooms above the re-encode "
                "limit to already exist as higher untouched inputs, "
                f"got decode_zoom={self.decode_zoom}, in_zooms={list(in_zooms)}, "
                f"and out_zooms={requested_output_zooms}."
            )

        return sorted(dict.fromkeys(requested_output_zooms + passthrough_zooms))

    def resolve_output_features(
        self,
        in_zooms: Sequence[int],
        in_features: Sequence[int],
    ) -> List[int]:
        output_zooms = self.resolve_output_zooms(in_zooms)
        feature_by_zoom = {
            int(zoom): int(feature)
            for zoom, feature in zip(in_zooms, in_features)
        }
        reencoded_highest_zoom = self.resolve_reencoded_highest_zoom(in_zooms)
        reencoded_feature = (
            None
            if reencoded_highest_zoom is None
            else feature_by_zoom.get(reencoded_highest_zoom)
        )
        if reencoded_highest_zoom is not None and reencoded_feature is None:
            raise ValueError(
                "ReencodeZoomsLayerConfig requires at least one re-encodable input zoom when "
                "computing output features."
            )

        output_features: List[int] = []
        for zoom in output_zooms:
            feature = feature_by_zoom.get(zoom, reencoded_feature)
            if feature is None:
                raise ValueError(
                    "ReencodeZoomsLayerConfig could not infer features for output zoom "
                    f"{zoom}."
                )
            output_features.append(feature)
        return output_features


class ReencodeZoomsLayer(nn.Module):
    """Decode a residual zoom pyramid and re-encode selected zoom levels."""

    def __init__(
        self,
        config: ReencodeZoomsLayerConfig,
        in_zooms: Sequence[int],
        in_features: Sequence[int],
    ) -> None:
        super().__init__()
        self.decode_zoom = config.decode_zoom
        self.requested_out_zooms = config.out_zooms
        self.out_zooms = config.resolve_output_zooms(in_zooms)
        self.out_features = config.resolve_output_features(in_zooms, in_features)

    def _partition_available_zooms(
        self,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
    ) -> tuple[List[int], List[int]]:
        available_zooms = sorted(
            {
                int(zoom)
                for group in x_zooms_groups
                if group
                for zoom in group
            }
        )
        if self.decode_zoom is None:
            return available_zooms, []
        return (
            [zoom for zoom in available_zooms if zoom <= self.decode_zoom],
            [zoom for zoom in available_zooms if zoom > self.decode_zoom],
        )

    def _resolve_reencoded_highest_zoom(
        self,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Mapping[str, Any]],
    ) -> Optional[int]:
        reencoded_zooms, passthrough_zooms = self._partition_available_zooms(x_zooms_groups)
        if reencoded_zooms:
            return max(reencoded_zooms)
        if passthrough_zooms:
            return None if self.decode_zoom is not None else max(passthrough_zooms)
        if sample_configs:
            sample_zooms = sorted(int(zoom) for zoom in sample_configs)
            if self.decode_zoom is None:
                return max(sample_zooms)
            eligible_zooms = [zoom for zoom in sample_zooms if zoom <= self.decode_zoom]
            return max(eligible_zooms) if eligible_zooms else None
        raise ValueError("ReencodeZoomsLayer could not infer a re-encoded zoom.")

    def _resolve_runtime_output_zooms(
        self,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        reencoded_highest_zoom: Optional[int],
    ) -> List[int]:
        _, passthrough_zooms = self._partition_available_zooms(x_zooms_groups)
        requested_output_zooms = _normalize_unique_sorted_zooms(self.requested_out_zooms)
        if not requested_output_zooms:
            output_zooms = []
            if reencoded_highest_zoom is not None:
                output_zooms.append(reencoded_highest_zoom)
            output_zooms.extend(passthrough_zooms)
            return sorted(dict.fromkeys(output_zooms))

        if reencoded_highest_zoom is None:
            invalid_output_zooms = [
                zoom for zoom in requested_output_zooms if zoom not in passthrough_zooms
            ]
        else:
            invalid_output_zooms = [
                zoom
                for zoom in requested_output_zooms
                if zoom not in passthrough_zooms and zoom > reencoded_highest_zoom
            ]
        if invalid_output_zooms:
            input_zooms = sorted(
                {
                    int(zoom)
                    for group in x_zooms_groups
                    if group
                    for zoom in group
                }
            )
            raise ValueError(
                "ReencodeZoomsLayer requires requested output zooms above the re-encode limit "
                "to already exist as higher untouched inputs, "
                f"got decode_zoom={self.decode_zoom}, input_zooms={input_zooms}, "
                f"and out_zooms={requested_output_zooms}."
            )
        return sorted(dict.fromkeys(requested_output_zooms + passthrough_zooms))

    def _partition_group_zooms(
        self,
        x_zooms: Dict[int, torch.Tensor],
    ) -> tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        if self.decode_zoom is None:
            return dict(x_zooms), {}
        return (
            {zoom: tensor for zoom, tensor in x_zooms.items() if zoom <= self.decode_zoom},
            {zoom: tensor for zoom, tensor in x_zooms.items() if zoom > self.decode_zoom},
        )

    def forward(
        self,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any],
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Sequence[Optional[Dict[int, torch.Tensor]]]:
        del mask_groups
        del emb_groups
        if not x_zooms_groups:
            return list(x_zooms_groups)

        int_sample_configs = {
            int(zoom): config
            for zoom, config in sample_configs.items()
            if isinstance(zoom, int)
        }
        reencoded_highest_zoom = self._resolve_reencoded_highest_zoom(
            x_zooms_groups,
            int_sample_configs,
        )
        output_zooms = self._resolve_runtime_output_zooms(
            x_zooms_groups,
            reencoded_highest_zoom,
        )
        required_sample_zooms = (
            ([] if reencoded_highest_zoom is None else [reencoded_highest_zoom])
            + output_zooms
        )
        missing_sample_configs = [
            zoom for zoom in required_sample_zooms if zoom not in int_sample_configs
        ]
        if missing_sample_configs:
            raise ValueError(
                "ReencodeZoomsLayer requires sample_configs for all decoded/output zooms, "
                f"missing {sorted(dict.fromkeys(missing_sample_configs))}."
            )

        _validate_matching_timestep_counts(
            zooms=output_zooms,
            sample_configs=int_sample_configs,
        )
        patch_index_zooms = _extract_patch_index_zooms(int_sample_configs)
        reencoded_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        for group_idx, x_zooms in enumerate(x_zooms_groups):
            if x_zooms is None:
                reencoded_groups.append(None)
                continue
            if not x_zooms:
                reencoded_groups.append({})
                continue

            reencoded_group, passthrough_group = self._partition_group_zooms(x_zooms)
            group_outputs = {
                zoom: tensor
                for zoom, tensor in passthrough_group.items()
                if zoom in output_zooms
            }
            reencoded_output_zooms = (
                []
                if reencoded_highest_zoom is None
                else [zoom for zoom in output_zooms if zoom <= reencoded_highest_zoom]
            )
            if reencoded_highest_zoom is not None and reencoded_output_zooms:
                decoded_group = decode_zooms(
                    reencoded_group,
                    sample_configs=int_sample_configs,
                    out_zoom=reencoded_highest_zoom,
                )
                if reencoded_highest_zoom not in decoded_group:
                    raise ValueError(
                        f"ReencodeZoomsLayer failed to decode group {group_idx} to zoom "
                        f"{reencoded_highest_zoom}."
                    )
                decoded_highest = decoded_group[reencoded_highest_zoom]
                encoded_inputs = {
                    zoom: (
                        decoded_highest
                        if zoom == reencoded_highest_zoom
                        else to_zoom(
                            decoded_highest,
                            in_zoom=reencoded_highest_zoom,
                            out_zoom=zoom,
                        )[0]
                    )
                    for zoom in reencoded_output_zooms
                }
                group_outputs.update(
                    encode_zooms(
                        encoded_inputs,
                        sample_configs=int_sample_configs,
                        patch_index_zooms=patch_index_zooms,
                    )
                )

            reencoded_groups.append(
                {zoom: group_outputs[zoom] for zoom in output_zooms if zoom in group_outputs}
            )
        return reencoded_groups


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
