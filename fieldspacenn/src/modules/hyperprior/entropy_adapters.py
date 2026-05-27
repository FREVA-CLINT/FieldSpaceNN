"""Adapters between FieldSpace tensors and CompressAI entropy models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from compressai.entropy_models import EntropyBottleneck, GaussianConditional
except Exception as exc:  # pragma: no cover - exercised only without compressai
    EntropyBottleneck = None
    GaussianConditional = None
    _COMPRESSAI_IMPORT_ERROR = exc
else:
    _COMPRESSAI_IMPORT_ERROR = None


@dataclass
class FieldTensorSpec:
    """Shape metadata needed to restore a packed FieldSpace tensor."""

    shape: Tuple[int, int, int, int, int, int]
    group_index: Optional[int] = None
    zoom: Optional[int] = None


def _require_compressai() -> None:
    if EntropyBottleneck is None or GaussianConditional is None:
        raise ImportError(
            "mg_hyperprior_autoencoder requires compressai. Install the official "
            "compressai package in the active environment."
        ) from _COMPRESSAI_IMPORT_ERROR


def _key(group_index: int, zoom: int) -> str:
    return f"group{int(group_index)}_zoom{int(zoom)}"


def field_to_entropy_tensor(x: torch.Tensor) -> Tuple[torch.Tensor, FieldTensorSpec]:
    """Pack ``(b, v, t, n, d, f)`` into CompressAI ``(b, f, 1, v*t*n*d)``."""

    if x.dim() != 6:
        raise ValueError(f"Expected FieldSpace tensor with 6 dimensions, got shape {tuple(x.shape)}.")
    b, v, t, n, d, f = x.shape
    packed = x.permute(0, 5, 1, 2, 3, 4).contiguous().view(b, f, 1, v * t * n * d)
    return packed, FieldTensorSpec(shape=(b, v, t, n, d, f))


def entropy_to_field_tensor(y: torch.Tensor, spec: FieldTensorSpec) -> torch.Tensor:
    """Unpack CompressAI ``(b, f, 1, v*t*n*d)`` back to FieldSpace layout."""

    b, v, t, n, d, f = spec.shape
    if y.shape[0] != b or y.shape[1] != f or y.shape[-1] != v * t * n * d:
        raise ValueError(
            "Entropy tensor shape does not match FieldSpace spec: "
            f"tensor={tuple(y.shape)}, spec={spec.shape}."
        )
    return y.contiguous().view(b, f, v, t, n, d).permute(0, 2, 3, 4, 5, 1).contiguous()


def pack_nested_field_tensors(
    groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
) -> Tuple[List[Optional[Dict[int, torch.Tensor]]], List[Optional[Dict[int, FieldTensorSpec]]]]:
    """Pack every FieldSpace tensor in a nested structure."""

    packed_groups: List[Optional[Dict[int, torch.Tensor]]] = []
    spec_groups: List[Optional[Dict[int, FieldTensorSpec]]] = []
    for group_idx, group in enumerate(groups):
        if group is None:
            packed_groups.append(None)
            spec_groups.append(None)
            continue
        packed_group: Dict[int, torch.Tensor] = {}
        spec_group: Dict[int, FieldTensorSpec] = {}
        for zoom, tensor in group.items():
            packed, spec = field_to_entropy_tensor(tensor)
            spec.group_index = group_idx
            spec.zoom = int(zoom)
            packed_group[int(zoom)] = packed
            spec_group[int(zoom)] = spec
        packed_groups.append(packed_group)
        spec_groups.append(spec_group)
    return packed_groups, spec_groups


def unpack_nested_entropy_tensors(
    groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
    specs: Sequence[Optional[Mapping[int, FieldTensorSpec]]],
) -> List[Optional[Dict[int, torch.Tensor]]]:
    """Unpack a nested structure of entropy tensors."""

    output: List[Optional[Dict[int, torch.Tensor]]] = []
    for group, spec_group in zip(groups, specs):
        if group is None:
            output.append(None)
            continue
        if spec_group is None:
            raise ValueError("Missing FieldTensorSpec group for entropy tensor group.")
        output.append({int(zoom): entropy_to_field_tensor(tensor, spec_group[int(zoom)]) for zoom, tensor in group.items()})
    return output


def flatten_likelihoods_for_bpp(
    likelihoods: Optional[Sequence[Optional[Mapping[int, torch.Tensor]]]],
) -> List[torch.Tensor]:
    """Collect likelihood tensors from a nested structure."""

    tensors: List[torch.Tensor] = []
    if likelihoods is None:
        return tensors
    for group in likelihoods:
        if not group:
            continue
        tensors.extend(group.values())
    return tensors


class FieldSpaceEntropyBottleneckAdapter(nn.Module):
    """Per-group/zoom ``EntropyBottleneck`` wrapper for FieldSpace hyperlatents."""

    def __init__(self, channels_by_key: Optional[Mapping[str, int]] = None) -> None:
        super().__init__()
        _require_compressai()
        self.entropy_bottlenecks = nn.ModuleDict()
        for key, channels in (channels_by_key or {}).items():
            self.entropy_bottlenecks[str(key)] = EntropyBottleneck(int(channels))

    def forward(
        self,
        z_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
    ) -> Tuple[List[Optional[Dict[int, torch.Tensor]]], List[Optional[Dict[int, torch.Tensor]]]]:
        """Quantize hyperlatents and return likelihoods in FieldSpace layout."""

        z_hat_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        likelihood_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        for group_idx, group in enumerate(z_groups):
            if group is None:
                z_hat_groups.append(None)
                likelihood_groups.append(None)
                continue
            z_hat_group: Dict[int, torch.Tensor] = {}
            likelihood_group: Dict[int, torch.Tensor] = {}
            for zoom, tensor in group.items():
                packed, spec = field_to_entropy_tensor(tensor)
                bottleneck = self._get_bottleneck(group_idx, int(zoom), packed.shape[1])
                z_hat, likelihoods = bottleneck(packed)
                z_hat_group[int(zoom)] = entropy_to_field_tensor(z_hat, spec)
                likelihood_group[int(zoom)] = entropy_to_field_tensor(likelihoods, spec)
            z_hat_groups.append(z_hat_group)
            likelihood_groups.append(likelihood_group)
        return z_hat_groups, likelihood_groups

    def compress(
        self,
        z_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
    ) -> Tuple[List[Optional[Dict[int, List[bytes]]]], List[Optional[Dict[int, Dict[str, Any]]]], List[Optional[Dict[int, FieldTensorSpec]]]]:
        """Entropy-code hyperlatents and return strings plus shape metadata."""

        strings_groups: List[Optional[Dict[int, List[bytes]]]] = []
        metadata_groups: List[Optional[Dict[int, Dict[str, Any]]]] = []
        spec_groups: List[Optional[Dict[int, FieldTensorSpec]]] = []
        for group_idx, group in enumerate(z_groups):
            if group is None:
                strings_groups.append(None)
                metadata_groups.append(None)
                spec_groups.append(None)
                continue
            strings_group: Dict[int, List[bytes]] = {}
            metadata_group: Dict[int, Dict[str, Any]] = {}
            spec_group: Dict[int, FieldTensorSpec] = {}
            for zoom, tensor in group.items():
                packed, spec = field_to_entropy_tensor(tensor)
                spec.group_index = group_idx
                spec.zoom = int(zoom)
                bottleneck = self._get_bottleneck(group_idx, int(zoom), packed.shape[1])
                strings_group[int(zoom)] = bottleneck.compress(packed)
                metadata_group[int(zoom)] = {
                    "key": _key(group_idx, int(zoom)),
                    "channels": int(packed.shape[1]),
                    "size": tuple(int(v) for v in packed.shape[-2:]),
                }
                spec_group[int(zoom)] = spec
            strings_groups.append(strings_group)
            metadata_groups.append(metadata_group)
            spec_groups.append(spec_group)
        return strings_groups, metadata_groups, spec_groups

    def decompress(
        self,
        strings_groups: Sequence[Optional[Mapping[int, List[bytes]]]],
        metadata_groups: Sequence[Optional[Mapping[int, Mapping[str, Any]]]],
        spec_groups: Sequence[Optional[Mapping[int, FieldTensorSpec]]],
        device: Optional[torch.device | str] = None,
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """Decode hyperlatent strings into FieldSpace tensors."""

        output: List[Optional[Dict[int, torch.Tensor]]] = []
        for group_idx, (strings_group, metadata_group, spec_group) in enumerate(zip(strings_groups, metadata_groups, spec_groups)):
            if strings_group is None:
                output.append(None)
                continue
            if metadata_group is None or spec_group is None:
                raise ValueError("Missing entropy bottleneck metadata for decompression.")
            output_group: Dict[int, torch.Tensor] = {}
            for zoom, strings in strings_group.items():
                metadata = metadata_group[int(zoom)]
                bottleneck = self._get_bottleneck(group_idx, int(zoom), int(metadata["channels"]))
                packed = bottleneck.decompress(strings, tuple(metadata["size"]))
                if device is not None:
                    packed = packed.to(device=device)
                output_group[int(zoom)] = entropy_to_field_tensor(packed, spec_group[int(zoom)])
            output.append(output_group)
        return output

    def aux_loss(self) -> torch.Tensor:
        """Sum auxiliary entropy bottleneck losses."""

        losses = [
            module.aux_loss() if hasattr(module, "aux_loss") else module.loss()
            for module in self.entropy_bottlenecks.values()
        ]
        if not losses:
            return torch.tensor(0.0)
        return torch.stack(losses).sum()

    def update(self, force: bool = False) -> bool:
        """Update entropy bottleneck CDF tables."""

        return all(module.update(force=force) for module in self.entropy_bottlenecks.values())

    def _get_bottleneck(self, group_index: int, zoom: int, channels: int) -> nn.Module:
        key = _key(group_index, zoom)
        if key not in self.entropy_bottlenecks:
            self.entropy_bottlenecks[key] = EntropyBottleneck(int(channels))
        return self.entropy_bottlenecks[key]


class FieldSpaceGaussianConditionalAdapter(nn.Module):
    """``GaussianConditional`` wrapper for primary FieldSpace latents."""

    def __init__(self, scale_min: float = 1e-9, scale_table: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        _require_compressai()
        self.scale_min = float(scale_min)
        if scale_table is None:
            scale_table = torch.exp(torch.linspace(torch.log(torch.tensor(0.11)), torch.log(torch.tensor(256.0)), 64))
        self.register_buffer("scale_table", scale_table.float(), persistent=False)
        self.gaussian_conditional = GaussianConditional(self.scale_table.tolist())

    def forward(
        self,
        y_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
        scales_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
        means_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
    ) -> Tuple[List[Optional[Dict[int, torch.Tensor]]], List[Optional[Dict[int, torch.Tensor]]]]:
        """Quantize primary latents and compute Gaussian likelihoods."""

        y_hat_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        likelihood_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        for y_group, scales_group, means_group in zip(y_groups, scales_groups, means_groups):
            if y_group is None:
                y_hat_groups.append(None)
                likelihood_groups.append(None)
                continue
            if scales_group is None or means_group is None:
                raise ValueError("Gaussian conditional requires scales and means for every latent group.")
            y_hat_group: Dict[int, torch.Tensor] = {}
            likelihood_group: Dict[int, torch.Tensor] = {}
            for zoom, y in y_group.items():
                scales = self.positive_scales(scales_group[int(zoom)])
                means = means_group[int(zoom)]
                packed_y, spec = field_to_entropy_tensor(y)
                packed_scales, _ = field_to_entropy_tensor(scales)
                packed_means, _ = field_to_entropy_tensor(means)
                y_hat, likelihoods = self.gaussian_conditional(packed_y, packed_scales, means=packed_means)
                y_hat_group[int(zoom)] = entropy_to_field_tensor(y_hat, spec)
                likelihood_group[int(zoom)] = entropy_to_field_tensor(likelihoods, spec)
            y_hat_groups.append(y_hat_group)
            likelihood_groups.append(likelihood_group)
        return y_hat_groups, likelihood_groups

    def positive_scales(
        self,
        scales_groups: torch.Tensor | Sequence[Optional[Mapping[int, torch.Tensor]]],
    ) -> torch.Tensor | List[Optional[Dict[int, torch.Tensor]]]:
        """Convert raw scale logits to strictly positive scales."""

        if torch.is_tensor(scales_groups):
            return F.softplus(scales_groups) + self.scale_min
        output: List[Optional[Dict[int, torch.Tensor]]] = []
        for group in scales_groups:
            if group is None:
                output.append(None)
                continue
            output.append({int(zoom): F.softplus(tensor) + self.scale_min for zoom, tensor in group.items()})
        return output

    def build_indexes(
        self,
        scales_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """Build CompressAI scale indexes for nested FieldSpace scale tensors."""

        output: List[Optional[Dict[int, torch.Tensor]]] = []
        for group in scales_groups:
            if group is None:
                output.append(None)
                continue
            index_group = {}
            for zoom, scales in group.items():
                packed_scales, spec = field_to_entropy_tensor(self.positive_scales(scales))
                index_group[int(zoom)] = entropy_to_field_tensor(self.gaussian_conditional.build_indexes(packed_scales), spec)
            output.append(index_group)
        return output

    def compress(
        self,
        y_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
        scales_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
        means_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
    ) -> Tuple[List[Optional[Dict[int, List[bytes]]]], List[Optional[Dict[int, Dict[str, Any]]]], List[Optional[Dict[int, FieldTensorSpec]]]]:
        """Entropy-code primary latents conditioned on Gaussian parameters."""

        self.update(force=False)
        strings_groups: List[Optional[Dict[int, List[bytes]]]] = []
        metadata_groups: List[Optional[Dict[int, Dict[str, Any]]]] = []
        spec_groups: List[Optional[Dict[int, FieldTensorSpec]]] = []
        for y_group, scales_group, means_group in zip(y_groups, scales_groups, means_groups):
            if y_group is None:
                strings_groups.append(None)
                metadata_groups.append(None)
                spec_groups.append(None)
                continue
            if scales_group is None or means_group is None:
                raise ValueError("Gaussian compression requires scales and means for every latent group.")
            strings_group: Dict[int, List[bytes]] = {}
            metadata_group: Dict[int, Dict[str, Any]] = {}
            spec_group: Dict[int, FieldTensorSpec] = {}
            for zoom, y in y_group.items():
                packed_y, spec = field_to_entropy_tensor(y)
                packed_scales, _ = field_to_entropy_tensor(self.positive_scales(scales_group[int(zoom)]))
                packed_means, _ = field_to_entropy_tensor(means_group[int(zoom)])
                indexes = self.gaussian_conditional.build_indexes(packed_scales)
                strings_group[int(zoom)] = self.gaussian_conditional.compress(packed_y, indexes, means=packed_means)
                metadata_group[int(zoom)] = {"size": tuple(int(v) for v in packed_y.shape[-2:])}
                spec_group[int(zoom)] = spec
            strings_groups.append(strings_group)
            metadata_groups.append(metadata_group)
            spec_groups.append(spec_group)
        return strings_groups, metadata_groups, spec_groups

    def decompress(
        self,
        strings_groups: Sequence[Optional[Mapping[int, List[bytes]]]],
        scales_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
        means_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
        spec_groups: Sequence[Optional[Mapping[int, FieldTensorSpec]]],
        device: Optional[torch.device | str] = None,
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """Decode primary latent strings conditioned on Gaussian parameters."""

        self.update(force=False)
        output: List[Optional[Dict[int, torch.Tensor]]] = []
        for strings_group, scales_group, means_group, spec_group in zip(strings_groups, scales_groups, means_groups, spec_groups):
            if strings_group is None:
                output.append(None)
                continue
            if scales_group is None or means_group is None or spec_group is None:
                raise ValueError("Missing Gaussian metadata for decompression.")
            output_group: Dict[int, torch.Tensor] = {}
            for zoom, strings in strings_group.items():
                packed_scales, _ = field_to_entropy_tensor(self.positive_scales(scales_group[int(zoom)]))
                packed_means, _ = field_to_entropy_tensor(means_group[int(zoom)])
                indexes = self.gaussian_conditional.build_indexes(packed_scales)
                packed = self.gaussian_conditional.decompress(strings, indexes, means=packed_means)
                if device is not None:
                    packed = packed.to(device=device)
                output_group[int(zoom)] = entropy_to_field_tensor(packed, spec_group[int(zoom)])
            output.append(output_group)
        return output

    def update(self, force: bool = False) -> bool:
        """Update Gaussian conditional scale tables."""

        return self.gaussian_conditional.update_scale_table(self.scale_table, force=force)
