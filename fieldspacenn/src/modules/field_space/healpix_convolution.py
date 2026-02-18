from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Literal
import copy
from collections.abc import Sequence as SequenceCollection

import torch
import torch.nn as nn

from ..grids.grid_layer import GridLayer

__all__ = ["MultiZoomHealpixConvConfig", "MultiZoomHealpixConvBase", "HealpixConvBlock"]


def _expand_to_list(
    value: Union[int, Sequence[int]],
    n_items: int,
    field_name: str
) -> List[Any]:
    if isinstance(value, SequenceCollection) and not isinstance(value, (str, bytes)):
        if len(value) != n_items:
            raise ValueError(
                f"Expected `{field_name}` to have length {n_items}, got {len(value)}."
            )
        return list(value)
    return [value for _ in range(n_items)]


def _get_mode_from_zoom_relation(in_zoom: int, target_zoom: int) -> Literal["down", "same", "up"]:
    if in_zoom > target_zoom:
        return "down"
    if in_zoom < target_zoom:
        return "up"
    return "same"


def _resolve_group_count(num_channels: int, requested_groups: int) -> int:
    groups = max(1, min(requested_groups, num_channels))
    while num_channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class MultiZoomHealpixConvConfig:
    def __init__(
        self,
        in_zooms: List[int],
        target_zooms: List[int],
        target_features: Union[int, List[int]] = 1,
        share_weights: bool = False,
        use_neighborhood: bool = True,
        norm: Literal["group", "layer"] = "group",
        act: Literal["silu", "gelu"] = "silu",
        residual: bool = True,
        num_groups: int = 8,
        eps: float = 1e-5,
        **kwargs: Any
    ) -> None:
        """
        Store configuration for multi-zoom HEALPix convolution blocks.

        A zoom level corresponds to HEALPix ``nside = 2**zoom``. Tensors use the
        standard framework layout ``(b, v, t, n, d, f)``, where ``n`` is the spatial
        HEALPix index count and ``f`` is the feature/channel axis.

        :param in_zooms: Input zoom levels for each mapping entry.
        :param target_zooms: Target zoom levels for each mapping entry.
        :param target_features: Output feature count per mapping entry.
        :param share_weights: If true, reuse compatible block weights across entries.
        :param use_neighborhood: If true, include HEALPix neighbors in each convolution.
        :param norm: Normalization type ("group" or "layer"). Default is "group".
        :param act: Activation type ("silu" or "gelu"). Default is "silu".
        :param residual: Whether to enable residual connection in each block.
        :param num_groups: Requested number of GroupNorm groups.
        :param eps: Numerical epsilon used by normalization layers.
        :param kwargs: Additional attributes forwarded to the base/block constructors.
        :return: None.
        """
        self.in_zooms: List[int]
        self.target_zooms: List[int]
        self.target_features: Union[int, List[int]]
        self.share_weights: bool
        self.use_neighborhood: bool
        self.norm: Literal["group", "layer"]
        self.act: Literal["silu", "gelu"]
        self.residual: bool
        self.num_groups: int
        self.eps: float

        inputs = copy.deepcopy(locals())
        for input_name, value in inputs.items():
            if input_name == "kwargs":
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input_name, value)


class MultiZoomHealpixConvBase(nn.Module):
    def __init__(
        self,
        in_zooms: Sequence[int],
        target_zooms: Sequence[int],
        in_features: Union[int, Sequence[int]],
        target_features: Union[int, Sequence[int]],
        share_weights: bool = False,
        **block_kwargs: Any
    ) -> None:
        """
        Multi-resolution HEALPix convolution base module.

        This module constructs one ``HealpixConvBlock`` per zoom mapping index.
        Mapping is index-based: ``in_zooms[i] -> target_zooms[i]``.

        Forward accepts:
        - ``Dict[int, Tensor]`` with tensors shaped ``(b, v, t, n, d, f)``
        - ``List[Dict[int, Tensor]]`` for variable groups used by the framework
        - ``List[Tensor]`` aligned by zoom index

        It returns the same outer structure with transformed tensors.

        :param in_zooms: Input zoom levels.
        :param target_zooms: Target zoom levels.
        :param in_features: Input feature counts (scalar or per-level list).
        :param target_features: Output feature counts (scalar or per-level list).
        :param share_weights: If true, share weights only across compatible mappings.
        :param block_kwargs: Additional kwargs forwarded to ``HealpixConvBlock``.
            For neighborhood convs, provide ``grid_layers`` (dict/module-dict keyed by zoom)
            or ``grid_layer`` (single layer).
        :return: None.
        """
        super().__init__()

        self.in_zooms: List[int] = [int(z) for z in in_zooms]
        self.out_zooms: List[int] = [int(z) for z in target_zooms]
        self.target_zooms: List[int] = self.out_zooms

        if len(self.in_zooms) != len(self.target_zooms):
            raise ValueError(
                "`in_zooms` and `target_zooms` must have the same length. "
                f"Got {len(self.in_zooms)} and {len(self.target_zooms)}."
            )
        if len(set(self.target_zooms)) != len(self.target_zooms):
            raise ValueError(
                "`target_zooms` must be unique so output mappings are unambiguous."
            )

        n_levels = len(self.in_zooms)
        self.in_features: List[int] = [int(v) for v in _expand_to_list(in_features, n_levels, "in_features")]
        self.out_features: List[int] = [int(v) for v in _expand_to_list(target_features, n_levels, "target_features")]

        self.share_weights: bool = share_weights

        block_kwargs = dict(block_kwargs)
        grid_layers = block_kwargs.pop("grid_layers", None)
        default_grid_layer = block_kwargs.pop("grid_layer", None)
        use_neighborhood = bool(block_kwargs.get("use_neighborhood", True))

        self.blocks: nn.ModuleList = nn.ModuleList()
        shared_blocks: Dict[Tuple[int, int, int, int, int], HealpixConvBlock] = {}

        for idx, (in_zoom, target_zoom, in_ch, out_ch) in enumerate(
            zip(self.in_zooms, self.target_zooms, self.in_features, self.out_features)
        ):
            grid_layer = self._resolve_grid_layer(
                target_zoom=target_zoom,
                grid_layers=grid_layers,
                default_grid_layer=default_grid_layer,
                use_neighborhood=use_neighborhood,
            )

            share_key = (in_zoom, target_zoom, in_ch, out_ch, id(grid_layer))
            if self.share_weights and share_key in shared_blocks:
                block = shared_blocks[share_key]
            else:
                block = HealpixConvBlock(
                    in_zoom=in_zoom,
                    target_zoom=target_zoom,
                    in_features=in_ch,
                    out_features=out_ch,
                    grid_layer=grid_layer,
                    **block_kwargs,
                )
                if self.share_weights:
                    shared_blocks[share_key] = block

            self.blocks.append(block)

    @staticmethod
    def _resolve_grid_layer(
        target_zoom: int,
        grid_layers: Optional[Mapping[Any, GridLayer]],
        default_grid_layer: Optional[GridLayer],
        use_neighborhood: bool,
    ) -> Optional[GridLayer]:
        if not use_neighborhood:
            return default_grid_layer

        if grid_layers is not None:
            if target_zoom in grid_layers:
                return grid_layers[target_zoom]
            if str(target_zoom) in grid_layers:
                return grid_layers[str(target_zoom)]

        if default_grid_layer is not None:
            return default_grid_layer

        raise ValueError(
            f"Missing grid layer for target_zoom={target_zoom}. "
            "Provide `grid_layers` keyed by zoom or `grid_layer`."
        )

    @staticmethod
    def _get_sample_config(
        sample_configs: Optional[Mapping[Any, Any]],
        in_zoom: int,
        target_zoom: int,
    ) -> Mapping[str, Any]:
        if sample_configs is None:
            return {}

        keys = (target_zoom, str(target_zoom), in_zoom, str(in_zoom))
        for key in keys:
            if key in sample_configs and isinstance(sample_configs[key], Mapping):
                return sample_configs[key]
        return {}

    @staticmethod
    def _get_input_tensor(inputs: Mapping[Any, torch.Tensor], zoom: int) -> torch.Tensor:
        if zoom in inputs:
            return inputs[zoom]
        if str(zoom) in inputs:
            return inputs[str(zoom)]
        raise KeyError(f"Input collection does not contain zoom {zoom}. Available keys: {list(inputs.keys())}.")

    def _forward_mapping(
        self,
        inputs: Mapping[Any, torch.Tensor],
        sample_configs: Optional[Mapping[Any, Any]],
    ) -> Dict[Any, torch.Tensor]:
        # Pass through zooms that this block does not consume.
        outputs: Dict[Any, torch.Tensor] = {}
        for zoom_key, tensor in inputs.items():
            zoom_key_int = int(zoom_key) if isinstance(zoom_key, str) and zoom_key.isdigit() else zoom_key
            if zoom_key_int not in self.in_zooms:
                outputs[zoom_key] = tensor

        # Apply configured zoom mappings.
        for idx, (in_zoom, target_zoom) in enumerate(zip(self.in_zooms, self.target_zooms)):
            x = self._get_input_tensor(inputs, in_zoom)
            sample_config = self._get_sample_config(sample_configs, in_zoom=in_zoom, target_zoom=target_zoom)
            outputs[target_zoom] = self.blocks[idx](x, sample_config=sample_config)
        return outputs

    def forward(
        self,
        inputs: List[Mapping[Any, torch.Tensor]],
        sample_configs: Optional[Mapping[Any, Any]] = None,
        **kwargs: Any
    ) -> List[Dict[Any, torch.Tensor]]:
        """
        Apply per-zoom HEALPix convolution blocks to zoom groups.

        Inputs must be a list of dictionaries mapping ``zoom -> tensor``, where
        tensors use shape ``(b, v, t, n, d, f)``.
        Zooms not listed in ``self.in_zooms`` are passed through unchanged.

        :param inputs: List of per-group zoom tensor mappings.
        :param sample_configs: Sampling config dictionary indexed by zoom.
        :param kwargs: Extra unused kwargs for compatibility with existing block calls.
        :return: List of output zoom mappings.
        """

        outputs_groups: List[Dict[Any, torch.Tensor]] = []
        for idx, group_inputs in enumerate(inputs):
            outputs_groups.append(self._forward_mapping(group_inputs, sample_configs=sample_configs))

        return outputs_groups


class HealpixConvBlock(nn.Module):
    def __init__(
        self,
        in_zoom: int,
        target_zoom: int,
        in_features: int,
        out_features: int,
        grid_layer: Optional[GridLayer] = None,
        norm: Literal["group", "layer"] = "group",
        act: Literal["silu", "gelu"] = "silu",
        residual: bool = True,
        use_neighborhood: bool = True,
        num_groups: int = 8,
        eps: float = 1e-5
    ) -> None:
        """
        HEALPix convolution block for a single zoom mapping.

        The block infers its mode from ``in_zoom`` and ``target_zoom``:
        - ``in_zoom == target_zoom``: same-resolution convolution
        - ``in_zoom > target_zoom``: downsample (mean over 4**delta children), then conv
        - ``in_zoom < target_zoom``: nearest-neighbor upsample, then conv

        The residual path uses identity when shapes match, otherwise a learnable
        projection (1x1-style channel projection) when channel count changes and/or
        when resampling is performed.

        Input/Output tensor shape: ``(b, v, t, n, d, f)``.

        Defaults follow modern ConvNet practice:
        - norm: GroupNorm (switchable to LayerNorm)
        - activation: SiLU (switchable to GELU)
        - residual: enabled

        :param in_zoom: Input zoom level.
        :param target_zoom: Output zoom level.
        :param in_features: Number of input features/channels.
        :param out_features: Number of output features/channels.
        :param grid_layer: GridLayer for neighborhood gathering at target zoom.
        :param norm: Normalization type ("group" or "layer").
        :param act: Activation type ("silu" or "gelu").
        :param residual: Whether to use residual connections.
        :param use_neighborhood: Whether to concatenate spatial neighbors in convs.
        :param num_groups: Requested GroupNorm groups (auto-clamped to valid divisor).
        :param eps: Epsilon for normalization.
        :return: None.
        """
        super().__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"`in_features` and `out_features` must be positive, got {in_features}, {out_features}."
            )
        if norm not in ("group", "layer"):
            raise ValueError(f"Unsupported norm `{norm}`. Use 'group' or 'layer'.")
        if act not in ("silu", "gelu"):
            raise ValueError(f"Unsupported activation `{act}`. Use 'silu' or 'gelu'.")

        self.in_zoom: int = int(in_zoom)
        self.target_zoom: int = int(target_zoom)
        self.in_features: int = int(in_features)
        self.out_features: int = int(out_features)
        self.mode: Literal["down", "same", "up"] = _get_mode_from_zoom_relation(self.in_zoom, self.target_zoom)
        self.norm: Literal["group", "layer"] = norm
        self.act_name: Literal["silu", "gelu"] = act
        self.residual: bool = residual
        self.use_neighborhood: bool = use_neighborhood
        self.grid_layer: Optional[GridLayer] = grid_layer

        if self.use_neighborhood:
            if self.grid_layer is None:
                raise ValueError(
                    "`grid_layer` is required when `use_neighborhood=True` in HealpixConvBlock."
                )
            if int(self.grid_layer.zoom) != self.target_zoom:
                raise ValueError(
                    f"`grid_layer.zoom` ({self.grid_layer.zoom}) must match target_zoom ({self.target_zoom})."
                )

        self.kernel_size: int = int(self.grid_layer.adjc.shape[-1]) if self.use_neighborhood else 1

        self.norm1: nn.Module = self._build_norm(self.in_features, num_groups, eps)
        self.norm2: nn.Module = self._build_norm(self.out_features, num_groups, eps)
        self.activation: nn.Module = nn.SiLU() if act == "silu" else nn.GELU()

        self.conv1: nn.Linear = nn.Linear(self.kernel_size * self.in_features, self.out_features, bias=True)
        self.conv2: nn.Linear = nn.Linear(self.kernel_size * self.out_features, self.out_features, bias=True)

        use_projection = (self.in_features != self.out_features) or (self.in_zoom != self.target_zoom)
        self.skip_projection: nn.Module = (
            nn.Linear(self.in_features, self.out_features, bias=False)
            if (self.residual and use_projection)
            else nn.Identity()
        )

    def _build_norm(self, channels: int, num_groups: int, eps: float) -> nn.Module:
        if self.norm == "layer":
            return nn.LayerNorm(channels, eps=eps)
        groups = _resolve_group_count(channels, num_groups)
        return nn.GroupNorm(groups, channels, eps=eps)

    def _apply_norm(self, x: torch.Tensor, norm_layer: nn.Module) -> torch.Tensor:
        if self.norm == "layer":
            return norm_layer(x)

        b, v, t, n, d, c = x.shape
        x_norm = x.permute(0, 1, 2, 5, 3, 4).contiguous().view(b * v * t, c, n, d)
        x_norm = norm_layer(x_norm)
        return x_norm.view(b, v, t, c, n, d).permute(0, 1, 2, 4, 5, 3).contiguous()

    def _resample(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "same":
            return x

        if self.mode == "down":
            factor = 4 ** (self.in_zoom - self.target_zoom)
            if x.shape[3] % factor != 0:
                raise ValueError(
                    f"Cannot downsample from zoom {self.in_zoom} to {self.target_zoom}: "
                    f"spatial dimension {x.shape[3]} is not divisible by {factor}."
                )
            return x.view(*x.shape[:3], -1, factor, *x.shape[-2:]).mean(dim=-3)

        factor = 4 ** (self.target_zoom - self.in_zoom)
        x_up = x.view(*x.shape[:3], -1, 1, *x.shape[-2:])
        x_up = x_up.expand(-1, -1, -1, -1, factor, -1, -1)
        return x_up.reshape(*x.shape[:3], -1, *x.shape[-2:])

    def _apply_conv(
        self,
        x: torch.Tensor,
        layer: nn.Linear,
        sample_config: Optional[Mapping[str, Any]] = None
    ) -> torch.Tensor:
        if not self.use_neighborhood:
            return layer(x)

        sample_config = {} if sample_config is None else sample_config
        x_nh, _ = self.grid_layer.get_nh(x, input_zoom=self.target_zoom, **sample_config)
        x_nh = x_nh.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
        x_nh = x_nh.view(*x_nh.shape[:5], -1)
        return layer(x_nh)

    def forward(
        self,
        x: torch.Tensor,
        sample_config: Optional[Mapping[str, Any]] = None
    ) -> torch.Tensor:
        """
        Apply pre-activation residual HEALPix convolution.

        :param x: Input tensor of shape ``(b, v, t, n, d, f_in)``.
        :param sample_config: Optional sampling config for neighborhood lookup
            (for example, patch indices and zoom sampling metadata).
        :return: Tensor of shape ``(b, v, t, n_target, d, f_out)``.
        """
        if x.ndim != 6:
            raise ValueError(
                f"Expected 6D input `(b, v, t, n, d, f)`, got shape {tuple(x.shape)}."
            )
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}."
            )

        x = self._resample(x)
        residual = self.skip_projection(x) if self.residual else None

        y = self._apply_norm(x, self.norm1)
        y = self.activation(y)
        y = self._apply_conv(y, self.conv1, sample_config=sample_config)

        y = self._apply_norm(y, self.norm2)
        y = self.activation(y)
        y = self._apply_conv(y, self.conv2, sample_config=sample_config)

        if residual is not None:
            y = y + residual

        return y
