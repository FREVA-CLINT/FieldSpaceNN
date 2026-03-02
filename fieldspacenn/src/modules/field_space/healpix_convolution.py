from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Literal
import copy
from collections.abc import Sequence as SequenceCollection

import torch
import torch.nn as nn

from ..base import LayerNorm, get_layer
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


def _expand_group_configs(
    value: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]],
    n_groups: int,
    field_name: str,
) -> List[Dict[str, Any]]:
    if isinstance(value, Mapping):
        return [copy.deepcopy(dict(value)) for _ in range(n_groups)]

    if isinstance(value, SequenceCollection) and not isinstance(value, (str, bytes)):
        if len(value) != n_groups:
            raise ValueError(
                f"Expected `{field_name}` to have length {n_groups}, got {len(value)}."
            )
        return [copy.deepcopy(dict(v)) for v in value]

    raise TypeError(
        f"`{field_name}` must be a dict or a list of dicts, got {type(value).__name__}."
    )


def _resolve_variable_index(
    emb: Optional[Mapping[str, Any]],
    batch_size: int,
    n_tensor_variables: int,
    n_variables: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if n_variables <= 1:
        return None

    var_idx_raw = None
    if emb is not None:
        var_idx_raw = emb.get("VariableEmbedder")

    if var_idx_raw is None:
        if n_tensor_variables > n_variables:
            raise ValueError(
                f"Input tensor contains {n_tensor_variables} variables but module was configured "
                f"with n_variables={n_variables}."
            )
        return torch.arange(n_tensor_variables, device=device, dtype=torch.long).view(1, -1).expand(batch_size, -1)

    if not torch.is_tensor(var_idx_raw):
        var_idx = torch.as_tensor(var_idx_raw, device=device, dtype=torch.long)
    else:
        var_idx = var_idx_raw.to(device=device, dtype=torch.long)

    if var_idx.ndim == 1:
        var_idx = var_idx.unsqueeze(0)
    if var_idx.ndim != 2:
        raise ValueError(
            f"`VariableEmbedder` must be a 1D or 2D tensor, got shape {tuple(var_idx.shape)}."
        )

    if var_idx.shape[0] == 1 and batch_size > 1:
        var_idx = var_idx.expand(batch_size, -1)

    if tuple(var_idx.shape) != (batch_size, n_tensor_variables):
        raise ValueError(
            f"`VariableEmbedder` shape must be {(batch_size, n_tensor_variables)}, "
            f"got {tuple(var_idx.shape)}."
        )

    if var_idx.numel() > 0 and (int(var_idx.min().item()) < 0 or int(var_idx.max().item()) >= n_variables):
        raise ValueError(
            f"`VariableEmbedder` indices must be in [0, {n_variables - 1}], "
            f"got min={int(var_idx.min().item())}, max={int(var_idx.max().item())}."
        )

    return var_idx


class _VariableGroupNorm(nn.Module):
    def __init__(
        self,
        channels: int,
        num_groups: int,
        eps: float,
        n_variables: int = 1,
    ) -> None:
        super().__init__()
        groups = _resolve_group_count(channels, num_groups)
        self.channels: int = int(channels)
        self.n_variables: int = int(n_variables)
        self.group_norm: nn.GroupNorm = nn.GroupNorm(groups, self.channels, eps=eps, affine=False)

        if self.n_variables <= 1:
            self.weight: nn.Parameter = nn.Parameter(torch.ones(self.channels))
            self.bias: nn.Parameter = nn.Parameter(torch.zeros(self.channels))
        else:
            self.weight = nn.Parameter(torch.ones(self.n_variables, self.channels))
            self.bias = nn.Parameter(torch.zeros(self.n_variables, self.channels))

    def forward(self, x: torch.Tensor, emb: Optional[Mapping[str, Any]] = None) -> torch.Tensor:
        b, v, t, n, d, c = x.shape
        x_norm = x.permute(0, 1, 2, 5, 3, 4).contiguous().view(b * v * t, c, n, d)
        x_norm = self.group_norm(x_norm)
        x_norm = x_norm.view(b, v, t, c, n, d).permute(0, 1, 2, 4, 5, 3).contiguous()

        if self.n_variables <= 1:
            weight = self.weight.view(1, 1, 1, 1, 1, c)
            bias = self.bias.view(1, 1, 1, 1, 1, c)
            return x_norm * weight + bias

        var_idx = _resolve_variable_index(
            emb=emb,
            batch_size=b,
            n_tensor_variables=v,
            n_variables=self.n_variables,
            device=x.device,
        )
        weight = self.weight[var_idx].unsqueeze(2).unsqueeze(3).unsqueeze(4)
        bias = self.bias[var_idx].unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return x_norm * weight + bias


class MultiZoomHealpixConvConfig:
    def __init__(
        self,
        in_zooms: List[int],
        target_zooms: List[int],
        out_zooms: Optional[List[int]] = None,
        target_features: Union[int, List[int]] = 1,
        share_weights: bool = False,
        use_neighborhood: bool = True,
        norm: Literal["group", "layer"] = "group",
        act: Literal["silu", "gelu"] = "silu",
        residual: bool = True,
        num_groups: int = 8,
        eps: float = 1e-5,
        n_groups_variables: List[int] = [1],
        rank_space: Optional[int] = None,
        rank_time: Optional[int] = None,
        rank_depth: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Store configuration for multi-zoom HEALPix convolution blocks.

        A zoom level corresponds to HEALPix ``nside = 2**zoom``. Tensors use the
        standard framework layout ``(b, v, t, n, d, f)``, where ``n`` is the spatial
        HEALPix index count and ``f`` is the feature/channel axis.

        :param in_zooms: Input zoom levels for each mapping entry.
        :param target_zooms: Target zoom levels for each mapping entry.
        :param out_zooms: Optional zoom levels to keep in block outputs.
        :param target_features: Output feature count per mapping entry.
        :param share_weights: If true, reuse compatible block weights across entries.
        :param use_neighborhood: If true, include HEALPix neighbors in each convolution.
        :param norm: Normalization type ("group" or "layer"). Default is "group".
        :param act: Activation type ("silu" or "gelu"). Default is "silu".
        :param residual: Whether to enable residual connection in each block.
        :param num_groups: Requested number of GroupNorm groups.
        :param eps: Numerical epsilon used by normalization layers.
        :param n_groups_variables: Number of variables in each variable group.
        :param rank_space: Optional rank applied on the neighborhood-space axis.
        :param rank_time: Optional rank applied on the time axis.
        :param rank_depth: Optional rank applied on the depth axis.
        :param kwargs: Additional attributes forwarded to the base/block constructors.
        :return: None.
        """
        self.in_zooms: List[int]
        self.target_zooms: List[int]
        self.out_zooms: Optional[List[int]]
        self.target_features: Union[int, List[int]]
        self.share_weights: bool
        self.use_neighborhood: bool
        self.norm: Literal["group", "layer"]
        self.act: Literal["silu", "gelu"]
        self.residual: bool
        self.num_groups: int
        self.eps: float
        self.n_groups_variables: List[int]
        self.rank_space: Optional[int]
        self.rank_time: Optional[int]
        self.rank_depth: Optional[int]

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
        x_zooms: List[int],
        in_zooms: Sequence[int],
        target_zooms: Sequence[int],
        in_features: Union[int, Sequence[int]],
        target_features: Union[int, Sequence[int]],
        out_zooms: Optional[Sequence[int]] = None,
        share_weights: bool = False,
        n_groups_variables: Sequence[int] = (1,),
        rank_space: Union[Optional[int], Sequence[Optional[int]]] = None,
        rank_time: Union[Optional[int], Sequence[Optional[int]]] = None,
        rank_depth: Union[Optional[int], Sequence[Optional[int]]] = None,
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

        :param x_zooms: Zoom levels present in inputs.
        :param in_zooms: Input zoom levels.
        :param target_zooms: Target zoom levels.
        :param out_zooms: Optional zoom levels to keep in returned outputs.
        :param in_features: Input feature counts (scalar or per-level list).
        :param target_features: Output feature counts (scalar or per-level list).
        :param share_weights: If true, share weights only across compatible mappings.
        :param n_groups_variables: Number of variables in each input variable group.
        :param rank_space: Optional per-group neighborhood-space rank(s).
        :param rank_time: Optional per-group time rank(s).
        :param rank_depth: Optional per-group depth rank(s).
        :param block_kwargs: Additional kwargs forwarded to ``HealpixConvBlock``.
            For neighborhood convs, provide ``grid_layers`` (dict/module-dict keyed by zoom)
            or ``grid_layer`` (single layer).
        :return: None.
        """
        super().__init__()

        self.x_zooms: List[int] = [int(z) for z in x_zooms]
        self.in_zooms: List[int] = [int(z) for z in in_zooms]
        self.target_zooms: List[int] = [int(z) for z in target_zooms]
        self.n_groups_variables: List[int] = [int(v) for v in n_groups_variables]
        self._output_filter_zooms: Optional[List[int]] = (
            [int(z) for z in out_zooms] if out_zooms is not None else None
        )
        # Keep constructor-chain behavior backward compatible when `out_zooms` is omitted.
        self.out_zooms: List[int] = (
            copy.deepcopy(self.target_zooms)
            if self._output_filter_zooms is None
            else copy.deepcopy(self._output_filter_zooms)
        )

        if len(self.n_groups_variables) == 0:
            raise ValueError("`n_groups_variables` must contain at least one group.")
        if any(n <= 0 for n in self.n_groups_variables):
            raise ValueError(f"`n_groups_variables` must contain positive values, got {self.n_groups_variables}.")
        rank_space_groups: List[Optional[int]] = _expand_to_list(rank_space, len(self.n_groups_variables), "rank_space")
        rank_time_groups: List[Optional[int]] = _expand_to_list(rank_time, len(self.n_groups_variables), "rank_time")
        rank_depth_groups: List[Optional[int]] = _expand_to_list(rank_depth, len(self.n_groups_variables), "rank_depth")

        if len(self.in_zooms) != len(self.target_zooms):
            raise ValueError(
                "`in_zooms` and `target_zooms` must have the same length. "
                f"Got {len(self.in_zooms)} and {len(self.target_zooms)}."
            )
        if len(set(self.target_zooms)) != len(self.target_zooms):
            raise ValueError(
                "`target_zooms` must be unique so output mappings are unambiguous."
            )

        self.in_features: List[int] = [int(v) for v in _expand_to_list(in_features, len(x_zooms), "in_features")]
        self.target_features: List[int] = [int(v) for v in _expand_to_list(target_features, len(target_zooms), "target_features")]
        self.in_features_dict: Dict[int, int] = dict(zip(self.x_zooms, self.in_features))
        self.target_features_dict: Dict[int, int] = dict(zip(self.target_zooms, self.target_features))
        self.out_features: List[int] = [
            self.target_features_dict[zoom] if zoom in self.target_features_dict.keys() else self.in_features_dict[zoom]
            for zoom in self.out_zooms
        ]

        self.share_weights: bool = share_weights

        block_kwargs = dict(block_kwargs)
        layer_confs_groups = _expand_group_configs(
            block_kwargs.pop("layer_confs", {}),
            len(self.n_groups_variables),
            "layer_confs",
        )
        grid_layers = block_kwargs.pop("grid_layers", None)
        default_grid_layer = block_kwargs.pop("grid_layer", None)
        use_neighborhood = bool(block_kwargs.get("use_neighborhood", True))

        self.blocks_groups: nn.ModuleList = nn.ModuleList()
        for group_idx, n_variables in enumerate(self.n_groups_variables):
            group_blocks: nn.ModuleList = nn.ModuleList()
            shared_blocks: Dict[Tuple[int, int, int, int, int], HealpixConvBlock] = {}
            group_layer_confs = copy.deepcopy(layer_confs_groups[group_idx])

            for idx, (in_zoom, target_zoom) in enumerate(zip(self.in_zooms, self.target_zooms)):
                in_ch = self.in_features_dict[in_zoom]
                out_ch = self.target_features_dict[target_zoom]
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
                        n_variables=n_variables,
                        rank_space=rank_space_groups[group_idx],
                        rank_time=rank_time_groups[group_idx],
                        rank_depth=rank_depth_groups[group_idx],
                        layer_confs=copy.deepcopy(group_layer_confs),
                        **block_kwargs,
                    )
                    if self.share_weights:
                        shared_blocks[share_key] = block

                group_blocks.append(block)

            self.blocks_groups.append(group_blocks)

        # Backward-compatible alias for single-group access patterns.
        self.blocks: nn.ModuleList = self.blocks_groups[0]

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
        blocks: Sequence["HealpixConvBlock"],
        sample_configs: Optional[Mapping[Any, Any]],
        emb: Optional[Dict[str, Any]] = None,
    ) -> Dict[Any, torch.Tensor]:
        # Pass through zooms that this block does not consume.
        outputs: Dict[Any, torch.Tensor] = {}
        for zoom_key, tensor in inputs.items():
            if int(zoom_key) not in self.in_zooms and int(zoom_key) in self.out_zooms:
                outputs[zoom_key] = tensor

        # Apply configured zoom mappings.
        for idx, (in_zoom, target_zoom) in enumerate(zip(self.in_zooms, self.target_zooms)):
            x = self._get_input_tensor(inputs, in_zoom)
            sample_config = self._get_sample_config(sample_configs, in_zoom=in_zoom, target_zoom=target_zoom)
            outputs[target_zoom] = blocks[idx](x, sample_config=sample_config, emb=emb)
        return outputs

    def _select_out_zooms(self, outputs: Mapping[Any, torch.Tensor]) -> Dict[int, torch.Tensor]:
        selected_outputs: Dict[int, torch.Tensor] = {}
        for zoom in self.out_zooms:
            selected_outputs[zoom] = self._get_input_tensor(outputs, zoom)
        return selected_outputs

    def forward(
        self,
        inputs: List[Mapping[Any, torch.Tensor]],
        sample_configs: Optional[Mapping[Any, Any]] = None,
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        **kwargs: Any
    ) -> List[Dict[Any, torch.Tensor]]:
        """
        Apply per-zoom HEALPix convolution blocks to zoom groups.

        Inputs must be a list of dictionaries mapping ``zoom -> tensor``, where
        tensors use shape ``(b, v, t, n, d, f)``.
        Zooms not listed in ``self.in_zooms`` are passed through unchanged.

        :param inputs: List of per-group zoom tensor mappings.
        :param sample_configs: Sampling config dictionary indexed by zoom.
        :param emb_groups: Optional list of embedding dictionaries per group.
        :param kwargs: Extra unused kwargs for compatibility with existing block calls.
        :return: List of output zoom mappings.
        """

        if emb_groups is None:
            emb_groups = [None] * len(inputs)

        if len(self.blocks_groups) not in (1, len(inputs)):
            raise ValueError(
                f"Expected number of input groups ({len(inputs)}) to match `n_groups_variables` "
                f"({len(self.blocks_groups)}), or define a single shared group."
            )

        outputs_groups: List[Dict[Any, torch.Tensor]] = []
        for idx, group_inputs in enumerate(inputs):
            group_blocks = self.blocks_groups[idx] if len(self.blocks_groups) > 1 else self.blocks_groups[0]
            emb = emb_groups[idx] if idx < len(emb_groups) else None
            outputs = self._forward_mapping(
                group_inputs,
                blocks=group_blocks,
                sample_configs=sample_configs,
                emb=emb,
            )
            if self._output_filter_zooms is None:
                outputs_groups.append(outputs)
            else:
                outputs_groups.append(self._select_out_zooms(outputs))

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
        eps: float = 1e-5,
        n_variables: int = 1,
        rank_space: Optional[int] = None,
        rank_time: Optional[int] = None,
        rank_depth: Optional[int] = None,
        layer_confs: Optional[Dict[str, Any]] = None,
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
        :param n_variables: Number of variables for variable-specific parameters.
        :param rank_space: Optional rank applied on the neighborhood-space axis.
        :param rank_time: Optional rank applied on the time axis.
        :param rank_depth: Optional rank applied on the depth axis.
        :param layer_confs: Optional `get_layer` configuration dictionary.
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
        self.n_variables: int = int(n_variables)
        self.rank_space: Optional[int] = rank_space
        self.rank_time: Optional[int] = rank_time
        self.rank_depth: Optional[int] = rank_depth
        self.layer_confs: Dict[str, Any] = copy.deepcopy(layer_confs) if layer_confs is not None else {}
        self.layer_confs["n_variables"] = self.n_variables
        self.layer_confs["ranks"] = [self.rank_time, self.rank_space, self.rank_depth, None, None]

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

        conv1_in_features = [1, self.kernel_size, 1, self.in_features]
        conv2_in_features = [1, self.kernel_size, 1, self.out_features]
        conv_out_features = [1, 1, 1, self.out_features]
        skip_in_features = [1, 1, 1, self.in_features]
        skip_out_features = [1, 1, 1, self.out_features]

        self.conv1: nn.Module = get_layer(
            conv1_in_features,
            conv_out_features,
            layer_confs=self.layer_confs,
            bias=True,
        )
        self.conv2: nn.Module = get_layer(
            conv2_in_features,
            conv_out_features,
            layer_confs=self.layer_confs,
            bias=True,
        )

        use_projection = (self.in_features != self.out_features) or (self.in_zoom != self.target_zoom)
        self.skip_projection: nn.Module = (
            get_layer(
                skip_in_features,
                skip_out_features,
                layer_confs=self.layer_confs,
                bias=False,
            )
            if (self.residual and use_projection)
            else nn.Identity()
        )

    def _build_norm(self, channels: int, num_groups: int, eps: float) -> nn.Module:
        if self.norm == "layer":
            return LayerNorm(channels, n_variables=self.n_variables, eps=eps, elementwise_affine=True)
        return _VariableGroupNorm(channels, num_groups=num_groups, eps=eps, n_variables=self.n_variables)

    @staticmethod
    def _apply_norm(
        x: torch.Tensor,
        norm_layer: nn.Module,
        emb: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        return norm_layer(x, emb=emb)

    @staticmethod
    def _apply_layer(
        layer: nn.Module,
        x: torch.Tensor,
        emb: Optional[Dict[str, Any]] = None,
        sample_config: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        if isinstance(layer, nn.Identity):
            return layer(x)
        return layer(
            x,
            emb=emb,
            sample_configs={} if sample_config is None else sample_config,
        )
        
    @staticmethod
    def _squeeze_trailing_singletons(x: torch.Tensor) -> torch.Tensor:
        while x.ndim > 6 and x.shape[-2] == 1:
            x = x.squeeze(-2)
        if x.ndim != 6:
            raise ValueError(
                f"Expected 6D tensor after layer projection, got shape {tuple(x.shape)}."
            )
        return x

    def _get_emb_for_variables(
        self,
        x: torch.Tensor,
        emb: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        var_idx = _resolve_variable_index(
            emb=emb,
            batch_size=x.shape[0],
            n_tensor_variables=x.shape[1],
            n_variables=self.n_variables,
            device=x.device,
        )
        if var_idx is None:
            return emb

        emb_out: Dict[str, Any] = {} if emb is None else dict(emb)
        emb_out["VariableEmbedder"] = var_idx
        return emb_out

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
        layer: nn.Module,
        emb: Optional[Dict[str, Any]] = None,
        sample_config: Optional[Mapping[str, Any]] = None
    ) -> torch.Tensor:
        if not self.use_neighborhood:
            y = self._apply_layer(layer, x, emb=emb, sample_config=sample_config)
            return self._squeeze_trailing_singletons(y)

        sample_config = {} if sample_config is None else sample_config
        x_nh, _ = self.grid_layer.get_nh(x, input_zoom=self.target_zoom, **sample_config)
        x_nh = x_nh.permute(0, 1, 2, 3, 5, 4, 6).contiguous()
        y = self._apply_layer(layer, x_nh, emb=emb, sample_config=sample_config)
        return self._squeeze_trailing_singletons(y)

    def forward(
        self,
        x: torch.Tensor,
        sample_config: Optional[Mapping[str, Any]] = None,
        emb: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Apply pre-activation residual HEALPix convolution.

        :param x: Input tensor of shape ``(b, v, t, n, d, f_in)``.
        :param sample_config: Optional sampling config for neighborhood lookup
            (for example, patch indices and zoom sampling metadata).
        :param emb: Optional embedding dictionary containing ``VariableEmbedder``.
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
        emb = self._get_emb_for_variables(x, emb)
        residual = (
            self._squeeze_trailing_singletons(self._apply_layer(self.skip_projection, x, emb=emb))
            if self.residual
            else None
        )

        y = self._apply_norm(x, self.norm1, emb=emb)
        y = self.activation(y)
        y = self._apply_conv(y, self.conv1, emb=emb, sample_config=sample_config)

        y = self._apply_norm(y, self.norm2, emb=emb)
        y = self.activation(y)
        y = self._apply_conv(y, self.conv2, emb=emb, sample_config=sample_config)

        if residual is not None:
            y = y + residual

        return y
