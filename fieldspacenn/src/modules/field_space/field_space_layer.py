from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

from einops import rearrange
from omegaconf import ListConfig
import torch
import torch.nn as nn

import copy
from ..base import get_layer, MLP_fac
from ..factorization import build_indexed_dims
from .field_space_base import (
    Tokenizer,
    add_time_overlap_from_neighbor_patches,
    add_depth_overlap_from_neighbor_patches,
)


def _is_sequence_value(value: Any) -> bool:
    return isinstance(value, (list, tuple, ListConfig))


def _normalize_axis_values(
    value: Any,
    keys: Sequence[int],
    name: str,
) -> Dict[int, Any]:
    keys = [int(key) for key in keys]
    if isinstance(value, Mapping):
        normalized = {int(key): item for key, item in value.items()}
        missing = [key for key in keys if key not in normalized]
        extra = [key for key in normalized if key not in keys]
        if missing or extra:
            raise ValueError(
                f"{name} keys must match {keys}; missing={missing}, extra={extra}"
            )
        return {key: normalized[key] for key in keys}
    if _is_sequence_value(value):
        values = list(value)
        if len(values) != len(keys):
            raise ValueError(f"{name} must have length {len(keys)}, got {len(values)}")
        return dict(zip(keys, values))
    return {key: value for key in keys}


def _normalize_group_values(value: Any, n_groups: int, name: str) -> List[Any]:
    if isinstance(value, Mapping):
        normalized = {int(key): item for key, item in value.items()}
        expected = list(range(n_groups))
        if sorted(normalized) != expected:
            raise ValueError(
                f"{name} group keys must be {expected}, got {sorted(normalized)}"
            )
        return [normalized[index] for index in expected]
    if _is_sequence_value(value):
        values = list(value)
        if len(values) != n_groups:
            raise ValueError(f"{name} must have length {n_groups}, got {len(values)}")
        return values
    return [value] * n_groups


def _collapse_shared_value(value: Any, n_groups: int, name: str) -> Any:
    if not _is_sequence_value(value) and not isinstance(value, Mapping):
        return value
    values = _normalize_group_values(value, n_groups, name)
    if not values:
        raise ValueError(f"{name} cannot be empty")
    first = values[0]
    if any(item != first for item in values[1:]):
        raise ValueError(
            f"{name} must be constant across groups; got {values}"
        )
    return first


def _normalize_ext_rank_depth(
    value: Any,
    n_groups: int,
    zooms: Sequence[int],
    name: str = "rank_depth",
) -> List[Dict[int, Any]]:
    if not _is_sequence_value(value) and not isinstance(value, Mapping):
        per_zoom = _normalize_axis_values(value, zooms, name)
        return [dict(per_zoom) for _ in range(n_groups)]
    if isinstance(value, Mapping):
        group_values = _normalize_group_values(value, n_groups, name)
    else:
        values = list(value)
        if not any(
            _is_sequence_value(item) or isinstance(item, Mapping)
            for item in values
        ):
            raise ValueError(
                f"Ext {name} must be a scalar or nested group-by-zoom values"
            )
        group_values = _normalize_group_values(values, n_groups, name)
    return [
        _normalize_axis_values(group_value, zooms, f"{name}[{group_index}]")
        for group_index, group_value in enumerate(group_values)
    ]


def _validate_numeric_leaves(
    value: Any,
    name: str,
    *,
    allow_none: bool,
    minimum: int,
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_numeric_leaves(
                item,
                f"{name}[{key}]",
                allow_none=allow_none,
                minimum=minimum,
            )
        return
    if _is_sequence_value(value):
        for index, item in enumerate(value):
            _validate_numeric_leaves(
                item,
                f"{name}[{index}]",
                allow_none=allow_none,
                minimum=minimum,
            )
        return
    if value is None and allow_none:
        return
    if value is None or int(value) < minimum:
        comparison = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"{name} must be {comparison}")


class FieldSpaceLayerConfig:
    def __init__(
        self,
        in_zooms: List[int],
        target_zooms: List[int],
        field_zoom: int,
        out_zooms: Optional[List[int]] = None,
        token_overlap_space: bool = False,
        token_overlap_time: bool = False,
        token_overlap_depth: bool = False,
        rank_space: Optional[int] = None,
        rank_time: Optional[int] = None,
        n_rank_time: Optional[int] = None,
        rank_depth: Optional[int] = None,
        n_rank_depth: Optional[int] = None,
        rank_variables: Optional[int] = None,
        n_times: int = 1,
        n_rank_space: Optional[int] = None,
        n_depths: Optional[int] = None,
        in_token_len_time: int = 1,
        in_token_len_depth: int = 1,
        out_token_len_time: int = 1,
        out_token_len_depth: int = 1,
        n_groups_variables: List[int] = [1],
        residual: bool = False,
        residual_gamma: bool = False,
        mult: int = 2,
        hidden_dim: int = None,
        hidden_dim_mixed: int = 0,
        use_indexed_input: bool = False,
        use_indexed_output: bool = False,
        use_indexed_mlp: bool = False,
        type: str = 'linear',
        block_type: Literal["legacy", "ext"] = "legacy",
        **kwargs: Any
    ) -> None:
        """
        Store configuration for field-space layers.

        :param in_zooms: Input zoom levels.
        :param target_zooms: Target zoom levels.
        :param field_zoom: Zoom level used for tokenization.
        :param out_zooms: Optional output zoom levels.
        :param token_overlap_space: Whether to overlap space tokens.
        :param token_overlap_time: Whether to overlap time tokens.
        :param token_overlap_depth: Whether to overlap depth tokens.
        :param rank_space: Optional rank for space.
        :param rank_time: Optional rank for time.
        :param n_rank_time: Optional indexed-tensor rank for time.
        :param rank_depth: Optional rank for depth.
        :param n_rank_depth: Optional indexed-tensor rank for depth.
        :param n_rank_space: Optional indexed-tensor rank for space.
        :param in_token_len_time: Input token length along time.
        :param in_token_len_depth: Input token length along depth.
        :param out_token_len_time: Output token length along time.
        :param out_token_len_depth: Output token length along depth.
        :param n_groups_variables: Number of variable groups.
        :param residual: Whether to add a residual connection around the layer.
        :param residual_gamma: Whether to scale the learned layer-output branch with
            a gamma initialized near zero before adding the residual skip.
        :param mult: MLP multiplier when using non-linear type.
        :param hidden_dim: Optional explicit hidden dimension for MLP.
        :param hidden_dim_mixed: Width of the shared cross-variable latent branch.
        :param type: Layer type ("linear" or "mlp").
        :param kwargs: Additional keyword arguments assigned as attributes.
        :return: None.
        """
        self.in_zooms: List[int]
        self.target_zooms: List[int]
        self.field_zoom: int
        self.out_zooms: Optional[List[int]]
        self.token_overlap_space: bool
        self.token_overlap_time: bool
        self.token_overlap_depth: bool
        self.rank_space: Optional[int]
        self.rank_time: Optional[int]
        self.n_rank_time: Optional[int]
        self.rank_depth: Optional[int]
        self.n_rank_depth: Optional[int]
        self.rank_variables: Optional[int]
        self.n_times: int
        self.n_rank_space: Optional[int]
        self.n_depths: Optional[int]
        self.in_token_len_time: int
        self.in_token_len_depth: int
        self.out_token_len_time: int
        self.out_token_len_depth: int
        self.n_groups_variables: List[int]
        self.residual: bool
        self.residual_gamma: bool
        self.mult: int
        self.hidden_dim: int
        self.hidden_dim_mixed: int
        self.use_indexed_input: bool
        self.use_indexed_output: bool
        self.use_indexed_mlp: bool
        self.type: str
        self.block_type: Literal["legacy", "ext"]

        hidden_dim_mixed = int(hidden_dim_mixed)
        if "att_dim_mixed" in kwargs:
            raise TypeError(
                "Field-space layers use hidden_dim_mixed; "
                "att_dim_mixed is reserved for field-space attention"
            )
        if block_type not in {"legacy", "ext"}:
            raise ValueError("block_type must be either 'legacy' or 'ext'")
        if hidden_dim_mixed < 0:
            raise ValueError("hidden_dim_mixed must be non-negative")
        _validate_numeric_leaves(
            rank_variables,
            "rank_variables",
            allow_none=True,
            minimum=0,
        )
        _validate_numeric_leaves(
            n_times,
            "n_times",
            allow_none=False,
            minimum=1,
        )
        _validate_numeric_leaves(
            n_rank_space,
            "n_rank_space",
            allow_none=True,
            minimum=0,
        )
        _validate_numeric_leaves(
            n_rank_time,
            "n_rank_time",
            allow_none=True,
            minimum=0,
        )
        _validate_numeric_leaves(
            n_rank_depth,
            "n_rank_depth",
            allow_none=True,
            minimum=0,
        )
        _validate_numeric_leaves(
            n_depths,
            "n_depths",
            allow_none=True,
            minimum=1,
        )
        if block_type == "ext" and (
            hidden_dim is None or int(hidden_dim) <= 0
        ):
            raise ValueError(
                "ExtFieldSpaceLayerBlock requires a positive hidden_dim"
            )
        if hidden_dim_mixed > 0 and (
            hidden_dim is None or int(hidden_dim) <= 0
        ):
            raise ValueError(
                "Field-space layers require a positive hidden_dim when "
                "hidden_dim_mixed > 0"
            )

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)

class FieldSpaceLayerModule(nn.Module):
    def __init__(self,
                 grid_layers: Dict[str, Any],
                 x_zooms: List[int],
                 in_zooms: List[int],
                 target_zooms: List[int],
                 field_zoom: int,
                 n_groups_variables: List[int] = [1],
                 n_groups_depths: Optional[List[int]] = None,
                 **kwargs: Any):
        """
        Initialize a field-space layer module with per-group blocks.

        :param grid_layers: Mapping from zoom string to grid layer.
        :param x_zooms: Zoom levels present in inputs.
        :param in_zooms: Input zoom levels.
        :param target_zooms: Target zoom levels.
        :param field_zoom: Zoom level used for tokenization.
        :param n_groups_variables: Number of variable groups.
        :param kwargs: Additional keyword arguments for block construction.
        :return: None.
        """
        super().__init__()
        self.blocks: nn.ModuleList = nn.ModuleList()
        n_groups = len(n_groups_variables)
        block_type = kwargs.get("block_type", "legacy")
        if block_type not in {"legacy", "ext"}:
            raise ValueError("block_type must be either 'legacy' or 'ext'")

        in_features_by_zoom = _normalize_axis_values(
            kwargs.get("in_features", 1),
            x_zooms,
            "in_features",
        )
        target_features_by_zoom = _normalize_axis_values(
            kwargs.get("target_features", 1),
            target_zooms,
            "target_features",
        )
        layer_zooms = list(dict.fromkeys([*in_zooms, *target_zooms]))

        in_token_len_depth = _normalize_group_values(
            kwargs.get("in_token_len_depth", 1),
            n_groups,
            "in_token_len_depth",
        )
        out_token_len_depth = _normalize_group_values(
            kwargs.get("out_token_len_depth", 1),
            n_groups,
            "out_token_len_depth",
        )
        token_overlap_depth = _normalize_group_values(
            kwargs.get("token_overlap_depth", False),
            n_groups,
            "token_overlap_depth",
        )
        n_groups_depths = _normalize_group_values(
            1 if n_groups_depths is None else n_groups_depths,
            n_groups,
            "n_groups_depths",
        )
        n_depths = _normalize_group_values(
            n_groups_depths if kwargs.get("n_depths") is None
            else kwargs["n_depths"],
            n_groups,
            "n_depths",
        )
        hidden_dim_shared = _collapse_shared_value(
            kwargs.get("hidden_dim"),
            n_groups,
            "hidden_dim",
        )
        hidden_dim_mixed_shared = int(
            _collapse_shared_value(
                kwargs.get("hidden_dim_mixed", 0),
                n_groups,
                "hidden_dim_mixed",
            )
        )

        if block_type == "ext":
            if hidden_dim_shared is None or int(hidden_dim_shared) <= 0:
                raise ValueError(
                    "ExtFieldSpaceLayerBlock requires a positive hidden_dim"
                )
            in_token_len_time = _normalize_axis_values(
                kwargs.get("in_token_len_time", 1),
                in_zooms,
                "in_token_len_time",
            )
            out_token_len_time = _normalize_axis_values(
                kwargs.get("out_token_len_time", 1),
                target_zooms,
                "out_token_len_time",
            )
            rank_space = _normalize_axis_values(
                kwargs.get("rank_space"),
                layer_zooms,
                "rank_space",
            )
            rank_time = _normalize_axis_values(
                kwargs.get("rank_time"),
                layer_zooms,
                "rank_time",
            )
            rank_variables = _normalize_axis_values(
                kwargs.get("rank_variables"),
                layer_zooms,
                "rank_variables",
            )
            n_times = _normalize_axis_values(
                kwargs.get("n_times", 1),
                layer_zooms,
                "n_times",
            )
            n_rank_space = _normalize_axis_values(
                kwargs.get("n_rank_space"),
                layer_zooms,
                "n_rank_space",
            )
            n_rank_time = _normalize_axis_values(
                kwargs.get("n_rank_time"),
                layer_zooms,
                "n_rank_time",
            )
            rank_depth = _normalize_ext_rank_depth(
                kwargs.get("rank_depth"),
                n_groups,
                layer_zooms,
            )
            n_rank_depth = _normalize_ext_rank_depth(
                kwargs.get("n_rank_depth"),
                n_groups,
                layer_zooms,
                name="n_rank_depth",
            )
            in_features = in_features_by_zoom
            target_features = target_features_by_zoom
        else:
            in_token_len_time = _collapse_shared_value(
                kwargs.get("in_token_len_time", 1),
                n_groups,
                "in_token_len_time",
            )
            out_token_len_time = _collapse_shared_value(
                kwargs.get("out_token_len_time", 1),
                n_groups,
                "out_token_len_time",
            )
            rank_space = _collapse_shared_value(
                kwargs.get("rank_space"),
                n_groups,
                "rank_space",
            )
            rank_time = _collapse_shared_value(
                kwargs.get("rank_time"),
                n_groups,
                "rank_time",
            )
            rank_variables = _collapse_shared_value(
                kwargs.get("rank_variables"),
                n_groups,
                "rank_variables",
            )
            n_times = _collapse_shared_value(
                kwargs.get("n_times", 1),
                n_groups,
                "n_times",
            )
            n_rank_space = _collapse_shared_value(
                kwargs.get("n_rank_space"),
                n_groups,
                "n_rank_space",
            )
            n_rank_time = _collapse_shared_value(
                kwargs.get("n_rank_time"),
                n_groups,
                "n_rank_time",
            )
            rank_depth = _normalize_group_values(
                kwargs.get("rank_depth"),
                n_groups,
                "rank_depth",
            )
            n_rank_depth = _normalize_group_values(
                kwargs.get("n_rank_depth"),
                n_groups,
                "n_rank_depth",
            )
            in_features = [
                in_features_by_zoom[int(zoom)] for zoom in x_zooms
            ]
            target_features = [
                target_features_by_zoom[int(zoom)] for zoom in target_zooms
            ]

        shared_values = {
            "token_overlap_space": _collapse_shared_value(
                kwargs.get("token_overlap_space", False),
                n_groups,
                "token_overlap_space",
            ),
            "token_overlap_time": _collapse_shared_value(
                kwargs.get("token_overlap_time", False),
                n_groups,
                "token_overlap_time",
            ),
            "residual": _collapse_shared_value(
                kwargs.get("residual", False),
                n_groups,
                "residual",
            ),
            "residual_gamma": _collapse_shared_value(
                kwargs.get("residual_gamma", False),
                n_groups,
                "residual_gamma",
            ),
            "type": _collapse_shared_value(
                kwargs.get("type", "linear"),
                n_groups,
                "type",
            ),
            "mult": _collapse_shared_value(
                kwargs.get("mult", 2),
                n_groups,
                "mult",
            ),
            "hidden_dim": hidden_dim_shared,
            "hidden_dim_mixed": hidden_dim_mixed_shared,
            "use_indexed_input": bool(_collapse_shared_value(
                kwargs.get("use_indexed_input", False),
                n_groups,
                "use_indexed_input",
            )),
            "use_indexed_output": bool(_collapse_shared_value(
                kwargs.get("use_indexed_output", False),
                n_groups,
                "use_indexed_output",
            )),
            "use_indexed_mlp": bool(_collapse_shared_value(
                kwargs.get("use_indexed_mlp", False),
                n_groups,
                "use_indexed_mlp",
            )),
            "fac_mode": _collapse_shared_value(
                kwargs.get("fac_mode", "Tucker"),
                n_groups,
                "fac_mode",
            ),
        }
        if shared_values["hidden_dim_mixed"] < 0:
            raise ValueError("hidden_dim_mixed must be non-negative")
        if shared_values["hidden_dim_mixed"] > 0:
            hidden_dim = shared_values["hidden_dim"]
            if hidden_dim is None or int(hidden_dim) <= 0:
                raise ValueError(
                    "Field-space layers require a positive hidden_dim when "
                    "hidden_dim_mixed > 0"
                )
            invalid_groups = [
                group_index
                for group_index, n_variables in enumerate(n_groups_variables)
                if int(n_variables) <= 1
            ]
            if invalid_groups:
                raise ValueError(
                    "hidden_dim_mixed > 0 requires n_variables > 1 for every "
                    f"active group; invalid groups={invalid_groups}"
                )

        for i in range(n_groups):
            block_kwargs = kwargs.copy()
            block_kwargs["n_variables"] = n_groups_variables[i]
            block_kwargs['in_features'] = in_features
            block_kwargs['target_features'] = target_features
            block_kwargs["in_token_len_depth"] = in_token_len_depth[i]
            block_kwargs["out_token_len_depth"] = out_token_len_depth[i]
            block_kwargs["token_overlap_depth"] = token_overlap_depth[i]
            block_kwargs["in_token_len_time"] = in_token_len_time
            block_kwargs["out_token_len_time"] = out_token_len_time
            block_kwargs["rank_space"] = rank_space
            block_kwargs["rank_time"] = rank_time
            block_kwargs["rank_depth"] = rank_depth[i]
            block_kwargs["rank_variables"] = rank_variables
            block_kwargs["n_times"] = n_times
            block_kwargs["n_rank_space"] = n_rank_space
            block_kwargs["n_rank_time"] = n_rank_time
            block_kwargs["n_depths"] = n_depths[i]
            block_kwargs["n_rank_depth"] = n_rank_depth[i]
            block_kwargs.update(shared_values)

            block_class = (
                ExtFieldSpaceLayerBlock
                if block_type == "ext"
                else FieldSpaceLayerBlock
            )
            block = block_class(
                grid_layers=grid_layers,
                x_zooms=x_zooms,
                in_zooms=in_zooms,
                target_zooms=target_zooms,
                field_zoom=field_zoom,
                **block_kwargs
            )
            self.out_zooms: Optional[List[int]] = block.out_zooms
            self.out_features: List[int] = block.out_features
            self.blocks.append(block)

    def forward(
        self,
        x_zooms_groups: List[Dict[int, torch.Tensor]],
        emb_groups: Optional[List[Optional[Dict[str, Any]]]] = None,
        sample_configs: Dict[str, Any] = {},
        **kwargs: Any
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Apply field-space blocks to each group.

        :param x_zooms_groups: List of zoom-to-tensor mappings shaped like
            ``(b, v, t, n, d, f)``.
        :param emb_groups: Optional list of embedding dictionaries per group.
        :param sample_configs: Sampling configuration dictionary.
        :param kwargs: Additional keyword arguments forwarded to blocks.
        :return: List of output zoom mappings with tensors shaped like ``(b, v, t, n, d, f)``.
        """
        if emb_groups is None:
            emb_groups = [None] * len(x_zooms_groups)

        output_groups = []
        for i, block in enumerate(self.blocks):
            output_groups.append(block(
                x_zooms=x_zooms_groups[i],
                emb=emb_groups[i],
                sample_configs=sample_configs,
                **kwargs
            ))
        return output_groups


class FieldSpaceLayerBlock(nn.Module):
  
    def __init__(
        self,
        grid_layers: Dict[str, Any],
        x_zooms: List[int],
        in_zooms: List[int],
        target_zooms: List[int],
        field_zoom: int,
        out_zooms: Optional[List[int]] = None,
        in_features: Union[List[int], int] = [1],
        target_features: Union[List[int], int] = [1],
        type: str = 'linear',
        in_token_len_time: int = 1,
        in_token_len_depth: int = 1,
        out_token_len_time: int = 1,
        out_token_len_depth: int = 1,
        token_overlap_space: bool = False,
        token_overlap_time: bool = False,
        token_overlap_depth: bool = False,
        rank_space: Optional[int] = None,
        rank_time: Optional[int] = None,
        n_rank_time: Optional[int] = None,
        rank_depth: Optional[int] = None,
        n_rank_depth: Optional[int] = None,
        rank_variables: Optional[int] = None,
        n_times: int = 1,
        n_rank_space: Optional[int] = None,
        n_depths: Optional[int] = None,
        mult: int = 2,
        hidden_dim: int = None,
        hidden_dim_mixed: int = 0,
        use_indexed_input: bool = False,
        use_indexed_output: bool = False,
        use_indexed_mlp: bool = False,
        residual: bool = False,
        residual_gamma: bool = False,
        n_variables: int = 1,
        fac_mode: str = "Tucker",
        block_type: Literal["legacy", "ext"] = "legacy",
    ) -> None:
        """
        Initialize a field-space layer block.

        :param grid_layers: Mapping from zoom string to grid layer.
        :param x_zooms: Zoom levels present in inputs.
        :param in_zooms: Input zoom levels.
        :param target_zooms: Target zoom levels.
        :param field_zoom: Zoom level used for tokenization.
        :param out_zooms: Optional output zoom levels.
        :param in_features: Input feature counts per zoom.
        :param target_features: Target feature counts per zoom.
        :param type: Layer type ("linear" or "mlp").
        :param in_token_len_time: Input token length along time.
        :param in_token_len_depth: Input token length along depth.
        :param out_token_len_time: Output token length along time.
        :param out_token_len_depth: Output token length along depth.
        :param token_overlap_space: Whether to overlap space tokens.
        :param token_overlap_time: Whether to overlap time tokens.
        :param token_overlap_depth: Whether to overlap depth tokens.
        :param rank_space: Optional rank for space.
        :param rank_time: Optional rank for time.
        :param n_rank_time: Optional indexed-tensor rank for time.
        :param rank_depth: Optional rank for depth.
        :param n_rank_depth: Optional indexed-tensor rank for depth.
        :param n_rank_space: Optional indexed-tensor rank for space.
        :param mult: MLP multiplier when using non-linear type.
        :param hidden_dim: Optional explicit hidden dimension for MLP.
        :param hidden_dim_mixed: Width of the shared cross-variable latent branch.
        :param residual: Whether to add a residual connection around the layer.
        :param residual_gamma: Whether to scale the learned layer-output branch with
            a gamma initialized near zero before adding the residual skip.
        :param layer_confs: Layer configuration dictionary.
        :return: None.
        """

        super().__init__()
        self.hidden_dim_mixed = int(hidden_dim_mixed)
        if self.hidden_dim_mixed < 0:
            raise ValueError("hidden_dim_mixed must be non-negative")
        self.n_variables = int(n_variables)
        self.use_indexed_input = bool(use_indexed_input)
        self.use_indexed_output = bool(use_indexed_output)
        self.use_indexed_mlp = bool(use_indexed_mlp)
        self.rank_variables = (
            None if rank_variables is None else int(rank_variables)
        )
        self.n_times = int(n_times)
        self.n_rank_space = (
            None if n_rank_space is None else int(n_rank_space)
        )
        self.n_rank_time = (
            None if n_rank_time is None else int(n_rank_time)
        )
        self.n_rank_depth = (
            None if n_rank_depth is None else int(n_rank_depth)
        )
        self.n_depths = 1 if n_depths is None else int(n_depths)
        if self.rank_variables is not None and self.rank_variables < 0:
            raise ValueError("rank_variables must be non-negative")
        if self.n_times <= 0:
            raise ValueError("n_times must be positive")
        if self.n_rank_space is not None and self.n_rank_space < 0:
            raise ValueError("n_rank_space must be non-negative")
        if self.n_rank_time is not None and self.n_rank_time < 0:
            raise ValueError("n_rank_time must be non-negative")
        if self.n_rank_depth is not None and self.n_rank_depth < 0:
            raise ValueError("n_rank_depth must be non-negative")
        if self.n_depths <= 0:
            raise ValueError("n_depths must be positive")
        self.hidden_dim = (
            None if hidden_dim is None else int(hidden_dim)
        )
        if self.hidden_dim_mixed > 0:
            if self.n_variables <= 1:
                raise ValueError(
                    "hidden_dim_mixed > 0 requires n_variables > 1"
                )
            if self.hidden_dim is None or self.hidden_dim <= 0:
                raise ValueError(
                    "FieldSpaceLayerBlock requires a positive hidden_dim "
                    "when hidden_dim_mixed > 0"
                )
        self.latent_dim_total = (
            None
            if self.hidden_dim is None
            else self.hidden_dim + self.hidden_dim_mixed
        )
        self.type = type
        self.mult = int(mult)
        if self.hidden_dim_mixed > 0 and self.type not in {"linear", "mlp"}:
            raise ValueError("type must be either 'linear' or 'mlp'")
        if (
            self.hidden_dim_mixed > 0
            and self.type == "mlp"
            and self.mult <= 0
        ):
            raise ValueError("mult must be positive")
        if isinstance(in_features, int):
            in_features = [in_features] * len(x_zooms)
        if isinstance(target_features, int):
            target_features = [target_features] * len(target_zooms)
        self.token_overlap_space: bool = token_overlap_space
        self.token_overlap_time: bool = token_overlap_time
        self.token_overlap_depth: bool = token_overlap_depth
        self.residual: bool = residual
        self.residual_gamma: bool = residual_gamma

        self.out_zooms: Optional[List[int]] = out_zooms
        self.in_zooms: List[int] = in_zooms
        self.field_zoom: int = int(field_zoom)
        self.in_token_len_depth = int(in_token_len_depth)
        self.out_token_len_depth = int(out_token_len_depth)
        if self.in_token_len_depth <= 0:
            raise ValueError("in_token_len_depth must be positive")
        if self.out_token_len_depth <= 0:
            raise ValueError("out_token_len_depth must be positive")
        uses_indexed_parameters = (
            self.use_indexed_input
            or self.use_indexed_output
            or (self.type == "mlp" and self.use_indexed_mlp)
        )
        if (
            uses_indexed_parameters
            and self.n_depths % self.in_token_len_depth != 0
        ):
            raise ValueError(
                "n_depths must be divisible by in_token_len_depth when "
                "indexed parameters are enabled"
            )
        self.n_channels_in: Dict[int, int] = {}

        self.in_features_dict: Dict[int, int] = dict(zip(x_zooms, in_features))
        self.target_features_dict: Dict[int, int] = dict(zip(target_zooms, target_features))

        self.out_features: List[int] = [
            self.target_features_dict[zoom] if zoom in self.target_features_dict.keys() else self.in_features_dict[zoom]
            for zoom in out_zooms
        ]

        self.tokenizer: Tokenizer = Tokenizer(in_zooms, 
                                   field_zoom,
                                   grid_layers=grid_layers,
                                   overlap_thickness=int(self.token_overlap_space),
                                   token_len_time=in_token_len_time,
                                   token_len_depth=in_token_len_depth)

        tokenizer_out = Tokenizer(target_zooms, 
                                  field_zoom,
                                  grid_layers=grid_layers,
                                  token_len_time=out_token_len_time,
                                  token_len_depth=out_token_len_depth)

        n_in_features_zooms, _ = self.tokenizer.get_features()
        n_out_features_zooms, _ = tokenizer_out.get_features()
        self.n_in_features_zooms: Dict[int, int] = n_in_features_zooms
        self.n_out_features_zooms: Dict[int, int] = n_out_features_zooms

        for z,f in self.n_in_features_zooms.items():
            self.n_in_features_zooms[z] = f * self.in_features_dict[z]
        
        for z,f in self.n_out_features_zooms.items():
            self.n_out_features_zooms[z] = f * self.target_features_dict[z]

        in_features_space = sum(self.n_in_features_zooms.values())
        out_features_space = sum(self.n_out_features_zooms.values())

        in_features_full = [
            in_token_len_time + 2 * int(self.token_overlap_time),
            in_features_space,
            in_token_len_depth + 2 * int(self.token_overlap_depth),
            1
        ]
        out_features_full = [out_token_len_time, out_features_space, out_token_len_depth, 1]

        ranks = [rank_time, rank_space, rank_depth, None, None]
        indexed_dims_input = self._build_indexed_dims(
            self.use_indexed_input,
            preserve_variable_default=True,
        )
        indexed_dims_output = self._build_indexed_dims(
            self.use_indexed_output,
            preserve_variable_default=True,
        )
        indexed_dims_mlp = self._build_indexed_dims(
            self.use_indexed_mlp,
            preserve_variable_default=False,
        )
        self.indexed_dims_input = indexed_dims_input
        self.indexed_dims_output = indexed_dims_output
        self.indexed_dims_mlp = indexed_dims_mlp

        if self.hidden_dim_mixed > 0:
            latent_shape = [1, 1, 1, self.hidden_dim]
            mixed_latent_shape = [1, 1, 1, self.hidden_dim_mixed]
            total_latent_shape = [1, 1, 1, self.latent_dim_total]
            mixed_input_shape = [
                *in_features_full[:-1],
                in_features_full[-1] * self.n_variables,
            ]
            self.input_projection_layer = get_layer(
                in_features_full,
                latent_shape,
                ranks=ranks,
                n_variables=self.n_variables,
                indexed_dims=indexed_dims_input,
                fac_mode=fac_mode,
                bias=False,
            )
            self.input_projection_layer_mixed = get_layer(
                mixed_input_shape,
                mixed_latent_shape,
                ranks=ranks,
                n_variables=1,
                indexed_dims={},
                fac_mode=fac_mode,
                bias=False,
            )
            if self.type == "mlp":
                mlp_hidden_shape = [
                    1,
                    1,
                    1,
                    self.mult * self.latent_dim_total,
                ]
                self.mlp_layer1 = get_layer(
                    total_latent_shape,
                    mlp_hidden_shape,
                    ranks=[None] * 5,
                    n_variables=self.n_variables,
                    indexed_dims=indexed_dims_mlp,
                    fac_mode=fac_mode,
                    bias=False,
                )
                self.mlp_layer2 = get_layer(
                    mlp_hidden_shape,
                    total_latent_shape,
                    ranks=[None] * 5,
                    n_variables=self.n_variables,
                    indexed_dims=indexed_dims_mlp,
                    fac_mode=fac_mode,
                    bias=True,
                )
                self.mlp_activation = nn.SiLU()
            self.output_projection_layer = get_layer(
                total_latent_shape,
                out_features_full,
                ranks=ranks,
                n_variables=self.n_variables,
                indexed_dims=indexed_dims_output,
                fac_mode=fac_mode,
                bias=False,
            )
            self.mixed_pattern = (
                "b v T N D t n d f -> b 1 T N D t n d (v f)"
            )
        elif type == 'linear':
            self.layer = get_layer(
                in_features_full,
                out_features_full,
                ranks=ranks,
                n_variables=n_variables,
                indexed_dims=self._build_indexed_dims(
                    self.use_indexed_input
                    or self.use_indexed_output,
                    preserve_variable_default=True,
                ),
                fac_mode=fac_mode,
            )
        else:
            self.layer = MLP_fac(
                in_features_full,
                out_features_full,
                mult=mult,
                hidden_dim=hidden_dim,
                ranks=ranks,
                n_variables=n_variables,
                indexed_dims_layer1=self._build_indexed_dims(
                    self.use_indexed_input or self.use_indexed_mlp,
                    preserve_variable_default=True,
                ),
                indexed_dims_layer2=self._build_indexed_dims(
                    self.use_indexed_output or self.use_indexed_mlp,
                    preserve_variable_default=True,
                ),
                fac_mode=fac_mode,
            )

        # Residual is taken from input `x_zooms` at matching output zoom keys.
        self.skip_projection_by_zoom: nn.ModuleDict = nn.ModuleDict()
        self.output_gamma_by_zoom: nn.ParameterDict = nn.ParameterDict()
        self.residual_source_zoom_by_target: Dict[int, int] = {}
        self.residual_zoom_mode_by_target: Dict[int, str] = {}
        self.residual_zoom_factor_by_target: Dict[int, int] = {}
        if self.residual:
            if len(x_zooms) == 0:
                raise ValueError("`x_zooms` must be non-empty when `residual=True`.")

            for target_zoom, out_features_zoom in self.target_features_dict.items():
                source_zoom = min(x_zooms, key=lambda z: abs(int(z) - int(target_zoom)))
                self.residual_source_zoom_by_target[target_zoom] = source_zoom

                in_features_zoom = self.in_features_dict[source_zoom]
                if source_zoom == target_zoom:
                    self.residual_zoom_mode_by_target[target_zoom] = "same"
                    self.residual_zoom_factor_by_target[target_zoom] = 1
                    use_projection = in_features_zoom != out_features_zoom
                    self.skip_projection_by_zoom[str(target_zoom)] = (
                        nn.Linear(in_features_zoom, out_features_zoom, bias=False)
                        if use_projection
                        else nn.Identity()
                    )
                elif source_zoom > target_zoom:
                    # Learn down-projection from child group features to parent features.
                    factor = 4 ** (source_zoom - target_zoom)
                    self.residual_zoom_mode_by_target[target_zoom] = "down"
                    self.residual_zoom_factor_by_target[target_zoom] = factor
                    self.skip_projection_by_zoom[str(target_zoom)] = nn.Linear(
                        factor * in_features_zoom,
                        out_features_zoom,
                        bias=False,
                    )
                else:
                    # Learn up-projection from parent features to child group features.
                    factor = 4 ** (target_zoom - source_zoom)
                    self.residual_zoom_mode_by_target[target_zoom] = "up"
                    self.residual_zoom_factor_by_target[target_zoom] = factor
                    self.skip_projection_by_zoom[str(target_zoom)] = nn.Linear(
                        in_features_zoom,
                        factor * out_features_zoom,
                        bias=False,
                    )

                if self.residual_gamma:
                    self.output_gamma_by_zoom[str(target_zoom)] = nn.Parameter(
                        torch.ones(out_features_zoom) * 1e-12,
                        requires_grad=True,
                    )

        self.pattern_tokens_reverse: str = 'b v T N D t (n f) d 1 -> b v (T t) (N n) (D d) f'

    def _build_indexed_dims(
        self,
        enabled: bool,
        *,
        preserve_variable_default: bool,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        if not enabled:
            if not preserve_variable_default:
                return {}
            return build_indexed_dims(
                n_variables=self.n_variables,
                same_values_variables=True,
            )
        indexed_n_depths = max(
            1,
            self.n_depths // self.in_token_len_depth,
        )
        indexed_n_space = (
            12 * 4**self.field_zoom
            if self.n_rank_space is not None
            and self.n_rank_space > 0
            and self.field_zoom >= 0
            else 1
        )
        return build_indexed_dims(
            n_variables=self.n_variables,
            rank_variables=self.rank_variables,
            same_values_variables=True,
            n_times=self.n_times,
            rank_time=self.n_rank_time,
            same_values_times=True,
            n_space=indexed_n_space,
            rank_space=(
                self.n_rank_space if indexed_n_space > 1 else None
            ),
            same_values_space=True,
            n_depths=indexed_n_depths,
            rank_depth=self.n_rank_depth,
            same_values_depths=True,
        )


    def update_time_embedder(self, emb: Dict[str, Any]) -> None:
        """
        Normalize zoom-keyed time embeddings to the max input zoom and copy
        them to newly produced output zooms.

        :param emb: Embedding dictionary containing zoom-keyed time embeddings.
        :return: None.
        """
        for emb_key in (
            "TimeEmbedder",
            "TimeProgressEmbedder",
            "TimeIndexEmbedder",
        ):
            if emb_key not in emb or not isinstance(emb[emb_key], dict):
                continue

            ref_zoom = max(self.in_zooms) if max(self.in_zooms) in emb[emb_key].keys() else max(emb[emb_key].keys())
            output_zooms = list(self.target_features_dict)
            for zoom in dict.fromkeys([*self.in_zooms, *output_zooms]):
                emb[emb_key][zoom] = emb[emb_key][ref_zoom]

    def get_time_depth_overlaps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time/depth overlap padding to tokenized tensors.

        :param x: Input tensor of shape ``(b, v, T, N, D, t, n, d, f)``.
        :return: Tensor with overlap padding applied.
        """
        if self.token_overlap_time:
            x = add_time_overlap_from_neighbor_patches(x, overlap=1, pad_mode="edge")

        if self.token_overlap_depth:
            x = add_depth_overlap_from_neighbor_patches(x, overlap=1, pad_mode="edge")

        return x

    def _apply_residual_projection(
        self,
        residual: torch.Tensor,
        target_zoom: int
    ) -> torch.Tensor:
        """
        Apply learned residual projection for same/down/up zoom mappings.

        :param residual: Source residual tensor of shape ``(b, v, t, n_src, d, f_src)``.
        :param target_zoom: Target zoom key used to select/create projection weights.
        :return: Projected tensor aligned to ``target_zoom`` output shape.
        """
        mode = self.residual_zoom_mode_by_target[target_zoom]
        factor = self.residual_zoom_factor_by_target[target_zoom]
        projection = self.skip_projection_by_zoom[str(target_zoom)]
        if mode == "same":
            return projection(residual)

        b, v, t, n_src, d, f_src = residual.shape
        f_tgt = self.target_features_dict[target_zoom]

        if mode == "down":
            if n_src % factor != 0:
                raise ValueError(
                    f"Cannot apply residual down-projection for zoom {target_zoom}: "
                    f"spatial dimension {n_src} is not divisible by {factor}."
                )
            n_tgt = n_src // factor
            x = residual.view(b, v, t, n_tgt, factor, d, f_src)
            x = x.permute(0, 1, 2, 3, 5, 4, 6).contiguous().view(b, v, t, n_tgt, d, factor * f_src)
            return projection(x)

        if mode == "up":
            x = projection(residual)
            x = x.view(b, v, t, n_src, d, factor, f_tgt)
            return x.reshape(b, v, t, n_src * factor, d, f_tgt)

        raise ValueError(f"Unsupported residual zoom mode `{mode}` for zoom {target_zoom}.")

    def _apply_output_gamma(
        self,
        x_out: torch.Tensor,
        target_zoom: int,
    ) -> torch.Tensor:
        """
        Apply per-feature learned scaling to the layer-output branch.

        :param x_out: Layer-output tensor of shape ``(b, v, t, n, d, f)``.
        :param target_zoom: Target zoom key used to select gamma weights.
        :return: Layer-output tensor scaled by the learned gamma.
        """
        if not self.residual_gamma:
            return x_out
        return x_out * self.output_gamma_by_zoom[str(target_zoom)]

    def forward(
        self,
        x_zooms: Dict[int, torch.Tensor],
        emb: Optional[Dict[str, Any]] = None,
        sample_configs: Dict[str, Any] = {},
        **kwargs: Any
    ) -> Dict[int, torch.Tensor]:
        """
        Apply the field-space layer to zoomed tensors.

        :param x_zooms: Mapping from zoom to tensors shaped like ``(b, v, t, n, d, f)``.
        :param emb: Optional embedding dictionary.
        :param sample_configs: Sampling configuration dictionary.
        :param kwargs: Additional keyword arguments (unused).
        :return: Updated zoom tensors shaped like ``(b, v, t, n, d, f)``.
        """
        nv = x_zooms[list(self.n_in_features_zooms.keys())[0]].shape[1]
        if self.hidden_dim_mixed > 0:
            runtime_variable_counts = {
                zoom: int(x_zooms[zoom].shape[1])
                for zoom in self.in_zooms
            }
            invalid_runtime_counts = {
                zoom: count
                for zoom, count in runtime_variable_counts.items()
                if count != self.n_variables
            }
            if invalid_runtime_counts:
                raise ValueError(
                    "hidden_dim_mixed requires the runtime variable count to "
                    f"equal configured n_variables={self.n_variables} for "
                    f"every input zoom; got {invalid_runtime_counts}"
                )
        residual_inputs: Dict[int, torch.Tensor] = {}
        if self.residual:
            for target_zoom, source_zoom in self.residual_source_zoom_by_target.items():
                if source_zoom not in x_zooms:
                    continue
                residual_inputs[target_zoom] = x_zooms[source_zoom]

        x = self.tokenizer(x_zooms, sample_configs=sample_configs)

        if emb:
            self.update_time_embedder(emb)

        x = self.get_time_depth_overlaps(x)

        layer_sample_config = sample_configs[self.field_zoom]
        if self.hidden_dim_mixed > 0:
            x_mixed = rearrange(x, self.mixed_pattern)
            latent = self.input_projection_layer(
                x,
                emb=emb,
                sample_configs=layer_sample_config,
            )
            latent_mixed = self.input_projection_layer_mixed(
                x_mixed,
                emb=emb,
                sample_configs=layer_sample_config,
            )
            expand_shape = (-1, nv, *([-1] * 7))
            latent = torch.cat(
                (latent, latent_mixed.expand(*expand_shape)),
                dim=-1,
            )
            if self.type == "mlp":
                latent = self.mlp_layer1(
                    latent,
                    emb=emb,
                    sample_configs=layer_sample_config,
                )
                latent = self.mlp_activation(latent)
                latent = self.mlp_layer2(
                    latent,
                    emb=emb,
                    sample_configs=layer_sample_config,
                )
            x = self.output_projection_layer(
                latent,
                emb=emb,
                sample_configs=layer_sample_config,
            )
        else:
            x = self.layer(
                x,
                emb=emb,
                sample_configs=layer_sample_config,
            )
        x = x.split(tuple(self.n_out_features_zooms.values()), dim=-3)
        
        for k, (zoom, n) in enumerate(self.n_out_features_zooms.items()):
            x_zoom_out = rearrange(x[k], self.pattern_tokens_reverse, f=self.target_features_dict[zoom], v=nv)
            x_zoom_out = self._apply_output_gamma(x_zoom_out, zoom)
            if zoom in residual_inputs:
                residual = self._apply_residual_projection(residual_inputs[zoom], zoom)
                x_zoom_out = x_zoom_out + residual
            x_zooms[zoom] = x_zoom_out
        
        if self.out_zooms is None:
            return x_zooms
        else:
            x_zooms_out = {}
            for zoom in self.out_zooms:
                x_zooms_out[zoom] = x_zooms[zoom]
            return x_zooms_out


class ExtFieldSpaceLayerBlock(FieldSpaceLayerBlock):
    """Field-space layer with independently projected per-zoom inputs and outputs."""

    def __init__(
        self,
        grid_layers: Dict[str, Any],
        x_zooms: List[int],
        in_zooms: List[int],
        target_zooms: List[int],
        field_zoom: int,
        out_zooms: Optional[List[int]] = None,
        in_features: Union[Mapping[int, int], Sequence[int], int] = 1,
        target_features: Union[Mapping[int, int], Sequence[int], int] = 1,
        type: str = "linear",
        in_token_len_time: Union[Mapping[int, int], Sequence[int], int] = 1,
        in_token_len_depth: int = 1,
        out_token_len_time: Union[Mapping[int, int], Sequence[int], int] = 1,
        out_token_len_depth: int = 1,
        token_overlap_space: bool = False,
        token_overlap_time: bool = False,
        token_overlap_depth: bool = False,
        rank_space: Union[
            Mapping[int, Optional[int]],
            Sequence[Optional[int]],
            Optional[int],
        ] = None,
        rank_time: Union[
            Mapping[int, Optional[int]],
            Sequence[Optional[int]],
            Optional[int],
        ] = None,
        n_rank_time: Union[
            Mapping[int, Optional[int]],
            Sequence[Optional[int]],
            Optional[int],
        ] = None,
        rank_depth: Union[
            Mapping[int, Optional[int]],
            Sequence[Optional[int]],
            Optional[int],
        ] = None,
        n_rank_depth: Union[
            Mapping[int, Optional[int]],
            Sequence[Optional[int]],
            Optional[int],
        ] = None,
        rank_variables: Union[
            Mapping[int, Optional[int]],
            Sequence[Optional[int]],
            Optional[int],
        ] = None,
        n_times: Union[
            Mapping[int, int],
            Sequence[int],
            int,
        ] = 1,
        n_rank_space: Union[
            Mapping[int, Optional[int]],
            Sequence[Optional[int]],
            Optional[int],
        ] = None,
        n_depths: Optional[int] = None,
        mult: int = 2,
        hidden_dim: Optional[int] = None,
        hidden_dim_mixed: int = 0,
        use_indexed_input: bool = False,
        use_indexed_output: bool = False,
        use_indexed_mlp: bool = False,
        residual: bool = False,
        residual_gamma: bool = False,
        n_variables: int = 1,
        fac_mode: str = "Tucker",
        block_type: Literal["legacy", "ext"] = "ext",
    ) -> None:
        nn.Module.__init__(self)

        if hidden_dim is None or int(hidden_dim) <= 0:
            raise ValueError(
                "ExtFieldSpaceLayerBlock requires a positive hidden_dim"
            )
        if int(hidden_dim_mixed) < 0:
            raise ValueError("hidden_dim_mixed must be non-negative")
        if int(hidden_dim_mixed) > 0 and int(n_variables) <= 1:
            raise ValueError(
                "hidden_dim_mixed > 0 requires n_variables > 1"
            )
        if type not in {"linear", "mlp"}:
            raise ValueError("type must be either 'linear' or 'mlp'")
        if not in_zooms:
            raise ValueError("in_zooms must be non-empty")
        if not target_zooms:
            raise ValueError("target_zooms must be non-empty")

        self.x_zooms = [int(zoom) for zoom in x_zooms]
        self.in_zooms = [int(zoom) for zoom in in_zooms]
        self.target_zooms = [int(zoom) for zoom in target_zooms]
        self.layer_zooms = list(
            dict.fromkeys([*self.in_zooms, *self.target_zooms])
        )
        missing_inputs = [
            zoom for zoom in self.in_zooms if zoom not in self.x_zooms
        ]
        if missing_inputs:
            raise ValueError(
                f"Input zooms {missing_inputs} are not present in x_zooms"
            )

        self.in_features_dict = {
            zoom: int(value)
            for zoom, value in _normalize_axis_values(
                in_features,
                self.x_zooms,
                "in_features",
            ).items()
        }
        self.target_features_dict = {
            zoom: int(value)
            for zoom, value in _normalize_axis_values(
                target_features,
                self.target_zooms,
                "target_features",
            ).items()
        }
        self.in_token_len_time_by_zoom = {
            zoom: int(value)
            for zoom, value in _normalize_axis_values(
                in_token_len_time,
                self.in_zooms,
                "in_token_len_time",
            ).items()
        }
        self.out_token_len_time_by_zoom = {
            zoom: int(value)
            for zoom, value in _normalize_axis_values(
                out_token_len_time,
                self.target_zooms,
                "out_token_len_time",
            ).items()
        }
        self.rank_space_by_zoom = _normalize_axis_values(
            rank_space,
            self.layer_zooms,
            "rank_space",
        )
        self.rank_time_by_zoom = _normalize_axis_values(
            rank_time,
            self.layer_zooms,
            "rank_time",
        )
        self.n_rank_time_by_zoom = {
            zoom: None if value is None else int(value)
            for zoom, value in _normalize_axis_values(
                n_rank_time,
                self.layer_zooms,
                "n_rank_time",
            ).items()
        }
        self.rank_depth_by_zoom = _normalize_axis_values(
            rank_depth,
            self.layer_zooms,
            "rank_depth",
        )
        self.n_rank_depth_by_zoom = {
            zoom: None if value is None else int(value)
            for zoom, value in _normalize_axis_values(
                n_rank_depth,
                self.layer_zooms,
                "n_rank_depth",
            ).items()
        }
        self.rank_variables_by_zoom = {
            zoom: None if value is None else int(value)
            for zoom, value in _normalize_axis_values(
                rank_variables,
                self.layer_zooms,
                "rank_variables",
            ).items()
        }
        self.n_times_by_zoom = {
            zoom: int(value)
            for zoom, value in _normalize_axis_values(
                n_times,
                self.layer_zooms,
                "n_times",
            ).items()
        }
        self.n_rank_space_by_zoom = {
            zoom: None if value is None else int(value)
            for zoom, value in _normalize_axis_values(
                n_rank_space,
                self.layer_zooms,
                "n_rank_space",
            ).items()
        }

        self.field_zoom = int(field_zoom)
        self.in_token_len_depth = int(in_token_len_depth)
        self.out_token_len_depth = int(out_token_len_depth)
        self.n_depths = 1 if n_depths is None else int(n_depths)
        self.use_indexed_input = bool(use_indexed_input)
        self.use_indexed_output = bool(use_indexed_output)
        self.use_indexed_mlp = bool(use_indexed_mlp)
        if self.in_token_len_depth <= 0:
            raise ValueError("in_token_len_depth must be positive")
        if self.out_token_len_depth <= 0:
            raise ValueError("out_token_len_depth must be positive")
        if self.n_depths <= 0:
            raise ValueError("n_depths must be positive")
        for zoom, value in self.rank_variables_by_zoom.items():
            if value is not None and value < 0:
                raise ValueError(
                    f"rank_variables[{zoom}] must be non-negative"
                )
        for zoom, value in self.n_times_by_zoom.items():
            if value <= 0:
                raise ValueError(f"n_times[{zoom}] must be positive")
        for zoom, value in self.n_rank_space_by_zoom.items():
            if value is not None and value < 0:
                raise ValueError(
                    f"n_rank_space[{zoom}] must be non-negative"
                )
        for zoom, value in self.n_rank_time_by_zoom.items():
            if value is not None and value < 0:
                raise ValueError(
                    f"n_rank_time[{zoom}] must be non-negative"
                )
        for zoom, value in self.n_rank_depth_by_zoom.items():
            if value is not None and value < 0:
                raise ValueError(
                    f"n_rank_depth[{zoom}] must be non-negative"
                )
        uses_indexed_parameters = (
            self.use_indexed_input
            or self.use_indexed_output
            or (type == "mlp" and self.use_indexed_mlp)
        )
        if (
            uses_indexed_parameters
            and self.n_depths % self.in_token_len_depth != 0
        ):
            raise ValueError(
                "n_depths must be divisible by in_token_len_depth when "
                "indexed parameters are enabled"
            )
        self.token_overlap_space = bool(token_overlap_space)
        self.token_overlap_time = bool(token_overlap_time)
        self.token_overlap_depth = bool(token_overlap_depth)
        self.hidden_dim = int(hidden_dim)
        self.hidden_dim_mixed = int(hidden_dim_mixed)
        self.latent_dim_total = self.hidden_dim + self.hidden_dim_mixed
        self.mult = int(mult)
        if self.mult <= 0:
            raise ValueError("mult must be positive")
        self.type = type
        self.residual = bool(residual)
        self.residual_gamma = bool(residual_gamma)
        self.n_variables = int(n_variables)
        self.fac_mode = fac_mode
        self.out_zooms = (
            None if out_zooms is None else [int(zoom) for zoom in out_zooms]
        )
        effective_out_zooms = (
            list(dict.fromkeys([*self.x_zooms, *self.target_zooms]))
            if self.out_zooms is None else self.out_zooms
        )
        missing_output_features = [
            zoom
            for zoom in effective_out_zooms
            if zoom not in self.target_features_dict
            and zoom not in self.in_features_dict
        ]
        if missing_output_features:
            raise ValueError(
                f"Missing output feature definitions for zooms "
                f"{missing_output_features}"
            )
        self.out_features = [
            self.target_features_dict.get(
                zoom, self.in_features_dict.get(zoom)
            )
            for zoom in effective_out_zooms
        ]

        self.input_tokenizers = nn.ModuleDict()
        self.input_projection_layers = nn.ModuleDict()
        self.input_projection_layers_mixed = nn.ModuleDict()
        self.target_tokenizers = nn.ModuleDict()
        self.output_projection_layers = nn.ModuleDict()
        self.input_shapes_by_zoom: Dict[int, List[int]] = {}
        self.output_shapes_by_zoom: Dict[int, List[int]] = {}
        self.indexed_dims_input_by_zoom = {
            zoom: self._build_indexed_dims_for_zoom(
                zoom,
                self.use_indexed_input,
                preserve_variable_default=True,
            )
            for zoom in self.in_zooms
        }
        self.indexed_dims_output_by_zoom = {
            zoom: self._build_indexed_dims_for_zoom(
                zoom,
                self.use_indexed_output,
                preserve_variable_default=True,
            )
            for zoom in self.target_zooms
        }
        self.indexed_dims_mlp: Optional[Dict[str, Dict[str, Any]]] = {}
        if self.type == "mlp":
            mlp_indexed_dims_by_zoom = [
                self._build_indexed_dims_for_zoom(
                    zoom,
                    self.use_indexed_mlp,
                    preserve_variable_default=False,
                )
                for zoom in self.in_zooms
            ]
            self.indexed_dims_mlp = mlp_indexed_dims_by_zoom[0]
            if any(
                indexed_dims != self.indexed_dims_mlp
                for indexed_dims in mlp_indexed_dims_by_zoom[1:]
            ):
                raise ValueError(
                    "Ext shared MLP requires identical indexed parameter "
                    "specifications across all input zooms"
                )

        for zoom in self.in_zooms:
            key = str(zoom)
            tokenizer = Tokenizer(
                [zoom],
                self.field_zoom,
                grid_layers=grid_layers,
                overlap_thickness=int(self.token_overlap_space),
                token_len_time=self.in_token_len_time_by_zoom[zoom],
                token_len_depth=self.in_token_len_depth,
            )
            self.input_tokenizers[key] = tokenizer
            n_space = tokenizer.get_features()[0][zoom]
            input_shape = [
                self.in_token_len_time_by_zoom[zoom]
                + 2 * int(self.token_overlap_time),
                n_space * self.in_features_dict[zoom],
                self.in_token_len_depth
                + 2 * int(self.token_overlap_depth),
                1,
            ]
            self.input_shapes_by_zoom[zoom] = input_shape
            self.input_projection_layers[key] = get_layer(
                input_shape,
                [1, 1, 1, self.hidden_dim],
                ranks=self._ranks_for_zoom(zoom),
                n_variables=self.n_variables,
                indexed_dims=self.indexed_dims_input_by_zoom[zoom],
                fac_mode=self.fac_mode,
                bias=False,
            )
            if self.hidden_dim_mixed > 0:
                mixed_input_shape = [
                    *input_shape[:-1],
                    input_shape[-1] * self.n_variables,
                ]
                self.input_projection_layers_mixed[key] = get_layer(
                    mixed_input_shape,
                    [1, 1, 1, self.hidden_dim_mixed],
                    ranks=self._ranks_for_zoom(zoom),
                    n_variables=1,
                    indexed_dims={},
                    fac_mode=self.fac_mode,
                    bias=False,
                )

        for zoom in self.target_zooms:
            key = str(zoom)
            tokenizer = Tokenizer(
                [zoom],
                self.field_zoom,
                grid_layers=grid_layers,
                overlap_thickness=0,
                token_len_time=self.out_token_len_time_by_zoom[zoom],
                token_len_depth=self.out_token_len_depth,
            )
            self.target_tokenizers[key] = tokenizer
            n_space = tokenizer.get_features()[1][zoom]
            output_shape = [
                self.out_token_len_time_by_zoom[zoom],
                n_space * self.target_features_dict[zoom],
                self.out_token_len_depth,
                1,
            ]
            self.output_shapes_by_zoom[zoom] = output_shape
            self.output_projection_layers[key] = get_layer(
                [1, 1, 1, self.latent_dim_total],
                output_shape,
                ranks=self._ranks_for_zoom(zoom),
                n_variables=self.n_variables,
                indexed_dims=self.indexed_dims_output_by_zoom[zoom],
                fac_mode=self.fac_mode,
                bias=True,
            )

        if self.type == "mlp":
            mlp_hidden_dim = self.mult * self.latent_dim_total
            self.mlp_layer1: Optional[nn.Module] = get_layer(
                [1, 1, 1, self.latent_dim_total],
                [1, 1, 1, mlp_hidden_dim],
                ranks=[None] * 5,
                n_variables=self.n_variables,
                indexed_dims=self.indexed_dims_mlp,
                fac_mode=self.fac_mode,
                bias=False,
            )
            self.mlp_layer2: Optional[nn.Module] = get_layer(
                [1, 1, 1, mlp_hidden_dim],
                [1, 1, 1, self.latent_dim_total],
                ranks=[None] * 5,
                n_variables=self.n_variables,
                indexed_dims=self.indexed_dims_mlp,
                fac_mode=self.fac_mode,
                bias=True,
            )
            self.mlp_activation: Optional[nn.Module] = nn.SiLU()
        else:
            self.mlp_layer1 = None
            self.mlp_layer2 = None
            self.mlp_activation = None

        self.skip_projection_by_zoom = nn.ModuleDict()
        self.output_gamma_by_zoom = nn.ParameterDict()
        self.residual_source_zoom_by_target: Dict[int, int] = {}
        self.residual_zoom_mode_by_target: Dict[int, str] = {}
        self.residual_zoom_factor_by_target: Dict[int, int] = {}
        self._initialize_ext_residuals()

        self.pattern_tokens_reverse = (
            "b v T N D t (n f) d 1 -> b v (T t) (N n) (D d) f"
        )

    def _ranks_for_zoom(self, zoom: int) -> List[Optional[int]]:
        return [
            self.rank_time_by_zoom[zoom],
            self.rank_space_by_zoom[zoom],
            self.rank_depth_by_zoom[zoom],
            None,
            None,
        ]

    def _build_indexed_dims_for_zoom(
        self,
        zoom: int,
        enabled: bool,
        *,
        preserve_variable_default: bool,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        if not enabled:
            if not preserve_variable_default:
                return {}
            return build_indexed_dims(
                n_variables=self.n_variables,
                same_values_variables=True,
            )
        indexed_n_depths = max(
            1,
            self.n_depths // self.in_token_len_depth,
        )
        n_rank_space = self.n_rank_space_by_zoom[zoom]
        indexed_n_space = (
            12 * 4**self.field_zoom
            if n_rank_space is not None
            and n_rank_space > 0
            and self.field_zoom >= 0
            else 1
        )
        return build_indexed_dims(
            n_variables=self.n_variables,
            rank_variables=self.rank_variables_by_zoom[zoom],
            same_values_variables=True,
            n_times=self.n_times_by_zoom[zoom],
            rank_time=self.n_rank_time_by_zoom[zoom],
            same_values_times=True,
            n_space=indexed_n_space,
            rank_space=(
                n_rank_space if indexed_n_space > 1 else None
            ),
            same_values_space=True,
            n_depths=indexed_n_depths,
            rank_depth=self.n_rank_depth_by_zoom[zoom],
            same_values_depths=True,
        )

    def _initialize_ext_residuals(self) -> None:
        if not self.residual:
            return
        if not self.x_zooms:
            raise ValueError("x_zooms must be non-empty when residual=True")

        for target_zoom, out_features_zoom in self.target_features_dict.items():
            source_zoom = min(
                self.x_zooms,
                key=lambda zoom: abs(int(zoom) - int(target_zoom)),
            )
            self.residual_source_zoom_by_target[target_zoom] = source_zoom
            in_features_zoom = self.in_features_dict[source_zoom]
            key = str(target_zoom)

            if source_zoom == target_zoom:
                self.residual_zoom_mode_by_target[target_zoom] = "same"
                self.residual_zoom_factor_by_target[target_zoom] = 1
                self.skip_projection_by_zoom[key] = (
                    nn.Linear(
                        in_features_zoom,
                        out_features_zoom,
                        bias=False,
                    )
                    if in_features_zoom != out_features_zoom
                    else nn.Identity()
                )
            elif source_zoom > target_zoom:
                factor = 4 ** (source_zoom - target_zoom)
                self.residual_zoom_mode_by_target[target_zoom] = "down"
                self.residual_zoom_factor_by_target[target_zoom] = factor
                self.skip_projection_by_zoom[key] = nn.Linear(
                    factor * in_features_zoom,
                    out_features_zoom,
                    bias=False,
                )
            else:
                factor = 4 ** (target_zoom - source_zoom)
                self.residual_zoom_mode_by_target[target_zoom] = "up"
                self.residual_zoom_factor_by_target[target_zoom] = factor
                self.skip_projection_by_zoom[key] = nn.Linear(
                    in_features_zoom,
                    factor * out_features_zoom,
                    bias=False,
                )

            if self.residual_gamma:
                self.output_gamma_by_zoom[key] = nn.Parameter(
                    torch.ones(out_features_zoom) * 1e-12,
                    requires_grad=True,
                )

    @staticmethod
    def _sum_latent_projections(
        projections: List[torch.Tensor],
    ) -> torch.Tensor:
        if not projections:
            raise ValueError("Ext field-space layer has no input projections")
        expected_shape = projections[0].shape
        for index, projection in enumerate(projections[1:], start=1):
            if projection.shape != expected_shape:
                raise ValueError(
                    "Ext input zoom projections must have identical shapes; "
                    f"projection 0 has {tuple(expected_shape)}, projection "
                    f"{index} has {tuple(projection.shape)}"
                )
        return torch.stack(projections, dim=0).sum(dim=0)

    def _preprocess_input_zoom(
        self,
        zoom: int,
        x_zooms: Dict[int, torch.Tensor],
        sample_configs: Dict[str, Any],
    ) -> torch.Tensor:
        x = self.input_tokenizers[str(zoom)](
            {zoom: x_zooms[zoom]},
            sample_configs=sample_configs,
        )
        if self.token_overlap_time:
            x = add_time_overlap_from_neighbor_patches(
                x,
                overlap=1,
                pad_mode="edge",
            )
        if self.token_overlap_depth:
            x = add_depth_overlap_from_neighbor_patches(
                x,
                overlap=1,
                pad_mode="edge",
            )
        return x

    def forward(
        self,
        x_zooms: Dict[int, torch.Tensor],
        emb: Optional[Dict[str, Any]] = None,
        sample_configs: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> Dict[int, torch.Tensor]:
        del kwargs
        residual_inputs: Dict[int, torch.Tensor] = {}
        if self.residual:
            for target_zoom, source_zoom in (
                self.residual_source_zoom_by_target.items()
            ):
                if source_zoom in x_zooms:
                    residual_inputs[target_zoom] = x_zooms[source_zoom]

        if emb:
            self.update_time_embedder(emb)

        projections: List[torch.Tensor] = []
        mixed_projections: List[torch.Tensor] = []
        layer_sample_config = sample_configs.get(self.field_zoom, {})
        runtime_variable_counts = {
            zoom: int(x_zooms[zoom].shape[1])
            for zoom in self.in_zooms
        }
        runtime_n_variables = runtime_variable_counts[self.in_zooms[0]]
        if self.hidden_dim_mixed > 0:
            invalid_runtime_counts = {
                zoom: count
                for zoom, count in runtime_variable_counts.items()
                if count != self.n_variables
            }
            if invalid_runtime_counts:
                raise ValueError(
                    "hidden_dim_mixed requires the runtime variable count to "
                    f"equal configured n_variables={self.n_variables} for "
                    f"every input zoom; got {invalid_runtime_counts}"
                )
        for zoom in self.in_zooms:
            x = self._preprocess_input_zoom(
                zoom,
                x_zooms,
                sample_configs,
            )
            projections.append(
                self.input_projection_layers[str(zoom)](
                    x,
                    emb=emb,
                    sample_configs=layer_sample_config,
                )
            )
            if self.hidden_dim_mixed > 0:
                x_mixed = rearrange(
                    x,
                    "b v T N D t n d f -> b 1 T N D t n d (v f)",
                )
                mixed_projections.append(
                    self.input_projection_layers_mixed[str(zoom)](
                        x_mixed,
                        emb=emb,
                        sample_configs=layer_sample_config,
                    )
                )

        latent = self._sum_latent_projections(projections)
        if mixed_projections:
            latent_mixed = self._sum_latent_projections(
                mixed_projections
            )
            expand_shape = (
                -1,
                runtime_n_variables,
                *([-1] * 7),
            )
            latent = torch.cat(
                (latent, latent_mixed.expand(*expand_shape)),
                dim=-1,
            )
        if self.type == "mlp":
            latent = self.mlp_layer1(
                latent,
                emb=emb,
                sample_configs=layer_sample_config,
            )
            latent = self.mlp_activation(latent)
            latent = self.mlp_layer2(
                latent,
                emb=emb,
                sample_configs=layer_sample_config,
            )

        n_variables = runtime_n_variables
        for zoom in self.target_zooms:
            x_zoom_out = self.output_projection_layers[str(zoom)](
                latent,
                emb=emb,
                sample_configs=layer_sample_config,
            )
            x_zoom_out = rearrange(
                x_zoom_out,
                self.pattern_tokens_reverse,
                f=self.target_features_dict[zoom],
                v=n_variables,
            )
            x_zoom_out = self._apply_output_gamma(x_zoom_out, zoom)
            if zoom in residual_inputs:
                residual = self._apply_residual_projection(
                    residual_inputs[zoom],
                    zoom,
                )
                x_zoom_out = x_zoom_out + residual
            x_zooms[zoom] = x_zoom_out

        if self.out_zooms is None:
            return x_zooms
        return {zoom: x_zooms[zoom] for zoom in self.out_zooms}
