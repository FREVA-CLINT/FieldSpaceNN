import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from ...modules.field_space.field_space_base import (
    GLOBAL_EMBEDDER_CACHE_KEY,
)
from ...modules.grids.grid_layer import GridLayer

ZoomGroup = Dict[int, torch.Tensor]
ZoomGroups = List[ZoomGroup]
MaskGroup = Optional[Dict[int, torch.Tensor]]
MaskGroups = Optional[Sequence[MaskGroup]]
EmbGroups = Optional[Sequence[Optional[Dict[str, Any]]]]


@dataclass
class BlockWrapContext:
    mask_groups: MaskGroups
    emb_groups: EmbGroups
    sample_configs: Mapping[int, Dict[str, Any]]


class BlockWrapOperation(nn.Module):
    operation_kind = "base"

    def pre(
        self,
        x_zooms_groups: ZoomGroups,
        context: BlockWrapContext,
    ) -> Tuple[ZoomGroups, Any]:
        return x_zooms_groups, None

    def post(
        self,
        x_zooms_groups: ZoomGroups,
        state: Any,
        context: BlockWrapContext,
    ) -> ZoomGroups:
        return x_zooms_groups


@dataclass
class MergeGroupsBlockWrapState:
    original_mask_groups: MaskGroups
    original_emb_groups: EmbGroups
    group_shapes: List[Dict[int, torch.Size]]
    zoom_order: List[int]
    selected_depth_indices: Dict[int, List[int]]
    remaining_depth_indices: Dict[int, List[int]]
    stage_group_original_indices: List[int]


class BlockWrapConfig:
    operation_kind = "base"

    def build(
        self,
        *,
        grid_layers: nn.ModuleDict,
    ) -> BlockWrapOperation:
        raise NotImplementedError

    def get_block_build_overrides(
        self,
        *,
        n_groups_variables: Sequence[int],
        n_groups_depths: Sequence[int],
        base_block_kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        del n_groups_variables
        del n_groups_depths
        del base_block_kwargs
        return {}

    def get_stage_input_zooms(self, current_in_zooms: Sequence[int]) -> List[int]:
        return [int(zoom) for zoom in current_in_zooms]

    def get_stage_input_features(
        self,
        *,
        current_in_zooms: Sequence[int],
        current_in_features: Sequence[int],
    ) -> List[int]:
        del current_in_zooms
        return [int(feature) for feature in current_in_features]


def create_block_wrap_operation(
    wrap_conf: Any,
    *,
    grid_layers: nn.ModuleDict,
) -> BlockWrapOperation:
    if not isinstance(wrap_conf, BlockWrapConfig):
        raise TypeError(
            "block_wrap_configs entries must instantiate BlockWrapConfig targets, "
            f"got {type(wrap_conf).__name__}."
        )

    operation = wrap_conf.build(grid_layers=grid_layers)
    if not isinstance(operation, BlockWrapOperation):
        raise TypeError(
            f"{type(wrap_conf).__name__}.build() must return a BlockWrapOperation, "
            f"got {type(operation).__name__}."
        )

    return operation


def validate_block_wrap_operation_sequence(operations: Mapping[str, BlockWrapOperation]) -> None:
    seen_kinds: Dict[str, str] = {}
    for name, operation in operations.items():
        kind = getattr(operation, "operation_kind", type(operation).__name__)
        if kind in seen_kinds:
            raise ValueError(
                f"Duplicate block wrap operation kind `{kind}` is not supported: "
                f"`{seen_kinds[kind]}` and `{name}`."
            )
        seen_kinds[kind] = name


def clone_zoom_groups(x_zooms_groups: Sequence[ZoomGroup]) -> ZoomGroups:
    return [
        {zoom: tensor.clone() for zoom, tensor in x_zooms.items()}
        for x_zooms in x_zooms_groups
    ]


def clone_mask_groups(mask_zooms_groups: MaskGroups) -> Optional[List[MaskGroup]]:
    if mask_zooms_groups is None:
        return None

    return [
        None if mask_zooms is None else {zoom: mask.clone() for zoom, mask in mask_zooms.items()}
        for mask_zooms in mask_zooms_groups
    ]


def apply_saved_residuals(
    *,
    x_zooms_groups: Sequence[ZoomGroup],
    saved_residual_groups: Optional[Sequence[ZoomGroup]],
    mask_zooms_groups: MaskGroups,
    saved_mask_groups: Optional[Sequence[MaskGroup]],
    mode: str,
) -> ZoomGroups:
    if saved_residual_groups is None:
        raise ValueError("Residual block wrap operation requires a saved residual state.")

    if len(x_zooms_groups) != len(saved_residual_groups):
        raise ValueError(
            "Residual block wrap operation requires the same number of groups in the saved and current states."
        )

    if mode not in {"add", "masked"}:
        raise ValueError(f"Unsupported residual mode `{mode}`.")

    x_zooms_groups_out = list(x_zooms_groups)
    for group_idx, (x_zooms, saved_zooms) in enumerate(zip(x_zooms_groups_out, saved_residual_groups)):
        current_zooms = set(x_zooms.keys())
        saved_zoom_keys = set(saved_zooms.keys())
        if current_zooms != saved_zoom_keys:
            raise ValueError(
                f"Residual block wrap operation requires matching zoom keys for group {group_idx}: "
                f"{sorted(saved_zoom_keys)} != {sorted(current_zooms)}."
            )

        current_masks = None if mask_zooms_groups is None else mask_zooms_groups[group_idx]
        saved_masks = None if saved_mask_groups is None else saved_mask_groups[group_idx]
        if mode == "masked" and (current_masks is None or saved_masks is None):
            raise ValueError(
                f"Residual block wrap operation in masked mode requires masks for group {group_idx}."
            )

        for zoom in x_zooms.keys():
            current = x_zooms[zoom]
            saved = saved_zooms[zoom]
            if current.shape != saved.shape:
                raise ValueError(
                    f"Residual block wrap operation requires matching tensor shapes for group {group_idx}, "
                    f"zoom {zoom}: {tuple(saved.shape)} != {tuple(current.shape)}."
                )

            if mode == "add":
                x_zooms[zoom] = saved + current
                continue

            assert current_masks is not None
            assert saved_masks is not None
            if zoom not in current_masks or zoom not in saved_masks:
                raise ValueError(
                    f"Residual block wrap operation in masked mode requires masks for group {group_idx}, zoom {zoom}."
                )

            mask = current_masks[zoom]
            saved_mask = saved_masks[zoom]
            if mask.shape != current.shape or saved_mask.shape != saved.shape:
                raise ValueError(
                    "Residual block wrap operation in masked mode requires mask shapes to match tensor shapes "
                    f"for group {group_idx}, zoom {zoom}."
                )

            if mask.dtype == torch.bool:
                x_zooms[zoom] = torch.where(mask, current, saved)
            else:
                x_zooms[zoom] = saved * (1 - mask) + current * mask

    return x_zooms_groups_out


class ResidualBlockWrapConfig(BlockWrapConfig):
    operation_kind = "residual"

    def __init__(self, mode: str = "add", **kwargs: Any) -> None:
        self.mode: str

        inputs = copy.deepcopy(locals())
        for input_name, value in inputs.items():
            if input_name == "kwargs":
                for kw_name, kw_value in value.items():
                    setattr(self, kw_name, kw_value)
            else:
                setattr(self, input_name, value)

    def build(
        self,
        *,
        grid_layers: nn.ModuleDict,
    ) -> BlockWrapOperation:
        del grid_layers
        return ResidualBlockWrapOperation(mode=self.mode)


class ResidualBlockWrapOperation(BlockWrapOperation):
    operation_kind = "residual"

    def __init__(self, mode: str = "add") -> None:
        super().__init__()
        self.mode = mode

    def pre(
        self,
        x_zooms_groups: ZoomGroups,
        context: BlockWrapContext,
    ) -> Tuple[ZoomGroups, Tuple[ZoomGroups, Optional[List[MaskGroup]]]]:
        saved_residual_groups = clone_zoom_groups(x_zooms_groups)
        saved_mask_groups = clone_mask_groups(context.mask_groups)
        return x_zooms_groups, (saved_residual_groups, saved_mask_groups)

    def post(
        self,
        x_zooms_groups: ZoomGroups,
        state: Tuple[ZoomGroups, Optional[List[MaskGroup]]],
        context: BlockWrapContext,
    ) -> ZoomGroups:
        saved_residual_groups, saved_mask_groups = state
        return apply_saved_residuals(
            x_zooms_groups=x_zooms_groups,
            saved_residual_groups=saved_residual_groups,
            mask_zooms_groups=context.mask_groups,
            saved_mask_groups=saved_mask_groups,
            mode=self.mode,
        )


class ShiftGroupsBlockWrapConfig(BlockWrapConfig):
    operation_kind = "shift_groups"

    def __init__(
        self,
        token_zoom: int,
        zooms: Optional[Sequence[int]] = None,
        q_zooms: Union[Sequence[int], int] = -1,
        kv_zooms: Union[Sequence[int], int] = -1,
        multi_shift: bool = False,
        direction: str = "east",
        **kwargs: Any,
    ) -> None:
        self.token_zoom: int
        self.zooms: Optional[Sequence[int]]
        self.q_zooms: Union[Sequence[int], int]
        self.kv_zooms: Union[Sequence[int], int]
        self.multi_shift: bool
        self.direction: str

        inputs = copy.deepcopy(locals())
        for input_name, value in inputs.items():
            if input_name == "kwargs":
                for kw_name, kw_value in value.items():
                    setattr(self, kw_name, kw_value)
            else:
                setattr(self, input_name, value)

    def build(
        self,
        *,
        grid_layers: nn.ModuleDict,
    ) -> BlockWrapOperation:
        zooms = _resolve_shift_zooms(self.zooms, self.q_zooms, self.kv_zooms)
        return ShiftGroupsBlockWrapOperation(
            grid_layers=grid_layers,
            token_zoom=self.token_zoom,
            zooms=zooms,
            multi_shift=self.multi_shift,
            direction=self.direction,
        )


class ShiftGroupsBlockWrapOperation(BlockWrapOperation):
    operation_kind = "shift_groups"

    def __init__(
        self,
        *,
        grid_layers: Mapping[str, GridLayer],
        token_zoom: int,
        zooms: Sequence[int],
        multi_shift: bool = False,
        direction: str = "east",
    ) -> None:
        super().__init__()
        self.grid_layers = grid_layers
        self.token_zoom = token_zoom
        self.zooms = list(zooms)
        self.multi_shift = multi_shift
        self.direction = direction

        if not self.zooms:
            raise ValueError("ShiftGroupsBlockWrapOperation requires at least one zoom.")

        if not multi_shift and str(int(token_zoom + 1)) not in grid_layers:
            raise ValueError(
                f"ShiftGroupsBlockWrapOperation requires grid layer `{token_zoom + 1}` when multi_shift is False."
            )

        if multi_shift:
            missing = [zoom for zoom in self.zooms if str(int(zoom)) not in grid_layers]
            if missing:
                raise ValueError(
                    "ShiftGroupsBlockWrapOperation requires grid layers for all shifted zooms, "
                    f"missing {missing}."
                )

    def pre(
        self,
        x_zooms_groups: ZoomGroups,
        context: BlockWrapContext,
    ) -> Tuple[ZoomGroups, None]:
        return self._apply_shift(x_zooms_groups, sample_configs=context.sample_configs, reverse=False), None

    def post(
        self,
        x_zooms_groups: ZoomGroups,
        state: None,
        context: BlockWrapContext,
    ) -> ZoomGroups:
        del state
        return self._apply_shift(x_zooms_groups, sample_configs=context.sample_configs, reverse=True)

    def _apply_shift(
        self,
        x_zooms_groups: ZoomGroups,
        *,
        sample_configs: Mapping[int, Dict[str, Any]],
        reverse: bool,
    ) -> ZoomGroups:
        for group_idx, x_zooms in enumerate(x_zooms_groups):
            for zoom in self.zooms:
                if zoom not in x_zooms:
                    raise ValueError(
                        f"ShiftGroupsBlockWrapOperation requires zoom {zoom} in group {group_idx}."
                    )

                if zoom not in sample_configs:
                    raise ValueError(
                        f"ShiftGroupsBlockWrapOperation requires sample_configs for zoom {zoom}."
                    )

                grid_layer = (
                    self.grid_layers[str(int(zoom))]
                    if self.multi_shift
                    else self.grid_layers[str(int(self.token_zoom + 1))]
                )
                x_zooms[zoom] = grid_layer.apply_shift(
                    x_zooms[zoom],
                    self.direction,
                    **sample_configs[zoom],
                    reverse=reverse,
                )[0]
            x_zooms_groups[group_idx] = x_zooms

        return x_zooms_groups


class MergeGroupsBlockWrapConfig(BlockWrapConfig):
    operation_kind = "merge_groups"

    def __init__(
        self,
        variable_embedder_mode: str = "unique_per_depth",
        depth_indices: Union[int, Sequence[Any], Mapping[int, Any]] = -1,
        block_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.variable_embedder_mode: str
        self.depth_indices: Union[int, Sequence[Any], Mapping[int, Any]]
        self.block_kwargs: Dict[str, Any]

        inputs = copy.deepcopy(locals())
        for input_name, value in inputs.items():
            if input_name == "kwargs":
                for kw_name, kw_value in value.items():
                    setattr(self, kw_name, kw_value)
            elif input_name == "block_kwargs":
                setattr(self, input_name, {} if value is None else dict(value))
            else:
                setattr(self, input_name, value)

    def build(
        self,
        *,
        grid_layers: nn.ModuleDict,
    ) -> BlockWrapOperation:
        del grid_layers
        return MergeGroupsBlockWrapOperation(
            variable_embedder_mode=self.variable_embedder_mode,
            depth_indices=self.depth_indices,
        )

    def get_block_build_overrides(
        self,
        *,
        n_groups_variables: Sequence[int],
        n_groups_depths: Sequence[int],
        base_block_kwargs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        merge_layout = _resolve_merge_group_layout(
            n_groups_variables=n_groups_variables,
            n_groups_depths=n_groups_depths,
            variable_embedder_mode=self.variable_embedder_mode,
            depth_indices=self.depth_indices,
        )
        shared_indexed_group_variables = list(
            base_block_kwargs.get("shared_indexed_group_variables", [False] * len(n_groups_variables))
        )
        shared_indexed_group_depths = list(
            base_block_kwargs.get("shared_indexed_group_depths", [False] * len(n_groups_variables))
        )
        shared_indexed_group_space = list(
            base_block_kwargs.get("shared_indexed_group_space", [False] * len(n_groups_variables))
        )
        overrides = {
            "n_groups_variables": merge_layout["n_groups_variables"],
            "n_groups_depths": merge_layout["n_groups_depths"],
            "shared_indexed_group_variables": [
                shared_indexed_group_variables[group_idx]
                for group_idx in merge_layout["stage_group_original_indices"]
            ],
            "shared_indexed_group_depths": [
                shared_indexed_group_depths[group_idx]
                for group_idx in merge_layout["stage_group_original_indices"]
            ],
            "shared_indexed_group_space": [
                shared_indexed_group_space[group_idx]
                for group_idx in merge_layout["stage_group_original_indices"]
            ],
        }
        overrides.update(self.block_kwargs)
        return overrides


class MergeGroupsBlockWrapOperation(BlockWrapOperation):
    operation_kind = "merge_groups"
    _VARIABLE_ID_KEYS = ("VariableEmbedder", "variables_sampled", "MGEmbedder")

    def __init__(
        self,
        variable_embedder_mode: str = "unique_per_depth",
        depth_indices: Union[int, Sequence[Any], Mapping[int, Any]] = -1,
    ) -> None:
        super().__init__()
        if variable_embedder_mode not in {"unique_per_depth", "shared_across_depth"}:
            raise ValueError(
                "variable_embedder_mode must be `unique_per_depth` or `shared_across_depth`, "
                f"got `{variable_embedder_mode}`."
            )
        self.variable_embedder_mode = variable_embedder_mode
        self.depth_indices = depth_indices

    def pre(
        self,
        x_zooms_groups: ZoomGroups,
        context: BlockWrapContext,
    ) -> Tuple[ZoomGroups, MergeGroupsBlockWrapState]:
        if len(x_zooms_groups) <= 1:
            return x_zooms_groups, MergeGroupsBlockWrapState(
                original_mask_groups=context.mask_groups,
                original_emb_groups=context.emb_groups,
                group_shapes=[{zoom: tensor.shape for zoom, tensor in x_zooms.items()} for x_zooms in x_zooms_groups],
                zoom_order=sorted(x_zooms_groups[0].keys()) if x_zooms_groups else [],
                selected_depth_indices={},
                remaining_depth_indices={},
                stage_group_original_indices=list(range(len(x_zooms_groups))),
            )

        zoom_order = _validate_group_zoom_keys(x_zooms_groups)
        group_shapes = [{zoom: tensor.shape for zoom, tensor in x_zooms.items()} for x_zooms in x_zooms_groups]
        group_n_variables, group_n_depths = _extract_group_layout_from_shapes(group_shapes, zoom_order)
        merge_layout = _resolve_merge_group_layout(
            n_groups_variables=group_n_variables,
            n_groups_depths=group_n_depths,
            variable_embedder_mode=self.variable_embedder_mode,
            depth_indices=self.depth_indices,
        )
        selected_depth_indices = merge_layout["selected_depth_indices"]
        remaining_depth_indices = merge_layout["remaining_depth_indices"]
        stage_group_original_indices = merge_layout["stage_group_original_indices"]

        if not any(selected_depth_indices.values()):
            return x_zooms_groups, MergeGroupsBlockWrapState(
                original_mask_groups=context.mask_groups,
                original_emb_groups=context.emb_groups,
                group_shapes=group_shapes,
                zoom_order=zoom_order,
                selected_depth_indices=selected_depth_indices,
                remaining_depth_indices=remaining_depth_indices,
                stage_group_original_indices=stage_group_original_indices,
            )

        if group_n_depths[0] != 1:
            raise ValueError(
                "MergeGroupsBlockWrapOperation requires group 0 to have depth size 1 when merging donor depths into it, "
                f"got n_groups_depths[0]={group_n_depths[0]}."
            )

        merged_group0: ZoomGroup = {}
        for zoom in zoom_order:
            pieces = [x_zooms_groups[0][zoom]]
            for group_idx in range(1, len(x_zooms_groups)):
                if not selected_depth_indices[group_idx]:
                    continue
                pieces.append(
                    _fold_depth_into_variable(
                        _select_depth_indices(x_zooms_groups[group_idx][zoom], selected_depth_indices[group_idx])
                    )
                )
            merged_group0[zoom] = torch.cat(pieces, dim=1)

        merged_groups: ZoomGroups = [merged_group0]
        for group_idx in stage_group_original_indices[1:]:
            merged_groups.append(
                {
                    zoom: _select_depth_indices(x_zooms_groups[group_idx][zoom], remaining_depth_indices[group_idx])
                    for zoom in zoom_order
                }
            )

        merged_mask_groups = None
        if context.mask_groups is not None:
            if any(mask_zooms is None for mask_zooms in context.mask_groups):
                raise ValueError(
                    "MergeGroupsBlockWrapOperation requires mask_groups to be either absent or present for every group."
                )
            merged_group0_masks: Dict[int, torch.Tensor] = {}
            for zoom in zoom_order:
                mask_pieces = [context.mask_groups[0][zoom]]
                for group_idx in range(1, len(context.mask_groups)):
                    if not selected_depth_indices[group_idx]:
                        continue
                    mask_pieces.append(
                        _fold_depth_into_variable(
                            _select_depth_indices(context.mask_groups[group_idx][zoom], selected_depth_indices[group_idx])
                        )
                    )
                merged_group0_masks[zoom] = torch.cat(mask_pieces, dim=1)

            merged_mask_groups = [merged_group0_masks]
            for group_idx in stage_group_original_indices[1:]:
                merged_mask_groups.append(
                    {
                        zoom: _select_depth_indices(context.mask_groups[group_idx][zoom], remaining_depth_indices[group_idx])
                        for zoom in zoom_order
                    }
                )

        merged_emb_groups = None
        if context.emb_groups is not None:
            merged_emb_groups = self._merge_emb_groups(
                context.emb_groups,
                group_shapes,
                selected_depth_indices,
                remaining_depth_indices,
                stage_group_original_indices,
            )

        state = MergeGroupsBlockWrapState(
            original_mask_groups=context.mask_groups,
            original_emb_groups=context.emb_groups,
            group_shapes=group_shapes,
            zoom_order=zoom_order,
            selected_depth_indices=selected_depth_indices,
            remaining_depth_indices=remaining_depth_indices,
            stage_group_original_indices=stage_group_original_indices,
        )
        context.mask_groups = merged_mask_groups
        context.emb_groups = merged_emb_groups
        return merged_groups, state

    def post(
        self,
        x_zooms_groups: ZoomGroups,
        state: MergeGroupsBlockWrapState,
        context: BlockWrapContext,
    ) -> ZoomGroups:
        context.mask_groups = state.original_mask_groups
        context.emb_groups = state.original_emb_groups

        if len(state.group_shapes) <= 1 or not any(state.selected_depth_indices.values()):
            return x_zooms_groups

        if len(x_zooms_groups) != len(state.stage_group_original_indices):
            raise ValueError(
                "MergeGroupsBlockWrapOperation.post expects the stage group count produced in pre, "
                f"got {len(x_zooms_groups)} but expected {len(state.stage_group_original_indices)}."
            )

        stage_groups_by_original = {
            original_idx: x_zooms_groups[stage_idx]
            for stage_idx, original_idx in enumerate(state.stage_group_original_indices)
        }
        merged_group0 = stage_groups_by_original[0]
        restored_groups: ZoomGroups = []
        for group_idx, group_shapes in enumerate(state.group_shapes):
            restored_group: ZoomGroup = {}
            for zoom in state.zoom_order:
                if zoom not in merged_group0:
                    continue
                shape = group_shapes[zoom]
                if group_idx == 0:
                    flat_width = int(shape[1])
                    restored_group[zoom] = merged_group0[zoom][:, :flat_width]
                    continue

                selected_indices = state.selected_depth_indices[group_idx]
                remaining_indices = state.remaining_depth_indices[group_idx]
                if not selected_indices:
                    restored_group[zoom] = stage_groups_by_original[group_idx][zoom]
                    continue

                selected_chunk = _extract_group0_merged_chunk(
                    merged_group0[zoom],
                    group_shapes_by_group=state.group_shapes,
                    selected_depth_indices=state.selected_depth_indices,
                    zoom=zoom,
                    group_idx=group_idx,
                )
                selected_tensor = _unfold_selected_depth_chunk(
                    selected_chunk,
                    n_variables=int(shape[1]),
                    n_depths_selected=len(selected_indices),
                )
                remaining_tensor = (
                    None
                    if not remaining_indices
                    else stage_groups_by_original[group_idx][zoom]
                )
                restored_group[zoom] = _restore_group_depth_layout(
                    n_depths_total=int(shape[4]),
                    selected_indices=selected_indices,
                    selected_tensor=selected_tensor,
                    remaining_indices=remaining_indices,
                    remaining_tensor=remaining_tensor,
                )
            restored_groups.append(restored_group)

        return restored_groups

    def _merge_emb_groups(
        self,
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        group_shapes: Sequence[Dict[int, torch.Size]],
        selected_depth_indices: Mapping[int, List[int]],
        remaining_depth_indices: Mapping[int, List[int]],
        stage_group_original_indices: Sequence[int],
    ) -> List[Optional[Dict[str, Any]]]:
        if not emb_groups:
            return [None for _ in stage_group_original_indices]

        merged_group0_sources = [
            group_idx
            for group_idx in range(len(emb_groups))
            if group_idx == 0 or selected_depth_indices.get(group_idx)
        ]
        merged_group0 = self._build_group0_emb_group(
            emb_groups,
            group_shapes,
            selected_depth_indices,
            merged_group0_sources,
        )

        merged_emb_groups: List[Optional[Dict[str, Any]]] = [merged_group0]
        for group_idx in stage_group_original_indices[1:]:
            emb_group = emb_groups[group_idx]
            if emb_group is None:
                merged_emb_groups.append(None)
                continue
            merged_emb_groups.append(
                _slice_emb_group_depths(
                    emb_group,
                    group_shapes[group_idx],
                    remaining_depth_indices[group_idx],
                )
            )

        return merged_emb_groups

    def _build_group0_emb_group(
        self,
        emb_groups: Sequence[Optional[Dict[str, Any]]],
        group_shapes: Sequence[Dict[int, torch.Size]],
        selected_depth_indices: Mapping[int, List[int]],
        source_group_indices: Sequence[int],
    ) -> Optional[Dict[str, Any]]:
        source_emb = next(
            (emb_groups[group_idx] for group_idx in source_group_indices if emb_groups[group_idx] is not None),
            None,
        )
        if source_emb is None:
            return None

        merged_emb = {
            key: copy.deepcopy(value)
            for key, value in source_emb.items()
            if key not in self._VARIABLE_ID_KEYS and key not in {"variable_names_sampled", GLOBAL_EMBEDDER_CACHE_KEY}
        }

        variable_name_pieces: List[str] = []
        merged_id_values: Dict[str, List[torch.Tensor]] = {key: [] for key in self._VARIABLE_ID_KEYS}
        next_unique_id = 0

        for group_idx in source_group_indices:
            emb_group = emb_groups[group_idx]
            if emb_group is None:
                continue

            group_shape = next(iter(group_shapes[group_idx].values()))
            n_variables = int(group_shape[1])
            if group_idx == 0:
                contribution_depths = 1
            else:
                contribution_depths = len(selected_depth_indices[group_idx])
            if contribution_depths <= 0:
                continue

            key_tensor = None
            for key in self._VARIABLE_ID_KEYS:
                if key in emb_group:
                    key_tensor = _get_variable_id_tensor({key: emb_group[key]}, n_variables=n_variables)
                    break

            if self.variable_embedder_mode == "unique_per_depth":
                if key_tensor is None:
                    raise KeyError(
                        "MergeGroupsBlockWrapOperation requires one of `variables_sampled`, `VariableEmbedder`, or "
                        "`MGEmbedder` for groups merged into group 0 when variable_embedder_mode='unique_per_depth'."
                    )
                merged_ids_unique = torch.arange(
                    next_unique_id,
                    next_unique_id + n_variables * contribution_depths,
                    device=key_tensor.device,
                    dtype=key_tensor.dtype,
                ).view(1, -1).expand(key_tensor.shape[0], -1)
                next_unique_id += n_variables * contribution_depths

            for key in self._VARIABLE_ID_KEYS:
                if key not in emb_group:
                    continue
                key_ids = _get_variable_id_tensor({key: emb_group[key]}, n_variables=n_variables)
                if self.variable_embedder_mode == "unique_per_depth":
                    merged_id_values[key].append(merged_ids_unique)
                elif group_idx == 0:
                    merged_id_values[key].append(key_ids)
                else:
                    merged_id_values[key].append(
                        key_ids.unsqueeze(-1).expand(-1, -1, contribution_depths).reshape(key_ids.shape[0], -1)
                    )

            if "variable_names_sampled" in emb_group:
                names = list(emb_group["variable_names_sampled"])
                if len(names) != n_variables:
                    raise ValueError(
                        f"variable_names_sampled must have length {n_variables}, got {len(names)} for group {group_idx}."
                    )
                if group_idx == 0:
                    variable_name_pieces.extend(names)
                else:
                    for name in names:
                        variable_name_pieces.extend([name] * contribution_depths)

        for key, pieces in merged_id_values.items():
            if pieces:
                merged_emb[key] = torch.cat(pieces, dim=1)

        if variable_name_pieces:
            merged_emb["variable_names_sampled"] = variable_name_pieces

        return merged_emb


def _resolve_shift_zooms(
    zooms: Optional[Sequence[int]],
    q_zooms: Union[Sequence[int], int],
    kv_zooms: Union[Sequence[int], int],
) -> List[int]:
    if zooms is not None:
        return [int(zoom) for zoom in zooms]

    ordered_zooms: List[int] = []
    for zoom_collection in (q_zooms, kv_zooms):
        if isinstance(zoom_collection, int):
            if zoom_collection == -1:
                continue
            ordered_zooms.append(int(zoom_collection))
            continue

        for zoom in zoom_collection:
            zoom_int = int(zoom)
            if zoom_int not in ordered_zooms:
                ordered_zooms.append(zoom_int)

    if not ordered_zooms:
        raise ValueError(
            "ShiftGroupsBlockWrapConfig requires either `zooms` or at least one configured q_zooms/kv_zooms entry."
        )

    return ordered_zooms


def _resolve_merge_depth_indices(
    depth_indices: Union[int, Sequence[Any], Mapping[int, Any]],
    *,
    n_groups: int,
    n_groups_depths: Sequence[int],
) -> Dict[int, List[int]]:
    if len(n_groups_depths) != n_groups:
        raise ValueError(
            "MergeGroupsBlockWrapOperation requires n_groups_depths to match the number of groups, "
            f"got {len(n_groups_depths)} and {n_groups}."
        )

    if depth_indices == -1:
        raw_indices: Dict[int, Any] = {group_idx: -1 for group_idx in range(1, n_groups)}
    elif isinstance(depth_indices, Mapping):
        raw_indices = {int(group_idx): value for group_idx, value in depth_indices.items()}
    elif isinstance(depth_indices, Sequence) and not isinstance(depth_indices, (str, bytes)):
        if len(depth_indices) != n_groups:
            raise ValueError(f"depth_indices must have length {n_groups}, got {len(depth_indices)}.")
        raw_indices = {group_idx: depth_indices[group_idx] for group_idx in range(n_groups)}
    else:
        raw_indices = {group_idx: depth_indices for group_idx in range(1, n_groups)}

    resolved: Dict[int, List[int]] = {0: []}
    for group_idx in range(1, n_groups):
        resolved[group_idx] = _normalize_depth_index_selection(
            raw_indices.get(group_idx, []),
            n_depths=int(n_groups_depths[group_idx]),
            group_idx=group_idx,
        )
    return resolved


def _normalize_depth_index_selection(
    selection: Any,
    *,
    n_depths: int,
    group_idx: int,
) -> List[int]:
    if selection in (None, False):
        return []
    if isinstance(selection, Sequence) and not isinstance(selection, (str, bytes)) and len(selection) == 0:
        return []
    if selection == -1:
        return list(range(n_depths))

    raw_indices = (
        list(selection)
        if isinstance(selection, Sequence) and not isinstance(selection, (str, bytes))
        else [selection]
    )
    resolved: List[int] = []
    seen: set[int] = set()
    for raw_idx in raw_indices:
        if not isinstance(raw_idx, int):
            raise TypeError(
                f"depth_indices entries must be integers for group {group_idx}, got {type(raw_idx).__name__}."
            )
        idx = raw_idx if raw_idx >= 0 else n_depths + raw_idx
        if idx < 0 or idx >= n_depths:
            raise IndexError(
                f"depth index {raw_idx} resolves to {idx} for group {group_idx}, but valid range is [0, {n_depths - 1}]."
            )
        if idx in seen:
            continue
        seen.add(idx)
        resolved.append(idx)
    return resolved


def _resolve_merge_group_layout(
    *,
    n_groups_variables: Sequence[int],
    n_groups_depths: Sequence[int],
    variable_embedder_mode: str,
    depth_indices: Union[int, Sequence[Any], Mapping[int, Any]],
) -> Dict[str, Any]:
    if len(n_groups_variables) != len(n_groups_depths):
        raise ValueError(
            "MergeGroupsBlockWrapConfig requires n_groups_variables and n_groups_depths to have the same length, "
            f"got {len(n_groups_variables)} and {len(n_groups_depths)}."
        )

    n_groups = len(n_groups_variables)
    selected_depth_indices = _resolve_merge_depth_indices(
        depth_indices,
        n_groups=n_groups,
        n_groups_depths=n_groups_depths,
    )
    remaining_depth_indices = {
        group_idx: [
            depth_idx
            for depth_idx in range(int(n_groups_depths[group_idx]))
            if depth_idx not in set(selected_depth_indices[group_idx])
        ]
        for group_idx in range(n_groups)
    }

    merged_group0_variables = int(n_groups_variables[0])
    for group_idx in range(1, n_groups):
        n_selected = len(selected_depth_indices[group_idx])
        if n_selected == 0:
            continue
        if variable_embedder_mode == "unique_per_depth":
            merged_group0_variables += int(n_groups_variables[group_idx]) * n_selected
        elif variable_embedder_mode == "shared_across_depth":
            merged_group0_variables += int(n_groups_variables[group_idx])
        else:
            raise ValueError(
                "variable_embedder_mode must be `unique_per_depth` or `shared_across_depth`, "
                f"got `{variable_embedder_mode}`."
            )

    stage_group_original_indices = [0]
    stage_n_groups_variables = [merged_group0_variables]
    stage_n_groups_depths = [int(n_groups_depths[0])]
    for group_idx in range(1, n_groups):
        if not remaining_depth_indices[group_idx]:
            continue
        stage_group_original_indices.append(group_idx)
        stage_n_groups_variables.append(int(n_groups_variables[group_idx]))
        stage_n_groups_depths.append(len(remaining_depth_indices[group_idx]))

    return {
        "n_groups_variables": stage_n_groups_variables,
        "n_groups_depths": stage_n_groups_depths,
        "selected_depth_indices": selected_depth_indices,
        "remaining_depth_indices": remaining_depth_indices,
        "stage_group_original_indices": stage_group_original_indices,
    }


def _fold_depth_into_variable(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 6:
        raise ValueError(f"Expected a 6D tensor shaped like (b, v, t, n, d, f), got {tuple(x.shape)}.")
    return x.permute(0, 1, 4, 2, 3, 5).reshape(x.shape[0], x.shape[1] * x.shape[4], x.shape[2], x.shape[3], 1, x.shape[5])


def _unfold_variable_into_depth(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if x.ndim != 6:
        raise ValueError(f"Expected a 6D tensor shaped like (b, vd, t, n, 1, f), got {tuple(x.shape)}.")
    b, v, t, n, d, f = shape
    return x.reshape(b, v, d, t, n, f).permute(0, 1, 3, 4, 2, 5)


def _select_depth_indices(x: torch.Tensor, depth_indices: Sequence[int]) -> torch.Tensor:
    if x.ndim != 6:
        raise ValueError(f"Expected a 6D tensor shaped like (b, v, t, n, d, f), got {tuple(x.shape)}.")
    if not depth_indices:
        raise ValueError("depth_indices must be non-empty when selecting depth slices.")
    depth_index_tensor = torch.as_tensor(depth_indices, device=x.device, dtype=torch.long)
    return x.index_select(4, depth_index_tensor)


def _extract_group_layout_from_shapes(
    group_shapes: Sequence[Dict[int, torch.Size]],
    zoom_order: Sequence[int],
) -> Tuple[List[int], List[int]]:
    n_groups_variables: List[int] = []
    n_groups_depths: List[int] = []
    for group_idx, shape_by_zoom in enumerate(group_shapes):
        if not zoom_order:
            n_groups_variables.append(0)
            n_groups_depths.append(0)
            continue
        reference_shape = shape_by_zoom[zoom_order[0]]
        n_variables = int(reference_shape[1])
        n_depths = int(reference_shape[4])
        for zoom in zoom_order[1:]:
            shape = shape_by_zoom[zoom]
            if int(shape[1]) != n_variables or int(shape[4]) != n_depths:
                raise ValueError(
                    "MergeGroupsBlockWrapOperation requires each group to keep the same variable and depth counts "
                    f"across zooms, but group {group_idx} has {(n_variables, n_depths)} at zoom {zoom_order[0]} "
                    f"and {(int(shape[1]), int(shape[4]))} at zoom {zoom}."
                )
        n_groups_variables.append(n_variables)
        n_groups_depths.append(n_depths)
    return n_groups_variables, n_groups_depths


def _extract_group0_merged_chunk(
    merged_group0_zoom: torch.Tensor,
    *,
    group_shapes_by_group: Sequence[Dict[int, torch.Size]],
    selected_depth_indices: Mapping[int, List[int]],
    zoom: int,
    group_idx: int,
) -> torch.Tensor:
    offset = int(group_shapes_by_group[0][zoom][1])
    for donor_group_idx in range(1, group_idx):
        offset += int(group_shapes_by_group[donor_group_idx][zoom][1]) * len(selected_depth_indices[donor_group_idx])

    width = int(group_shapes_by_group[group_idx][zoom][1]) * len(selected_depth_indices[group_idx])
    return merged_group0_zoom[:, offset:offset + width]


def _unfold_selected_depth_chunk(
    x: torch.Tensor,
    *,
    n_variables: int,
    n_depths_selected: int,
) -> torch.Tensor:
    if x.ndim != 6:
        raise ValueError(f"Expected a 6D tensor shaped like (b, vk, t, n, 1, f), got {tuple(x.shape)}.")
    return x.reshape(x.shape[0], n_variables, n_depths_selected, x.shape[2], x.shape[3], x.shape[5]).permute(0, 1, 3, 4, 2, 5)


def _restore_group_depth_layout(
    *,
    n_depths_total: int,
    selected_indices: Sequence[int],
    selected_tensor: torch.Tensor,
    remaining_indices: Sequence[int],
    remaining_tensor: Optional[torch.Tensor],
) -> torch.Tensor:
    if selected_tensor.ndim != 6:
        raise ValueError(
            f"Expected selected_tensor to have shape (b, v, t, n, d_selected, f), got {tuple(selected_tensor.shape)}."
        )
    b, v, t, n, _, f = selected_tensor.shape
    restored = selected_tensor.new_empty((b, v, t, n, n_depths_total, f))
    if remaining_indices:
        if remaining_tensor is None:
            raise ValueError("remaining_tensor must be provided when remaining_indices is non-empty.")
        if remaining_tensor.ndim != 6:
            raise ValueError(
                f"Expected remaining_tensor to have shape (b, v, t, n, d_remaining, f), got {tuple(remaining_tensor.shape)}."
            )
        if (
            remaining_tensor.shape[0] != b
            or remaining_tensor.shape[1] != v
            or remaining_tensor.shape[2] != t
            or remaining_tensor.shape[3] != n
            or remaining_tensor.shape[5] != f
        ):
            raise ValueError(
                "remaining_tensor must match selected_tensor across batch, variable, time, space, and feature axes, "
                f"got {tuple(remaining_tensor.shape)} vs {tuple(selected_tensor.shape)}."
            )
        restored[:, :, :, :, list(remaining_indices), :] = remaining_tensor
    restored[:, :, :, :, list(selected_indices), :] = selected_tensor
    return restored


def _slice_emb_group_depths(
    emb_group: Dict[str, Any],
    group_shape: Dict[int, torch.Size],
    remaining_depth_indices: Sequence[int],
) -> Dict[str, Any]:
    sliced = copy.deepcopy(emb_group)
    reference_shape = next(iter(group_shape.values()))
    n_depths = int(reference_shape[4])

    if "PressureLevelEmbedder" in sliced and torch.is_tensor(sliced["PressureLevelEmbedder"]):
        pressure_levels = sliced["PressureLevelEmbedder"]
        if pressure_levels.ndim == 1 and pressure_levels.shape[0] == n_depths:
            depth_index_tensor = torch.as_tensor(remaining_depth_indices, device=pressure_levels.device, dtype=torch.long)
            sliced["PressureLevelEmbedder"] = pressure_levels.index_select(0, depth_index_tensor)
        elif pressure_levels.ndim > 1 and pressure_levels.shape[-1] == n_depths:
            depth_index_tensor = torch.as_tensor(remaining_depth_indices, device=pressure_levels.device, dtype=torch.long)
            sliced["PressureLevelEmbedder"] = pressure_levels.index_select(-1, depth_index_tensor)

    if "GroupDepthEmbedder" in sliced:
        group_depth_value = sliced["GroupDepthEmbedder"]
        if isinstance(group_depth_value, tuple) and len(group_depth_value) == 2:
            group_id, depth_ids = group_depth_value
            if torch.is_tensor(depth_ids) and depth_ids.ndim == 1 and depth_ids.shape[0] == n_depths:
                depth_index_tensor = torch.as_tensor(remaining_depth_indices, device=depth_ids.device, dtype=torch.long)
                sliced["GroupDepthEmbedder"] = (group_id, depth_ids.index_select(0, depth_index_tensor))

    return sliced


def _validate_group_zoom_keys(x_zooms_groups: Sequence[ZoomGroup]) -> List[int]:
    if not x_zooms_groups:
        return []

    zoom_order = sorted(x_zooms_groups[0].keys())
    for group_idx, x_zooms in enumerate(x_zooms_groups[1:], start=1):
        current_zoom_order = sorted(x_zooms.keys())
        if current_zoom_order != zoom_order:
            raise ValueError(
                "MergeGroupsBlockWrapOperation requires matching zoom keys across groups, "
                f"but group 0 has {zoom_order} and group {group_idx} has {current_zoom_order}."
            )
    return zoom_order


def _get_variable_id_tensor(emb_group: Mapping[str, Any], *, n_variables: int) -> torch.Tensor:
    for key in ("variables_sampled", "VariableEmbedder", "MGEmbedder"):
        if key not in emb_group:
            continue
        value = emb_group[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"`{key}` must be a tensor, got {type(value).__name__}.")
        if value.ndim == 1:
            value = value.view(1, -1)
        if value.ndim != 2 or value.shape[1] != n_variables:
            raise ValueError(
                f"`{key}` must have shape (batch, {n_variables}), got {tuple(value.shape)}."
            )
        return value
    raise KeyError("MergeGroupsBlockWrapOperation requires one of `variables_sampled`, `VariableEmbedder`, or `MGEmbedder`.")


def _embedding_values_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, Mapping):
        if left.keys() != right.keys():
            return False
        return all(_embedding_values_equal(left[key], right[key]) for key in left.keys())
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_embedding_values_equal(left_item, right_item) for left_item, right_item in zip(left, right))
    return left == right


def _merge_group_variable_counts(
    *,
    n_groups_variables: Sequence[int],
    n_groups_depths: Sequence[int],
    variable_embedder_mode: str,
) -> List[int]:
    if len(n_groups_variables) != len(n_groups_depths):
        raise ValueError(
            "MergeGroupsBlockWrapConfig requires n_groups_variables and n_groups_depths to have the same length, "
            f"got {len(n_groups_variables)} and {len(n_groups_depths)}."
        )

    if variable_embedder_mode == "unique_per_depth":
        return [int(sum(int(n_variables) * int(n_depths) for n_variables, n_depths in zip(n_groups_variables, n_groups_depths)))]

    if variable_embedder_mode == "shared_across_depth":
        return [int(sum(int(n_variables) for n_variables in n_groups_variables))]

    raise ValueError(
        "variable_embedder_mode must be `unique_per_depth` or `shared_across_depth`, "
        f"got `{variable_embedder_mode}`."
    )
