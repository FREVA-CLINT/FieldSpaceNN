import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from ...modules.field_space.field_space_base import ConservativeLayer,ConservativeLayerConfig
from ...modules.field_space.field_space_layer import FieldSpaceLayerModule, FieldSpaceLayerConfig
from ...modules.field_space.field_space_attention import FieldSpaceAttentionModule,FieldSpaceAttentionConfig
from ...modules.field_space.healpix_convolution import MultiZoomHealpixConvBase, MultiZoomHealpixConvConfig
from ...modules.grids.grid_layer import GridLayer
from ...utils.helpers import check_get

defaults = {
    "predict_var":False,
    'n_head_channels': 32,
    'att_dim': 256,
    'layer_confs': {},
    'layer_confs_emb': {},
    'input_layer_confs': {},
    'embed_confs': {},
    'dropout': 0,
    'learn_residual': False,
    'with_residual': False,
    'masked_residual': False,
    'use_mask': False
}

def create_missing_zooms(
    in_zooms_groups: Optional[Sequence[Optional[Mapping[Any, torch.Tensor]]]],
    in_zooms: Sequence[int],
    mask_zooms_groups: Optional[Sequence[Optional[Mapping[Any, torch.Tensor]]]] = None,
    embedding_groups: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    sample_configs: Optional[Mapping[Any, Any]] = None,
) -> Union[
    List[Optional[Dict[int, torch.Tensor]]],
    Tuple[
        List[Optional[Dict[int, torch.Tensor]]],
        Optional[List[Optional[Dict[int, torch.Tensor]]]],
        Optional[List[Optional[Dict[str, Any]]]],
        Optional[Dict[Any, Any]],
    ],
]:
    """
    Create missing zoom levels for data groups and optionally for mask/embedding/config groups.

    Data tensors are filled with zeros. Mask tensors are filled with either ``True``
    (for bool masks) or ``1.0`` (for floating masks), preserving mask dtype.
    Embedding dictionaries are only extended for zoom-keyed sub-mappings
    (e.g. ``TimeEmbedder``); non-zoom entries (e.g. ``VariableEmbedder`` tensors)
    are left unchanged. For ``sample_configs``, missing zoom entries are copied from
    the highest existing zoom level.
    """

    def _normalize_zoom_tensor_dict(group: Optional[Mapping[Any, torch.Tensor]]) -> Dict[int, torch.Tensor]:
        return {int(zoom): tensor for zoom, tensor in group.items()}

    def _compute_n_cells(ref_ncells: int, ref_zoom: int, zoom: int) -> int:
        delta = int(zoom) - int(ref_zoom)
        if delta >= 0:
            return int(ref_ncells * (4 ** delta))
        return int(ref_ncells // (4 ** (-delta)))

    def _copy_embedding_zoom_entries(
        emb_group: Optional[Dict[str, Any]],
        missing_zooms: Sequence[int],
        existing_zooms: Sequence[int],
    ) -> Dict[str, Any]:
        emb_group_out = dict(emb_group)
        existing_zooms_set = {int(zoom) for zoom in existing_zooms}

        for emb_key, emb_val in emb_group.items():
            if not isinstance(emb_val, Mapping):
                continue

            zoom_map = {}
            for zoom_key, zoom_value in emb_val.items():
                try:
                    zoom_map[int(zoom_key)] = zoom_value
                except (TypeError, ValueError):
                    zoom_map = {}
                    break

            if not zoom_map:
                continue

            # Do not touch mappings that are not zoom keyed.
            if not (set(zoom_map.keys()) & existing_zooms_set):
                continue

            ref_zoom_emb = max(zoom_map.keys())
            for zoom in missing_zooms:
                if zoom not in zoom_map:
                    zoom_map[zoom] = zoom_map[ref_zoom_emb]

            emb_group_out[emb_key] = {zoom: zoom_map[zoom] for zoom in sorted(zoom_map.keys())}

        return emb_group_out

    output_groups, output_mask_groups, output_embedding_groups = [], [], []
    output_sample_configs = {}
    for key, value in sample_configs.items():
        output_sample_configs[key] = copy.deepcopy(value)

    for group_idx, group in enumerate(in_zooms_groups):
        x_zooms = _normalize_zoom_tensor_dict(group)

        ref_zoom = max(x_zooms.keys())
        ref_tensor = x_zooms[ref_zoom]

        missing_zooms: List[int] = []
        ref_ncells = int(ref_tensor.shape[3])
        for zoom in in_zooms:
            if zoom in x_zooms:
                continue

            missing_zooms.append(zoom)
            n_cells = _compute_n_cells(ref_ncells, ref_zoom, zoom)
            target_shape = list(ref_tensor.shape)
            target_shape[3] = n_cells
            x_zooms[zoom] = ref_tensor.new_zeros(tuple(target_shape))

        output_groups.append({zoom: x_zooms[zoom] for zoom in sorted(x_zooms.keys())})

        if output_mask_groups is not None:
            mask_group_in = mask_zooms_groups[group_idx] if group_idx < len(mask_zooms_groups) else None
            mask_zooms = _normalize_zoom_tensor_dict(mask_group_in)
            ref_zoom_mask = max(mask_zooms.keys())
            ref_mask = mask_zooms[ref_zoom_mask]
            for zoom in missing_zooms:
                if zoom in mask_zooms:
                    continue
                target_mask_shape = list(x_zooms[zoom].shape)
                fill_value: Union[bool, float] = True if ref_mask.dtype == torch.bool else 1.0
                mask_zooms[zoom] = ref_mask.new_full(tuple(target_mask_shape), fill_value)

            output_mask_groups.append({zoom: mask_zooms[zoom] for zoom in sorted(mask_zooms.keys())})

        emb_group_in = embedding_groups[group_idx] if group_idx < len(embedding_groups) else None
        output_embedding_groups.append(
            _copy_embedding_zoom_entries(emb_group_in, missing_zooms, existing_zooms=x_zooms.keys())
        )

    sample_zoom_keys = [key for key in output_sample_configs.keys() if isinstance(key, int)]
    ref_zoom_cfg = max(sample_zoom_keys)
    for zoom in in_zooms:
        if zoom not in output_sample_configs:
            output_sample_configs[zoom] = copy.deepcopy(output_sample_configs[ref_zoom_cfg])

    return output_groups, output_mask_groups, output_embedding_groups, output_sample_configs

def create_encoder_decoder_block(
    block_conf: Any,
    in_zooms: Sequence[int],
    in_features: Sequence[int],
    n_groups_variables: Sequence[int],
    grid_layers: nn.ModuleDict,
    **kwargs: Any,
) -> nn.Module:
    """
    Build an encoder or decoder block based on the configuration type.

    :param block_conf: Block configuration object (e.g., FieldLayerConfig).
    :param in_zooms: Input zoom levels for the block.
    :param in_features: Feature counts per zoom.
    :param n_groups_variables: Number of variable groups for attention layers.
    :param grid_layers: Grid layers used to map spatial neighborhoods.
    :param kwargs: Additional configuration overrides.
    :return: Instantiated block module with ``out_features`` set.
    """
    embed_confs = check_get([block_conf, kwargs, defaults], "embed_confs")
    layer_confs = check_get([block_conf, kwargs, defaults], "layer_confs")
    layer_confs_emb = check_get([block_conf, kwargs, defaults], "layer_confs_emb")
    dropout = check_get([block_conf, kwargs, defaults], "dropout")
    out_zooms = check_get([block_conf, {'out_zooms':in_zooms}], "out_zooms")
    use_mask = check_get([block_conf, kwargs, defaults], "use_mask")
    n_head_channels = check_get([block_conf,kwargs,defaults], "n_head_channels")
    att_dim = check_get([block_conf,kwargs,defaults], "att_dim")

    # Select the correct block implementation based on the config type.
    if isinstance(block_conf, ConservativeLayerConfig):
        block = ConservativeLayer(in_zooms)
        block.out_features = in_features
    
    elif isinstance(block_conf, FieldSpaceAttentionConfig):
        block = FieldSpaceAttentionModule(
                grid_layers,
                in_zooms,
                out_zooms,
                token_zoom = block_conf.token_zoom,
                q_zooms  = block_conf.q_zooms,
                kv_zooms = block_conf.kv_zooms,
                target_zooms = block_conf.target_zooms,
                use_mask = use_mask,
                refine_zooms= block_conf.refine_zooms,
                shift= block_conf.shift,
                multi_shift= block_conf.multi_shift,
                att_dim = att_dim,
                n_groups_variables = n_groups_variables,
                token_len_time = block_conf.token_len_time,
                token_len_depth = block_conf.token_len_depth,
                token_overlap_space = block_conf.token_overlap_space,
                token_overlap_time = block_conf.token_overlap_time,
                token_overlap_depth = block_conf.token_overlap_depth,
                token_overlap_mlp_time = block_conf.token_overlap_mlp_time,
                token_overlap_mlp_depth = block_conf.token_overlap_mlp_depth,
                rank_variables = block_conf.rank_variables,
                rank_space = block_conf.rank_space,
                rank_time = block_conf.rank_time,
                rank_depth = block_conf.rank_depth,
                rank_features = block_conf.rank_features,
                seq_len_zoom = block_conf.seq_len_zoom,
                seq_len_time =  block_conf.seq_len_time,
                seq_len_depth = block_conf.seq_len_depth,
                seq_overlap_space = block_conf.seq_overlap_space,
                seq_overlap_time = block_conf.seq_overlap_time,
                seq_overlap_depth = block_conf.seq_overlap_depth,
                with_var_att= block_conf.with_var_att,
                update = block_conf.update,
                dropout = dropout,
                n_head_channels = n_head_channels,
                embed_confs = embed_confs,
                separate_mlp_norm = block_conf.separate_mlp_norm,
                layer_confs=layer_confs,
                layer_confs_emb = layer_confs_emb)
        block.out_features = in_features

    elif isinstance(block_conf, MultiZoomHealpixConvConfig):
        block = MultiZoomHealpixConvBase(
            x_zooms=in_zooms,
            in_zooms=block_conf.in_zooms,
            target_zooms=block_conf.target_zooms,
            out_zooms=check_get([block_conf, {"out_zooms": block_conf.target_zooms}], "out_zooms"),
            in_features=in_features,
            target_features=check_get([block_conf, {"target_features": in_features}], "target_features"),
            share_weights=check_get([block_conf, {"share_weights": False}], "share_weights"),
            grid_layers=grid_layers,
            use_neighborhood=check_get([block_conf, {"use_neighborhood": True}], "use_neighborhood"),
            norm=check_get([block_conf, {"norm": "group"}], "norm"),
            act=check_get([block_conf, {"act": "silu"}], "act"),
            residual=check_get([block_conf, {"residual": True}], "residual"),
            num_groups=check_get([block_conf, {"num_groups": 8}], "num_groups"),
            eps=check_get([block_conf, {"eps": 1e-5}], "eps"),
        )

    elif isinstance(block_conf, FieldSpaceLayerConfig):
        block = FieldSpaceLayerModule(
                grid_layers,
                in_zooms,
                block_conf.in_zooms,
                block_conf.target_zooms,
                block_conf.field_zoom,
                out_zooms=block_conf.out_zooms,
                in_features=in_features,
                target_features=check_get([block_conf,{"target_features": in_features}], "target_features"),
                mult = block_conf.mult,
                hidden_dim = block_conf.hidden_dim,
                in_token_len_time = block_conf.in_token_len_time,
                in_token_len_depth = block_conf.in_token_len_depth,
                out_token_len_time = block_conf.out_token_len_time,
                out_token_len_depth = block_conf.out_token_len_depth,
                token_overlap_space = block_conf.token_overlap_space,
                token_overlap_time = block_conf.token_overlap_time,
                token_overlap_depth = block_conf.token_overlap_depth,
                residual = check_get([block_conf, {"residual": False}], "residual"),
                type= block_conf.type,
                layer_confs=layer_confs)
    return block

class MG_base_model(nn.Module):
    """
    Base class for multi-grid models with shared grid layers.
    """

    def __init__(
        self,
        mgrids: Sequence[Mapping[str, Any]],
    ) -> None:
        """
        Initialize grid layers for each zoom level in the multi-grid configuration.

        :param mgrids: List of grid dictionaries containing adjacency and coordinate data.
        :return: None.
        """
        super().__init__()

        # Create grid layers for each unique global level
        zooms = []
        self.grid_layers: nn.ModuleDict = nn.ModuleDict()
        for zoom, mgrid in enumerate(mgrids):
            self.grid_layers[str(int(zoom))] = GridLayer(zoom, mgrid['adjc'], mgrid['adjc_mask'], mgrid['coords'], coord_system='polar')
            zooms.append(zoom)

        self.register_buffer('zooms', torch.tensor(zooms), persistent=False)
        self.zoom_max: int = int(self.zooms[-1])

        self.grid_layer_max: GridLayer = self.grid_layers[str(int(self.zooms[-1]))]
