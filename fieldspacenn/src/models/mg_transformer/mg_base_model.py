from typing import Any, Mapping, Sequence

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
        in_features_by_zoom = dict(zip(in_zooms, in_features))
        missing_zooms = [zoom for zoom in block_conf.in_zooms if zoom not in in_features_by_zoom]
        if missing_zooms:
            raise ValueError(
                f"MultiZoomHealpixConvConfig expects input zooms {missing_zooms} "
                f"which are not available in current in_zooms={list(in_zooms)}."
            )

        conv_in_features = [in_features_by_zoom[zoom] for zoom in block_conf.in_zooms]
        conv_target_features = check_get(
            [block_conf, {"target_features": conv_in_features}],
            "target_features",
        )

        block = MultiZoomHealpixConvBase(
            in_zooms=block_conf.in_zooms,
            target_zooms=block_conf.target_zooms,
            in_features=conv_in_features,
            target_features=conv_target_features,
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
