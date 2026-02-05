import torch
import torch.nn as nn
from ...modules.multi_grid.mg_base import ConservativeLayer,ConservativeLayerConfig
from ...modules.multi_grid.field_layer import FieldLayerModule, FieldLayerConfig
from ...modules.multi_grid.field_attention import FieldAttentionModule,FieldAttentionConfig
from ...modules.grids.grid_layer import GridLayer
from ...utils.helpers import check_get
from .confs import defaults


def create_encoder_decoder_block(block_conf, in_zooms, in_features, n_groups_variables,  grid_layers, **kwargs):
    embed_confs = check_get([block_conf, kwargs, defaults], "embed_confs")
    layer_confs = check_get([block_conf, kwargs, defaults], "layer_confs")
    layer_confs_emb = check_get([block_conf, kwargs, defaults], "layer_confs_emb")
    dropout = check_get([block_conf, kwargs, defaults], "dropout")
    out_zooms = check_get([block_conf, {'out_zooms':in_zooms}], "out_zooms")
    use_mask = check_get([block_conf, kwargs, defaults], "use_mask")
    n_head_channels = check_get([block_conf,kwargs,defaults], "n_head_channels")
    att_dim = check_get([block_conf,kwargs,defaults], "att_dim")

    if isinstance(block_conf, ConservativeLayerConfig):
        block = ConservativeLayer(in_zooms)
        block.out_features = in_features
    
    elif isinstance(block_conf, FieldAttentionConfig):
        block = FieldAttentionModule(
                grid_layers,
                in_zooms,
                out_zooms,
                token_zoom = block_conf.token_zoom,
                q_zooms  = block_conf.q_zooms,
                kv_zooms = block_conf.kv_zooms,
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

    elif isinstance(block_conf, FieldLayerConfig):
        block = FieldLayerModule(
                grid_layers,
                in_zooms,
                block_conf.in_zooms,
                block_conf.target_zooms,
                block_conf.field_zoom,
                out_zooms=block_conf.out_zooms,
                in_features=in_features,
                target_features=check_get([block_conf,{"target_features": in_features}], "target_features"),
                mult = block_conf.mult,
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
    def __init__(self,
                 mgrids
                 ) -> None:
        
                
        super().__init__()

        # Create grid layers for each unique global level
        zooms = []
        self.grid_layers = nn.ModuleDict()
        for zoom, mgrid in enumerate(mgrids):
            self.grid_layers[str(int(zoom))] = GridLayer(zoom, mgrid['adjc'], mgrid['adjc_mask'], mgrid['coords'], coord_system='polar')
            zooms.append(zoom)

        self.register_buffer('zooms', torch.tensor(zooms), persistent=False)
        self.zoom_max = int(self.zooms[-1])

        self.grid_layer_max = self.grid_layers[str(int(self.zooms[-1]))]