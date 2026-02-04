import torch
import torch.nn as nn

from hydra.utils import instantiate
from omegaconf import ListConfig
from typing import List,Dict

from ...utils.helpers import check_get
from einops import rearrange

from ...modules.multi_grid.mg_base import ConservativeLayer,ConservativeLayerConfig, DiffDecoder
from ...modules.multi_grid.field_layer import FieldLayer, FieldLayerConfig
from ...modules.multi_grid.field_attention import FieldAttentionModule,FieldAttentionConfig

from ...modules.grids.grid_utils import decode_zooms

from .confs import defaults

from .mg_base_model import MG_base_model

class MG_Transformer(MG_base_model):
    def __init__(self, 
                 mgrids,
                 block_configs: List,
                 in_zooms: List,
                 in_features: int=1,
                 n_groups_variables: List = [1],
                 **kwargs
                 ) -> None: 
        
        
        super().__init__(mgrids)

        self.in_zooms = in_zooms
        self.in_features = in_features 
       

        self.Blocks = nn.ModuleDict()

        in_features = [in_features]*len(in_zooms)


        for block_key, block_conf in block_configs.items():
            assert isinstance(block_key, str), "block keys should be strings"

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
                     self.grid_layers,
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
                     in_features= in_features[0],
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
        
                block = FieldLayer(
                        self.grid_layers,
                        in_zooms,
                        block_conf.in_zooms,
                        block_conf.target_zooms,
                        block_conf.field_zoom,
                        out_zooms=block_conf.out_zooms,
                        in_features=in_features,
                        target_features=check_get([block_conf,{"target_features": in_features}], "target_features"),
                        mult = block_conf.mult,
                        overlap= block_conf.overlap,
                        type= block_conf.type,
                        layer_confs=layer_confs)

            self.Blocks[block_key] = block     

            in_features = block.out_features
            in_zooms = block.out_zooms        

        block.out_features = [in_features[0]]
        
        self.masked_residual = check_get([kwargs, defaults], "masked_residual")
        self.learn_residual = check_get([kwargs, defaults], "learn_residual") if not self.masked_residual else True

        self.decoder = DiffDecoder()

        
    def decode(self, x_zooms:Dict[str, torch.Tensor], sample_configs: Dict, out_zoom: int=None, emb={}):
        return self.decoder(x_zooms, emb=emb, sample_configs=sample_configs, out_zoom=out_zoom)


    def forward(self, 
                x_zooms_groups: List[Dict[int, torch.Tensor]] = None,
                mask_zooms_groups: List[Dict[int, torch.Tensor]] = None,
                emb_groups: List[Dict] = None,
                sample_configs={},
                out_zoom=None):

        if x_zooms_groups is None:
            x_zooms_groups = []
        if mask_zooms_groups is None:
            mask_zooms_groups = [None] * len(x_zooms_groups)
        if emb_groups is None:
            emb_groups = [{} for _ in range(len(x_zooms_groups))]

        x_zooms_groups_res = None
        #if self.learn_residual:
        x_zooms_groups_res = [x.copy() for x in x_zooms_groups]

        mask_zooms_groups_res = None
        if self.masked_residual:
            mask_zooms_groups_res = [
                mask.copy() if mask is not None else None for mask in mask_zooms_groups
            ]

        for block in self.Blocks.values():
            x_zooms_groups = block(
                x_zooms_groups,
                sample_configs=sample_configs,
                mask_groups=mask_zooms_groups,
                emb_groups=emb_groups,
            )
            pass

        for i, x_zooms in enumerate(x_zooms_groups):

            x_res = x_zooms_groups_res[i] if self.learn_residual else None
            x_zooms_groups[i] = self.apply_residuals(x_zooms, x_res, mask_zooms_groups, mask_zooms_groups_res, sample_configs)

        if out_zoom is not None:
            for i, x_zooms in enumerate(x_zooms_groups):
                x_zooms_groups[i] = (
                    self.decoder(x_zooms, sample_configs=sample_configs, out_zoom=out_zoom)
                    if x_zooms
                    else {}
                )

        return x_zooms_groups


    def apply_residuals(self, x_zooms, x_zooms_res, mask_zooms, mask_res, sample_configs):
        if not x_zooms:
            return x_zooms

        if mask_res is None:
            mask_res = mask_zooms

        if self.learn_residual:
            for i, x_zooms_res in enumerate(x_zooms_res):
                if len(x_zooms_res) == 1 :
                    x_zooms_res = decode_zooms(
                        x_zooms_res,
                        sample_configs=sample_configs,
                        out_zoom=list(x_zooms_res.keys())[0],
                    )

        elif self.learn_residual:
            for zoom in x_zooms.keys():
                if not self.masked_residual or mask_zooms is None:
                    x_zooms[zoom] = x_zooms_res[zoom] + x_zooms[zoom]
                elif mask_zooms[zoom].dtype == torch.bool:
                    x_zooms[zoom] = (1 - 1. * mask_zooms[zoom]) * x_zooms_res[zoom] + (mask_zooms[zoom]) * x_zooms[zoom]
                else:
                    x_zooms[zoom] = mask_zooms[zoom] * x_zooms_res[zoom] + (1 - mask_res[zoom]) * x_zooms[zoom]

        return x_zooms
