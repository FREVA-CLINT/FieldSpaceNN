import torch
import torch.nn as nn

from hydra.utils import instantiate
from omegaconf import ListConfig
from typing import List,Dict

from ...utils.helpers import check_get

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
                 out_features: int=1,
                 **kwargs
                 ) -> None: 
        
        
        super().__init__(mgrids)

        self.in_zooms = in_zooms
        self.in_features = in_features 
        predict_var = kwargs.get("predict_var", defaults['predict_var'])

        if predict_var:
            out_features = out_features * 2
            self.activation_var = nn.Softplus()

        self.out_features = out_features
        self.predict_var = predict_var

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
                block = ConservativeLayer(in_zooms,
                                          first_feature_only=self.predict_var)
                block.out_features = in_features
                        
            
            elif isinstance(block_conf, FieldAttentionConfig):

                block = FieldAttentionModule(
                     self.grid_layers,
                     in_zooms,
                     out_zooms,
                     token_zoom = block_conf.token_zoom,
                     seq_zoom = block_conf.seq_zoom,
                     q_zooms  = block_conf.q_zooms,
                     kv_zooms = block_conf.kv_zooms,
                     use_mask = use_mask,
                     refine_zooms= block_conf.refine_zooms,
                     shift= block_conf.shift,
                     rev_shift= block_conf.rev_shift,
                     multi_shift= block_conf.multi_shift,
                     att_dim = att_dim,
                     token_len_td = block_conf.token_len_td,
                     token_overlap_std = block_conf.token_overlap_std,
                     seq_len_td =  block_conf.seq_len_td,
                     seq_nh_std= block_conf.seq_nh_std,
                     mlp_token_overlap_td = block_conf.mlp_token_overlap_td,
                     with_var_att= block_conf.with_var_att,
                     update = block_conf.update,
                     dropout = dropout,
                     n_head_channels = n_head_channels,
                     embed_confs = embed_confs,
                     ranks_std = block_conf.ranks_std,
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


    def forward(self, x_zooms: Dict[int, torch.Tensor], sample_configs={}, mask_zooms: Dict[int, torch.Tensor]= None, emb=None, out_zoom=None):

        b, nv, nt, n, nd, f = x_zooms[list(x_zooms.keys())[0]].shape

        emb['MGEmbedder'] = emb['GroupEmbedder']

        if self.learn_residual:
            x_zooms_res = x_zooms.copy()
        
        if self.masked_residual:
            mask_res = mask_zooms.copy()

        for k, block in enumerate(self.Blocks.values()):
            
            x_zooms = block(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)


        if self.learn_residual and len(x_zooms.keys())==1:
            x_zooms_res = decode_zooms(x_zooms_res, sample_configs=sample_configs, out_zoom=list(x_zooms.keys())[0])

        if self.predict_var and self.learn_residual:
            for zoom, x in x_zooms.items():
                x, x_var = x.chunk(2,dim=-1) 

                if not self.masked_residual:
                    x = x_zooms_res[zoom] + x
                elif mask_zooms[zoom].dtype == torch.bool:
                    x = (1 - mask_zooms[zoom]) * x_zooms_res[zoom] + (mask_zooms[zoom]) * x
                else:
                    x = mask_zooms[zoom] * x_zooms_res[zoom] + (1 - mask_zooms[zoom]) * x

                x_zooms[zoom] = torch.concat((x, self.activation_var(x_var)),dim=-1)

        elif self.predict_var:
            for zoom, x in x_zooms.items():
                x, x_var = x.chunk(2,dim=-1) 
                x_zooms[zoom] = torch.concat((x, self.activation_var(x_var)),dim=-1)

        elif self.learn_residual:
            for zoom in x_zooms.keys():

                if not self.masked_residual:
                    x_zooms[zoom] = x_zooms_res[zoom] + x_zooms[zoom]
                elif mask_zooms[zoom].dtype == torch.bool:
                    x_zooms[zoom] = (1 - 1.*mask_zooms[zoom]) * x_zooms_res[zoom] + (mask_zooms[zoom]) * x_zooms[zoom]
                else:
                    x_zooms[zoom] = mask_zooms[zoom] * x_zooms_res[zoom] + (1 - mask_res[zoom]) * x_zooms[zoom]

        x_zooms = self.decoder(x_zooms, sample_configs=sample_configs, emb=emb, out_zoom=out_zoom)

        return x_zooms

