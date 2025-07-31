from typing import List, Dict
import warnings

from omegaconf import ListConfig

import torch
import torch.nn as nn

from ..transformer.transformer_base import TransformerBlock
from ..base import LinEmbLayer, MLP_fac

from ..grids.grid_layer import GridLayer

from ...modules.embedding.embedder import get_embedder
from .mg_attention import GridSelfAttention
from .mg_base import NHConv,ResNHConv

class MG_SingleBlock(nn.Module):
  
    def __init__(self,
                 grid_layers: Dict[str,GridLayer],
                 in_zooms: List[int],
                 layer_settings: Dict,
                 in_features_list: List[int],
                 out_features_list: List[int],
                 mg_emb_zoom: int,
                 zooms = None,
                 layer_confs = {},
                 layer_confs_emb={},
                 use_mask=False
                ) -> None: 
      
        super().__init__()

        self.in_zooms = in_zooms
        self.out_features = out_features_list
        self.out_zooms = in_zooms

        self.blocks = nn.ModuleDict()
        self.grid_layers = grid_layers
        self.use_mask = use_mask

        zooms = in_zooms if zooms is None else zooms 

        if not (len(zooms) == len(out_features_list)):
            warnings.warn(
                f"Length mismatch: zooms({len(zooms)}), "
                f"in_features_list({len(in_features_list)}), "
                f"out_features_list({len(out_features_list)})"
            )

        k = 0
        for zoom, in_features, out_features in zip(zooms, in_features_list, out_features_list):
            
            type = layer_settings.get('type','TransformerBlock')

            embedders = get_embedder(**layer_settings.get('embed_confs', {}), grid_layer=grid_layers[str(mg_emb_zoom)],zoom=zoom)

            if type == 'TransformerBlock':
                seq_length = layer_settings.get('seq_lengths', [10]*len(in_zooms))[k]
                zoom_block = max([zoom - seq_length,0])
                
                block = TransformerBlock(
                            in_features,
                            out_features,
                            layer_settings['blocks'],
                            seq_lengths=seq_length,
                            num_heads=layer_settings.get('num_heads', 2),
                            n_head_channels=layer_settings.get('n_head_channels', None),
                            mlp_mult=layer_settings.get('mlp_mult', 1),
                            dropout=layer_settings.get('dropout', 1),
                            spatial_dim_count=layer_settings.get('spatial_dim_count', 1),
                            embedders=embedders,
                            layer_confs=layer_confs,
                            grid_layer=grid_layers[str(zoom_block)],
                            layer_confs_emb=layer_confs_emb
                        )
                    
            elif type == 'nh_conv':
                ranks_spatial = layer_settings.get('ranks_spatial', [])

                block = NHConv(
                    grid_layers[str(zoom)],
                    in_features,
                    out_features,
                    ranks_spatial=ranks_spatial,
                    layer_confs=layer_confs
                )
            
            elif type == 'res_nh_conv':
                ranks_spatial = layer_settings.get('ranks_spatial', [])

                block = ResNHConv(
                    grid_layers[str(zoom)],
                    in_features,
                    out_features,
                    ranks_spatial=ranks_spatial,
                    layer_confs=layer_confs
                )

            elif type == 'linear':
                ranks_spatial = layer_settings.get('ranks_spatial', [])

                block = LinEmbLayer(
                    in_features,
                    out_features, 
                    layer_norm=False, 
                    identity_if_equal=False,
                    embedder=embedders,
                    layer_confs=layer_confs,
                    layer_confs_emb=layer_confs_emb)
                
            elif type == 'mlp':

                block = MLP_fac(
                    in_features,
                    out_features, 
                    mult=layer_settings.get('mlp_mult', 1), 
                    dropout=layer_settings.get('dropout', 0), 
                    layer_confs=layer_confs)

            self.blocks[str(zoom)] = block
        

    def forward(self, x_zooms: Dict, sample_dict=None,  emb=None, mask_zooms={}, **kwargs):

        for zoom, block in self.blocks.items():
            x = x_zooms[int(zoom)]

            x = block(x, emb=emb, sample_dict=sample_dict, mask=mask_zooms[int(zoom)] if self.use_mask else None)

            x_zooms[int(zoom)] = x

        return x_zooms
    

class MG_MultiBlock(nn.Module):
  
    def __init__(self, 
                 grid_layers: Dict[str, GridLayer],
                 in_zooms: List[int],
                 out_zooms: List[int],
                 layer_settings: Dict,
                 in_features: int,
                 out_features: int,
                 mg_emb_zoom: int,
                 q_zooms:List|int = -1,
                 kv_zooms:List|int = -1,
                 layer_confs={},
                 layer_confs_emb={},
                 use_mask = False
                 ) -> None:
        super().__init__()
        
        self.out_features = [out_features]*len(out_zooms)
        self.out_zooms = out_zooms
        self.use_mask = use_mask

        self.zero_zooms_gen = nn.ParameterDict()

        for zoom in out_zooms:
            if zoom not in in_zooms:
                self.zero_zooms_gen[str(zoom)] = nn.Parameter(torch.zeros((1,1,1,12*4**zoom,1)), requires_grad=False)


        self.blocks = nn.ModuleDict()

        if isinstance(in_features, (List,ListConfig)) and len(set(in_features)) > 1:
            raise ValueError("features of levels should be the same.")
        elif isinstance(in_features, (int)):
            in_features = in_features
        else:
            in_features = in_features[0]

        if not isinstance(q_zooms, (List,ListConfig)) and (q_zooms == -1):
            q_zooms = in_zooms
        
        if not isinstance(kv_zooms,(List,ListConfig)) and (kv_zooms == -1):
            kv_zooms = in_zooms

        grid_layer = grid_layers[str(in_zooms[0])]

        embedders_kv = {}
        embedders_q = {}

        for k, zoom in enumerate(q_zooms):
            #if zoom not in in_zooms:
            #    raise ValueError(f"Zoom level {zoom} at index {k} of q_zooms not found in in_zooms")
            
            embedder_q = get_embedder(**layer_settings.get('embed_confs', {}), grid_layer=grid_layers[str(mg_emb_zoom)], zoom=zoom)
            embedders_q[str(zoom)] = embedder_q
            
        
        for k, zoom in enumerate(kv_zooms):
            if zoom not in in_zooms:
                raise ValueError(f"Zoom level {zoom} at index {k} of kv_zooms not found in in_zooms")
            
            embedder_kv = get_embedder(**layer_settings.get('embed_confs', {}), grid_layer=grid_layers[str(mg_emb_zoom)], zoom=zoom)
            embedders_kv[str(zoom)] = embedder_kv
     
        grid_layer = grid_layers[str(min(q_zooms + kv_zooms))] 
            
        block = GridSelfAttention(grid_layer,
                                in_features,
                                out_features,
                                mult = layer_settings.get("mlp_mult",1),
                                num_heads = layer_settings.get("num_heads",1),
                                n_head_channels = layer_settings.get("n_head_channels",None),
                                with_nh= layer_settings.get("with_nh",True),
                                var_att= layer_settings.get("var_att",False),
                                seq_length= layer_settings.get("seq_length",0),
                                common_kv = layer_settings.get("common_kv",False),
                                common_q = layer_settings.get("common_q",False),
                                embedders_q=embedders_q,
                                embedders_kv=embedders_kv,
                                layer_confs=layer_confs,
                                layer_confs_emb=layer_confs_emb
                                )
        self.block = block

    def generate_zoom(self, x: torch.Tensor, zoom, sample_dict={}):
        b,nv,t,s,f = x.shape

        x_zoom = self.zero_zooms_gen[str(zoom)]

        if 'zoom_patch_sample' in sample_dict.keys():
            x_zoom = x_zoom.view(1,1,1,12*4**sample_dict['zoom_patch_sample'],-1,1)[:,:,:,0]

        return x_zoom.expand(b,nv,t,-1,f)


    def forward(self, x_zooms, emb=None, mask_zooms={}, sample_dict={}):

        for zoom in self.out_zooms:
            if zoom not in x_zooms.keys():
                x = self.generate_zoom(list(x_zooms.values())[0], zoom, sample_dict=sample_dict)
                x_zooms[zoom] = x
    
        x_zooms = self.block(x_zooms, emb=emb, mask_zooms=mask_zooms if self.use_mask else {}, sample_dict=sample_dict)

        x_zooms_out = {}
        for zoom in self.out_zooms:
            x_zooms_out[zoom] = x_zooms[zoom]

        return x_zooms_out