from typing import List, Dict

import torch
import torch.nn as nn

from ..transformer.transformer_base import TransformerBlock
from ..base import LinEmbLayer

from ..grids.grid_layer import GridLayer

from ...modules.embedding.embedder import get_embedder
from .mg_attention import GridCrossAttention,GridSelfAttention
from .mg_base import NHConv,ResNHConv

class MG_Block(nn.Module):
  
    def __init__(self,
                 grid_layers: Dict[str,GridLayer],
                 in_zooms: List[int],
                 layer_settings: Dict,
                 in_features_list: List[int],
                 out_features_list: List[int],
                 layer_confs = {}
                ) -> None: 
      
        super().__init__()

        self.in_zooms = in_zooms
        self.out_features = out_features_list
        self.out_zooms = in_zooms

        self.blocks = nn.ModuleDict()
        self.grid_layers = grid_layers

        min_zoom = layer_settings.get("min_zoom", max(in_zooms))

        for in_features, out_features, zoom in zip(in_features_list, out_features_list, in_zooms):
            
            type = layer_settings.get('type','TransformerBlock')

            if zoom > min_zoom:
                block = LinEmbLayer(
                    in_features,
                    out_features, 
                    layer_norm=False, 
                    identity_if_equal=True,
                    layer_confs=layer_confs)
            
            else:
                embedders = get_embedder(**layer_settings.get('embed_confs', {}), zoom=zoom)


                if type == 'TransformerBlock':
                    zoom_block = max([zoom - layer_settings.get('seq_lengths', 0),0])
                                    
                    block = TransformerBlock(
                                in_features,
                                out_features,
                                layer_settings['blocks'],
                                seq_lengths=layer_settings.get('seq_lengths', None),
                                num_heads=layer_settings.get('num_heads', 2),
                                n_head_channels=layer_settings.get('n_head_channels', None),
                                mlp_mult=layer_settings.get('mlp_mult', 1),
                                dropout=layer_settings.get('dropout', 1),
                                spatial_dim_count=layer_settings.get('spatial_dim_count', 1),
                                embedders=embedders,
                                layer_confs=layer_confs,
                                grid_layer=grid_layers[str(zoom_block)]
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
        
            self.blocks[str(zoom)] = block
        

    def forward(self, x_zooms: Dict, sample_dict=None,  emb=None, **kwargs):

        for zoom, block in self.blocks.items():
            x = x_zooms[int(zoom)]

            x = block(x, emb=emb, sample_dict=sample_dict)

            x_zooms[int(zoom)] = x

        return x_zooms
    


class MG_CrossBlock(nn.Module):
  
    def __init__(self, 
                 grid_layers: Dict[str, GridLayer],
                 in_zooms: List[int],
                 layer_settings: Dict,
                 in_features_list: List[int],
                 layer_confs={}
                 ) -> None:
        super().__init__()
        
        max_zoom_diff = layer_settings.get("max_zoom_diff", max(in_zooms)-min(in_zooms))
        asc = layer_settings['type'] == 'ascending'

        self.out_features = in_features_list
        self.out_zooms = in_zooms
        self.blocks = nn.ModuleDict()

        for k, in_zoom in enumerate(in_zooms):
            embedder_q = get_embedder(**layer_settings.get('embed_confs', {}), zoom=in_zoom)

            grid_layer_cross = grid_layers[str(in_zoom)]
            embedders_kv = {}
            for in_zoom_cross in in_zooms:
                
                zoom_diff = in_zoom - in_zoom_cross
                if (asc and zoom_diff > 0) or (zoom_diff < 0) and (abs(zoom_diff) > max_zoom_diff):
                    grid_layer_cross = grid_layers[str(in_zoom_cross)] if grid_layers[str(in_zoom_cross)].zoom < grid_layer_cross.zoom else grid_layer_cross
                    embedders_kv[str(in_zoom_cross)] = get_embedder(**layer_settings.get('embed_confs', {}), zoom=in_zoom_cross)

            if len(embedders_kv)>0:
                block = GridCrossAttention(grid_layers[str(in_zoom)],
                                        grid_layer_cross,
                                        in_features_list[k],
                                        in_features_list[k],
                                        mult = layer_settings.get("mlp_mult",1),
                                        num_heads = layer_settings.get("num_heads",1),
                                        n_head_channels = layer_settings.get("n_head_channels",None),
                                        embedder_q=embedder_q,
                                        embedders_kv=embedders_kv,
                                        layer_confs=layer_confs,
                                        )
                self.blocks[str(in_zoom)] = block

    def forward(self, x_zooms, emb=None, mask_zooms=None, sample_dict={}):

        for block in self.blocks.values():
            x_zooms = block(x_zooms, emb=emb, mask_zooms=mask_zooms, sample_dict=sample_dict)

        return x_zooms
    

class MG_SelfBlock(nn.Module):
  
    def __init__(self, 
                 grid_layers: Dict[str, GridLayer],
                 in_zooms: List[int],
                 layer_settings: Dict,
                 in_features_list: List[int],
                 out_features: int,
                 layer_confs={}
                 ) -> None:
        super().__init__()
        
        self.out_features = [out_features]*len(in_features_list)
        self.out_zooms = in_zooms
        self.blocks = nn.ModuleDict()

        grid_layer = grid_layers[str(in_zooms[0])]

        embedders_kv = {}
        embedders_q = {}

        for k, zoom in enumerate(in_zooms):
            embedder_q = get_embedder(**layer_settings.get('embed_confs', {}), zoom=zoom)
            embedder_kv = get_embedder(**layer_settings.get('embed_confs', {}), zoom=zoom)

            grid_layer = grid_layers[str(zoom)] if grid_layers[str(zoom)].zoom < grid_layer.zoom else grid_layer

            embedders_q[str(zoom)] = embedder_q
            embedders_kv[str(zoom)] = embedder_kv

       #     in_features[str(zoom)] = in_features_list[k]

            
        block = GridSelfAttention(grid_layer,
                                in_features_list[0],
                                out_features,
                                mult = layer_settings.get("mlp_mult",1),
                                num_heads = layer_settings.get("num_heads",1),
                                n_head_channels = layer_settings.get("n_head_channels",None),
                                with_nh= layer_settings.get("with_nh",True),
                                common_kv = layer_settings.get("common_kv",False),
                                common_q = layer_settings.get("common_q",False),
                                embedders_q=embedders_q,
                                embedders_kv=embedders_kv,
                                layer_confs=layer_confs,
                                )
        self.block = block

    def forward(self, x_zooms, emb=None, mask_zooms=None, sample_dict={}):

    
        x_zooms = self.block(x_zooms, emb=emb, mask_zooms=mask_zooms, sample_dict=sample_dict)

        return x_zooms