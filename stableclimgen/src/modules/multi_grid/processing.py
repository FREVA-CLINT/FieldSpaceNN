from typing import List, Dict

import torch.nn as nn

from ..transformer.transformer_base import TransformerBlock

from ..grids.grid_layer import GridLayer

from ...modules.embedding.embedder import get_embedder

class MG_Block(nn.Module):
  
    def __init__(self,
                 grid_layers: Dict[str,GridLayer],
                 in_zooms: List[int],
                 layer_settings: Dict,
                 in_features_list: List[int],
                 out_features_list: List[int]
                ) -> None: 
      
        super().__init__()

        self.in_zooms = in_zooms
        self.out_features = out_features_list
        self.out_zooms = in_zooms

        self.blocks = nn.ModuleDict()
        self.grid_layers = grid_layers

        max_lvl = layer_settings.get("max_lvl", max(in_zooms))

        for in_features, out_features, zoom in zip(in_features_list, out_features_list, in_zooms):
            
            type = layer_settings.get('type','TransformerBlock')

            if zoom > max_lvl:
                continue
            
            embedders = get_embedder(**layer_settings.get('embed_confs', {}), zoom=zoom)

            if type == 'TransformerBlock':
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
                            layer_confs=layer_settings.get('layer_confs', {}),
                        )
    
            self.blocks[str(zoom)] = block
        

    def forward(self, x_zooms: Dict, sample_dict=None,  emb=None, **kwargs):

        for zoom, block in self.blocks.items():
            x = x_zooms[int(zoom)]

            x = block(x, emb=emb, sample_dict=sample_dict)

            x_zooms[int(zoom)] = x

        return x_zooms