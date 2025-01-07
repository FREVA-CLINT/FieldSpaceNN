import copy

import torch
import torch.nn as nn

from .grid_layer import RelativeCoordinateManager, GridLayer
from ..transformer.transformer_base import TransformerBlock


class GridAttention(nn.Module):

    def __init__(self,
                 grid_layer: GridLayer,
                 ch_in,
                 ch_out,
                 n_head_channels,
                 spatial_attention_configs=None,
                 rotate_coord_system=True,
                 p_dropout=0
                 ) -> None:

        super().__init__()

        spatial_attention_configs = copy.deepcopy(spatial_attention_configs)
        nh = spatial_attention_configs.pop('nh')

        seq_lvl = spatial_attention_configs.pop('seq_lvl')

        if seq_lvl != -1 or nh:
            self.rel_coord_mngr = RelativeCoordinateManager(
                grid_layer,
                grid_layer,
                nh_in=nh,
                nh_ref=nh,
                seq_lvl=seq_lvl,
                precompute=True,
                coord_system="cartesian",
                rotate_coord_system=rotate_coord_system)

        spatial_attention_configs['seq_lengths'] = 4 ** seq_lvl if seq_lvl != -1 else None
        self.attention_layer = TransformerBlock(ch_in,
                                                ch_out,
                                                n_head_channels=n_head_channels,
                                                dropout=p_dropout,
                                                **spatial_attention_configs)

        self.grid_layer = grid_layer
        self.global_level = int(grid_layer.global_level)

    def get_coordinates(self, indices_layers, emb):

        if hasattr(self, 'rel_coord_mngr'):
            coords = self.rel_coord_mngr(indices_ref=indices_layers[self.global_level] if indices_layers else None)
            coords = torch.stack(coords, dim=-1)
        else:
            coords = self.grid_layer.get_coordinates_from_grid_indices(
                indices_layers[self.global_level] if indices_layers else None)

        if emb is None:
            emb = {}
        emb['CoordinateEmbedder'] = coords
        return emb

    def forward(self, x, indices_sample=None, mask=None, emb=None, *args, **kwargs):

        emb = self.get_coordinates(indices_sample["indices_layers"] if indices_sample else None, emb)

        x = self.attention_layer(x, emb=emb, mask=mask)

        return x