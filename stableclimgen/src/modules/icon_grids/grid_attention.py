import copy

import torch
import torch.nn as nn
from numpy.ma.core import indices

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

        self.seq_lvl = spatial_attention_configs.pop('seq_lvl')
        self.timesteps = spatial_attention_configs.pop('timesteps') if 'timesteps' in spatial_attention_configs.keys() else 1

        spatial_attention_configs['seq_lengths'] = 4 ** self.seq_lvl  if self.seq_lvl != -1 else None
        self.attention_layer = TransformerBlock(ch_in,
                                                ch_out,
                                                n_head_channels=n_head_channels,
                                                dropout=p_dropout,
                                                **spatial_attention_configs)

        self.grid_layer = grid_layer
        self.global_level = int(grid_layer.global_level)

    def get_coordinates(self, indices_layers, emb):

        coords = self.grid_layer.get_coordinates_from_grid_indices(
            indices_layers[self.global_level] if indices_layers else None)
        

        if emb is None:
            emb = {}
        emb['CoordinateEmbedder'] = coords
        return emb

    def forward(self, x, indices_sample=None, mask=None, emb=None, *args, **kwargs):
        b = x.shape[0]
        emb = self.get_coordinates(indices_sample["indices_layers"] if indices_sample else None, emb)
        x, mask, emb = self.prepare_batch(x, mask, emb.copy())
        x = self.attention_layer(x, emb=emb, mask=mask)
        return x.view(b, *x.shape[2:])

    def prepare_batch(self, x, mask, emb):
        bt, nt = x.shape[0], self.timesteps
        x = x.view(bt // nt, nt, *x.shape[1:])
        if emb is not None:
            for key, value in emb.items():
                if not isinstance(value, tuple) and not isinstance(value, dict) and  (value.shape[0] == bt):
                    emb[key] = value.view(bt // nt, nt, *value.shape[1:])
        return x, mask, emb