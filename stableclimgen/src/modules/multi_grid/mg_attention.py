from typing import List,Dict
from einops import rearrange

import torch
import torch.nn as nn

from ..base import get_layer, IdentityLayer, LinEmbLayer, MLP_fac
from ...modules.grids.grid_layer import GridLayer
from ...modules.transformer.transformer_base import SelfAttention

from ...modules.embedding.embedder import EmbedderSequential









class Asc_GridCrossAttention(nn.Module):
  
    def __init__(self, 
                 grid_layer: GridLayer,
                 grid_layer_cross: GridLayer,
                 in_features: int,
                 out_features: int,
                 mult: int=1,
                 dropout: float=0.0,
                 num_heads: int=1,
                 n_head_channels: int=None, 
                 embedder_q: EmbedderSequential = None,
                 embedders_kv: Dict[str, EmbedderSequential] = {},
                 layer_confs: Dict = {}) -> None: 
        
        super().__init__()
        
        self.q_layer = LinEmbLayer(in_features, in_features, layer_confs=layer_confs, identity_if_equal=False, embedder=embedder_q, layer_norm=True)
        self.mlp_emb_layer = LinEmbLayer(in_features, in_features, layer_confs=layer_confs, identity_if_equal=False, embedder=embedder_q, layer_norm=True)

        self.kv_layers = nn.ModuleDict()
        zooms = [int(grid_layer.zoom)]

        for in_zoom_cross in embedders_kv.keys():
            self.kv_layers[in_zoom_cross] = LinEmbLayer(in_features, in_features*2, layer_confs=layer_confs, identity_if_equal=False, embedder=embedders_kv[in_zoom_cross], layer_norm=True)
            zooms.append(int(in_zoom_cross))

        zooms = torch.tensor(zooms) 
       
        seq_lengths = list(4**(zooms - zooms.min()))
        zooms_str = [str(int(zoom)) for zoom in list(zooms)]
        self.seq_lengths = dict(zip(zooms_str, [int(seq_length) for seq_length in list(seq_lengths)]))

        num_heads = num_heads if not n_head_channels else in_features // n_head_channels
        self.attention = SelfAttention(in_features, in_features, num_heads=num_heads, dropout=dropout, cross=True)

        self.out_layer = get_layer(in_features, out_features, layer_confs=layer_confs, bias=False)

        self.grid_layer = grid_layer
        self.grid_layer_cross = grid_layer_cross

        self.mlp = MLP_fac(in_features, out_features, mult, dropout, layer_confs=layer_confs, gamma=True)

        self.pattern = 'b v t s n c -> (b v t s) (n) c'
        self.nh_mask_pattern = 'b v t s n -> (b v t s) 1 1 n'
        self.reverse_pattern = '(b v t s) n c -> b v t (s n) c'

        self.gamma = torch.nn.Parameter(torch.ones(in_features) * 1E-6)

    def forward(self, x_zooms, mask_zooms=None, emb=None, sample_dict={}):
        
        in_zoom = self.grid_layer.zoom

        x = x_zooms[in_zoom]
        q = self.q_layer(x, emb=emb, sample_dict=sample_dict)

        x_cross = []
        mask_cross = []
        for zoom, kv_layer in self.kv_layers.items():
            x_cross_ = kv_layer(x_zooms[int(zoom)], emb=emb, sample_dict=sample_dict)

            x_cross_nh, mask_nh = self.grid_layer_cross.get_nh(x_cross_, **sample_dict, with_nh=True, mask=None)
            x_cross.append(x_cross_nh)
            mask_cross.append(mask_nh)

        x_cross = torch.concat(x_cross, dim=-2)

        if mask_cross[0] is not None:
            mask_cross = torch.concat(mask_cross, dim=-1)
            mask_cross = mask_cross == False
            mask_cross = rearrange(mask_cross, self.nh_mask_pattern)
        else:
            mask_cross = None

        q = q.view(*x_cross.shape[:4], -1, q.shape[-1])

        b, v, t, s, n, c = q.shape

        q = rearrange(q, self.pattern)
        x_cross = rearrange(x_cross, self.pattern)

        q = self.attention(q, x_cross, mask=mask_cross)

        q = rearrange(q, self.reverse_pattern, b=b, t=t, s=s, n=n, v=v, c=c) 

        x = x + self.gamma*q

        x_mlp = self.mlp_emb_layer(x, emb=emb, sample_dict=sample_dict)

        x = x + self.mlp(x_mlp, emb=emb, sample_dict=sample_dict)

        x_zooms[in_zoom] = x

        return x_zooms