from typing import List,Dict
import math

import string

from einops import rearrange
import torch
import torch.nn as nn

import copy
from ..base import get_layer, MLP_fac
from .mg_base import Tokenizer

_AXIS_POOL = list("g") + list(string.ascii_lowercase.replace("g","")) + list(string.ascii_uppercase)


class FieldLayerConfig:
    def __init__(self, 
                 in_zooms: List,
                 target_zooms: List,
                 field_zoom: int,
                 out_zooms: List=None,
                 overlap: int = 0,
                 mult: int = 2,
                 type: str ='linear',
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)

class FieldLayer(nn.Module):
  
    def __init__(self, 
                 grid_layers,
                 x_zooms: List,
                 in_zooms: List,
                 target_zooms: List,
                 field_zoom: int,
                 overlap: int=0,
                 out_zooms: List=None,
                 in_features: int=1,
                 target_features: List=[1],
                 type = 'linear',
                 mult: int=2,
                 with_nh=False,
                 layer_confs: Dict={}) -> None: 
        
        super().__init__()

        self.with_nh = with_nh
        self.out_zooms = out_zooms
        self.in_zooms = in_zooms
        self.field_zoom = field_zoom
        self.n_channels_in = {}

        self.in_features_dict = dict(zip(x_zooms, in_features))
        self.target_features_dict = dict(zip(target_zooms, target_features))

        self.out_features = [self.target_features_dict[zoom] if zoom in self.target_features_dict.keys() else self.in_features_dict[zoom] for zoom in out_zooms]

        self.tokenizer = Tokenizer(grid_layers, 
                                in_zooms, 
                                field_zoom, 
                                overlap_thickness=overlap)
        
        tokenizer_out = Tokenizer(grid_layers, 
                                target_zooms, 
                                field_zoom)
        
        self.n_in_features_zooms, _  = self.tokenizer.get_features()
        self.n_out_features_zooms, _ = tokenizer_out.get_features()

        for z,f in self.n_in_features_zooms.items():
            self.n_in_features_zooms[z] = f * self.in_features_dict[z]
        
        for z,f in self.n_out_features_zooms.items():
            self.n_out_features_zooms[z] = f * self.target_features_dict[z]

        in_features = sum(self.n_in_features_zooms.values()) 
        out_features = sum(self.n_out_features_zooms.values())

        if type == 'linear':
            self.layer = get_layer(in_features, out_features, layer_confs=layer_confs)
        else: 
            self.layer = MLP_fac(in_features, out_features, mult=mult)

        self.pattern_tokens = 'b v T N n D f ->  b v T N D (n f)'
        self.pattern_tokens_reverse = 'b v T N D (n f) ->  b v T (N n) D f'

    def update_time_embedder(self, emb):
        for zoom in self.in_zooms:
            emb['TimeEmbedder'][zoom] = emb['TimeEmbedder'][max(self.in_zooms)]

    def forward(self, x_zooms, emb=None, sample_configs={}, **kwargs):
        
        nv = x_zooms[list(self.n_in_features_zooms.keys())[0]].shape[1]

        x = self.tokenizer(x_zooms, sample_configs=sample_configs)

        self.update_time_embedder(emb)

        x = rearrange(x, self.pattern_tokens)

        x = self.layer(x, emb=emb, sample_configs=sample_configs[self.field_zoom])

        x = x.split(tuple(self.n_out_features_zooms.values()), dim=-1)
        
        for k, (zoom, n) in enumerate(self.n_out_features_zooms.items()):
            x_zooms[zoom] = rearrange(x[k], self.pattern_tokens_reverse, f=self.target_features_dict[zoom], v=nv)

        if self.out_zooms is None:
            return x_zooms
        else:
            x_zooms_out = {}
            for zoom in self.out_zooms:
                x_zooms_out[zoom] = x_zooms[zoom]
            return x_zooms_out