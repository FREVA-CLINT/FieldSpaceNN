from typing import List,Dict
import math

import string

from einops import rearrange
import torch
import torch.nn as nn

import copy
from ..base import get_layer, MLP_fac
from ...utils.helpers import check_value
from .mg_base import Tokenizer, add_time_overlap_from_neighbor_patches, add_depth_overlap_from_neighbor_patches

_AXIS_POOL = list("g") + list(string.ascii_lowercase.replace("g","")) + list(string.ascii_uppercase)


class FieldLayerConfig:
    def __init__(self, 
                 in_zooms: List,
                 target_zooms: List,
                 field_zoom: int,
                 out_zooms: List=None,
                 token_overlap_space: bool = False,
                 token_overlap_time: bool = False,
                 token_overlap_depth: bool = False,
                 rank_space: int = None,
                 rank_time: int = None,
                 rank_depth: int = None,
                 in_token_len_time: int = 1,
                 in_token_len_depth: int = 1,
                 out_token_len_time: int = 1,
                 out_token_len_depth: int = 1,
                 n_groups_variables: List = [1],
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

class FieldLayerModule(nn.Module):
    def __init__(self,
                 grid_layers,
                 x_zooms: List,
                 in_zooms: List,
                 target_zooms: List,
                 field_zoom: int,
                 n_groups_variables: List = [1],
                 **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList()
        n_groups = len(n_groups_variables)

        # Ensure layer_confs is a list for each group
        layer_confs = kwargs.get('layer_confs', {})
        if not isinstance(layer_confs, list):
            layer_confs = [copy.deepcopy(layer_confs) for _ in range(n_groups)]
        
        # Handle other kwargs that might be group-specific
        in_features = check_value(kwargs.get('in_features', 1), n_groups)
        target_features = check_value(kwargs.get('target_features', [1]), n_groups)

        for i in range(n_groups):
            block_kwargs = kwargs.copy()
            block_kwargs['layer_confs'] = layer_confs[i]
            block_kwargs['in_features'] = in_features[i]
            block_kwargs['target_features'] = target_features[i]

            block = FieldLayerBlock(
                grid_layers=grid_layers,
                x_zooms=x_zooms,
                in_zooms=in_zooms,
                target_zooms=target_zooms,
                field_zoom=field_zoom,
                **block_kwargs
            )
            self.out_zooms = block.out_zooms
            self.out_features = block.out_features
            self.blocks.append(block)

    def forward(self, x_zooms_groups: List[Dict], emb_groups: List = None, sample_configs: Dict = {}, **kwargs):
        if emb_groups is None:
            emb_groups = [None] * len(x_zooms_groups)

        output_groups = []
        for i, block in enumerate(self.blocks):
            output_groups.append(block(
                x_zooms=x_zooms_groups[i],
                emb=emb_groups[i],
                sample_configs=sample_configs,
                **kwargs
            ))
        return output_groups


class FieldLayerBlock(nn.Module):
  
    def __init__(self, 
                 grid_layers,
                 x_zooms: List,
                 in_zooms: List,
                 target_zooms: List,
                 field_zoom: int,
                 out_zooms: List = None,
                 in_features: List = [1],
                 target_features: List = [1],
                 type: str = 'linear',
                 in_token_len_time: int = 1,
                 in_token_len_depth: int = 1,
                 out_token_len_time: int = 1,
                 out_token_len_depth: int = 1,
                 token_overlap_space: bool = False,
                 token_overlap_time: bool = False,
                 token_overlap_depth: bool = False,
                 rank_space: int = None,
                 rank_time: int = None,
                 rank_depth: int = None,
                 mult: int = 2,
                 layer_confs: Dict = {}) -> None: 

        super().__init__()

        if isinstance(in_features, int):
            in_features = [in_features] * len(x_zooms)
        if isinstance(target_features, int):
            target_features = [target_features] * len(target_zooms)
        self.token_overlap_space = token_overlap_space
        self.token_overlap_time = token_overlap_time
        self.token_overlap_depth = token_overlap_depth

        self.out_zooms = out_zooms
        self.in_zooms = in_zooms
        self.field_zoom = field_zoom
        self.n_channels_in = {}

        self.in_features_dict = dict(zip(x_zooms, in_features))
        self.target_features_dict = dict(zip(target_zooms, target_features))

        self.out_features = [self.target_features_dict[zoom] if zoom in self.target_features_dict.keys() else self.in_features_dict[zoom] for zoom in out_zooms]

        self.tokenizer = Tokenizer(in_zooms, 
                                   field_zoom,
                                   grid_layers=grid_layers,
                                   overlap_thickness=int(self.token_overlap_space),
                                   token_len_time=in_token_len_time,
                                   token_len_depth=in_token_len_depth)

        tokenizer_out = Tokenizer(target_zooms, 
                                  field_zoom,
                                  grid_layers=grid_layers,
                                  token_len_time=out_token_len_time,
                                  token_len_depth=out_token_len_depth)

        self.n_in_features_zooms, _  = self.tokenizer.get_features()
        self.n_out_features_zooms, _ = tokenizer_out.get_features()

        for z,f in self.n_in_features_zooms.items():
            self.n_in_features_zooms[z] = f * self.in_features_dict[z]
        
        for z,f in self.n_out_features_zooms.items():
            self.n_out_features_zooms[z] = f * self.target_features_dict[z]

        in_features_space = sum(self.n_in_features_zooms.values())
        out_features_space = sum(self.n_out_features_zooms.values())

        in_features_full = [
            in_token_len_time + 2 * int(self.token_overlap_time),
            in_features_space,
            in_token_len_depth + 2 * int(self.token_overlap_depth),
            1
        ]
        out_features_full = [out_token_len_time, out_features_space, out_token_len_depth, 1]

        layer_confs_ = layer_confs.copy()
        layer_confs_['ranks'] = [rank_time, rank_space, rank_depth, None, None]

        if type == 'linear':
            self.layer = get_layer(in_features_full, out_features_full, layer_confs=layer_confs_)
        else: 
            self.layer = MLP_fac(in_features_full, out_features_full, mult=mult, layer_confs=layer_confs_)

        self.pattern_tokens_reverse = 'b v T N D t n d f -> b v (T t) (N n) (D d) f'


    def update_time_embedder(self, emb):
        for zoom in self.in_zooms:
            emb['TimeEmbedder'][zoom] = emb['TimeEmbedder'][max(self.in_zooms)]

    def get_time_depth_overlaps(self, x: torch.Tensor):
        if self.token_overlap_time:
            x = add_time_overlap_from_neighbor_patches(x, overlap=1, pad_mode="edge")

        if self.token_overlap_depth:
            x = add_depth_overlap_from_neighbor_patches(x, overlap=1, pad_mode="edge")

        return x

    def forward(self, x_zooms: Dict, emb: Dict = None, sample_configs: Dict = {}, **kwargs):
        nv = x_zooms[list(self.n_in_features_zooms.keys())[0]].shape[1]

        x = self.tokenizer(x_zooms, sample_configs=sample_configs)

        if emb:
            self.update_time_embedder(emb)

        x = self.get_time_depth_overlaps(x)

        x = self.layer(x, emb=emb, sample_configs=sample_configs[self.field_zoom])

        x = x.split(tuple(self.n_out_features_zooms.values()), dim=-3)
        
        for k, (zoom, n) in enumerate(self.n_out_features_zooms.items()):
            x_zooms[zoom] = rearrange(x[k], self.pattern_tokens_reverse, f=self.target_features_dict[zoom], v=nv)
        
        if self.out_zooms is None:
            return x_zooms
        else:
            x_zooms_out = {}
            for zoom in self.out_zooms:
                x_zooms_out[zoom] = x_zooms[zoom]
            return x_zooms_out