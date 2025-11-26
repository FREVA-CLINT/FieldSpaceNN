from typing import List,Dict
import math

import string

from einops import rearrange
import torch
import torch.nn as nn

import copy
from ..base import get_layer, IdentityLayer, LinEmbLayer, MLP_fac, LayerNorm
from ...modules.grids.grid_layer import GridLayer, Interpolator, get_nh_idx_of_patch, get_idx_of_patch

from ...modules.embedding.embedder import EmbedderSequential


from ..grids.grid_utils import insert_matching_time_patch, get_matching_time_patch, decode_zooms

_AXIS_POOL = list("g") + list(string.ascii_lowercase.replace("g","")) + list(string.ascii_uppercase)

def combine_zooms(x_zooms, out_zoom, zooms=None):
    zooms = list(x_zooms.keys()) if zooms is None else zooms
    x_out = []
    for zoom in zooms:
        x = x_zooms[zoom]
        if zoom < out_zoom:
            x = refine_zoom(x, zoom, out_zoom).unsqueeze(dim=-2)
        else:
            x = x.view(*x.shape[:3],-1, 4**(zoom - out_zoom),x.shape[-1])
        x_out.append(x)
    return torch.concat(x_out, dim=-2)


def refine_zoom(x, in_zoom, out_zoom):
    x = x.view(*x.shape[:3],-1, 1, x.shape[-1])
    x = x.expand(-1,-1,-1, -1,4**(out_zoom - in_zoom),-1).reshape(*x.shape[:3],-1, x.shape[-1])
    return x


def coarsen_zoom(x, in_zoom, out_zoom):
    x = x.view(*x.shape[:3],-1, 4**(in_zoom - out_zoom), x.shape[-1]).mean(dim=-2)
    return x

class MGFieldLayer(nn.Module):
  
    def __init__(self, 
                 grid_layer,
                 x_zooms: List,
                 in_zooms: List,
                 target_zooms: List,
                 field_zoom: int,
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
        self.grid_layer = grid_layer
        self.in_zooms = in_zooms
        self.field_zoom = field_zoom
        self.n_channels_in = {}

        self.in_features_dict = dict(zip(x_zooms, in_features))
        self.target_features_dict = dict(zip(target_zooms, target_features))

        self.out_features = [self.target_features_dict[zoom] if zoom in self.target_features_dict.keys() else self.in_features_dict[zoom] for zoom in out_zooms]

        self.total_dim_in = []
        self.n_in = {}
        for zoom in in_zooms:
            n_in = 4**(zoom - field_zoom)
            self.n_in[zoom] = n_in
            self.total_dim_in.append(n_in * self.in_features_dict[zoom])

        self.total_dim_out = []
        self.n_out = {}
        for zoom in target_zooms:
            n_out = 4**(zoom - field_zoom)
            self.n_out[zoom] = n_out
            self.total_dim_out.append(n_out * self.target_features_dict[zoom])

        in_features =  sum(self.total_dim_in) * grid_layer.adjc.shape[1] if with_nh else sum(self.total_dim_in)
        out_features = sum(self.total_dim_out)

        if type == 'linear':
            self.layer = get_layer(in_features, out_features, layer_confs=layer_confs)
        else: 
            self.layer = MLP_fac(in_features, out_features, mult=mult)

        self.pattern_channel = 'b v t N n f ->  b v t N (n f)'
        self.pattern_channel_reverse = 'b v t N (n f) ->  b v t (N n) f'

    def forward(self, x_zooms, emb=None, sample_configs={}, **kwargs):
        
        nv = x_zooms[list(self.n_in.keys())[0]].shape[1]

        x = combine_zooms(x_zooms, out_zoom=self.field_zoom, zooms=self.in_zooms)

        x = rearrange(x, self.pattern_channel)

        if self.with_nh:
            x, _ = self.grid_layer.get_nh(x, **sample_configs[self.grid_layer.zoom], with_nh=True, mask=None)

        x = self.layer(x, emb=emb, sample_configs=sample_configs[self.field_zoom])

        x = x.split(tuple(self.total_dim_out), dim=-1)
        
        for k, (zoom, n) in enumerate(self.n_out.items()):
            x_zooms[zoom] = rearrange(x[k], self.pattern_channel_reverse, n=n, f=self.target_features_dict[zoom],v=nv)

        if self.out_zooms is None:
            return x_zooms
        else:
            x_zooms_out = {}
            for zoom in self.out_zooms:
                x_zooms_out[zoom] = x_zooms[zoom]
            return x_zooms_out


class LinearReductionLayer(nn.Module):
  
    def __init__(self, 
                 in_features: List,
                 out_features: int,
                 layer_confs: Dict) -> None: 
        super().__init__()
        
        if len(in_features)>1:
            self.layer = get_layer(
                [len(in_features), in_features[0]],
                [out_features],
                layer_confs=layer_confs)

        else:
            self.layer = IdentityLayer()

    def forward(self, x_levels, emb=None):

        x_out = self.layer(torch.stack(x_levels, dim=-2), emb=emb)
        x_out = x_out.view(*x_out.shape[:4],-1)

        return x_out


class SumReductionLayer(nn.Module):
  
    def __init__(self) -> None: 
        super().__init__()


    def forward(self, x_levels, mask_levels=None, emb=None):

        if mask_levels is not None and mask_levels[0] is not None:
            mask_out = torch.stack(mask_levels, dim=-1).sum(dim=-1) == len(mask_levels)
        else:
            mask_out = None
        
        x_out = torch.stack(x_levels, dim=-1).sum(dim=-1)

        return x_out, mask_out


class DecodeLayer(nn.Module):
  
    def __init__(self,
                 out_zoom: int
                ) -> None: 
      
        super().__init__()

        self.out_zooms = [out_zoom]  
        self.out_zoom = out_zoom

    def forward(self, x_zooms, sample_configs={}, **kwargs):

        x = {self.out_zoom: decode_zooms(x_zooms, self.out_zoom, sample_configs=sample_configs)}

        return x

class ConservativeLayer(nn.Module):
  
    def __init__(self,
                 in_zooms: List[int],
                 first_feature_only=False
                ) -> None: 
      
        super().__init__()

        self.ffo = first_feature_only

        self.proj_layers = nn.ModuleDict()
        self.out_zooms = in_zooms
        
        zooms_sorted = [int(t) for t in torch.tensor(in_zooms).sort(descending=True).values]
        
        self.cons_dict = dict(zip(zooms_sorted[:-1],zooms_sorted[1:]))
        self.cons_dict[zooms_sorted[-1]] = zooms_sorted[-1]

        self.in_zooms = in_zooms
    

    def forward(self, x_zooms, sample_configs={}, **kwargs):

        for zoom in sorted(x_zooms.keys()):
            
            x = x_zooms[zoom]
            zoom_level_cons = zoom - self.cons_dict[zoom]

            if zoom_level_cons > 0:
                x = x.view(*x.shape[:3], -1, 4**zoom_level_cons, x.shape[-1]) 

                mean = x.mean(dim=-2)
                x = (x-mean.unsqueeze(dim=-2)).view(*x.shape[:3], -1, x.shape[-1])

                x_patch = get_matching_time_patch(x_zooms[self.cons_dict[zoom]], self.cons_dict[zoom], zoom, sample_configs) + mean

                x_zooms[self.cons_dict[zoom]] = insert_matching_time_patch(x_zooms[self.cons_dict[zoom]], x_patch, self.cons_dict[zoom], zoom, sample_configs)

                x_zooms[zoom] = x

        return x_zooms
        

class ProjLayer(nn.Module):
  
    def __init__(self,
                 in_features,
                 out_features,
                 in_zoom,
                 out_zoom=None
                ) -> None: 
      
        super().__init__() 

        self.zoom_diff = 0 if out_zoom is None else out_zoom - in_zoom
        self.lin_layer = nn.Linear(in_features, out_features, bias=True) if in_features!= out_features else nn.Identity()

    def get_sum_residual(self, x, mask=None, **kwargs):
        if self.zoom_diff < 0:
            x = x.view(*x.shape[:3], -1, 4**(-1*self.zoom_diff), x.shape[-1])

            if mask is not None:
                weights = mask.view(*x.shape[:3],-1, 4**self.zoom_diff, 1)==False
                weights = weights.sum(dim=-3, keepdim=True)
                x = (x/(weights+1e-10)).sum(dim=-3) 
                x = x * (weights.sum(dim=-3)!=0)

            else:
                x = x.mean(dim=-2)

        elif self.zoom_diff > 0:
            #x = x.unsqueeze(dim=-2).repeat_interleave(4**(self.zoom_diff), dim=-2)
            x = x.unsqueeze(dim=-2).expand(-1,-1,-1,-1,4**(self.zoom_diff),-1)
            x = x.reshape(*x.shape[:3],-1,x.shape[-1])
        
        else:
            x = x

        return x

    def forward(self, x, mask=None, **kwargs):

        return self.lin_layer(self.get_sum_residual(x, mask=mask))
    

class IWD_ProjLayer(nn.Module):
  
    def __init__(self,
                 grid_layers: List[GridLayer],
                 zoom_input: int,
                 zoom_output: int,
                 interpolator_confs: dict = {},
                ) -> None: 
      
        super().__init__()

        self.interpolators = nn.ModuleDict()
        self.lin_layers = nn.ModuleDict()
        self.output_zooms = [0]
        
        self.interpolator = Interpolator(grid_layers, 
                                        search_zoom_rel=0, 
                                        input_zoom=zoom_input, 
                                        target_zoom=zoom_output, 
                                        **interpolator_confs)


    def forward(self, x, sample_configs={}):
      
        x,_ = self.interpolator(x.unsqueeze(dim=-2), calc_density=False, sample_configs=sample_configs)

        return x
    

class Conv(nn.Module):
    """
    Rearranges input to make the variable dimension centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, grid_layer: GridLayer, in_features, out_features, ranks_spatial=[], layer_confs={}, out_zoom=None):
        super().__init__()
        self.grid_layer = grid_layer
        
        rank_spatial_dict = {}
        if len(ranks_spatial)>0:
            for k,rank in enumerate(ranks_spatial):
                if grid_layer.zoom-k >= 0:
                    rank_spatial_dict[grid_layer.zoom-k] = rank

        layer_confs_ = copy.deepcopy(layer_confs)
        layer_confs_['ranks_spatial'] = rank_spatial_dict

        if out_zoom is None or out_zoom==self.grid_layer.in_zoom:
            self.layer = get_layer([self.grid_layer.adjc.shape[1], in_features], [1, out_features], layer_confs=layer_confs_)
            self.with_nh = True

        elif out_zoom > self.grid_layer.in_zoom:
            out_dims = [4] * (out_zoom - self.grid_layer.in_zoom)
            in_dims = [1] * len(out_dims)
            self.layer = get_layer([*in_dims, in_features], [*out_dims, out_features], layer_confs=layer_confs_)
            self.with_nh = False

        else:
            in_dims = [4] * (self.grid_layer.in_zoom * out_zoom)
            out_dims = [1] * len(out_dims)
            self.layer = get_layer([*in_dims, in_features], [*out_dims, out_features], layer_confs=layer_confs_)
            self.with_nh = False

    def forward(self, x: torch.Tensor, emb: Dict = None, mask: torch.Tensor = None, sample_configs: Dict = {}) -> torch.Tensor:
        
        if self.with_nh:
            x_nh, mask_nh = self.grid_layer.get_nh(x, **sample_configs, with_nh=True, mask=mask)
            x = self.layer(x_nh, emb=emb, sample_configs=sample_configs)
        else:
            x = self.layer(x, emb=emb, sample_configs=sample_configs)

        x = x.view(*x.shape[:3],-1,x.shape[-1])

        return x
    
class ResConv(nn.Module):
    """
    Rearranges input to make the variable dimension centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, grid_layer_in: GridLayer, in_features, out_features, grid_layer_out=None, ranks_spatial=[], layer_confs={}, layer_confs_emb={}, embedder: EmbedderSequential=None, use_skip_conv=False, with_gamma=False):
        super().__init__()

        grid_layer_out = grid_layer_in if grid_layer_out is None else grid_layer_out

        self.norm1 = LayerNorm(in_features, n_groups=layer_confs.get("n_groups",1))
        self.norm2 = LayerNorm(out_features, n_groups=layer_confs.get("n_groups",1))

        self.pattern_norm = 'b v t s f -> (b v t) f s'
        self.pattern_unnorm = '(b v t) f s -> b v t s f'   

        self.activation = nn.SiLU()
        
        if (grid_layer_out is None) or (grid_layer_out.zoom == grid_layer_in.zoom) or not use_skip_conv:
            self.skip_layer = ProjLayer(in_features, out_features, in_zoom=grid_layer_in.zoom, out_zoom=grid_layer_out.zoom)
            self.updownconv = ProjLayer(in_features, in_features, in_zoom=grid_layer_in.zoom, out_zoom=grid_layer_out.zoom)
            in_features_conv = in_features
        else:
            self.skip_layer = Conv(grid_layer_out, in_features, out_features, ranks_spatial=ranks_spatial, layer_confs=layer_confs, out_zoom=grid_layer_out.zoom)
            self.updownconv = Conv(grid_layer_out, in_features, in_features, ranks_spatial=ranks_spatial, layer_confs=layer_confs, out_zoom=grid_layer_out.zoom)
            in_features_conv = in_features

        if embedder is not None:
            self.emb_layer = LinEmbLayer(in_features, out_features, identity_if_equal=True, embedder=embedder, layer_confs_emb=layer_confs_emb)
        else:
            self.emb_layer = IdentityLayer()
    
        self.conv1 = Conv(grid_layer_out, in_features_conv, out_features, ranks_spatial=ranks_spatial, layer_confs=layer_confs)    
        self.conv2 = Conv(grid_layer_out, out_features, out_features, ranks_spatial=ranks_spatial, layer_confs=layer_confs)  

        self.with_gamma = with_gamma

        if with_gamma:
            self.gamma = nn.Parameter(torch.ones(out_features)*1e-6, requires_grad=True)

    def forward(self, x: torch.Tensor, emb: Dict = None, mask: torch.Tensor = None, 
                sample_configs: Dict={}, sample_configs_in: Dict = {}, sample_configs_out: Dict={}) -> torch.Tensor:
        
        sample_configs_in = sample_configs if len(sample_configs_in)==0 else sample_configs_in
        sample_configs_out = sample_configs if len(sample_configs_out)==0 else sample_configs_out

        x_res = self.skip_layer(x)
        
        b, v, t, s, f = x.shape

        x = self.norm1(x, emb=emb)

        x = self.activation(x)

        x = self.updownconv(x, emb=emb, sample_configs=sample_configs_in)

        x = self.conv1(x, emb=emb, sample_configs=sample_configs_out)

        x = self.emb_layer(x, emb=emb, sample_configs=sample_configs_out)

        b, v, t, s, f = x.shape

        x = self.norm2(x, emb=emb)

        x = self.activation(x)

        x = self.conv2(x, emb=emb, sample_configs=sample_configs_out)

        if self.with_gamma:
            x = self.gamma * x + x_res
        else:
            x = x + x_res

        return x
    

class Conv_EncoderDecoder(nn.Module):
  
    def __init__(self,
                 grid_layers,
                 in_zooms,
                 zoom_map: Dict[int,List],
                 in_features_list,
                 out_zooms: List=None,
                 aggregation = 'sum',
                 layer_confs: dict={},
                 use_skip_conv=False,
                 with_gamma=False,
                ) -> None: 
      
        super().__init__()

        self.aggregation = aggregation
        self.out_zooms = in_zooms if out_zooms is None else out_zooms
        self.out_features = in_features_list
        
        feature_dict = dict(zip(in_zooms, in_features_list))

        self.out_features = []
        for out_zoom in self.out_zooms:
            if out_zoom in feature_dict.keys():
                features_out = feature_dict[out_zoom] if aggregation == 'sum' else 2*feature_dict[out_zoom]
            else:
                features_out = feature_dict[zoom_map[out_zoom][0]]
            self.out_features.append(features_out)
        
        self.out_layers = nn.ModuleDict()
        for out_zoom, input_zooms in zoom_map.items():
            for in_zoom in input_zooms:
                
                in_layers = nn.ModuleDict()
                out_features = feature_dict[out_zoom] if out_zoom in feature_dict.keys() else feature_dict[in_zoom]

                if in_zoom == out_zoom:
                    layer = IdentityLayer()
                else:
                    layer = ResConv(grid_layers[str(in_zoom)],
                                    feature_dict[in_zoom],
                                    out_features,
                                    grid_layer_out=grid_layers[str(out_zoom)],
                                    layer_confs=layer_confs,
                                    use_skip_conv=use_skip_conv,
                                    with_gamma=with_gamma)

                in_layers[str(in_zoom)] = layer

            self.out_layers[str(out_zoom)] = in_layers

        self.aggregation = aggregation


    def forward(self, x_zooms, emb=None, sample_configs={},**kwargs):
        
        for out_zoom, layers in self.out_layers.items():
            x_out = [] if int(out_zoom) not in x_zooms.keys() else [x_zooms[int(out_zoom)]]

            for in_zoom, layer in layers.items():  
                x = layer(x_zooms[int(in_zoom)], emb=emb, sample_configs_in=sample_configs[int(in_zoom)], sample_configs_out=sample_configs[int(out_zoom)])
                x_out.append(x)

            if self.aggregation == 'sum':
                x = torch.stack(x_out, dim=-1).sum(dim=-1)
        
            else:
                x = torch.cat(x_out, dim=-1)

            x_zooms[int(out_zoom)] = x

        x_zooms_out = {}
        for out_zoom in self.out_zooms:
            x_zooms_out[out_zoom] = x_zooms[out_zoom]

        return x_zooms_out
