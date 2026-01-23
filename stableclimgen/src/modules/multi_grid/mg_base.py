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


class ConservativeLayerConfig:
    pass



class Tokenizer(nn.Module):
  
    def __init__(self,
                 grid_layers: Dict,
                 input_zooms,
                 token_zoom,
                 overlap_thickness=0) -> None: 
               
        super().__init__()

        self.overlap_thickness = overlap_thickness
        self.token_zoom = token_zoom
        self.input_zooms = input_zooms

        self.grid_layers_overlap = nn.ModuleDict()
        self.features_zoom_w_overlap = []
        self.features_zoom = []
        for input_zoom in input_zooms:
            n_patch = 4**(input_zoom - self.token_zoom)
            if overlap_thickness > 0:
                grid_layer = grid_layers[str(input_zoom + (overlap_thickness - 1))]
                self.grid_layers_overlap[str(input_zoom)] = grid_layer

                n_tot = grid_layer.get_number_of_points_in_patch(token_zoom)
            else:
                n_tot = n_patch

            self.features_zoom_w_overlap.append(n_tot)
            self.features_zoom.append(n_patch)

        if overlap_thickness> 0:
            self.token_fcn = self.get_token_w_overlap
        else:
            self.token_fcn = self.get_token

    def get_features(self):
        return dict(zip(self.input_zooms, self.features_zoom_w_overlap)), dict(zip(self.input_zooms, self.features_zoom))
    
    def get_patch_features_zoom(self, input_zoom, overlap_thickness):
        n_overlap = 4*overlap_thickness * 2**(input_zoom - self.token_zoom) + 4*overlap_thickness**2
        n_patch = 4**(input_zoom - self.token_zoom)

        return n_patch + n_overlap

    def get_token(self, x_zooms: Dict, **kwargs): 
        return combine_zooms(x_zooms, out_zoom=self.token_zoom, zooms=self.input_zooms)

    def get_token_w_overlap(self, x_zooms: Dict, sample_configs={}, mask=None): 
    
        x_out = []
        for zoom in self.input_zooms:
            x = x_zooms[zoom]
            x, mask = self.grid_layers_overlap[str(zoom)].get_nh(x, zoom, **sample_configs[zoom], mask=mask, zoom_patch_out=self.token_zoom)
            x_out.append(x)

        return torch.concat(x_out, dim=-3)
    
    def forward(self, x_zooms, sample_configs, mask=None):

        return self.token_fcn(x_zooms, sample_configs=sample_configs, mask=mask)


class DiffDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_zooms: dict, sample_configs: Dict, out_zoom: int = None, **kwargs):

        if out_zoom is None:
            return x_zooms
        
        return decode_zooms(x_zooms, sample_configs=sample_configs, out_zoom=out_zoom)


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
                x = x.view(*x.shape[:3], -1, 4**zoom_level_cons, *x.shape[-2:]) 

                mean = x.mean(dim=-3)
                x = (x-mean.unsqueeze(dim=-3)).view(*x.shape[:3], -1, *x.shape[-2:])

                x_patch = get_matching_time_patch(x_zooms[self.cons_dict[zoom]], self.cons_dict[zoom], zoom, sample_configs) + mean

                x_zooms[self.cons_dict[zoom]] = insert_matching_time_patch(x_zooms[self.cons_dict[zoom]], x_patch, self.cons_dict[zoom], zoom, sample_configs)

                x_zooms[zoom] = x

        return x_zooms



def combine_zooms(x_zooms, out_zoom, zooms=None):
    zooms = list(x_zooms.keys()) if zooms is None else zooms
    x_out = []
    for zoom in zooms:
        x = x_zooms[zoom]
        if zoom < out_zoom:
            x = refine_zoom(x, zoom, out_zoom).unsqueeze(dim=-3)
        elif out_zoom==0:
            x = x.view(*x.shape[:3],-1, 12*4**(zoom - out_zoom),*x.shape[-2:])
        else:
            x = x.view(*x.shape[:3],-1, 4**(zoom - out_zoom),*x.shape[-2:])
        x_out.append(x)
    return torch.concat(x_out, dim=-3)



def refine_zoom(x, in_zoom, out_zoom):
    x = x.view(*x.shape[:3],-1, 1, *x.shape[-2:])
    x = x.expand(-1,-1,-1, -1,4**(out_zoom - in_zoom),-1,-1).reshape(*x.shape[:3],-1, *x.shape[-2:])
    return x



def coarsen_zoom(x, in_zoom, out_zoom):
    x = x.view(*x.shape[:3],-1, 4**(in_zoom - out_zoom), *x.shape[-2:]).mean(dim=-3)
    return x
