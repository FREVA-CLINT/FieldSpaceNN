from typing import List,Dict
import math

import string

from einops import rearrange
import torch
import torch.nn as nn

import copy
from ..base import get_layer, IdentityLayer, MLP_fac, LayerNorm
from ...modules.grids.grid_layer import GridLayer, Interpolator, get_nh_idx_of_patch, get_idx_of_patch

from ...modules.embedding.embedder import EmbedderSequential


from ..grids.grid_utils import insert_matching_time_patch, get_matching_time_patch, decode_zooms

_AXIS_POOL = list("g") + list(string.ascii_lowercase.replace("g","")) + list(string.ascii_uppercase)


def add_depth_overlap_from_neighbor_patches(
    x: torch.Tensor,
    overlap: int = 1,
    pad_mode: str = "zeros",  

) -> torch.Tensor:
  
    o = overlap
    if o == 0:
        return x

    b, v, T, N, D, t, n, d, f = x.shape
    assert o <= d, f"overlap={o} must be <= d={d}"

    out = x.new_empty(b, v, T, N, D, t, n, d + 2 * o, f)

    # center
    out[..., o:o + d, :] = x

    if D > 1:
        out[:, :, :, :, 1:, :, :, :o] = x[:, :, :, :, :-1, :, :, d - o : d]
        out[:, :, :, :, :-1, :, :, o + d :] = x[:, :, :, :, 1:, :, :, :o]

    # boundaries
    if pad_mode == "zeros":
        out[:, :, :, :, 0,  :, :, :o] = 0
        out[:, :, :, :, -1, :, :, o + d :] = 0

    elif pad_mode == "edge":
        left_edge  = x[:, :, :, :, 0,  :, :, :1].expand(b, v, T, N, t, n, o, f)
        right_edge = x[:, :, :, :, -1, :, :, -1:].expand(b, v, T, N, t, n, o, f)
        out[:, :, :, :, 0,  :, :, :o] = left_edge
        out[:, :, :, :, -1, :, :, o + d :] = right_edge

    else:
        raise ValueError("pad_mode must be 'zeros' or 'edge'")

    return out

def add_time_overlap_from_neighbor_patches(
    x: torch.Tensor,
    overlap: int = 1,
    pad_mode: str = "zeros",  

) -> torch.Tensor:
  
    o = overlap
    if o == 0:
        return x

    b, v, T, N, D, t, n, d, f = x.shape
    assert o <= t, f"overlap={o} must be <= t={t}"

    out = x.new_empty(b, v, T, N, D, t + 2 * o, n, d, f)

    # center
    out[..., o:o + t,:,:,:] = x

    if T > 1:
        out[:, :, 1:, :, :, :o] = x[:, :, :-1, :, :, t - o : t]
        out[:, :, :-1, :, :, o + t :] = x[:, :, 1:, :, :, :o]

    # boundaries
    if pad_mode == "zeros":
        out[:, :, 0, :, :,  :o] = 0
        out[:, :, -1, :, :, o + t :] = 0

    elif pad_mode == "edge":
        left_edge  = x[:, :, 0, :, :,  :1].expand(b, v, N, D, o, n, d, f)
        right_edge = x[:, :, -1, :, :, -1:].expand(b, v, N, D, o, n, d, f)
        out[:, :, 0, :, :,  :o] = left_edge
        out[:, :, -1, :, :, o + t :] = right_edge

    else:
        raise ValueError("pad_mode must be 'zeros' or 'edge'")

    return out



class ConservativeLayerConfig:
    pass



class Tokenizer(nn.Module):
  
    def __init__(self,
                 input_zooms = [],
                 token_zoom = -1,
                 overlap_thickness=0,
                 grid_layers: Dict={}, 
                 token_len_time=1,
                 token_len_depth=1) -> None: 
               
        super().__init__()

        if token_zoom==-1:
            overlap_thickness = 0

        self.overlap_thickness = overlap_thickness
        self.token_zoom = token_zoom
        self.input_zooms = input_zooms

        self.grid_layers_overlap = nn.ModuleDict()
        self.features_zoom_w_overlap = []
        self.features_zoom = []
        for input_zoom in input_zooms:

            n_patch = 4**(input_zoom - self.token_zoom) if token_zoom > -1 else 12*4**(input_zoom)
            if overlap_thickness > 0:
                grid_layer = grid_layers[str(input_zoom + (overlap_thickness - 1))]
                self.grid_layers_overlap[str(input_zoom)] = grid_layer

                n_tot = grid_layer.get_number_of_points_in_patch(token_zoom)
            else:
                n_tot = n_patch

            self.features_zoom_w_overlap.append(n_tot)
            self.features_zoom.append(n_patch)

        if overlap_thickness> 0 and len(input_zooms)>0:
            self.token_fcn = self.get_token_w_overlap
        else:
            self.token_fcn = self.get_token
        
        if len(input_zooms) > 0:
            self.token_size = [token_len_time, sum(self.features_zoom_w_overlap), token_len_depth]
        else:
            self.token_size = [token_len_time, 1, token_len_depth]

        self.pattern_tokens = 'b v (T t) N n (D d) f ->  b v T N D t n d f'

    def get_features(self):
        return dict(zip(self.input_zooms, self.features_zoom_w_overlap)), dict(zip(self.input_zooms, self.features_zoom))
    
    def get_patch_features_zoom(self, input_zoom, overlap_thickness):
        n_overlap = 4*overlap_thickness * 2**(input_zoom - self.token_zoom) + 4*overlap_thickness**2
        n_patch = 4**(input_zoom - self.token_zoom)

        return n_patch + n_overlap

    def get_token(self, x_zooms: Dict, sample_configs={}, **kwargs): 
        return combine_zooms(x_zooms, out_zoom=self.token_zoom, zooms=self.input_zooms, sample_configs=sample_configs)

    def get_token_w_overlap(self, x_zooms: Dict, sample_configs={}, mask=None): 
    
        x_out = []
        for zoom in self.input_zooms:
            x = x_zooms[zoom]
            x, mask = self.grid_layers_overlap[str(zoom)].get_nh(x, zoom, **sample_configs[zoom], mask=mask, zoom_patch_out=self.token_zoom)

            x = get_matching_time_patch(x, zoom, max(self.input_zooms), sample_configs)

            x_out.append(x)

        return torch.concat(x_out, dim=-3)
    
    def forward(self, x_zooms: Dict, sample_configs, mask=None):
        
        if self.token_size[1] > 1:
            if not isinstance(x_zooms, Dict):
                x_zooms = {self.input_zooms[0]: x_zooms}
            x = self.token_fcn(x_zooms, sample_configs=sample_configs, mask=mask)
        else:
            x = x_zooms.unsqueeze(dim=-3)

        x = rearrange(x, self.pattern_tokens, t=self.token_size[0], n=self.token_size[1], d=self.token_size[2])
        return x


class EmbLayer(nn.Module):
    def __init__(self,
                 out_features,
                 embedder: EmbedderSequential,
                 in_features = None,
                 layer_confs_emb: dict={},
                 spatial_dim_count=1,
                 field_tokenizer = None,
                 output_zoom=None
                ) -> None: 
         
        super().__init__()

        aggregation = layer_confs_emb.get("aggregation","shift_scale")
        self.embedder = embedder
        self.field_tokenizer = field_tokenizer
        self.spatial_dim_count = spatial_dim_count
        self.output_zoom = output_zoom

        if not isinstance(out_features, list):
            out_features_ = [out_features]
        else:
            out_features_ = out_features
        
        self.out_features = out_features_

        if in_features is None:
            in_features = [1] * (len(out_features_) - 1)   

        self.get_emb_fcn = self.get_emb
        if field_tokenizer is not None:
            in_features = field_tokenizer.token_size
            self.get_emb_fcn = self.get_emb_and_tokenize

        if aggregation == 'shift_scale':
            layer_confs_emb['ranks'] = layer_confs_emb.get('ranks', [None]*(len(in_features) + 1)) + [None]

            self.embedding_layer = get_layer([*in_features, self.embedder.get_out_channels, 1], [*out_features_, 2], layer_confs=layer_confs_emb)
            self.forward_fcn = self.forward_w_shift_scale

        elif aggregation == 'shift':
            self.embedding_layer = get_layer([*in_features, self.embedder.get_out_channels], [*out_features_], layer_confs=layer_confs_emb)
            self.forward_fcn = self.forward_w_shift
        
        elif aggregation == 'scale':
            self.embedding_layer = get_layer([*in_features, self.embedder.get_out_channels], [*out_features_], layer_confs=layer_confs_emb)
            self.forward_fcn = self.forward_w_scale

        elif aggregation == 'concat':
            self.embedding_layer = get_layer([*in_features, self.embedder.get_out_channels], [*out_features_], layer_confs=layer_confs_emb)
            self.forward_fcn = self.forward_w_concat

        self.aggregation = aggregation
    
    def get_emb(self, emb, sample_configs={}):
        emb_ = self.embedder(emb, sample_configs, output_zoom=self.output_zoom)
        return emb_
    
    def get_emb_and_tokenize(self, emb, sample_configs={}):
        emb = self.get_emb(emb, sample_configs=sample_configs)
        emb_tokenized = self.field_tokenizer(emb, sample_configs=sample_configs)
        return emb_tokenized
    
    def forward_w_shift(self, x, emb=None, sample_configs={}):
        
        emb_ = self.get_emb_fcn(emb, sample_configs)
        shift = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb)

        n = shift.shape[-1]
        shift = shift.view(*shift.shape[:3], -1, n)

        x = x + shift

        return x
    
    def forward_w_scale(self, x, emb=None, sample_configs={}):
        
        emb_ = self.get_emb_fcn(emb, sample_configs)
        scale = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb)

        n = scale.shape[-1]
        scale = scale.view(*scale.shape[:3], -1, n)

        x = x * (1 + scale)

        return x

    def forward_w_concat(self, x, emb=None, sample_configs={}):
        
        emb_ = self.get_emb_fcn(emb, sample_configs)
        e = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb)

        n = e.shape[-1]
        e = e.view(*e.shape[:3], -1, n)

        x = torch.concat((x, e), dim=-1)

        return x
    
    def forward_w_shift_scale(self, x, emb=None, sample_configs={}):
        
        emb_ = self.get_emb_fcn(emb, sample_configs)
        scale, shift = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb).chunk(2, dim=-1)

        scale = scale.squeeze(dim=-1)
        shift = shift.squeeze(dim=-1)

        x = x * (scale + 1) + shift

        return x

    def forward(self, x, emb, sample_configs={}):
        return self.forward_fcn(x, emb=emb, sample_configs=sample_configs)



class LinEmbLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 layer_norm = False,
                 identity_if_equal = False,
                 layer_confs: dict={},
                 layer_confs_emb: dict={},
                 embedder=None,
                 field_tokenizer=None,
                 output_zoom = None,
                 spatial_dim_count=1
                ) -> None: 
         
        super().__init__()

        in_features = out_features if in_features is None else in_features

        self.embedder = embedder
        self.spatial_dim_count = spatial_dim_count

        if not isinstance(out_features, list):
            out_features_ = [out_features]
        else:
            out_features_ = out_features
        
        self.out_features = out_features
        
        if not isinstance(in_features, list):
            in_features_ = [in_features]
        else:
            in_features_ = in_features

        if self.embedder is not None:
            
            self.embedding_layer = EmbLayer(out_features, 
                                            embedder=embedder,
                                            layer_confs_emb=layer_confs_emb,
                                            spatial_dim_count=spatial_dim_count,
                                            field_tokenizer = field_tokenizer,
                                            output_zoom=output_zoom)
           
            concat = layer_confs_emb.get('aggregation','sum') == 'concat'

            self.out_features = self.embedding_layer.out_features + out_features if concat else out_features

        else:
            self.embedding_layer = IdentityLayer()

        if layer_norm:
            self.layer_norm = LayerNorm(out_features_, elementwise_affine=True, n_variables=layer_confs.get("n_variables",1))
        else:
            self.layer_norm = IdentityLayer()

        if identity_if_equal and (torch.tensor(in_features_)-torch.tensor(out_features_)==0).all():
            self.layer = IdentityLayer()
        else:
            self.layer = get_layer(in_features_, out_features_, layer_confs=layer_confs)


    def forward(self, x: torch.Tensor, emb: Dict={}, sample_configs: Dict={}, x_stats: torch.Tensor = None, **kwargs):
        
        x = self.layer(x, emb=emb, sample_configs=sample_configs)

        x = self.layer_norm(x, emb=emb, x_stats=x_stats)

        x = self.embedding_layer(x, emb=emb, sample_configs=sample_configs)

        return x
    


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
    

    def forward(self, x_zooms_groups, sample_configs={}, **kwargs):
        
        for k, x_zooms in enumerate(x_zooms_groups):
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
            x_zooms_groups[k] = x_zooms
        return x_zooms_groups



def combine_zooms(x_zooms, out_zoom, zooms=None, sample_configs=None):
    zooms = list(x_zooms.keys()) if zooms is None else zooms
    x_out = []
    for zoom in zooms:
        x = x_zooms[zoom]

        x = get_matching_time_patch(x, zoom, max(zooms), sample_configs)

        if zoom < out_zoom:
            x = refine_zoom(x, zoom, out_zoom).unsqueeze(dim=-3)
        elif out_zoom==-1:
            x = x.view(*x.shape[:3],1, -1,*x.shape[-2:])
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
