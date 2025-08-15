from typing import List,Dict
import math
from scipy.special import sph_harm_y
import string

import torch
import torch.nn as nn

import copy
from ..base import get_layer, IdentityLayer, LinEmbLayer
from ...modules.grids.grid_layer import GridLayer, Interpolator, get_nh_idx_of_patch, get_idx_of_patch
from ...modules.embedding.embedding_layers import RandomFourierLayer

from ..grids.grid_utils import insert_matching_time_patch, get_matching_time_patch,estimate_healpix_cell_radius_rad, decode_zooms

_AXIS_POOL = list("g") + list(string.ascii_lowercase.replace("g","")) + list(string.ascii_uppercase)

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


def get_mg_embedding(
        grid_layer_emb: GridLayer, 
        features, 
        n_groups, 
        init_mode='fourier_sphere',
        wavelength=1,
        amplitude=1):
    
    coords = grid_layer_emb.get_coordinates()
    
    if init_mode=='random':
        embs = amplitude*torch.randn(1, coords.shape[-2], features)
    
    elif 'fourier_sphere' == init_mode:
        fourier_layer = RandomFourierLayer(in_features=2, n_neurons=features, wave_length=2*wavelength*torch.pi)
        embs = amplitude*fourier_layer(coords)

    elif 'fourier' == init_mode:

        clon, clat = coords[...,0], coords[...,1]
        x = torch.cos(clat) * torch.cos(clon)
        y = torch.cos(clat) * torch.sin(clon)
        z = torch.sin(clat)

        coords_3d = torch.stack((x, y, z), dim=-1).float()

        fourier_layer = RandomFourierLayer(in_features=3, n_neurons=features, wave_length=wavelength)
        embs = amplitude*fourier_layer(coords_3d)
    
    elif "spherical_harmonics" == init_mode:
        l_min = int(1/wavelength -1)
        l_max = int(math.sqrt(math.pi*grid_layer_emb.adjc.shape[0])/4 - 1/2)

        clon, clat = coords[...,0], coords[...,1]

        ls = torch.randint(l_min, int(l_max), (features,))
        ms = torch.stack([torch.randint(0, max([1,l]), (1,)) for l in ls])
        
        embs = torch.zeros(1, coords.shape[-2], features)

        for k in range(features):

            Y_lm = sph_harm_y(ls[k], ms[k], clat, clon)
            embs[...,k] = amplitude*torch.tensor(Y_lm.real).view(1,-1)


    embs = embs.repeat_interleave(n_groups, dim=0)
    
    embs = nn.Parameter(embs, requires_grad=True)

    return embs

class MGEmbedding(nn.Module):
  
    def __init__(self,
                 grid_layer_emb: GridLayer,
                 features: int,
                 zooms: List,
                 n_groups: int =1,
                 init_mode='fourier_sphere',
                 layer_confs={}
                ) -> None: 
      
        super().__init__()
        self.grid_layer_emb = grid_layer_emb
        self.out_features = [features]*len(zooms)

        emb_zoom = grid_layer_emb.zoom

        if n_groups > 1:
            self.get_embedding_fcn = self.get_embeddings_from_var_idx
        else:
            self.get_embedding_fcn = self.get_embeddings

        self.embeddings = get_mg_embedding(grid_layer_emb, features, n_groups=n_groups, init_mode=init_mode)

        layer_confs_ = copy.deepcopy(layer_confs)

        self.layer_dict = nn.ModuleDict()
        self.fcn_dict = {}

        nh_dim = grid_layer_emb.adjc.shape[1]


        for k,zoom in enumerate(zooms):
            if zoom == emb_zoom:
                self.layer_dict[str(zoom)] = get_layer(features, [2, features], layer_confs=layer_confs_)
                self.fcn_dict[zoom] = self.downsample_embs

            elif zoom > emb_zoom:
                self.layer_dict[str(zoom)] = get_layer([nh_dim, features], [4**(zoom - emb_zoom), 2, features], layer_confs=layer_confs_)
                self.fcn_dict[zoom] = self.upsample_embs

            else:
                self.layer_dict[str(zoom)] = get_layer([4**(emb_zoom - zoom), features], [2, features], layer_confs=layer_confs_)
                self.fcn_dict[zoom] = self.downsample_embs
    
    def donothing(self, x, **kwargs):
        return x

    def get_embeddings(self, emb=None):
        return self.embeddings[emb['GroupEmbedder']*0]
    
    def get_embeddings_from_var_idx(self, emb=None):
        return self.embeddings[emb['GroupEmbedder']]
    
    
    def downsample_embs(self, layer, sample_configs={}, emb=None):
        embs = self.get_embedding_fcn(emb=emb)

        idx = self.grid_layer_emb.get_idx_of_patch(**sample_configs, return_local=False)

        idx = idx.view(idx.shape[0],1,1,-1,1)

        embs = torch.gather(embs, dim=-2, index=idx.expand(*embs.shape[:3], idx.shape[-2], embs.shape[-1]))
        
        embs = layer(embs, sample_configs=sample_configs, emb=emb)

        return embs
    
    def upsample_embs(self, layer, sample_configs={}, emb=None):
        embs = self.get_embedding_fcn(emb=emb)

        idx = get_nh_idx_of_patch(self.grid_layer_emb.adjc, **sample_configs, return_local=False)[0]
 
        idx = idx.view(idx.shape[0],1,1,-1,1)

        embs = torch.gather(embs, dim=-2, index=idx.expand(*embs.shape[:3], idx.shape[-2], embs.shape[-1]))
       
        embs = layer(embs, sample_configs=sample_configs, emb=emb)

    
        return embs


    def forward(self, x_zooms, sample_configs={}, emb=None,**kwargs):
        
        for zoom in x_zooms.keys():
            embs = self.fcn_dict[zoom](self.layer_dict[str(zoom)],sample_configs=sample_configs, emb=emb)

            embs = embs.view(*embs.shape[:3],-1, 2*embs.shape[-1])

            scale, shift = embs.chunk(2,dim=-1)

            x_zooms[zoom] = x_zooms[zoom]*scale + shift


        return x_zooms


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
                 zoom_diff
                ) -> None: 
      
        super().__init__() 

        self.zoom_diff = zoom_diff
        self.lin_layer = nn.Linear(in_features, out_features, bias=True) if in_features!= out_features else nn.Identity()

    def get_sum_residual(self, x, mask=None):
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
    

class NHConv(nn.Module):
    """
    Rearranges input to make the variable dimension centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, grid_layer: GridLayer, in_features, out_features, ranks_spatial=[], layer_confs={}):
        super().__init__()
        self.grid_layer = grid_layer
        
        rank_spatial_dict = {}
        if len(ranks_spatial)>0:
            for k,rank in enumerate(ranks_spatial):
                if grid_layer.zoom-k >= 0:
                    rank_spatial_dict[grid_layer.zoom-k] = rank

        layer_confs_ = copy.deepcopy(layer_confs)
        layer_confs_['ranks_spatial'] = rank_spatial_dict

        self.layer = get_layer([self.grid_layer.adjc.shape[1], in_features], out_features, layer_confs=layer_confs_)

    
    def forward(self, x: torch.Tensor, emb: Dict = None, mask: torch.Tensor = None, sample_configs: Dict = {}) -> torch.Tensor:

        x_nh, mask_nh = self.grid_layer.get_nh(x, **sample_configs, with_nh=True, mask=mask)

        x = self.layer(x_nh, emb=emb, sample_configs=sample_configs)

        return x
    

class ResNHConv(nn.Module):
    """
    Rearranges input to make the variable dimension centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, grid_layer: GridLayer, in_features, out_features, ranks_spatial=[], layer_confs={}):
        super().__init__()

        self.layer1 = NHConv(grid_layer, in_features, out_features, ranks_spatial=ranks_spatial, layer_confs=layer_confs)    
        self.layer2 = NHConv(grid_layer, out_features, out_features, ranks_spatial=ranks_spatial, layer_confs=layer_confs)    
        
        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(out_features)

        self.activation = nn.SiLU()
        
        self.skip_layer = LinEmbLayer(in_features, out_features, identity_if_equal=True)
    
    def forward(self, x: torch.Tensor, emb: Dict = None, mask: torch.Tensor = None, sample_configs: Dict = {}) -> torch.Tensor:
        
        x_res = self.skip_layer(x)

        x = self.norm1(x)

        x = self.activation(x)

        x = self.layer1(x, emb=emb, sample_configs=sample_configs)

        x = self.norm2(x)

        x = self.activation(x)

        x = self.layer2(x, emb=emb, sample_configs=sample_configs)

        x = x + x_res

        return x
    

class UpDownLayer(nn.Module):
   

    def __init__(self, grid_layer: GridLayer, in_features, out_features, in_zoom = None, out_zoom = None, with_nh=False, layer_confs={}):
        super().__init__()

        in_zoom = grid_layer.zoom if in_zoom is None else in_zoom

        self.grid_layer = grid_layer

        self.out_features = out_features

        if not isinstance(out_features, list):
            out_features = [out_features]

        if not isinstance(in_features, list):
            in_features = [in_features]
        
        if with_nh: 
            in_features = [self.grid_layer.adjc.shape[1]] + in_features

        if out_zoom is not None:
            zoom_diff = out_zoom - in_zoom

            if zoom_diff > 0:
                out_features = [4]*zoom_diff + out_features 

            elif zoom_diff < 0:
                in_features = [4] * abs(zoom_diff) + in_features

        self.layer = get_layer(in_features, out_features, layer_confs=layer_confs)
    

    def forward(self, x: torch.Tensor, emb= None, sample_configs: Dict = {}) -> torch.Tensor:
        
        x, mask_nh = self.grid_layer.get_nh(x, **sample_configs)
        
        x = self.layer(x, emb=emb, sample_configs=sample_configs)

        x = x.reshape(*x.shape[:3],-1,self.out_features)

        return x


class Res_UpDownLayer(nn.Module):
   

    def __init__(self, grid_layers: Dict, in_features, out_features, in_zoom, out_zoom, with_nh=False, ranks_spatial=[], layer_confs={}, interpolator_confs={}):
        super().__init__()

        self.grid_layer_in = grid_layers[str(in_zoom)]
        self.grid_layer_out = grid_layers[str(out_zoom)]

        self.up_down_conv = UpDownLayer(self.grid_layer_in, in_features, out_features, out_zoom=out_zoom, with_nh=True, layer_confs=layer_confs)
        self.nh_conv = NHConv(self.grid_layer_out, out_features, out_features, ranks_spatial=ranks_spatial, layer_confs=layer_confs)

        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(out_features)

        self.out_features = out_features

        self.activation = nn.SiLU()
        
        self.skip_layer = LinEmbLayer(in_features, out_features, identity_if_equal=True)

        #self.proj_layer = IWD_ProjLayer(grid_layers, in_zoom, out_zoom, interpolator_confs=interpolator_confs)
        self.proj_layer = ProjLayer(1, 1, zoom_diff=out_zoom-in_zoom)


    def forward(self, x: torch.Tensor, emb= None, sample_configs: Dict = {}) -> torch.Tensor:

        x_res = self.skip_layer(self.proj_layer(x, sample_configs=sample_configs), emb=emb, sample_configs=sample_configs)
        
        x = self.norm1(x)

        x = self.activation(x)

        x = self.up_down_conv(x, emb=emb, sample_configs=sample_configs)

        x = self.norm2(x)

        x = self.activation(x)

        x = self.nh_conv(x, emb=emb, sample_configs=sample_configs)

        x = x + x_res

        return x