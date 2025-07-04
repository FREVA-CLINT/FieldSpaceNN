from typing import List,Dict
import math

import torch
import torch.nn as nn

import copy
from ..base import get_layer, IdentityLayer, LinEmbLayer
from ...modules.grids.grid_layer import GridLayer, Interpolator, get_nh_idx_of_patch, get_idx_of_patch
from ...modules.embedding.embedding_layers import RandomFourierLayer

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
        n_vars_total, 
        init_mode='fourier_sphere'):
    
    coords = grid_layer_emb.get_coordinates()
    
    
    if init_mode=='random':
        embs = torch.randn(1, coords.shape[-2], features)
    
    elif init_mode=='fourier_sphere':
        fourier_layer = RandomFourierLayer(in_features=2, n_neurons=features)
        embs = fourier_layer(coords)

    elif init_mode=='fourier':
        clon, clat = coords[...,0], coords[...,1]
        x = torch.cos(clat) * torch.cos(clon)
        y = torch.cos(clat) * torch.sin(clon)
        z = torch.sin(clat)

        coords_3d = torch.stack((x, y, z), dim=-1).float()

        fourier_layer = RandomFourierLayer(in_features=3, n_neurons=features)
        embs = fourier_layer(coords_3d)

    
    embs = embs.repeat_interleave(n_vars_total, dim=0)
    
    embs = nn.Parameter(embs, requires_grad=True)

    return embs

class MGEmbedding(nn.Module):
  
    def __init__(self,
                 grid_layer_emb: GridLayer,
                 features: int,
                 zooms: List,
                 n_vars_total: int =1,
                 init_mode='fourier_sphere',
                 layer_confs={}
                ) -> None: 
      
        super().__init__()
        self.grid_layer_emb = grid_layer_emb
        self.out_features = [features]*len(zooms)

        emb_zoom = grid_layer_emb.zoom

        if n_vars_total > 1:

            self.get_embedding_fcn = self.get_embeddings_from_var_idx
        else:

            self.get_embedding_fcn = self.get_embeddings

        self.embeddings = get_mg_embedding(grid_layer_emb, features, n_vars_total=n_vars_total, init_mode=init_mode)

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
        return self.embeddings[emb['VariableEmbedder']*0]
    
    def get_embeddings_from_var_idx(self, emb=None):
        return self.embeddings[emb['VariableEmbedder']]
    
    def get_embs(self, sample_dict={}, emb=None):
        embs = self.get_embedding_fcn(emb=emb)

        if 'patch_index' in sample_dict:
            idx = get_idx_of_patch(self.grid_layer_emb.adjc, **sample_dict, return_local=False)
        else:
            idx = self.grid_layer_emb.adjc[:,[0]].unsqueeze(dim=0)

        idx = idx.view(idx.shape[0],1,1,-1,1)

        embs = torch.gather(embs, dim=-2, index=idx.expand(*embs.shape[:3], idx.shape[-2], embs.shape[-1]))

        return embs
    
    def get_embs_with_nh(self, sample_dict={}, emb=None):
        embs = self.get_embedding_fcn(emb=emb)

        if 'patch_index' in sample_dict:
            idx, mask = get_nh_idx_of_patch(self.grid_layer_emb.adjc, **sample_dict, return_local=False)
        else:
            idx = self.grid_layer_emb.adjc.unsqueeze(dim=0)
            mask = torch.zeros_like(idx, device=idx.device, dtype=int)

        idx = idx.view(idx.shape[0],1,1,-1,1)
        mask = mask.view(idx.shape[0],1,1,-1,1)

        embs = torch.gather(embs, dim=-2, index=idx.expand(*embs.shape[:3], idx.shape[-2], embs.shape[-1]))

        return embs

    def sample_embs(self, layer, sample_dict=None, emb=None):
        embs = self.get_embs(sample_dict=sample_dict, emb=emb)
        
        emb = layer(embs, sample_dict=sample_dict, emb=emb)

        return emb
    
    def downsample_embs(self, layer, sample_dict=None, emb=None):
        embs = self.get_embs(sample_dict=sample_dict, emb=emb)
        
        embs = layer(embs, sample_dict=sample_dict, emb=emb)

        return embs
    
    def upsample_embs(self, layer, sample_dict=None, emb=None):
        embs = self.get_embedding_fcn(emb=emb)

        if 'patch_index' in sample_dict:
            idx, mask = get_nh_idx_of_patch(self.grid_layer_emb.adjc, **sample_dict, return_local=False)
        else:
            idx = self.grid_layer_emb.adjc
            mask = torch.zeros_like(idx, device=idx.device, dtype=int)

        idx = idx.view(idx.shape[0],1,1,-1,1)
        mask = mask.view(idx.shape[0],1,1,-1,1)

        embs = torch.gather(embs, dim=-2, index=idx.expand(*embs.shape[:3], idx.shape[-2], embs.shape[-1]))
       
        embs = layer(embs, sample_dict=sample_dict, emb=emb)

    
        return embs


    def forward(self, x_zooms, sample_dict={}, emb=None,**kwargs):
        
        for zoom in x_zooms.keys():
            embs = self.fcn_dict[zoom](self.layer_dict[str(zoom)],sample_dict=sample_dict, emb=emb)

            embs = embs.view(*embs.shape[:3],-1, 2*embs.shape[-1])

            scale, shift = embs.chunk(2,dim=-1)

            x_zooms[zoom] = x_zooms[zoom]*scale + shift


        return x_zooms

class ConservativeLayer(nn.Module):
  
    def __init__(self,
                 in_zooms: List[int],
                 first_feature_only=False
                ) -> None: 
      
        super().__init__()

        self.ffo = first_feature_only
        if first_feature_only:
            self.fwd_fcn = self.forward_ffo
        else:
            self.fwd_fcn = self.forward_all

        self.proj_layers = nn.ModuleDict()
        self.out_zooms = in_zooms
        
        zooms_sorted = [int(t) for t in torch.tensor(in_zooms).sort(descending=True).values]
        
        self.cons_dict = dict(zip(zooms_sorted[:-1],zooms_sorted[1:]))
        self.cons_dict[zooms_sorted[-1]] = zooms_sorted[-1]

        self.in_zooms = in_zooms


    def forward_all(self, x_zooms, sample_dict=None, **kwargs):

        for zoom in sorted(x_zooms.keys()):
            
            x = x_zooms[zoom]
            zoom_level_cons = zoom - self.cons_dict[zoom]

            if zoom_level_cons > 0:
                x = x.view(*x.shape[:3], -1, 4**zoom_level_cons, x.shape[-1]) 

                mean = x.mean(dim=-2)
                x = (x-mean.unsqueeze(dim=-2)).view(*x.shape[:3], -1, x.shape[-1])

                x_zooms[self.cons_dict[zoom]] = x_zooms[self.cons_dict[zoom]] + mean

                x_zooms[zoom] = x

        return x_zooms


    def forward_ffo(self, x_zooms, sample_dict=None, **kwargs):

        for zoom in sorted(x_zooms.keys()):
            
            x = x_zooms[zoom]
            zoom_level_cons = zoom - self.cons_dict[zoom]

            x = x[...,0]
            if zoom_level_cons > 0:
                x = x.view(*x.shape[:3], -1, 4**zoom_level_cons)

                mean = x.mean(dim=-1)
                x = (x - mean.unsqueeze(dim=-1)).view(*x.shape[:3], -1)
                
                x_zooms[self.cons_dict[zoom]][...,0] = x_zooms[self.cons_dict[zoom]][...,0] + mean
                x_zooms[zoom][...,0] = x

        return x_zooms
    

    def forward(self, x_zooms, sample_dict=None, **kwargs):

        x_zooms = self.fwd_fcn(x_zooms)

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
            x = x.unsqueeze(dim=-2).repeat_interleave(4**(self.zoom_diff), dim=-2)
            x = x.view(*x.shape[:3],-1,x.shape[-1])
            
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


    def forward(self, x, sample_dict=None):
      
        x,_ = self.interpolator(x.unsqueeze(dim=-2), calc_density=False, sample_dict=sample_dict)

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

    
    def forward(self, x: torch.Tensor, emb: Dict = None, mask: torch.Tensor = None, sample_dict: Dict = {}) -> torch.Tensor:

        x_nh, mask_nh = self.grid_layer.get_nh(x, **sample_dict, with_nh=True, mask=mask)

        x = self.layer(x_nh, emb=emb, sample_dict=sample_dict)

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
    
    def forward(self, x: torch.Tensor, emb: Dict = None, mask: torch.Tensor = None, sample_dict: Dict = {}) -> torch.Tensor:
        
        x_res = self.skip_layer(x)

        x = self.norm1(x)

        x = self.activation(x)

        x = self.layer1(x, emb=emb, sample_dict=sample_dict)

        x = self.norm2(x)

        x = self.activation(x)

        x = self.layer2(x, emb=emb, sample_dict=sample_dict)

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
    

    def forward(self, x: torch.Tensor, emb= None, sample_dict: Dict = {}) -> torch.Tensor:
        
        x, mask_nh = self.grid_layer.get_nh(x, **sample_dict)
        
        x = self.layer(x, emb=emb, sample_dict=sample_dict)

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


    def forward(self, x: torch.Tensor, emb= None, sample_dict: Dict = {}) -> torch.Tensor:

        x_res = self.skip_layer(self.proj_layer(x, sample_dict=sample_dict), emb=emb, sample_dict=sample_dict)
        
        x = self.norm1(x)

        x = self.activation(x)

        x = self.up_down_conv(x, emb=emb, sample_dict=sample_dict)

        x = self.norm2(x)

        x = self.activation(x)

        x = self.nh_conv(x, emb=emb, sample_dict=sample_dict)

        x = x + x_res

        return x