from typing import List,Dict
import math
from scipy.special import sph_harm_y
import string

from einops import rearrange
import torch
import torch.nn as nn

import copy
from ..base import get_layer, IdentityLayer, LinEmbLayer, MLP_fac
from ...modules.grids.grid_layer import GridLayer, Interpolator, get_nh_idx_of_patch, get_idx_of_patch
from ...modules.embedding.embedding_layers import RandomFourierLayer
from ...modules.embedding.embedder import EmbedderSequential,MGEmbedder

from ...modules.embedding.embedder import get_embedder

from ..grids.grid_utils import insert_matching_time_patch, get_matching_time_patch,estimate_healpix_cell_radius_rad, decode_zooms, get_distance_angle,rotate_coord_system

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


def get_mg_embeddings(mg_emb_confs, grid_layers):
    mg_emeddings = nn.ParameterDict()
    diff_mode = mg_emb_confs.get('diff_mode', True)
    
    amplitude = 1
    wavelength_max = None
    for zoom, features, n_groups, init_method in zip(mg_emb_confs['zooms'], mg_emb_confs['features'], mg_emb_confs["n_groups"], mg_emb_confs['init_methods']):
        
        wavelength_min = estimate_healpix_cell_radius_rad(grid_layers[str(zoom)].adjc.shape[0])

        mg_emeddings[str(zoom)] = get_mg_embedding(
            grid_layers[str(zoom)],
            features,
            n_groups,
            init_mode=init_method,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            amplitude=amplitude)
        
        if diff_mode:
            wavelength_max = wavelength_min
            amplitude = 1e-3
        else:
            wavelength_max = None
            amplitude = 1

    return mg_emeddings


def get_mg_embedding(
        grid_layer_emb: GridLayer, 
        features, 
        n_groups, 
        init_mode='fourier_sphere',
        wavelength=1,
        wavelength_min=None,
        wavelength_max=None,
        random_rotation=False,
        amplitude=1):
    
    coords = grid_layer_emb.get_coordinates()       

    clon, clat = coords[...,0], coords[...,1]

    if init_mode=='random':
        embs = amplitude*torch.randn(1, coords.shape[-2], features)
    
    elif 'fourier_sphere' == init_mode:
        fourier_layer = RandomFourierLayer(in_features=2, n_neurons=features, wave_length=2*wavelength*torch.pi)
        embs = amplitude*fourier_layer(coords).squeeze(dim=-2)

    elif 'fourier' == init_mode:

        x = torch.cos(clat) * torch.cos(clon)
        y = torch.cos(clat) * torch.sin(clon)
        z = torch.sin(clat)

        coords_3d = torch.stack((x, y, z), dim=-1).float()

        fourier_layer = RandomFourierLayer(in_features=3, n_neurons=features, wave_length=wavelength)
        embs = amplitude*fourier_layer(coords_3d)
    
    elif "spherical_harmonics" == init_mode:

        if wavelength_min is not None:
            L_nyq = int(math.floor(math.pi / (2.0 * wavelength_min)))
        else:
            dtheta = estimate_healpix_cell_radius_rad(grid_layer_emb.adjc.shape[0])
            L_nyq = int(math.floor(math.pi / (2.0 * dtheta)))

        l_min = 0 if wavelength_max is None else math.floor(math.pi / (2.0 * wavelength_max))
        l_max = max(l_min + 1, L_nyq)


        ls = torch.randint(l_min, int(l_max), (features,))
        # sample m in [-l, l], excluding the empty interval if l=0
        ms = torch.stack([
                (-int(l.item()) + 2 * torch.randint(0, int(l.item()) + 1, (1,))).squeeze(0)
                for l in ls
            ])
        
        embs = torch.zeros(1, coords.shape[1], features)

        for k in range(features):
            l = int(ls[k].item())
            m = int(ms[k].item())

            if random_rotation:
                rotation_lon = torch.rand((1,))*torch.pi
                rotation_lat = torch.rand((1,))*torch.pi/2-torch.pi/4
                clon, clat = rotate_coord_system(clon, clat, rotation_lon, rotation_lat)
                clon, clat = clon.unsqueeze(dim=-1),clat.unsqueeze(dim=-1)

            Ylm = sph_harm_y(l, abs(m), clat, clon)

            if m == 0:
                Y_real = torch.as_tensor(Ylm.real, dtype=torch.float32)
            elif m > 0:
                Y_real = math.sqrt(2.0) * torch.as_tensor(Ylm.real, dtype=torch.float32)
            else:  
                Y_real = math.sqrt(2.0) * torch.as_tensor(Ylm.imag, dtype=torch.float32)

            embs[..., k] = amplitude * Y_real.view(1, -1)


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


class MFieldLayer(nn.Module):
  
    def __init__(self,
                 in_features: List,
                 out_features: List,
                 in_zooms: List[int],
                 grid_layers: GridLayer,
                 with_nh=True,
                 embed_confs={},
                 N = 2,
                 kmin = 0,
                 kmax= 0.5,
                 layer_confs = {}
                ) -> None: 
      
        super().__init__()

        self.proj_layers = nn.ModuleDict()
        self.out_zooms = in_zooms
        self.with_nh = with_nh
        self.in_features = in_features
        self.out_features = out_features
        
        zooms_sorted = [int(t) for t in torch.tensor(in_zooms).sort(descending=True).values]
        
        self.layers = nn.ModuleDict()
        for k,zoom in enumerate(in_zooms):
            embedder = get_embedder(**embed_confs, grid_layers=grid_layers, zoom=zoom)
            self.layers[str(zoom)] = FieldLayer(
                in_features[k], 
                out_features[k], 
                estimate_healpix_cell_radius_rad(12*4**zoom),
                embedder=embedder, 
                N=N, 
                kmin=kmin, 
                kmax=kmax, 
                layer_confs=layer_confs)

        self.cons_dict = dict(zip(zooms_sorted[:-1],zooms_sorted[1:]))
        self.cons_dict[zooms_sorted[-1]] = zooms_sorted[-1]

        self.in_zooms = in_zooms
        self.grid_layers = grid_layers

        
    
    def get_coordinates(self, in_zoom, out_zoom, sample_configs, coordinates_out=None):
        coordinates = self.grid_layers[str(in_zoom)].get_coordinates(**sample_configs[in_zoom])
        
        if self.with_nh:
            coordinates_in, mask_nh = self.grid_layers[str(in_zoom)].get_nh(coordinates.unsqueeze(dim=1),**sample_configs[in_zoom], with_nh=True)
        else:
            coordinates_in = coordinates.unsqueeze(dim=1)
        
        if coordinates_out is None and out_zoom==in_zoom:
            coordinates_out = coordinates.unsqueeze(dim=1).unsqueeze(dim=-2)

        elif out_zoom != in_zoom:
            coordinates_out = self.grid_layers[str(out_zoom)].get_coordinates(**sample_configs[out_zoom]).unsqueeze(dim=1)
            coordinates_out = coordinates_out.view(*coordinates_out.shape[:2], -1, 4**(out_zoom-in_zoom),2)

        coordinates_in = coordinates_in.unsqueeze(dim=-2)
        coordinates_out = coordinates_out.unsqueeze(dim=-3)

        dphi, dtheta = get_distance_angle(coordinates_in[...,0],coordinates_in[...,1], coordinates_out[...,0], coordinates_out[...,1], base='cartesian')

        return dphi, dtheta
    

    def forward(self, x_zooms, emb={}, sample_configs={}, out_zoom=None, **kwargs):
        
        decode = out_zoom is not None

        #sample_cfg = get_sample_configs(sample_configs, self.zoom)
        x_out = 0

        zooms = sorted(list(x_zooms.keys()), reverse=True)

        for zoom in zooms:
           # zoom_in = zooms[k]
           # zoom_out = zooms[k] if zoom_out is None else zoom_out

            out_zoom_ = zoom if out_zoom is None else out_zoom
            dphi, dtheta = self.get_coordinates(zoom, out_zoom_, sample_configs=sample_configs)

            if self.with_nh:
                x, _ = self.grid_layers[str(zoom)].get_nh(x_zooms[zoom],**sample_configs[zoom], with_nh=True)
            else:
                x = x_zooms[zoom].unsqueeze(dim=-2)

            x = self.layers[str(zoom)](x, dphi, dtheta, emb=emb, sample_configs=sample_configs[zoom])

            if decode:
                x_out += x
            else:
                x_zooms[zoom] = x
        
        if decode:
            return {out_zoom: x}
        else:
            return x_zooms
    

class FieldLayer(nn.Module):
  
    def __init__(self,
                 in_features,
                 out_features,
                 grid_dist,
                 N = 2,
                 kmin = 0,
                 kmax = 0.5,
                 embedder:EmbedderSequential=None,
                 layer_confs = {},
                 **kwargs
                ) -> None: 
      
        super().__init__() 

        self.N = N
        self.kmin = kmin
        self.kmax = kmax
        self.grid_dist = grid_dist

        self.proj_layer_c = get_layer([1, in_features], [N*4, out_features], layer_confs=layer_confs) if in_features!=out_features else IdentityLayer()


        if embedder is not None:
            self.emb_layer = embedder.embedders['MGEmbedder'] if embedder is not None else None

            self.wavenumber_mlp = MLP_fac(in_features=embedder.get_out_channels, out_features=N*4, layer_confs=layer_confs)
            self.encode_fcn = self.enc_angles_learned
        else:
            self.encode_fcn = self.enc_angles
        # different activation
       # self.wave_number_activation = nn.Sigmoid()

    
    def enc_angles(self, phi, theta, **kwargs):
        feats = []
        for k in torch.linspace(self.kmin, self.kmax, self.N):
            feats += [torch.sin(k*phi), torch.cos(k*phi),
                    torch.sin(k*theta), torch.cos(k*theta)]
        return torch.stack(feats, dim=-1)
    
    def enc_angles_learned(self, phi, theta, emb, sample_configs):
        #mg_embeddings -> k_numbers
        
        embs = self.emb_layer(emb['MGEmbedder'], sample_configs=sample_configs)

        wave_numbers = self.wavenumber_mlp(embs, emb=emb, sample_configs=sample_configs)

        wave_numbers = wave_numbers.clamp(min=0, max=1)
        c = wave_numbers.shape[-1]
        
        c1 = torch.cos(torch.pi * wave_numbers[...,:c//4].unsqueeze(dim=-2).unsqueeze(dim=-2)*phi.unsqueeze(dim=1).unsqueeze(dim=-1)/self.grid_dist)
        c2 = torch.sin(torch.pi * wave_numbers[...,c//4:c//2].unsqueeze(dim=-2).unsqueeze(dim=-2)*phi.unsqueeze(dim=1).unsqueeze(dim=-1)/self.grid_dist)
        c3 = torch.cos(torch.pi * wave_numbers[...,c//2:3*c//4].unsqueeze(dim=-2).unsqueeze(dim=-2)*theta.unsqueeze(dim=1).unsqueeze(dim=-1)/self.grid_dist)
        c4 = torch.sin(torch.pi * wave_numbers[...,3*c//4:].unsqueeze(dim=-2).unsqueeze(dim=-2)*theta.unsqueeze(dim=1).unsqueeze(dim=-1)/self.grid_dist)


        return torch.concat((c1,c2,c3,c4), dim=-1)


    def forward(self, x, dphi, dtheta, emb={}, sample_configs=None):
        
        
        # get frequencies
       # out = self.enc_angles(dphi, dtheta, self.N)

        out = self.encode_fcn(dphi, dtheta, emb=emb, sample_configs=sample_configs)

        b,v,t,n,nh,n_out,c = out.shape

        b,v,t,n,nh,c = x.shape

        x_out = self.proj_layer_c(x, emb=emb, sample_configs=sample_configs)

        x_out = (x_out.view(b,v,t,n,nh,1,out.shape[-1],-1) * out.unsqueeze(dim=-1)).sum(dim=[-2,-4])/nh
        
        x_out = x_out.view(*x_out.shape[:3],-1,x_out.shape[-1])

        return x_out
        

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
            self.layer = get_layer([1, in_features], [4**(out_zoom - self.grid_layer.in_zoom), out_features], layer_confs=layer_confs_)
            self.with_nh = False

        else:
            self.layer = get_layer([4**(self.grid_layer.in_zoom - out_zoom), in_features], [1, out_features], layer_confs=layer_confs_)
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

    def __init__(self, grid_layer_in: GridLayer, in_features, out_features, grid_layer_out=None, ranks_spatial=[], layer_confs={}, layer_confs_emb={}, embedder: EmbedderSequential=None, use_skip_conv=False):
        super().__init__()

        grid_layer_out = grid_layer_in if grid_layer_out is None else grid_layer_out

        
       # self.norm1 = nn.LayerNorm(in_features)
       # self.norm2 = nn.LayerNorm(out_features)

        self.norm1 = nn.GroupNorm(32, in_features)
        self.norm2 = nn.GroupNorm(32, out_features)

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

    def forward(self, x: torch.Tensor, emb: Dict = None, mask: torch.Tensor = None, 
                sample_configs: Dict={}, sample_configs_in: Dict = {}, sample_configs_out: Dict={}) -> torch.Tensor:
        
        sample_configs_in = sample_configs if len(sample_configs_in)==0 else sample_configs_in
        sample_configs_out = sample_configs if len(sample_configs_out)==0 else sample_configs_out

        x_res = self.skip_layer(x)
        
        b, v, t, s, f = x.shape

        x = rearrange(x, self.pattern_norm)
        x = self.norm1(x)
        x = rearrange(x, self.pattern_unnorm, b=b, v=v, t=t, s=s, f=f)

        x = self.activation(x)

        x = self.updownconv(x, emb=emb, sample_configs=sample_configs_in)

        x = self.conv1(x, emb=emb, sample_configs=sample_configs_out)

        x = self.emb_layer(x, emb=emb, sample_configs=sample_configs_out)

        b, v, t, s, f = x.shape

        x = rearrange(x, self.pattern_norm)
        x = self.norm2(x)
        x = rearrange(x, self.pattern_unnorm, b=b, v=v, t=t, s=s, f=f)

        x = self.activation(x)

        x = self.conv2(x, emb=emb, sample_configs=sample_configs_out)

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
                 use_skip_conv=False
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
                                    use_skip_conv=use_skip_conv)

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

class UpDownConvLayer(nn.Module):
   

    def __init__(self, in_features, out_features, in_zoom = None, out_zoom = None, with_nh=False, layer_confs={}):
        super().__init__()

        
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

        self.proj_layer = ProjLayer(in_features, out_features, zoom_diff=out_zoom-in_zoom)
    

    def forward(self, x: torch.Tensor, emb= None, sample_configs: Dict = {}) -> torch.Tensor:
        
                
        x = self.layer(x, emb=emb, sample_configs=sample_configs)

        x = x.reshape(*x.shape[:3],-1,self.out_features)

        return x

class UpDownNHConvLayer(nn.Module):
   

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