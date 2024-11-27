import torch
import torch.nn.functional as F
import torch.nn as nn

from ..transformer.attention import ChannelVariableAttention,ResLayer
from ...utils.grid_utils_icon import get_distance_angle
from ..icon_grids.icon_grids import RelativeCoordinateManager


class NoLayer(nn.Module):

    def __init__(self, 
                 grid_layers, 
                 global_level_in, 
                 global_level_no,
                 nh_projection=False,
                 nh_backprojection=True,
                 seq_level_attention=2, 
                 precompute_coordinates=True,
                 rotate_coord_system=True,
                 coord_system='polar') -> None: 
        
        super().__init__()
        self.grid_layers = grid_layers
        self.global_level_in = global_level_in
        self.global_level_no = global_level_no
        self.nh_projection = nh_projection
        self.nh_backprojection = nh_backprojection
        self.rotate_coord_system = rotate_coord_system
        self.coord_system = coord_system

        self.rel_coord_mngr = RelativeCoordinateManager(
            self.grid_layers[str(global_level_in)],
            self.grid_layers[str(global_level_no)],
            nh_in= nh_projection,
            nh_ref=nh_backprojection,
            precompute = precompute_coordinates,
            coord_system=coord_system,
            rotate_coord_system=rotate_coord_system)


    def transform(self, x, indices_layers=None, coordinates=None, coordinates_ref=None, sample_dict=None, mask=None):
  
        coordinates_rel = self.rel_coord_mngr(indices_in=indices_layers[self.global_level_in],
                                              indices_ref=indices_layers[self.global_level_no],
                                              coordinates_in=coordinates,
                                              coordinates_ref=coordinates_ref,
                                              sample_dict=sample_dict,
                                              back=False)

        if self.nh_projection:
            x, mask = self.grid_layers[str(self.global_level_in)].get_nh(x, 
                                                                indices_layers[self.global_level_in], 
                                                                sample_dict, 
                                                                mask=mask)
            
            
        return self.transform_(x, coordinates_rel, mask=mask)
        

    def inverse_transform(self, x, indices_layers=None, coordinates=None, coordinates_ref=None, sample_dict=None, mask=None):

        coordinates_rel = self.rel_coord_mngr(indices_in=indices_layers[self.global_level_in],
                                              indices_ref=indices_layers[self.global_level_no],
                                              coordinates_in=coordinates,
                                              coordinates_ref=coordinates_ref,
                                              sample_dict=sample_dict,
                                              back=True)

        if self.nh_backprojection:
            x, mask = self.grid_layers[str(self.global_level_no)].get_nh(x, 
                                                                indices_layers[self.global_level_no], 
                                                                sample_dict, 
                                                                mask=mask)
        else: 
            x = x.unsqueeze(dim=2)
        
        return self.inverse_transform_(x, coordinates_rel, mask=mask)


    def transform_(self, x, coordinates_rel, mask=None):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def inverse_transform_(self, x, coordinates_rel, mask=None):
        raise NotImplementedError("This method should be implemented by subclasses.")



class Normal_VM_NoLayer(NoLayer):

    def __init__(self,
                 grid_layers, 
                 global_level_in, 
                 global_level_no,
                 n_phi=4,
                 n_dist=4,
                 kappa_init=2,
                 sigma_learnable=True,
                 kappa_learnable=False,
                 dist_learnable=False,
                 nh_projection=False, 
                 nh_backprojection=True,
                 seq_level_attention=2, 
                 precompute_coordinates=True,
                 rotate_coord_system=True
                ) -> None: 
    
        super().__init__(grid_layers, 
                 global_level_in, 
                 global_level_no, 
                 nh_projection=nh_projection, 
                 nh_backprojection=nh_backprojection,
                 seq_level_attention=seq_level_attention, 
                 precompute_coordinates=precompute_coordinates,
                 rotate_coord_system=rotate_coord_system)
        

        self.n_params = [n_phi, n_dist]

        phis = torch.linspace(-torch.pi, torch.pi, n_phi+1)[:-1]
        self.phis =  nn.Parameter(phis, requires_grad=False)

        self.kappa = nn.Parameter(torch.tensor(kappa_init), requires_grad=kappa_learnable)

        mus = torch.linspace(0, grid_layers[str(global_level_no)].min_dist, n_dist)
        self.mus = nn.Parameter(mus, requires_grad=dist_learnable)
        
        self.min_sigma = 1e-10
        
        self.sigma = nn.Parameter(grid_layers[str(global_level_in)].min_dist, requires_grad=sigma_learnable)
        
    
    def vm_weights(self, coordinates_rel):
        dists, angles = coordinates_rel
        
        angles = torch.cos(angles.unsqueeze(dim=-1) - self.phis.view(1,1,1,-1))

        weights = torch.exp(self.kappa * angles)/torch.exp(self.kappa)

        weights[dists==0]=1#/angles.shape[-1]

        return weights
           

    def normal_dist(self, dists, mus):

        sigma = self.sigma.clamp(min=self.min_sigma)
        dists = dists.unsqueeze(dim=-1) - mus.view(1,1,1,-1)

        weights = torch.exp(-0.5 * (dists / sigma) ** 2)
        
        return weights

    def get_weights(self, coordinates_rel):

        weights_dists = self.normal_dist(coordinates_rel[0], self.mus)
        weights_vm = self.vm_weights(coordinates_rel)

        weights = weights_vm.unsqueeze(dim=-1) * (weights_dists).unsqueeze(dim=-2)

        return weights


    def transform_(self, x, coordinates_rel, mask=None):
        b, n, seq_ref, seq_in, nh_in = coordinates_rel[0].shape
        nv, nc = x.shape[-2:]
        c_shape = (b, n, seq_ref, seq_in*nh_in, nv)

        weights = self.get_weights(coordinates_rel)

        weights = weights.unsqueeze(dim=-3).repeat_interleave(nv, dim=-3)
        
        weights = weights.view(*c_shape, len(self.phis), len(self.mus))

        if mask is not None:
            mask = mask.view(*c_shape,1,1)
            weights_weights = (mask == False)
            weights = (weights * weights_weights) + 1e-10

            mask = mask.view(*c_shape)
            mask = mask.sum(dim=-2)==mask.shape[-2]
            mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_ref, dim=-2)

        
        weights = weights/weights.sum(dim=3,keepdim=True)

        x = x.view(*c_shape,1, 1, nc)
        x = (x * weights.unsqueeze(dim=-1)).sum(dim=3)
        
        x = x.view(b, n, nv, len(self.phis), len(self.mus), nc)

        if mask is not None:
            mask = mask.view(b,n,-1)

        return x, mask
    
    def inverse_transform_(self, x, coordinates_rel, mask=None):

        b, n, seq_ref, seq_in, _ = coordinates_rel[0].shape
        b, n, seq_ref, nv, n_lon, n_lat, nc = x.shape
        c_shape = (b, n, seq_ref, seq_in, nv)

        weights = self.get_weights(coordinates_rel)

        weights = weights.unsqueeze(dim=-3).repeat_interleave(nv, dim=-3)
        weights = weights.view(*c_shape, len(self.phis), len(self.mus))

        if mask is not None:
            mask = mask.sum(dim=-2)==mask.shape[-2]
            mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_in, dim=-2)
        
        x = x.view(b, n, seq_ref, 1, nv, n_lon, n_lat, nc)
        
        weights = weights.view(b,n,seq_ref,-1,nv,n_lon,n_lat,1)
        weights = weights/(weights.sum(dim=[2,-2,-3], keepdim=True)+1e-10)

        x = (x * weights).sum(dim=[2, -2,-3])

        if mask is not None:
            mask = mask.view(b,-1,nv)

        return x.view(b,-1,nv, nc), mask
    

class Normal_NoLayer(NoLayer):

    def __init__(self,
                 grid_layers, 
                 global_level_in, 
                 global_level_no,
                 n_dist_lon=4,
                 n_dist_lat=4,
                 sigma_learnable=True,
                 dist_learnable=False,
                 nh_projection=False, 
                 nh_backprojection=True,
                 seq_level_attention=2, 
                 precompute_coordinates=True,
                 rotate_coord_system=True
                ) -> None: 
    
        super().__init__(grid_layers, 
                 global_level_in, 
                 global_level_no, 
                 nh_projection=nh_projection, 
                 nh_backprojection=nh_backprojection,
                 seq_level_attention=seq_level_attention, 
                 precompute_coordinates=precompute_coordinates,
                 coord_system='cartesian',
                 rotate_coord_system=rotate_coord_system)
        
        self.n_params = [n_dist_lon, n_dist_lat]
      
        grid_dist_out = grid_layers[str(global_level_no)].min_dist

        mus_lon = torch.linspace(-grid_dist_out, grid_dist_out, n_dist_lon)
        self.mus_lon = nn.Parameter(mus_lon, requires_grad=dist_learnable)
        
        mus_lat = torch.linspace(-grid_dist_out, grid_dist_out, n_dist_lat)
        self.mus_lat = nn.Parameter(mus_lat, requires_grad=dist_learnable)

        self.min_sigma = 1e-10
        
        self.sigma = nn.Parameter(grid_layers[str(global_level_in)].min_dist, requires_grad=sigma_learnable)
                  

    def normal_dist(self, dists, mus):

        sigma = self.sigma.clamp(min=self.min_sigma)
        dists = dists.unsqueeze(dim=-1) - mus.view(1,1,1,-1)

        weights = torch.exp(-0.5 * (dists / sigma) ** 2)
        
        return weights

    def get_weights(self, coordinates_rel):

        weights_dists_lon = self.normal_dist(coordinates_rel[0], self.mus_lon)
        weights_dists_lat = self.normal_dist(coordinates_rel[1], self.mus_lat)

        weights = weights_dists_lon.unsqueeze(dim=-1) * (weights_dists_lat).unsqueeze(dim=-2)

        return weights


    def transform_(self, x, coordinates_rel, mask=None):
        b, n, seq_ref, seq_in, nh_in = coordinates_rel[0].shape
        nv, nc = x.shape[-2:]
        c_shape = (b, n, seq_ref, seq_in*nh_in, nv)

        weights = self.get_weights(coordinates_rel)

        weights = weights.unsqueeze(dim=-3).repeat_interleave(nv, dim=-3)
        
        weights = weights.view(*c_shape, len(self.mus_lon), len(self.mus_lat))

        if mask is not None:
            mask = mask.view(*c_shape,1,1)
            weights_weights = (mask == False)
            weights = (weights * weights_weights) + 1e-10

            mask = mask.view(*c_shape)
            mask = mask.sum(dim=-2)==mask.shape[-2]
            mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_ref, dim=-2)
        else:
            norm=1
        
        weights = weights/weights.sum(dim=3,keepdim=True)

        x = x.view(*c_shape,1, 1, nc)
        x = (x * weights.unsqueeze(dim=-1)).sum(dim=3)
        
        x = x.view(b, n, nv, len(self.mus_lon), len(self.mus_lat), nc)

        if mask is not None:
            mask = mask.view(b,n,-1)

        return x, mask
    
    def inverse_transform_(self, x, coordinates_rel, mask=None):

        b, n, seq_ref, seq_in, _ = coordinates_rel[0].shape
        b, n, seq_ref, nv, n_lon, n_lat, nc = x.shape
        c_shape = (b, n, seq_ref, seq_in, nv)

        weights = self.get_weights(coordinates_rel)

        weights = weights.unsqueeze(dim=-3).repeat_interleave(nv, dim=-3)
        weights = weights.view(*c_shape, len(self.mus_lon), len(self.mus_lat))

        x = x.view(b, n, seq_ref, 1, nv, n_lon, n_lat, nc)
        weights = weights.view(b,n,seq_ref,-1,nv,n_lon,n_lat,1)

        if mask is not None:
            mask = mask.view(b,n,seq_ref,nv)
            mask = mask.sum(dim=-2)==mask.shape[-2]
            mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_in, dim=-2)
        
        weights = weights/weights.sum(dim=[2,-2,-3], keepdim=True)

        x = (x * weights).sum(dim=[2, -2,-3])

        if mask is not None:
            mask = mask.view(b,-1,nv)

        return x.view(b,-1,nv, nc), mask
    

class FT_NOLayer(NoLayer):

    def __init__(self,
                 grid_layers, 
                 global_level_in, 
                 global_level_no,
                 n_freq_lon=4,
                 n_freq_lat=4,
                 freq_learnable=False,
                 nh_projection=False, 
                 nh_backprojection=True,
                 seq_level_attention=2, 
                 precompute_coordinates=True,
                 rotate_coord_system=True
                ) -> None: 
    
        super().__init__(grid_layers, 
                 global_level_in, 
                 global_level_no, 
                 nh_projection=nh_projection, 
                 nh_backprojection=nh_backprojection,
                 seq_level_attention=seq_level_attention, 
                 precompute_coordinates=precompute_coordinates,
                 coord_system='polar',
                 rotate_coord_system=rotate_coord_system)
        
        self.n_params = [n_freq_lon, n_freq_lat]

        self.min_freq = 0

        max_wl = (2*grid_layers[str(global_level_no)].min_dist)
        min_wl = (0.5*grid_layers[str(global_level_in)].min_dist)

        freqs = 1/torch.linspace(min_wl, max_wl, n_freq_lon)
  
        self.freqs = nn.Parameter(freqs, requires_grad=freq_learnable)
        
        angles = torch.linspace(0, torch.pi, n_freq_lat)
        self.angles = nn.Parameter(angles, requires_grad=False)

        self.max_freq = 1/min_wl

    def get_fft_weights(self, dists, mask=None, inv=False):
        
        dists, angles = dists[0], dists[1]

        freqs = self.freqs.clamp(max=self.max_freq)

        dir1 = torch.cos(angles.unsqueeze(dim=-1) - self.angles.view(1,1,1,-1))
   
        amp = dists.unsqueeze(dim=-1) * freqs.view(1,1,1,-1)

        p1 = amp.unsqueeze(dim=-1)*dir1.unsqueeze(dim=-2)

        if inv:
            ft = torch.exp(-1j* 2*torch.pi*(p1))
        else:
            ft = torch.exp(1j* 2*torch.pi*(p1))
        
        return ft



    def transform_(self, x, coordinates_rel, mask=None):
        b, n, seq_ref, seq_in, nh_in = coordinates_rel[0].shape
        nv, nc = x.shape[-2:]
        c_shape = (b, n, seq_ref, seq_in*nh_in, nv)

        weights = self.get_fft_weights(coordinates_rel, mask=mask.view(*c_shape))

        weights = weights.unsqueeze(dim=-3).repeat_interleave(nv, dim=-3)
        
        weights = weights.view(*c_shape, len(self.freqs), len(self.angles))

        if mask is not None:
            mask = mask.view(*c_shape,1,1)
            #norm = (mask == False).sum(dim=3, keepdim=True)
            weights_weights = (mask == False)
            weights = (weights * weights_weights) + 1e-20

            mask = mask.view(*c_shape)
            mask = mask.sum(dim=-2)==mask.shape[-2]
            mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_ref, dim=-2)

        
        weights = weights/(weights.abs().sum(dim=3, keepdim=True))
 
        x = x.view(*c_shape,1, 1, nc)
        x = (x * weights.unsqueeze(dim=-1)).sum(dim=3)
        
        x = x.view(b, n, nv, len(self.freqs), len(self.angles), nc)

        if mask is not None:
            mask = mask.view(b,n,-1)

        return x, mask
    
    def inverse_transform_(self, x, coordinates_rel, mask=None):

        b, n, seq_ref, seq_in, nh_in = coordinates_rel[0].shape
        nv, n_lon, n_lat, nc = x.shape[-4:]
        c_shape = (b, n*seq_ref, seq_in*nh_in, nv)

        weights = self.get_fft_weights(coordinates_rel, inv=False)

        weights = weights.unsqueeze(dim=-3).repeat_interleave(nv, dim=-3)
        weights = weights.view(*c_shape, len(self.freqs), len(self.angles))

        if mask is not None:
            mask = mask.sum(dim=-2)==mask.shape[-2]
            mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_in, dim=-2)
        
        x = x.view(b, n*seq_ref, 1, nv, n_lon, n_lat, nc)

        weights = weights/(weights.abs().sum(dim=[-1,-2], keepdim=True))

        x = (x * weights.unsqueeze(dim=-1)).sum(dim=[-2,-3])/n_lon
        
        x = x.view(b, n, seq_ref, seq_in, nv, nc).sum(dim=2)

        x = x.real

        if mask is not None:
            mask = mask.view(b,-1,nv)

        return x.view(b,-1,nv, nc), mask