import torch
import torch.nn.functional as F
import torch.nn as nn

from ..icon_grids.grid_layer import RelativeCoordinateManager


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

        self.nh_dist = self.grid_layers[str(global_level_no)].nh_dist

    def transform(self, x, coordinates=None, coordinates_ref=None, indices_sample=None, mask=None):
        
        indices_in = indices_sample["indices_layers"][self.global_level_in] if indices_sample is not None else None
        indices_ref = indices_sample["indices_layers"][self.global_level_no] if indices_sample is not None else None

        coordinates_rel = self.rel_coord_mngr(indices_in=indices_in,
                                              indices_ref=indices_ref,
                                              coordinates_in=coordinates,
                                              coordinates_ref=coordinates_ref,
                                              sample_dict=indices_sample,
                                              back=False)

        if self.nh_projection:
            x, mask = self.grid_layers[str(self.global_level_in)].get_nh(x, 
                                                                indices_in, 
                                                                indices_sample, 
                                                                mask=mask)
            
            
        return self.transform_(x, coordinates_rel, mask=mask)
        

    def inverse_transform(self, x, coordinates=None, coordinates_ref=None, indices_sample=None,  mask=None):
        
        indices_in = indices_sample["indices_layers"][self.global_level_in] if indices_sample is not None else None
        indices_ref = indices_sample["indices_layers"][self.global_level_no] if indices_sample is not None else None

        coordinates_rel = self.rel_coord_mngr(indices_in=indices_in,
                                              indices_ref=indices_ref,
                                              coordinates_in=coordinates,
                                              coordinates_ref=coordinates_ref,
                                              sample_dict=indices_sample,
                                              back=True)

        if self.nh_backprojection:
            x, mask = self.grid_layers[str(self.global_level_no)].get_nh(x, 
                                                                indices_ref, 
                                                                indices_sample, 
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
        
        self.sigma = nn.Parameter(grid_layers[str(global_level_no)].min_dist/(n_dist), requires_grad=sigma_learnable)
        
    
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
        c_shape = (-1, n, seq_ref, seq_in*nh_in, nv)

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
        c_shape = (-1, n, seq_ref, seq_in, nv)

        weights = self.get_weights(coordinates_rel)

        weights = weights.unsqueeze(dim=-3).repeat_interleave(nv, dim=-3)
        weights = weights.view(*c_shape, len(self.phis), len(self.mus), 1)

        if mask is not None:
            mask = mask.sum(dim=-2)==mask.shape[-2]
            mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_in, dim=-2)
        
        x = x.view(b, n, seq_ref, 1, nv, n_lon, n_lat, nc)
        
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
      
        grid_dist_out = self.nh_dist/2**0.5

        mus_lon = torch.linspace(-grid_dist_out, grid_dist_out, n_dist_lon)
        self.mus_lon = nn.Parameter(mus_lon, requires_grad=dist_learnable)
        
        mus_lat = torch.linspace(-grid_dist_out, grid_dist_out, n_dist_lat)
        self.mus_lat = nn.Parameter(mus_lat, requires_grad=dist_learnable)

        self.min_sigma = 1e-10

        sigma = torch.tensor([grid_dist_out/((n_dist_lon+n_dist_lat)/2), grid_layers[str(global_level_in)].nh_dist/2**0.5]).max()
        self.sigma = nn.Parameter(sigma, requires_grad=sigma_learnable)
                  

    def normal_dist(self, dists, mus):

        sigma = self.sigma.clamp(min=self.min_sigma)
        dists = dists.unsqueeze(dim=-1) - mus.view(1,1,1,-1)

        weights = torch.exp(-0.5 * (dists / sigma) ** 2)
        
        return weights

    def get_weights(self, coordinates_rel):

        weights_dists_lon = self.normal_dist(coordinates_rel[0], self.mus_lon)
        weights_dists_lat = self.normal_dist(coordinates_rel[1], self.mus_lat)

        weights = weights_dists_lon.unsqueeze(dim=-1) * (weights_dists_lat).unsqueeze(dim=-2)

        mask = torch.logical_or(coordinates_rel[0].abs() > (self.mus_lon.abs().max() + 1.5*self.sigma),
                                coordinates_rel[1].abs() > (self.mus_lat.abs().max() + 1.5*self.sigma))
        return weights, mask


    def transform_(self, x, coordinates_rel, mask=None):
        b, n, seq_ref, seq_in, nh_in = coordinates_rel[0].shape
        nv, nc = x.shape[-2:]
        c_shape = (-1, n, seq_ref, seq_in*nh_in, nv)
        weights, mask_cut = self.get_weights(coordinates_rel)

        weights = weights.unsqueeze(dim=-3).repeat_interleave(nv, dim=-3)
        
        weights = weights.view(*c_shape, len(self.mus_lon), len(self.mus_lat))

        if mask is not None:
       #     mask = mask.view(*c_shape)
       #     mask = torch.logical_or(mask, mask_cut)
            mask = mask.view(*c_shape,1,1)
        else:
            mask = mask_cut.repeat_interleave(nv, dim=-1).view(*c_shape,1,1)
        
        weights_weights = (mask == False)
        weights = (weights * weights_weights)

        mask = mask.view(*c_shape)
        mask = mask.sum(dim=-2)==mask.shape[-2]
        mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_ref, dim=-2)
        
        weights = weights/(weights.sum(dim=3,keepdim=True)+1e-10)

        x = x.view(*c_shape,1, 1, nc)
        x = (x * weights.unsqueeze(dim=-1)).sum(dim=3)
        
        x = x.view(-1, n, nv, len(self.mus_lon), len(self.mus_lat), nc)

        if mask is not None:
            mask = mask.view(b,n,-1)

        return x, mask
    
    def inverse_transform_(self, x, coordinates_rel, mask=None):

        b, n, seq_ref, seq_in, _ = coordinates_rel[0].shape
        b, n, seq_ref, nv, n_lon, n_lat, nc = x.shape
        c_shape = (-1, n, seq_ref, seq_in, nv)

        weights,_ = self.get_weights(coordinates_rel)

        weights = weights.unsqueeze(dim=-3).repeat_interleave(nv, dim=-3)
        weights = weights.view(*c_shape, len(self.mus_lon), len(self.mus_lat), 1)

        x = x.view(-1, n, seq_ref, 1, nv, n_lon, n_lat, nc)

        if mask is not None:
            mask = mask.view(b,n,seq_ref,nv)
            mask = mask.sum(dim=-2)==mask.shape[-2]
            mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_in, dim=-2)
        
        weights = weights/(weights.sum(dim=[2,-2,-3], keepdim=True) + 1e-10)

        x = (x * weights).sum(dim=[2, -2,-3])

        if mask is not None:
            mask = mask.view(b,-1,nv)

        return x.view(b,-1,nv, nc), mask
    
class polNormal_NoLayer(NoLayer):

    def __init__(self,
                 grid_layers, 
                 global_level_in, 
                 global_level_no,
                 n_dist=4,
                 n_phi=4,
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
        
        self.n_params = [n_dist, n_phi]
      
        grid_dist_out = self.nh_dist

        phis = torch.linspace(-torch.pi, torch.pi, n_phi+1)[:-1]
        self.phis =  nn.Parameter(phis, requires_grad=False)
        
        dists = torch.linspace(grid_layers[str(global_level_in)].min_dist, grid_dist_out, n_dist)
        self.dists = nn.Parameter(dists, requires_grad=dist_learnable)

        self.min_sigma = 1e-10

        sigma = torch.tensor([grid_dist_out/n_dist, grid_layers[str(global_level_in)].min_dist]).max()
        self.sigma = nn.Parameter(sigma, requires_grad=sigma_learnable)
                  

    def get_weights(self, coordinates_rel):
        
        mus_lon = torch.cos(self.phis).view(1,1,-1,1) * self.dists.view(1,1,1,-1)
        mus_lat = torch.sin(self.phis).view(1,1,-1,1) * self.dists.view(1,1,1,-1)

        sigma = self.sigma.clamp(min=self.min_sigma)

        dx = coordinates_rel[0].unsqueeze(dim=-1).unsqueeze(dim=-1) - mus_lon
        dy = coordinates_rel[1].unsqueeze(dim=-1).unsqueeze(dim=-1) - mus_lat
        
        weights = torch.exp(-0.5 * ((dx**2 + dy**2) / sigma** 2) )

        mask = (coordinates_rel[0]**2 + coordinates_rel[1]**2).sqrt() > (self.dists.max() + 1.5*self.sigma)

        return weights, mask


    def transform_(self, x, coordinates_rel, mask=None):
        _, n, seq_ref, seq_in, nh_in = coordinates_rel[0].shape
        nv, nc = x.shape[-2:]
        b = x.shape[0]

        c_shape = (-1, n, seq_ref, seq_in*nh_in, nv)

        weights, mask_cut = self.get_weights(coordinates_rel)

        weights = weights.unsqueeze(dim=-3).repeat_interleave(nv, dim=-3)
        
        weights = weights.view(*c_shape, len(self.phis), len(self.dists))

        if mask is not None:
       #     mask = mask.view(*c_shape)
        #    mask = torch.logical_or(mask, mask_cut.view(*c_shape[:-1],1))
            mask = mask.view(*c_shape,1,1)
        else:
            mask = mask_cut.repeat_interleave(nv, dim=-1).view(*c_shape[:-1],1)
        
        weights_weights = (mask == False)
        weights = (weights * weights_weights)

        mask = mask.view(*c_shape)
        mask = mask.sum(dim=-2)==mask.shape[-2]
        mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_ref, dim=-2)

        weights = weights/(weights.sum(dim=3,keepdim=True)+1e-10)

        x = x.view(*c_shape,1, 1, nc)
        x = (x * weights.unsqueeze(dim=-1)).sum(dim=3)
        
        x = x.view(b, n, nv, len(self.phis), len(self.dists), nc)

        if mask is not None:
            mask = mask.view(b,n,-1)

        return x, mask
    
    def inverse_transform_(self, x, coordinates_rel, mask=None):

        b, n, seq_ref, seq_in, _ = coordinates_rel[0].shape
        b, n, seq_ref, nv, n_lon, n_lat, nc = x.shape
        c_shape = (-1, n, seq_ref, seq_in, nv)

        weights,_ = self.get_weights(coordinates_rel)

        weights = weights.unsqueeze(dim=-3).repeat_interleave(nv, dim=-3)
        weights = weights.view(*c_shape, len(self.phis), len(self.dists), 1)

        x = x.view(b, n, seq_ref, 1, nv, n_lon, n_lat, nc)

        if mask is not None:
            mask = mask.view(b,n,seq_ref,nv)
            mask = mask.sum(dim=-2)==mask.shape[-2]
            mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_in, dim=-2)
        
        weights = weights/(weights.sum(dim=[2,-2,-3], keepdim=True) + 1e-10)

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
        c_shape = (-1, n, seq_ref, seq_in*nh_in, nv)

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
        c_shape = (-1, n*seq_ref, seq_in*nh_in, nv)

        weights = self.get_fft_weights(coordinates_rel, inv=False)

        weights = weights.unsqueeze(dim=-3).repeat_interleave(nv, dim=-3)
        weights = weights.view(*c_shape, len(self.freqs), len(self.angles), 1)

        if mask is not None:
            mask = mask.sum(dim=-2)==mask.shape[-2]
            mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_in, dim=-2)
        
        x = x.view(b, n*seq_ref, 1, nv, n_lon, n_lat, nc)

        weights = weights/(weights.abs().sum(dim=[-2,-3], keepdim=True))

        x = (x * weights).sum(dim=[-2,-3])/n_lon
        
        x = x.view(b, n, seq_ref, seq_in, nv, nc).sum(dim=2)

        x = x.real

        if mask is not None:
            mask = mask.view(b,-1,nv)

        return x.view(b,-1,nv, nc), mask


def get_no_layer(neural_operator_type, 
                 grid_layers, 
                 global_level_in, 
                 global_level_no,
                 n_params=[],
                 params_init=[],
                 params_learnable=[],
                 nh_projection=False,
                 nh_backprojection=False,
                 precompute_coordinates=True,
                 rotate_coordinate_system=True):
    
    if neural_operator_type == 'Normal_VM':
        no_layer = Normal_VM_NoLayer(grid_layers,
                            global_level_in,
                            global_level_no,
                            n_phi=n_params[0],
                            n_dist=n_params[1],
                            kappa_init=params_init[0],
                            kappa_learnable=params_learnable[0],
                            dist_learnable=params_learnable[1],
                            sigma_learnable=params_learnable[2],
                            nh_projection=nh_projection,
                            nh_backprojection=nh_backprojection,
                            precompute_coordinates=precompute_coordinates,
                            rotate_coord_system=rotate_coordinate_system)
                    
    elif neural_operator_type == 'Normal':
        no_layer = Normal_NoLayer(grid_layers,
                            global_level_in,
                            global_level_no,
                            n_dist_lon=n_params[0],
                            n_dist_lat=n_params[1],
                            dist_learnable=params_learnable[0],
                            sigma_learnable=params_learnable[1],
                            nh_projection=nh_projection,
                            nh_backprojection=nh_backprojection,
                            precompute_coordinates=precompute_coordinates,
                            rotate_coord_system=rotate_coordinate_system)
    
    elif neural_operator_type == 'polNormal':
        no_layer = polNormal_NoLayer(grid_layers,
                            global_level_in,
                            global_level_no,
                            n_phi=n_params[0],
                            n_dist=n_params[1],
                            dist_learnable=params_learnable[0],
                            sigma_learnable=params_learnable[1],
                            nh_projection=nh_projection,
                            nh_backprojection=nh_backprojection,
                            precompute_coordinates=precompute_coordinates,
                            rotate_coord_system=rotate_coordinate_system)
        
    elif neural_operator_type == 'FT':
        no_layer = FT_NOLayer(grid_layers,
                            global_level_in,
                            global_level_no,
                            n_freq_lon=n_params[0],
                            n_freq_lat=n_params[1],
                            freq_learnable=params_learnable[0],
                            nh_projection=nh_projection,
                            nh_backprojection=nh_backprojection,
                            precompute_coordinates=precompute_coordinates,
                            rotate_coord_system=rotate_coordinate_system)
    
    return no_layer