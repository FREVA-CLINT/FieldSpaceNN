import torch
import torch.nn as nn
import math
from .neural_operator import NoLayer

class polNormal_NoLayer(NoLayer):

    def __init__(self,
                 rcm,
                 zoom_encode,
                 zoom_no,
                 zoom_decode,
                 n_phi=4,
                 n_dist=4,
                 n_sigma=1,
                 dist_learnable=False,
                 sigma_learnable=True,
                 nh_in_encode=False, 
                 nh_in_decode=True,
                 precompute_encode=True,
                 precompute_decode=True,
                 normalize_to_mask=True
                ) -> None: 
    
        super().__init__(
            rcm,
            zoom_encode, 
            zoom_no,
            zoom_decode,
            nh_in_encode=nh_in_encode, 
            nh_in_decode=nh_in_decode,
            precompute_encode=precompute_encode,
            precompute_decode=precompute_decode,
            coord_system='cartesian')

        self.normalize_to_mask = normalize_to_mask

        self.n_no_tot = n_phi*n_dist*n_sigma

        self.n_params_no = [n_phi, n_dist, n_sigma]
                
        self.min_sigma = 1e-10
        self.min_var = 1e-10

        self.sd_activation = nn.Identity()
  
        phis = torch.linspace(-torch.pi, torch.pi, n_phi+1)[:-1]
    
        dists = torch.linspace(1/(n_dist*2), 1-1/(n_dist*2), n_dist)

        if n_sigma>1:
            sigma = torch.arange(1,2*n_sigma, 2)/(n_sigma*2)
        else:
            sigma = torch.tensor(1/(2*n_dist*math.sqrt(2*math.log(2)))).view(-1)

        self.dist_unit = self.no_nh_dist

        self.phis =  nn.Parameter(phis, requires_grad=False)
        self.dists = nn.Parameter(dists, requires_grad=dist_learnable)
        self.sigma = nn.Parameter(sigma, requires_grad=sigma_learnable)
        
     
    def get_spatial_weights(self, coordinates_rel, sigma):


        dists = self.dist_unit * self.sd_activation(self.dists)

        mus_lon = torch.cos(self.phis).view(1,1,-1,1) * dists.view(1,1,1,-1)
        mus_lat = torch.sin(self.phis).view(1,1,-1,1) * dists.view(1,1,1,-1)

        dx = coordinates_rel[0].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-3) - mus_lon
        dy = coordinates_rel[1].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-3) - mus_lat
        
        dx = dx.unsqueeze(dim=-1)
        dy = dy.unsqueeze(dim=-1)

        sigma = self.dist_unit * self.sd_activation(sigma)
        sigma = sigma.clamp(min=self.min_sigma).view(1,-1)
    
        weights = torch.exp(-0.5 * ((dx**2 + dy**2) / sigma** 2))

        return weights


    def mult_weights_t(self, x, weights, mask=None):
        _, n_out, seq_in, seq_in_nh, nvw, n_phi, n_dist, n_sigma = weights.shape
        b, n_in, nh_in, nv, nc = x.shape

        if n_out > n_in:
            weights = weights.view(weights.shape[0],n_in,-1,*weights.shape[3:])
            _, _, seq_in, seq_in_nh, nvw, n_phi, n_dist, n_sigma = weights.shape
            c_shape = (-1, n_in, 1, seq_in_nh)
        else:
            c_shape = (-1, n_out, seq_in, seq_in_nh)

        if mask is not None:
            if self.normalize_to_mask:
                norm = (mask.view(*c_shape, nv, *(1,)*(weights.dim()-5)) == False)
                weights = weights*norm
        
        if n_out>n_in:
            weights = weights/(weights.sum(dim=[3], keepdim=True)+1e-20)
            x = x.view(*c_shape, nv, *(1,)*(weights.dim()-5),nc)
            x = x * weights.unsqueeze(dim=-1)
            x = x.sum(dim=[3])
        else:
            weights = weights/(weights.sum(dim=[2,3], keepdim=True)+1e-20)
            x = x.view(*c_shape, nv, *(1,)*(weights.dim()-5),nc)
            x = x * weights.unsqueeze(dim=-1)
            x = x.sum(dim=[2,3]).unsqueeze(dim=3)

        return x


    def mult_weights_invt(self, x, weights, mask=None):
        _, n_out, seq_out, seq_in, nvw, n_phi, n_dist, n_sigma = weights.shape
        nv = x.shape[3]
        n_in = x.shape[1]

        if n_out > n_in:
            weights = weights.view(weights.shape[0],n_in,-1,*weights.shape[3:])
            _, _, seq_in, seq_in_nh, nvw, n_phi, n_dist, n_sigma = weights.shape
            c_shape = (-1, n_in, 1, seq_in_nh)
        else:
            c_shape = (-1, n_out, seq_out, seq_in)

        if mask is not None:
            if self.normalize_to_mask:
                norm = (mask.view(*c_shape, nv, *(1,)*(weights.dim()-5)) == False)
                weights = weights*norm

        weights = weights/(weights.sum(dim=[3,-1,-2,-3], keepdim=True)+1e-20)

        x = x.unsqueeze(dim=2) * weights.unsqueeze(dim=-1)

        return x.sum(dim=[3,-2,-3,-4])

    def transform_(self, x, coordinates_rel, mask=None, emb=None):
        _, n_out, seq_out, seq_in = coordinates_rel[0].shape
        nv, nc = x.shape[-2:]
        b = x.shape[0]

        weights = self.get_spatial_weights(coordinates_rel, self.sigma)

        x = self.mult_weights_t(x, weights, mask=mask)

        x = x.view(b, n_out, nv, *self.n_params_no, nc)

        return x
    
    def inverse_transform_(self, x, coordinates_rel, mask=None, emb=None):
        
        _, n_out, seq_out, seq_in = coordinates_rel[0].shape
        b, n, seq_in, nv, n_phi, n_dist, n_sigma, nc = x.shape
        c_shape = (-1, n, seq_in, nv)

        weights = self.get_spatial_weights(coordinates_rel, self.sigma)

        x = self.mult_weights_invt(x, weights, mask=mask)

        x = x.view(b,-1,nv,nc)

        return x