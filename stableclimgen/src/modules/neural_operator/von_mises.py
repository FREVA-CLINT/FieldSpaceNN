import torch
import torch.nn as nn
import math
from .neural_operator import NoLayer

class VonMises_NoLayer(NoLayer):

    def __init__(self,
                 grid_layer_encode,
                 grid_layer_no,
                 grid_layer_decode,
                 n_phi=4,
                 kappa=0.5,
                 sigma=0.2,
                 kappa_learnable=True,
                 sigma_learnable=False,
                 nh_in_encode=False, 
                 nh_in_decode=True,
                 precompute_encode=True,
                 precompute_decode=True,
                 rotate_coord_system=True
                ) -> None: 
    
        super().__init__(grid_layer_encode, 
                 grid_layer_no,
                 grid_layer_decode,
                 nh_in_encode=nh_in_encode, 
                 nh_in_decode=nh_in_decode,
                 precompute_encode=precompute_encode,
                 precompute_decode=precompute_decode,
                 coord_system='polar',
                 rotate_coord_system=rotate_coord_system)
        
        self.grid_layer_no = grid_layer_no
        self.n_no_tot = n_phi

        self.n_params_no = [n_phi]

        self.sd_activation = nn.Identity()
  
        phis = torch.linspace(-torch.pi, torch.pi, n_phi+1)[:-1]

        self.phis =  nn.Parameter(phis, requires_grad=False)
        self.kappa = nn.Parameter(torch.tensor(kappa), requires_grad=kappa_learnable)

        self.dist_unit = grid_layer_no.nh_dist

        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=sigma_learnable)
        
     
    def get_vm_dist_weights(self, coordinates_rel):

        dists, angles = coordinates_rel

        angles_weights = torch.cos(angles.unsqueeze(dim=-1) - self.phis.view(1,1,-1))

        vm_weights = torch.exp(self.kappa * angles_weights)

        sigma = self.sigma*self.dist_unit

        alpha = torch.exp(-0.5 * (dists**2/ sigma** 2))

        return vm_weights, alpha.unsqueeze(dim=-1)


    def mult_weights_t(self, x, weights, alpha, mask=None):

        _, n_out, seq_in, seq_in_nh, n_p = weights.shape
        b, n_in, nh_in, nv, nc = x.shape

        if n_out > n_in:
            weights = weights.view(weights.shape[0],n_in,-1,*weights.shape[3:])
            _, _, seq_in, seq_in_nh, nvw, n_phi, n_dist, n_sigma = weights.shape
            c_shape = (-1, n_in, 1, seq_in_nh)
        else:
            c_shape = (-1, n_out, seq_in, seq_in_nh)

        if mask is not None:
            norm = (mask.view(*c_shape, nv, *(1,)*(weights.dim()-5)) == False)
            weights = weights*norm

            mask = mask.view(*c_shape,nv)
            mask = mask.sum(dim=[-2,-3])==(mask.shape[-2]*mask.shape[-3])
            
            if n_out>n_in:
                mask = mask.unsqueeze(dim=-2).repeat_interleave(n_out//n_in, dim=-2)
        
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

            total_weight = (alpha).unsqueeze(dim=-1)
            total_weight_alpha = total_weight/total_weight.sum(dim=-2,keepdim=True)
            data_t_mean = (total_weight_alpha*data).sum(dim=-2)

            total_weight= (1-alpha.unsqueeze(dim=-1))*weights + alpha.unsqueeze(dim=-1)
            total_weight = total_weight/total_weight.sum(dim=-2,keepdim=True)
            data_t = (total_weight.unsqueeze(dim=-1)*data.unsqueeze(dim=-2)).sum(dim=-3)

            data_tot = torch.concat((data_t_mean.unsqueeze(dim=-2), data_t-data_t_mean.unsqueeze(dim=-2)),dim=1)
        return x, mask

    

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
            norm = (mask.view(*c_shape, nv, *(1,)*(weights.dim()-5)) == False)
            weights = weights*norm

            mask = mask.view(*c_shape,nv)

            mask = mask.sum(dim=[-2,-3])==(mask.shape[-2]*mask.shape[-3])
           
            mask = mask.unsqueeze(dim=-2).repeat_interleave(n_out//n_in, dim=-2)
 

        weights = weights/(weights.sum(dim=[3,-1,-2,-3], keepdim=True)+1e-20)

        x = x.unsqueeze(dim=2) * weights.unsqueeze(dim=-1)

        return x.sum(dim=[3,-2,-3,-4]), mask

    def transform_(self, x, coordinates_rel, mask=None, emb=None):
        _, n_out, seq_out, seq_in = coordinates_rel[0].shape
        nv, nc = x.shape[-2:]
        b = x.shape[0]

        weights, alpha = self.get_vm_dist_weights(coordinates_rel)

        x, mask = self.mult_weights_t(x, weights, alpha, mask=mask)

        if mask is not None:
            x = x.masked_fill_(mask.view(*x.shape[:4],*(1,)*(x.dim()-4)), 0.0)

        x = x.view(b, n_out, nv, *self.n_params_no, nc)

        if mask is not None:
            mask = mask.view(b,n_out,-1)

        return x, mask
    
    def inverse_transform_(self, x, coordinates_rel, mask=None, emb=None):
        
        _, n_out, seq_out, seq_in = coordinates_rel[0].shape
        b, n, seq_in, nv, n_phi, n_dist, n_sigma, nc = x.shape
        c_shape = (-1, n, seq_in, nv)

        weights = self.get_spatial_weights(coordinates_rel, self.sigma)

        x, mask = self.mult_weights_invt(x, weights, mask=mask)

        x = x.view(b,-1,nv,nc)

        if mask is not None:
            mask = mask.view(x.shape[:-1])
            x = x.masked_fill_(mask.unsqueeze(dim=-1), 0.0)

        return x, mask