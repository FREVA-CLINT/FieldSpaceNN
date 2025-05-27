import torch
import torch.nn as nn
import math
from .neural_operator import NoLayer

class VonMises_NoLayer(NoLayer):

    def __init__(self,
                 rcm,
                 zoom_encode,
                 zoom_no,
                 zoom_decode,
                 n_phi=4,
                 kappa=0.5,
                 sigma=0.2,
                 kappa_learnable=True,
                 sigma_learnable=False,
                 nh_in_encode=False, 
                 nh_in_decode=True,
                 precompute_encode=True,
                 precompute_decode=True,
                 diff=True,
                 normalize_to_mask=True
                ) -> None: 
    
        super().__init__(rcm,
            zoom_encode, 
            zoom_no,
            zoom_decode,
            nh_in_encode=nh_in_encode, 
            nh_in_decode=nh_in_decode,
            precompute_encode=precompute_encode,
            precompute_decode=precompute_decode,
            coord_system='polar')
        
        self.normalize_to_mask = normalize_to_mask
        self.diff_mode=diff
        self.n_no_tot = n_phi

        self.n_params_no = [n_phi +1]

        self.sd_activation = nn.Identity()
  
        phis = torch.linspace(-torch.pi, torch.pi, n_phi+1)[:-1]

        self.phis =  nn.Parameter(phis, requires_grad=False)
        self.kappa = nn.Parameter(torch.tensor(kappa), requires_grad=kappa_learnable)

        self.dist_unit = self.no_nh_dist

        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=sigma_learnable)
        
     
    def get_vm_dist_weights(self, coordinates_rel):

        dists, angles = coordinates_rel

        angles_weights = torch.cos(angles.unsqueeze(dim=-1) - self.phis.view(1,1,-1))

        vm_weights = torch.exp(self.kappa * angles_weights)
        
        sigma = self.sigma*self.dist_unit

        alpha = torch.exp(-0.5 * (dists**2/ sigma** 2))

        return vm_weights.unsqueeze(dim=-2), alpha.unsqueeze(dim=-1).unsqueeze(dim=-1)


    def mult_weights_t(self, x, weights, alpha, mask=None):
        
        weights = weights/weights.sum(dim=[-1], keepdim=True)

        weights = weights * (1 - alpha)

        weights = torch.concat((alpha, weights), dim=-1)

        _, n_out, seq_in, seq_in_nh, _, n_p = weights.shape
        b, nt, n_in, nh_in, nv, nc = x.shape

        if n_out > n_in:
            weights = weights.view(weights.shape[0], 1, n_in,-1,*weights.shape[3:])
            _, _, seq_in, seq_in_nh = weights.shape[:4]
            c_shape = (-1, nt, n_in, 1, nh_in)
        else:
            weights = weights.unsqueeze(dim=1)
            c_shape = (-1, nt, n_out, seq_in, nh_in)

        if mask is not None:
            if self.normalize_to_mask:
                norm = (mask.view(-1, mask.shape[1], n_in, 1, nh_in, nv, *(1,)*(weights.dim()-6)) == False)
                weights = weights*norm

        
        if n_out>n_in:
            weights = weights/(weights.sum(dim=[4], keepdim=True)+1e-20)
            x = x.view(*c_shape, nv, *(1,)*(weights.dim()-6),nc)
            x = x * weights.unsqueeze(dim=-1)
            x = x.sum(dim=[4])
        else:
            weights = weights/(weights.sum(dim=[3,4], keepdim=True)+1e-20)
            x = x.view(*c_shape, nv, *(1,)*(weights.dim()-6),nc)
            x = x * weights.unsqueeze(dim=-1)
            x = x.sum(dim=[3,4]).unsqueeze(dim=3)

        if self.diff_mode:
            x = torch.concat((x[...,[0],:], x[...,[0],:] - x[...,1:,:]), dim=-2)
        
        return x



    def mult_weights_invt(self, x, weights, alpha, mask=None):
        _, n_out, seq_out, seq_in = weights.shape[:4]

        b, nt, n_in, _, nv, n_p, nc = x.shape

        weights = weights/weights.sum(dim=[-1], keepdim=True)

        weights = weights * (1-alpha) 
        weights = torch.concat((alpha, weights), dim=-1)

        if n_out >= n_in:
            weights = weights.view(weights.shape[0],1,n_in,-1,*weights.shape[3:],1)
            _, _, seq_in, seq_in_nh = weights.shape[:4]
            c_shape = (-1, nt, n_in, 1, seq_in_nh)
        else:
            weights = weights.unsqueeze(dim=1)
            c_shape = (-1, nt, n_out, seq_out, seq_in)

        if mask is not None:
            if self.normalize_to_mask:
                norm = (mask.view(*c_shape, nv, *(1,)*(weights.dim()-5)) == False)
                weights = weights*norm

        weights = weights/(weights.sum(dim=[4,-2], keepdim=True)+1e-20)

        if self.diff_mode:
            x = torch.concat((x[...,[0],:], x[...,[0],:] + x[...,1:,:]), dim=-2)

        x = x.unsqueeze(dim=3)

        x = (weights * x).sum(dim=[4,-2])

        return x

    def transform_(self, x, coordinates_rel, mask=None, emb=None):
        _, n_out, seq_out, seq_in = coordinates_rel[0].shape
        nv, nc = x.shape[-2:]
        b, nt = x.shape[:2]

        weights, alpha = self.get_vm_dist_weights(coordinates_rel)

        x = self.mult_weights_t(x, weights, alpha, mask=mask)

        x = x.view(b, nt, n_out, nv, *self.n_params_no, nc)

        return x
    
    def inverse_transform_(self, x, coordinates_rel, mask=None, emb=None):
        
        _, n_out, seq_out, seq_in = coordinates_rel[0].shape
        b, nt, n, seq_in, nv, n_p, nc = x.shape

        weights, alpha = self.get_vm_dist_weights(coordinates_rel)

        x = self.mult_weights_invt(x, weights, alpha, mask=mask)

        x = x.view(b,nt,-1,nv,nc)

        return x