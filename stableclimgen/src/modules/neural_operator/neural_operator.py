import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from ..icon_grids.grid_layer import RelativeCoordinateManager


class NoLayer(nn.Module):

    def __init__(self, 
                 grid_layer_in, 
                 grid_layer_no,
                 grid_layer_out=None,
                 nh_projection=False,
                 nh_backprojection=True,
                 precompute_coordinates=True,
                 rotate_coord_system=True,
                 coord_system='polar') -> None: 
        
        super().__init__()
        self.grid_layer_in = grid_layer_in
        self.grid_layer_no = grid_layer_no

        self.global_level_in = int(grid_layer_in.global_level)
        self.global_level_no = int(grid_layer_no.global_level)
        self.nh_projection = nh_projection
        self.nh_backprojection = nh_backprojection
        self.rotate_coord_system = rotate_coord_system
        self.coord_system = coord_system

        self.rel_coord_mngr = RelativeCoordinateManager(
            grid_layer_in,
            grid_layer_no,
            nh_in= nh_projection,
            nh_ref=nh_backprojection,
            precompute = precompute_coordinates,
            coord_system=coord_system,
            rotate_coord_system=rotate_coord_system)

        self.nh_dist = grid_layer_no.nh_dist

    def transform(self, x, coordinates=None, coordinates_ref=None, indices_sample=None, mask=None, emb=None):
        
        indices_in  = indices_sample["indices_layers"][self.global_level_in] if indices_sample is not None else None
        indices_ref = indices_sample["indices_layers"][self.global_level_no] if indices_sample is not None else None

        coordinates_rel = self.rel_coord_mngr(indices_in=indices_in,
                                              indices_ref=indices_ref,
                                              coordinates_in=coordinates,
                                              coordinates_ref=coordinates_ref,
                                              sample_dict=indices_sample,
                                              back=False)

        if self.nh_projection:
            x, mask = self.grid_layer_in.get_nh(x, indices_in, indices_sample, mask=mask)
            
            
        return self.transform_(x, coordinates_rel, mask=mask, emb=emb)
        

    def inverse_transform(self, x, coordinates=None, coordinates_ref=None, indices_sample=None,  mask=None, emb=None):
        
        indices_out = indices_sample["indices_layers"][self.global_level_in] if indices_sample is not None else None
        indices_ref = indices_sample["indices_layers"][self.global_level_no] if indices_sample is not None else None

        coordinates_rel = self.rel_coord_mngr(indices_in=indices_out,
                                              indices_ref=indices_ref,
                                              coordinates_in=coordinates,
                                              coordinates_ref=coordinates_ref,
                                              sample_dict=indices_sample,
                                              back=True)

        if self.nh_backprojection:
            x, mask = self.grid_layer_no.get_nh(x, indices_ref, indices_sample, mask=mask)
        else: 
            x = x.unsqueeze(dim=2)
        
        
        return self.inverse_transform_(x, coordinates_rel, mask=mask, emb=emb)



    def transform_(self, x, coordinates_rel, mask=None, emb=None):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def inverse_transform_(self, x, coordinates_rel, mask=None, emb=None):
        raise NotImplementedError("This method should be implemented by subclasses.")


    
class polNormal_NoLayer(NoLayer):

    def __init__(self,
                 grid_layer_in,
                 grid_layer_no,
                 n_amplitudes_in: int,
                 n_amplitudes_out: int,
                 n_amplitdues_inv_in: int,
                 n_amplitudes_inv_out: int,
                 n_phi=4,
                 n_dist=4,
                 n_sigma=1,
                 avg_phi=False,
                 avg_dist=False,
                 avg_sigma=False,
                 dist_learnable=False,
                 sigma_learnable=True,
                 amplitudes_learnable=False,
                 nh_projection=False, 
                 nh_backprojection=True,
                 precompute_coordinates=True,
                 rotate_coord_system=True,
                 pretrained_weights=None,
                 n_var_amplitudes=1,
                 with_res=False,
                ) -> None: 
    
        super().__init__(grid_layer_in, 
                 grid_layer_no,
                 nh_projection=nh_projection, 
                 nh_backprojection=nh_backprojection,
                 precompute_coordinates=precompute_coordinates,
                 coord_system='cartesian',
                 rotate_coord_system=rotate_coord_system)
        
        self.grid_layer_no = grid_layer_no
        self.n_no_tot = n_phi*n_dist*n_sigma

        self.n_params_in = [n_phi, n_dist, n_sigma, n_amplitudes_out, n_amplitudes_in]

        self.n_params_no = [n_phi if not avg_phi else 1,
                            n_dist if not avg_dist else 1,
                            n_sigma if not avg_sigma else 1,
                            n_amplitudes_out]
        
        self.n_params_inv_in = [n_phi if not avg_phi else 1,
                            n_dist if not avg_dist else 1,
                            n_sigma if not avg_sigma else 1,
                            n_amplitdues_inv_in]
        
        self.n_params_out = [n_phi if not avg_phi else 1,
                            n_dist if not avg_dist else 1,
                            n_sigma if not avg_sigma else 1,
                            n_amplitudes_inv_out]
        
        norm_dims = [n_phi if avg_phi else 1,
                     n_dist if  avg_dist else 1,
                     n_sigma if avg_sigma else 1]
        
        self.norm_dims = norm_dims
        
        self.norm_factor = nn.Parameter(torch.tensor(norm_dims).prod(), requires_grad=False)

        self.sum_dims_params = []
        if avg_phi:
            self.sum_dims_params.append(-4)

        if avg_dist:
            self.sum_dims_params.append(-3)
        
        if avg_sigma:
            self.sum_dims_params.append(-2)

        self.n_params_inv_out = n_amplitudes_inv_out

        self.avg_phi, self.avg_dist, self.avg_sigma = avg_phi, avg_dist, avg_sigma
        grid_dist_out = self.nh_dist

        self.min_sigma = 1e-10

    
        if pretrained_weights is None:
            phis = torch.linspace(-torch.pi, torch.pi, n_phi+1)[:-1]
            dists = torch.linspace(0, grid_layer_no.nh_dist, n_dist+2)[1:-1]
            sigma = torch.cumsum(dists.diff(), dim=0)[:n_sigma]
            sigma_inv = sigma
           # sigma = (torch.randn(n_amplitudes_out)*grid_layer_in.nh_dist).abs()
           # sigma_inv = (torch.randn(n_amplitudes_inv_out)*grid_layer_no.nh_dist).abs()

          #  sigma = torch.logspace(math.log10(grid_layer_in.nh_dist/10), math.log10(grid_layer_no.nh_dist),n_amplitudes_out)
          #  sigma_inv = torch.logspace(math.log10(grid_layer_in.nh_dist/10), math.log10(grid_layer_no.nh_dist),n_amplitudes_inv_out)

            #sigma = torch.linspace(grid_layer_no.nh_dist/2, grid_layer_no.nh_dist, n_sigma)

        else:
            phis = pretrained_weights['phis']
            dists = pretrained_weights['dists']
            sigma = pretrained_weights['sigma']

        self.phis =  nn.Parameter(phis, requires_grad=False)
        self.dists = nn.Parameter(dists, requires_grad=dist_learnable)
        self.sigma = nn.Parameter(sigma, requires_grad=True)
        self.sigma_inv = nn.Parameter(sigma_inv, requires_grad=True)
        
        self.lin_layer= nn.Linear(n_amplitudes_in, n_amplitudes_out)

        inv_dim = torch.tensor(norm_dims).prod()
        self.inv_lin_layer = nn.Linear(n_amplitdues_inv_in, inv_dim*n_amplitudes_inv_out)

        self.with_res = with_res

    def get_amplitudes_no(self, emb):
        amps = self.amplitudes_no

        if amps.shape[0] > 1:            
            amps =  amps[emb['VariableEmbedder']]
            amps = amps.view(amps.shape[0],1,1,1,1,*amps.shape[1:])
        
        return amps.unsqueeze(dim=-3)

    def get_amplitudes_out(self, emb):
        amps = self.amplitudes_out

        if amps.shape[0] > 1:            
            amps =  amps[emb['VariableEmbedder']]
        
        amps = amps.view(amps.shape[0],1,1,1,*amps.shape[1:])

        return amps.unsqueeze(dim=-2)
    
    def get_spatial_weights(self, coordinates_rel):
        
        mus_lon = torch.cos(self.phis).view(1,1,-1,1) * self.dists.view(1,1,1,-1)
        mus_lat = torch.sin(self.phis).view(1,1,-1,1) * self.dists.view(1,1,1,-1)

        dx = coordinates_rel[0].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-3) - mus_lon
        dy = coordinates_rel[1].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-3) - mus_lat
        
        dx = dx.unsqueeze(dim=-1)
        dy = dy.unsqueeze(dim=-1)

        sigma = self.sigma.clamp(min=self.min_sigma).view(1,-1)
    
        weights = torch.exp(-0.5 * ((dx**2 + dy**2) / sigma** 2))#/(2*torch.pi*sigma**2).sqrt()

        return weights


    def get_spatial_weights_dphi(self, coordinates_rel):
        
        f1 = torch.cos(self.phis).view(1,1,-1,1) * self.dists.view(1,1,1,-1)
        f2 = torch.sin(self.phis).view(1,1,-1,1) * self.dists.view(1,1,1,-1)

        f1 = coordinates_rel[1].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-3) * f1
        f2 = coordinates_rel[0].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-3) * f2
        
        f1 = f1.unsqueeze(dim=-1)
        f2 = f2.unsqueeze(dim=-1)

        sigma = self.sigma.clamp(min=self.min_sigma).view(1,-1)
    
        weights_dphi = (f2 -f1) / sigma** 2

        return weights_dphi


    def get_spatial_weights_dd(self, coordinates_rel):
        
        f1 = coordinates_rel[0].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-3) * torch.cos(self.phis).view(1,1,-1,1)
        f2 = coordinates_rel[1].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-3) * torch.sin(self.phis).view(1,1,-1,1)
        
        f1 = f1.unsqueeze(dim=-1)
        f2 = f2.unsqueeze(dim=-1)

        sigma = self.sigma.clamp(min=self.min_sigma).view(1,-1)
    
        weights_dphi = (f2 +f1 - self.dists.view(1,1,1,-1)) / sigma** 2

        return weights_dphi

    def get_spatial_weights_res(self, coordinates_rel):
        
        d = (coordinates_rel[0]**2 + coordinates_rel[1]**2)
        
        sigma = self.sigma_res.clamp(min=self.min_sigma).view(1,-1)
    
        weights = torch.exp(-0.5 * ((d) / sigma** 2))

        weights = weights.view(*weights.shape,1,1,1,1)
        #weights = weights.view(*weights.shape,1,1,1,1)

        return weights


    def mult_weights_t(self, x, weights, mask=None, norm_seq=False):
        _, n, seq_out, seq_in, nh_in, nvw, n_phi, n_dist, n_sigma = weights.shape
        nv, np ,nc = x.shape[-3:]
        c_shape = (-1, n, seq_out, seq_in*nh_in, nv)

        #weights = weights.view(weights.shape[0], n, seq_out, seq_in*nh_in, nvw, n_phi, n_dist, n_sigma, 1, 1)
        weights = weights.view(weights.shape[0], n, seq_out, seq_in*nh_in, nvw, n_phi, n_dist, 1, n_sigma)
        if mask is not None:
            norm = (mask.view(*c_shape,1,1,1,1) == False)
            weights = weights*norm

            mask = mask.view(*c_shape)
            mask = mask.sum(dim=-2)==mask.shape[-2]
            mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_out, dim=-2)
        
        if norm_seq:
            weights = weights/(weights.sum(dim=[3], keepdim=True)+1e-20)

        x = x.view(*c_shape, 1, 1, np, nc)
        x = x * weights

        return x.sum(dim=3), mask

    def mult_weights_invt(self, x, weights, mask=None, norm_seq=False):
        _, n, seq_out, seq_in, nh_in, nvw, n_phi, n_dist, n_sigma = weights.shape
        nv, np ,nc = x.shape[3],x.shape[-2],x.shape[-1]
        c_shape = (-1, n, seq_out, seq_in*nh_in, nv)

        #weights = weights.view(weights.shape[0], n, seq_out, seq_in*nh_in, nvw, n_phi, n_dist, n_sigma, 1, 1)
        weights = weights.view(weights.shape[0], n, seq_out, seq_in*nh_in, nvw, n_phi, n_dist, 1, n_sigma)

        if mask is not None:
            norm = (mask.view(-1, n, seq_in*nh_in, 1, nv, 1, 1, 1, 1) == False)
            weights = weights*norm

            mask = mask.view(-1,n,seq_out,nv)
            mask = mask.sum(dim=-2)==mask.shape[-2]
            mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_in, dim=-2)

        if norm_seq:
            #weights = weights/(weights.sum(dim=[2,-3,-4,-5], keepdim=True)+1e-20)
            weights = weights/(weights.sum(dim=[2,-2,-3,-4], keepdim=True)+1e-20)

        x = x.unsqueeze(dim=3).squeeze(dim=-3)
        #x = x.view(x.shape[0], n, seq_in*nh_in, 1, nv, *self.n_params_no[:-2], np, nc)
        x = x * weights

        return x.sum(dim=2), mask

    def transform_(self, x, coordinates_rel, mask=None, emb=None):
        _, n, seq_ref, seq_in, nh_in = coordinates_rel[0].shape
        nv, np ,nc = x.shape[-3:]
        b = x.shape[0]

        weights = self.get_spatial_weights(coordinates_rel)

        if self.with_res:
            res_weights = self.get_spatial_weights_res(coordinates_rel)
            x_res = self.mult_weights_t(x, res_weights, mask=mask, norm_seq=True)[0]

        x = self.lin_layer(x)

        x, mask = self.mult_weights_t(x, weights, mask=mask, norm_seq=True)
        
        #if self.with_res:
        #    x = x-x_res

        #amplitudes = self.get_amplitudes_no(emb)
        #amplitudes = (1+self.gamma_res*amplitudes)
        #x = (x.unsqueeze(dim=-1) * (amplitudes.view(1,1,1,*amplitudes.shape))).mean(dim=-2)
        

        if len(self.sum_dims_params)>0:
            x = x.sum(dim=self.sum_dims_params, keepdim=True)/self.norm_factor # sum over NO params

        if self.with_res:
            x = x_res + x#self.gamma_res*x

        if mask is not None:
            x = x.masked_fill_(mask.view(*x.shape[:4],1,1,1,1), 0.0)

       # x = x.view(b, n, nv, *self.n_params_no[:-1], np, self.n_params_no[-1])
        x = x.view(b, n, nv, *self.n_params_no[:-2], 1, np, self.n_params_no[-1])

        if mask is not None:
            mask = mask.view(b,n,-1)

        return x, mask
    
    def inverse_transform_(self, x, coordinates_rel, mask=None, emb=None):

        b, n, seq_ref, seq_in, _ = coordinates_rel[0].shape
        b, n, seq_ref, nv, n_phi, n_dist, n_sigma, np, nc = x.shape
        c_shape = (-1, n, seq_ref, seq_in, nv)

        weights = self.get_spatial_weights(coordinates_rel)

        
        x = self.inv_lin_layer(x)

        x = x.transpose(-1,-2).reshape(b,n,seq_in,nv,*self.n_params_in[:3],self.n_params_out[-1],np).transpose(-2,-1)


        x, mask = self.mult_weights_invt(x, weights, mask=mask, norm_seq=True)

        #amplitudes = self.get_amplitudes_out(emb)

       # if self.with_res:
       # weights = weights.view(weights.shape[0], n, seq_ref, seq_in, 1, len(self.phis), len(self.dists), len(self.sigma), 1 ,1)
        
        #weights = weights/(weights.sum(dim=[-3,-4,-5], keepdim=True)+1e-20)

   #     amplitudes = self.get_amplitudes(emb, self.amplitudes_out)
   #     weights = weights * (amplitudes)#/x.shape[2]

        #x = (x * amplitudes)
   #     x = torch.matmul(x, weights)
        x = x.sum(dim=[-3, -4])
        
     #   if self.with_res:
      #      x = x + x_res
        x= x.view(b,-1, nv, np, self.n_params_inv_out)

        if mask is not None:
            mask = mask.view(b,-1,nv)
            x = x.masked_fill_(mask.unsqueeze(dim=-1).unsqueeze(dim=-1), 0.0)


        return x, mask