import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from ..icon_grids.grid_layer import RelativeCoordinateManager
from ..transformer.attention import ResLayer, AdaptiveLayerNorm

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


class no_res_layer(nn.Module):
    def __init__(self, model_dim,  x_dims_stat=None, res_block=False, layer_norm=False):
        super().__init__()

        if layer_norm: 
            self.ada_ln = AdaptiveLayerNorm([model_dim], model_dim)
        else:
            self.ada_ln = nn.Identity()

        if res_block:
            self.layer = ResLayer(model_dim, 
                                with_res=False, 
                                p_dropout=0)
        else:
            self.layer= nn.Linear(model_dim, model_dim)

        self.x_dims_stat = x_dims_stat
        self.model_dim = model_dim

    def forward(self, x):
        
        x_res = x

        b,n,_,nv,n_phi,n_dist,np,nc = x.shape
        x = x.permute(0,1,2,3,-2,-4,-3,-1)
        x = x.reshape(*x.shape[:5],*self.x_dims_stat,-1)
        x = self.ada_ln(x)
        x = self.layer(x)
        x = x.view(*x.shape[:5],n_phi,n_dist,-1)
        x = x.permute(0,1,2,3,-3,-2,-4,-1)

        x = x_res + x

        return x
        
    
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
                 non_linear_encode=False,
                 non_linear_decode=False,
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

        self.n_params_in = [n_phi, n_dist, n_sigma, n_amplitudes_in]
        self.n_params_out_na = [n_phi, n_dist, n_sigma, n_amplitudes_out]

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
        
      #  if avg_sigma:
      #      self.sum_dims_params.append(-2)

        self.n_params_inv_out = n_amplitudes_inv_out

        self.avg_phi, self.avg_dist, self.avg_sigma = avg_phi, avg_dist, avg_sigma
        grid_dist_out = self.nh_dist

        self.min_sigma = 1e-10
        self.min_var = 1e-10

        self.sd_activation = nn.Identity()#nn.Sigmoid()
        if pretrained_weights is None:
            phis = torch.linspace(-torch.pi, torch.pi, n_phi+1)[:-1]

            max_dist_fac = (grid_layer_no.global_level - grid_layer_in.global_level)
     
            dists = max_dist_fac*torch.arange(1,2*n_dist, 2)/(n_dist*2)

            sigma = torch.tensor(1/(2*n_dist*math.sqrt(2*math.log(2)))).view(-1)
            sigma_inv = torch.tensor(1/(2*n_dist*math.sqrt(2*math.log(2)))).view(-1)

            self.dist_unit = grid_layer_in.nh_dist

        else:
            phis = pretrained_weights['phis']
            dists = pretrained_weights['dists']
            sigma = pretrained_weights['sigma']

        self.phis =  nn.Parameter(phis, requires_grad=False)
        self.dists = nn.Parameter(dists, requires_grad=True)
        self.sigma = nn.Parameter(sigma, requires_grad=True)
        self.sigma_inv = nn.Parameter(sigma_inv, requires_grad=True)
        
        x_dims_stat = [n_param for n_param,avg_param in zip([n_phi, n_dist],[avg_phi,avg_dist]) if not avg_param]
        x_dims = [n_param for n_param,avg_param in zip([n_phi, n_dist],[avg_phi,avg_dist]) if avg_param]
        x_dims_inv = x_dims + [n_amplitudes_inv_out]
        x_dims = x_dims + [n_amplitudes_out]

        model_dim = int(torch.tensor(x_dims).prod())
        model_dim_inv = int(torch.tensor(x_dims_inv).prod())

        self.lin_layer =  nn.Linear(n_amplitudes_in, n_amplitudes_out, bias=False)
        if len(x_dims)>1:
            self.proc_layer = no_res_layer(model_dim, layer_norm=False, res_block=non_linear_encode, x_dims_stat=x_dims_stat)
        else:
            self.proc_layer = nn.Identity()

        self.lin_layer_inv = nn.Linear(n_amplitdues_inv_in, n_amplitudes_inv_out, bias=False)
        if len(x_dims)>1:
            self.proc_layer_inv = no_res_layer(model_dim_inv, layer_norm=False, res_block=non_linear_decode, x_dims_stat=x_dims_stat)
        else:
            self.proc_layer_inv = nn.Identity()
        
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
    
    def get_spatial_weights(self, coordinates_rel, sigma):

        #dists = F.softplus(self.dists)
        #dists = math.sqrt(2*math.log(2))*(self.sigma)*self.dists

        dists = self.dist_unit * self.dists#self.sd_activation(self.dists)

        mus_lon = torch.cos(self.phis).view(1,1,-1,1) * dists.view(1,1,1,-1)
        mus_lat = torch.sin(self.phis).view(1,1,-1,1) * dists.view(1,1,1,-1)

        dx = coordinates_rel[0].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-3) - mus_lon
        dy = coordinates_rel[1].unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-3) - mus_lat
        
        dx = dx.unsqueeze(dim=-1)
        dy = dy.unsqueeze(dim=-1)

        #sigma = F.softplus(self.sigma)
        sigma = self.dist_unit * self.sd_activation(sigma)
        sigma = sigma.clamp(min=self.min_sigma).view(1,-1)
    
        weights = torch.exp(-0.5 * ((dx**2 + dy**2) / sigma** 2))

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

        weights = self.get_spatial_weights(coordinates_rel, self.sigma)

        x, mask = self.mult_weights_t(x, weights, mask=mask, norm_seq=True)

        x = self.lin_layer(x)
        x = self.proc_layer(x)

        if len(self.sum_dims_params)>0:
            x = x.sum(dim=self.sum_dims_params, keepdim=True)/self.norm_factor # sum over NO params

        if mask is not None:
            x = x.masked_fill_(mask.view(*x.shape[:4],1,1,1,1), 0.0)

        x = x.view(b, n, nv, *self.n_params_no[:-2], 1, np, self.n_params_no[-1])

        if mask is not None:
            mask = mask.view(b,n,-1)

        return x, mask
    
    def inverse_transform_(self, x, coordinates_rel, mask=None, emb=None):

        b, n, seq_ref, seq_in, _ = coordinates_rel[0].shape
        b, n, seq_ref, nv, n_phi, n_dist, n_sigma, np, nc = x.shape
        c_shape = (-1, n, seq_ref, seq_in, nv)

        weights = self.get_spatial_weights(coordinates_rel, self.sigma_inv)

        x, mask = self.mult_weights_invt(x, weights, mask=mask, norm_seq=True)

        x = self.lin_layer_inv(x)

        x = (self.proc_layer_inv(x)).sum([-4,-3]) #+ x

        x= x.view(b,-1, nv, np, self.n_params_inv_out)

        if mask is not None:
            mask = mask.view(b,-1,nv)
            x = x.masked_fill_(mask.unsqueeze(dim=-1).unsqueeze(dim=-1), 0.0)


        return x, mask
    

class ReshapeAtt:
    def __init__(self, shape, param_att, cross_var=True):
        self.shape = shape
        self.param_att = param_att
        self.cross_var = cross_var

    def shape_to_att(self, x, mask=None):
        b,n,nv = x.shape[:3]

        if not self.cross_var:
            x = x.reshape(b,n*nv,1,*self.shape)

        b,n2,nv2 = x.shape[:3]

        if self.param_att == 0:
            x = x.reshape(b,n2,nv2*self.shape[0],-1)

        elif self.param_att==1:
            x = x.transpose(3,4).contiguous()
            x = x.view(b,n2,nv2*self.shape[1],-1)

        elif self.param_att==2:
            x = x.transpose(3,5).contiguous()
            x = x.view(b,n2,nv2*self.shape[2],-1)
        
        else:
            x = x.reshape(b,n2,nv2,-1)

        if mask is not None:
            mask = mask.view(b,n2,nv2,1).repeat_interleave(x.shape[2]//nv2, dim=-1)
            mask = mask.view(b,n2,-1)

        return x, mask
    
    def shape_to_x(self, x, nv_dim):
        b,n = x.shape[:2]

        if self.param_att == 0 or self.param_att is None:
            x = x.view(b,n,-1,*self.shape)

        elif self.param_att == 1:
            shape = (self.shape[1], self.shape[0], self.shape[2], self.shape[3], self.shape[4])
            x = x.view(b,n,-1,*shape).transpose(3,4)

        elif self.param_att == 2:
            shape = (self.shape[2], self.shape[1], self.shape[0], self.shape[3], self.shape[4])
            x = x.view(b,n,-1,*shape).transpose(3,5)
        
        x = x.view(b,-1,nv_dim,*self.shape)

        return x