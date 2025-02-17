import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from ..icon_grids.grid_layer import RelativeCoordinateManager, GridLayer
from ..transformer.attention import ResLayer, AdaptiveLayerNorm

from ...utils.helpers import check_get_missing_key
class NoLayer(nn.Module):

    def __init__(self, 
                 grid_layer_encode: GridLayer, 
                 grid_layer_no: GridLayer,
                 grid_layer_decode: GridLayer,
                 nh_in_encode=False,
                 nh_in_decode=True,
                 precompute_encode=True,
                 precompute_decode=True,
                 rotate_coord_system=True,
                 coord_system='polar') -> None: 
        
        super().__init__()
        self.grid_layer_encode = grid_layer_encode
        self.grid_layer_decode = grid_layer_decode
        self.grid_layer_no = grid_layer_no

        self.global_level_encode = int(grid_layer_encode.global_level)
        self.global_level_decode = int(grid_layer_decode.global_level)
        self.global_level_no = int(grid_layer_no.global_level)
        self.nh_in_encode = nh_in_encode
        self.nh_in_decode = nh_in_decode
        self.rotate_coord_system = rotate_coord_system
        self.coord_system = coord_system

        self.rel_coord_mngr_encode = RelativeCoordinateManager(
            grid_layer_encode,
            grid_layer_no,
            nh_in= nh_in_encode,
            precompute = precompute_encode,
            coord_system=coord_system,
            rotate_coord_system=rotate_coord_system,
            ref='out')
        
        self.rel_coord_mngr_decode = RelativeCoordinateManager(
            grid_layer_no,
            grid_layer_decode,
            nh_in= nh_in_decode,
            precompute = precompute_decode,
            coord_system=coord_system,
            rotate_coord_system=rotate_coord_system,
            ref='in')

        self.nh_dist = grid_layer_no.nh_dist

    def transform(self, x, coords_encode=None, coords_no=None, indices_sample=None, mask=None, emb=None):
        
        indices_encode  = indices_sample["indices_layers"][self.global_level_encode] if indices_sample is not None else None
        indices_no = indices_sample["indices_layers"][self.global_level_no] if indices_sample is not None else None

        coordinates_rel = self.rel_coord_mngr_encode(indices_in=indices_encode,
                                              indices_out=indices_no,
                                              coordinates_in=coords_encode,
                                              coordinates_out=coords_no,
                                              sample_dict=indices_sample)

        if self.nh_in_encode:
            x, mask = self.grid_layer_encode.get_nh(x, indices_encode, indices_sample, mask=mask)
        else:
            x = x.unsqueeze(dim=2)
            
        return self.transform_(x, coordinates_rel, mask=mask, emb=emb)
        

    def inverse_transform(self, x, coords_no=None, coords_decode=None, indices_sample=None,  mask=None, emb=None):
        
        indices_no = indices_sample["indices_layers"][self.global_level_no] if indices_sample is not None else None
        indices_decode = indices_sample["indices_layers"][self.global_level_decode] if indices_sample is not None else None

        #switch in and out, since out is the reference
        coordinates_rel = self.rel_coord_mngr_decode(indices_in=indices_no,
                                              indices_out=indices_decode,
                                              coordinates_in=coords_no,
                                              coordinates_out=coords_decode,
                                              sample_dict=indices_sample)

        self.global_level_decode
        if self.nh_in_decode:
            x, mask = self.grid_layer_no.get_nh(x, indices_no, indices_sample, mask=mask)
        else:
            x = x.unsqueeze(dim=2)
            if mask is not None:
                mask = mask.unsqueeze(dim=2)
        
        nh = x.shape[2]

        #seq_dim_in = 4**(self.global_level_no - self.global_level_decode)*nh


        return self.inverse_transform_(x, coordinates_rel, mask=mask, emb=emb)

    def transform_(self, x, coordinates_rel, mask=None, emb=None):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def inverse_transform_(self, x, coordinates_rel, mask=None, emb=None):
        raise NotImplementedError("This method should be implemented by subclasses.")


    
class polNormal_NoLayer(NoLayer):

    def __init__(self,
                 grid_layer_encode,
                 grid_layer_no,
                 grid_layer_decode,
                 n_phi=4,
                 n_dist=4,
                 n_sigma=1,
                 dist_learnable=False,
                 sigma_learnable=True,
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
                 coord_system='cartesian',
                 rotate_coord_system=rotate_coord_system)
        
        self.grid_layer_no = grid_layer_no
        self.n_no_tot = n_phi*n_dist*n_sigma

        self.n_params_no = [n_phi, n_dist, n_sigma]
                
        self.min_sigma = 1e-10
        self.min_var = 1e-10

        self.sd_activation = nn.Sigmoid()
  
        phis = torch.linspace(-torch.pi, torch.pi, n_phi+1)[:-1]

        max_dist_fac = (grid_layer_no.global_level - grid_layer_encode.global_level)
    
        dists = torch.arange(1,2*n_dist, 2)/(n_dist*2)

        if n_sigma>1:
            sigma = torch.arange(1,2*n_sigma, 2)/(n_sigma*2)
        else:
            sigma = torch.tensor(1/(2*n_dist*math.sqrt(2*math.log(2)))).view(-1)

        self.dist_unit = max_dist_fac*grid_layer_encode.nh_dist

        dists = -torch.log(1/dists-1)
        sigma = -torch.log(1/sigma-1)

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

        weights = self.get_spatial_weights(coordinates_rel, self.sigma)

        x, mask = self.mult_weights_t(x, weights, mask=mask)

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
    

def get_no_layer(type,
                 grid_layer_encode, 
                 grid_layer_no, 
                 grid_layer_decode, 
                 precompute_encode, 
                 precompute_decode, 
                 rotate_coordinate_system,
                 layer_settings,
                 ):
    n_params = check_get_missing_key(layer_settings, "n_params", ref=type)
    global_params_learnable = check_get_missing_key(layer_settings, "global_params_learnable", ref=type)

    if type == 'polNormal':

        assert len(n_params)==3, "len(n_params) should be equal to 3 for polNormal_NoLayer"
        assert len(global_params_learnable)==2, "len(global_params_learnable) should be equal to 2 for polNormal_NoLayer"

        no_layer = polNormal_NoLayer(
                grid_layer_encode,
                grid_layer_no,
                grid_layer_decode,
                n_phi=n_params[0],
                n_dist=n_params[1],
                n_sigma=n_params[2],
                dist_learnable=global_params_learnable[0],
                sigma_learnable=global_params_learnable[1],
                nh_in_encode=layer_settings.get("nh_in_encode",True), 
                nh_in_decode=layer_settings.get("nh_in_decode",True),
                precompute_encode=precompute_encode,
                precompute_decode=precompute_decode,
                rotate_coord_system=rotate_coordinate_system
            )
    return no_layer